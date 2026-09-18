"""Run serial, isolated MedNeXt training comparisons on one visible GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys
import time


CASES = {
    'eager_ckpt': ['--checkpoint'],
    'eager': ['--no-checkpoint'],
    'autotune': ['--no-checkpoint', '--cudnn-benchmark'],
    'channels_last': ['--no-checkpoint', '--cudnn-benchmark', '--channels-last'],
    'compile': ['--no-checkpoint', '--cudnn-benchmark', '--compile'],
    'eager_expanded': ['--checkpoint-style', 'expanded'],
    'compile_expanded': [
        '--checkpoint-style', 'expanded', '--cudnn-benchmark', '--compile'],
    'compile_expanded_l0': [
        '--checkpoint-style', 'expanded', '--checkpoint-levels', '0',
        '--cudnn-benchmark', '--compile'],
    'compile_expanded_l01': [
        '--checkpoint-style', 'expanded', '--checkpoint-levels', '0', '1',
        '--cudnn-benchmark', '--compile'],
    'compile_expanded_l012': [
        '--checkpoint-style', 'expanded', '--checkpoint-levels', '0', '1', '2',
        '--cudnn-benchmark', '--compile'],
    'compile_ckpt': ['--checkpoint', '--cudnn-benchmark', '--compile'],
}

SOURCE_FILES = (
    'activation_checkpoint.py',
    'mednext.py',
    'profile_mednext.py',
    'run_profile_matrix.py',
    'summarize_profiles.py',
    'pointwise_gemm.py',
    'depthwise_split.py',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=['small', 'base', 'medium', 'large'], default='base')
    parser.add_argument('--shape', type=int, nargs='+', default=[1, 1, 128, 128, 128])
    parser.add_argument('--classes', type=int, nargs='+', default=[3, 8])
    parser.add_argument('--filters', type=int, default=32)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--deep-supervision', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--precisions', nargs='+', choices=['auto', 'fp32', 'fp16', 'bf16'], default=['auto'])
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    parser.add_argument('--include-fp32-baseline', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--pointwise-gemm', action='store_true')
    parser.add_argument('--depthwise-split', action='store_true')
    parser.add_argument('--depthwise-stride2-dx', action='store_true',
                        help='Opt in to the experimental stride-two downsample dX replacement')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--profile-steps', type=int, default=3,
                        help='Trace first repeat only; 0 disables traces')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--tf32', choices=['default', 'on', 'off'], default='default')
    parser.add_argument('--timeout', type=int, default=1800, help='Seconds per child, including compilation')
    parser.add_argument('--resume', action='store_true', help='Skip successful runs from an identical manifest')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without CUDA access or execution')
    parser.add_argument('--dry-run-cc', default='12.0', help='Simulated compute capability for dry-run only')
    args = parser.parse_args()
    if min(args.repeats, args.warmup, args.steps, args.timeout) < 1 or args.profile_steps < 0:
        parser.error('repeats/warmup/steps/timeout must be positive; profile-steps must be nonnegative')
    if len(args.shape) not in (4, 5) or any(v <= 0 for v in args.shape) or any(v % 16 for v in args.shape[2:]):
        parser.error('shape must be N C H W or N C D H W, with positive spatial multiples of 16')
    if any(c < 2 for c in args.classes):
        parser.error('class counts must be >= 2')
    if 'auto' in args.precisions and len(args.precisions) != 1:
        parser.error('auto cannot be combined with explicit precisions')
    here = Path(__file__).resolve().parent
    if args.dry_run:
        cc = tuple(int(v) for v in args.dry_run_cc.split('.'))
        environment = {'simulated_compute_capability': cc}
    else:
        import torch
        if not torch.cuda.is_available():
            parser.error('No GPU visible. Run on a GPU host with a CUDA-enabled PyTorch environment.')
        cc = torch.cuda.get_device_capability()
        environment = {'gpu': torch.cuda.get_device_name(), 'compute_capability': cc,
                       'torch': torch.__version__, 'cuda': torch.version.cuda,
                       'cudnn': torch.backends.cudnn.version(), 'python': platform.python_version(),
                       'hostname': platform.node(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
        try:
            environment['driver_inventory'] = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,driver_version,uuid', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            environment['driver_inventory'] = None
    precisions = (['bf16', 'fp16'] if cc[0] >= 8 else ['fp16']) if args.precisions == ['auto'] else args.precisions
    if 'bf16' in precisions and cc[0] < 8:
        parser.error('BF16 requested on hardware without native BF16 support; choose fp16')
    configs = [('fp32', 'eager_ckpt')] if args.include_fp32_baseline else []
    configs += [(precision, case) for precision in precisions for case in args.cases]
    configs = list(dict.fromkeys(configs))
    output = args.output.resolve()
    plan = []
    # Rotate configuration order between repeats to reduce a fixed ordering/thermal bias.
    jobs = [(classes, precision, case) for classes in dict.fromkeys(args.classes) for precision, case in configs]
    for repeat in range(args.repeats):
        ordered = jobs[repeat % len(jobs):] + jobs[:repeat % len(jobs)]
        for classes, precision, case in ordered:
            name = f'c{classes}_{precision}_{case}_r{repeat + 1}'
            command = [sys.executable, str(here / 'profile_mednext.py'),
                       '--variant', args.variant, '--shape', *map(str, args.shape),
                       '--classes', str(classes), '--filters', str(args.filters),
                       '--kernel-size', str(args.kernel_size), '--precision', precision,
                       '--deep-supervision' if args.deep_supervision else '--no-deep-supervision',
                       *CASES[case], '--warmup', str(args.warmup), '--steps', str(args.steps),
                       '--profile-steps', str(args.profile_steps if repeat == 0 else 0),
                       '--seed', str(args.seed), '--lr', str(args.lr), '--tf32', args.tf32,
                       '--output', str(output / name)]
            if args.pointwise_gemm:
                command.append('--pointwise-gemm')
            if args.depthwise_split:
                command.append('--depthwise-split')
            if args.depthwise_stride2_dx:
                command.append('--depthwise-stride2-dx')
            plan.append({'name': name, 'command': command})
    print(json.dumps(environment, indent=2), flush=True)
    print(f'{len(plan)} serial runs; precisions={precisions}; deep_supervision={args.deep_supervision}', flush=True)
    if args.dry_run:
        for job in plan:
            print(shlex.join(job['command']))
        return
    # Initialize/query CUDA only in the parent, never allocate model tensors there.
    environment = json.loads(json.dumps(environment))
    manifest = {'environment': environment, 'runs': plan,
                'source_sha256': {name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                                  for name in SOURCE_FILES}}
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        if not args.resume:
            parser.error('Output already contains a matrix; use a new output directory or --resume')
        if json.loads(manifest_path.read_text()) != manifest:
            parser.error('Resume manifest differs in commands, hardware/software environment, or source files')
    elif output.exists() and any(output.iterdir()):
        parser.error('Output directory is not empty and has no matching manifest')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    failures = []
    for index, job in enumerate(plan, 1):
        directory = output / job['name']
        directory.mkdir(exist_ok=True)
        status_file = directory / 'status.json'
        if args.resume and status_file.exists() and json.loads(status_file.read_text())['status'] == 'ok':
            if (directory / 'summary.json').exists():
                print(f'[{index}/{len(plan)}] skip {job["name"]}', flush=True)
                continue
        # Failed/retried children must not leave stale summaries or traces in the new result.
        for name in ['summary.json', 'trace.json', 'operators.txt', 'kernel_breakdown.json']:
            (directory / name).unlink(missing_ok=True)
        print(f'[{index}/{len(plan)}] {job["name"]}', flush=True)
        start = time.monotonic()
        status = {'status': 'running', 'command': job['command']}
        status_file.write_text(json.dumps(status, indent=2) + '\n')
        with (directory / 'run.log').open('w') as log:
            process = subprocess.Popen(job['command'], stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                code = process.wait(timeout=args.timeout)
                status['status'] = 'ok' if code == 0 else 'failed'
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
                status['status'] = 'timeout'
            except KeyboardInterrupt:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                status['status'] = 'interrupted'
                status_file.write_text(json.dumps(status, indent=2) + '\n')
                raise
        status.update(returncode=code, elapsed_seconds=time.monotonic() - start)
        summary_file = directory / 'summary.json'
        if status['status'] == 'ok':
            if not summary_file.exists() or not json.loads(summary_file.read_text()).get('valid_training_timing'):
                status['status'] = 'invalid'
        status_file.write_text(json.dumps(status, indent=2) + '\n')
        if status['status'] != 'ok':
            failures.append(job['name'])
            print(f'  {status["status"]}; see {directory / "run.log"}', flush=True)
        else:
            summary = json.loads(summary_file.read_text())
            print(f'  {summary["mean_wall_step_ms"]:.2f} ms/step', flush=True)
    if list(output.glob('*/summary.json')):
        subprocess.run([sys.executable, str(here / 'summarize_profiles.py'), str(output)], check=True)
    print(f'Finished: {len(plan) - len(failures)} successful/skipped, {len(failures)} failed.', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
