"""Synthetic training profiler; does not modify MedNeXt or its output interface.

Use the actual training shape/settings. CPU mode is only a harness smoke test.
The surrogate loss is mean cross entropy across returned supervision heads;
replace loss_fn with your real training loss for representative loss timings.
"""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

import mednext
from activation_checkpoint import CHECKPOINT_STYLES


def loss_fn(prediction, target, deep_supervision):
    if deep_supervision:
        return sum(F.cross_entropy(head, target)
                   for head in prediction.unbind(1)) / prediction.shape[1]
    return F.cross_entropy(prediction, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['small', 'base', 'medium', 'large'], required=True)
    parser.add_argument('--shape', type=int, nargs='+', required=True,
                        help='N C H W or N C D H W; spatial sizes must be multiples of 16')
    parser.add_argument('--classes', type=int, required=True)
    parser.add_argument('--filters', type=int, default=32)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--precision', choices=['fp32', 'bf16', 'fp16'], required=True)
    parser.add_argument('--deep-supervision', action=argparse.BooleanOptionalAction, required=True)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        '--checkpoint', dest='legacy_checkpoint', action='store_const', const=True)
    checkpoint_group.add_argument(
        '--no-checkpoint', dest='legacy_checkpoint', action='store_const', const=False)
    checkpoint_group.add_argument('--checkpoint-style', choices=CHECKPOINT_STYLES)
    parser.add_argument('--checkpoint-levels', type=int, nargs='+')
    parser.add_argument('--channels-last', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--pointwise-gemm', action='store_true',
                        help='Experimental GEMM replacement for eligible Conv3d pointwise layers')
    parser.add_argument('--depthwise-split', action='store_true',
                        help='Experimental depthwise dW/dX replacements')
    parser.add_argument('--depthwise-stride2-dx', action='store_true',
                        help='Opt in to the experimental stride-two downsample dX replacement')
    parser.add_argument('--cudnn-benchmark', action='store_true')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--tf32', choices=['default', 'on', 'off'], default='default')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--profile-steps', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('profile_results'))
    args = parser.parse_args()
    if args.checkpoint_style is None:
        args.effective_checkpoint_style = (
            'block' if args.legacy_checkpoint else 'none')
    else:
        args.effective_checkpoint_style = args.checkpoint_style
    if (args.checkpoint_levels is not None
            and args.effective_checkpoint_style != 'expanded'):
        parser.error(
            '--checkpoint-levels is only valid with --checkpoint-style expanded')
    # Retain the historical summary field for existing result consumers.
    args.checkpoint = args.effective_checkpoint_style == 'block'
    if len(args.shape) not in (4, 5) or any(v <= 0 for v in args.shape):
        parser.error('--shape must contain four or five positive dimensions')
    if any(v % 16 for v in args.shape[2:]):
        parser.error('spatial dimensions must be multiples of 16 for these factories')
    if min(args.warmup, args.steps) < 1 or args.profile_steps < 0:
        parser.error('warmup and steps must be positive; profile-steps must be nonnegative')
    if args.classes < 2 or args.filters < 1 or args.kernel_size < 1 or args.kernel_size % 2 == 0:
        parser.error('classes >= 2, filters >= 1, and a positive odd kernel size are required')
    cuda = args.device == 'cuda'
    if cuda and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Run on the GPU host; CPU timings cannot rank RTX bottlenecks.')
    if cuda and args.precision == 'bf16' and torch.cuda.get_device_capability()[0] < 8:
        parser.error('Native BF16 is unavailable on this GPU; use fp16 with scaling or fp32.')
    if not cuda and args.precision != 'fp32':
        parser.error('CPU smoke tests support fp32 only')
    torch.manual_seed(args.seed)
    if args.tf32 != 'default':
        torch.backends.cudnn.allow_tf32 = args.tf32 == 'on'
        torch.backends.cuda.matmul.allow_tf32 = args.tf32 == 'on'
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    model = getattr(mednext, f'mednext_{args.variant}')(
        spatial_dims=len(args.shape) - 2, in_channels=args.shape[1],
        out_channels=args.classes, filters=args.filters, kernel_size=args.kernel_size,
        deep_supervision=args.deep_supervision,
        checkpoint_style=args.effective_checkpoint_style,
        checkpoint_levels=(
            None if args.checkpoint_levels is None
            else tuple(args.checkpoint_levels)
        ),
    ).to(args.device).train()
    args.effective_checkpoint_levels = (
        None if model.checkpoint_levels is None
        else list(model.checkpoint_levels)
    )
    pointwise_replacements = 0
    if args.pointwise_gemm:
        from pointwise_gemm import replace_pointwise_convs
        pointwise_replacements = replace_pointwise_convs(model)
    depthwise_replacements = 0
    if args.depthwise_split:
        from depthwise_split import replace_highres_depthwise_convs
        depthwise_replacements = replace_highres_depthwise_convs(
            model, include_stride2_input_grad=args.depthwise_stride2_dx)
    x = torch.randn(args.shape, device=args.device)
    target = torch.randint(args.classes, (args.shape[0], *args.shape[2:]), device=args.device)
    if args.channels_last:
        fmt = torch.channels_last_3d if x.ndim == 5 else torch.channels_last
        model.to(memory_format=fmt)
        x = x.contiguous(memory_format=fmt)
    if args.compile:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer_updates = 0

    def count_update(*unused):
        nonlocal optimizer_updates
        optimizer_updates += 1

    optimizer.register_step_post_hook(count_update)
    scaler = torch.amp.GradScaler('cuda', enabled=cuda and args.precision == 'fp16')
    dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]

    def step(measure=False, annotate=False):
        events = [torch.cuda.Event(enable_timing=True) for _ in range(6)] if cuda and measure else []

        def mark(index):
            if events:
                events[index].record()

        def region(name):
            return record_function(name) if annotate else contextlib.nullcontext()

        mark(0)
        with region('train/zero_grad'):
            optimizer.zero_grad(set_to_none=True)
        mark(1)
        with torch.autocast(args.device, dtype=dtype, enabled=args.precision != 'fp32'):
            with region('train/forward'):
                prediction = model(x)
            mark(2)
            with region('train/loss'):
                loss = loss_fn(prediction, target, args.deep_supervision)
        mark(3)
        with region('train/backward'):
            scaler.scale(loss).backward()
        mark(4)
        with region('train/optimizer'):
            scaler.step(optimizer)
            scaler.update()
        mark(5)
        return events, loss.detach()

    for _ in range(args.warmup):
        step()
    updates_before_timing = optimizer_updates
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    records = []
    start = time.perf_counter()
    for _ in range(args.steps):
        events, last_loss = step(measure=True)
        records.append(events)
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_allocated = torch.cuda.max_memory_allocated() if cuda else None
    peak_reserved = torch.cuda.max_memory_reserved() if cuda else None
    finite_loss = torch.isfinite(last_loss).item()
    finite_gradients = all(torch.isfinite(p.grad).all().item()
                           for p in model.parameters() if p.grad is not None)
    measured_updates = optimizer_updates - updates_before_timing
    valid = finite_loss and finite_gradients and measured_updates == args.steps
    try:
        driver = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,uuid',
                                 '--format=csv,noheader'], capture_output=True,
                                text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        driver = None
    result = {
        'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'workload': 'synthetic resident inputs, AdamW, mean cross entropy across heads; no data loading',
        'torch': torch.__version__, 'cuda_build': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'device': torch.cuda.get_device_name() if cuda else 'CPU SMOKE TEST ONLY',
        'compute_capability': torch.cuda.get_device_capability() if cuda else None,
        'driver_inventory': driver,
        'hostname': platform.node(), 'python': platform.python_version(),
        'cpu_threads': torch.get_num_threads(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'model_sha256': hashlib.sha256(Path(mednext.__file__).read_bytes()).hexdigest(),
        'activation_checkpoint_sha256': hashlib.sha256(
            Path(__file__).with_name('activation_checkpoint.py').read_bytes()
        ).hexdigest(),
        'profiler_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'pointwise_replacements': pointwise_replacements,
        'depthwise_replacements': depthwise_replacements,
        'pointwise_gemm_sha256': (hashlib.sha256(Path(__file__).with_name('pointwise_gemm.py').read_bytes()).hexdigest()
                                  if args.pointwise_gemm else None),
        'depthwise_split_sha256': (hashlib.sha256(Path(__file__).with_name('depthwise_split.py').read_bytes()).hexdigest()
                                   if args.depthwise_split else None),
        'valid_training_timing': valid, 'finite_final_gradients': finite_gradients,
        'measured_optimizer_updates': measured_updates,
        'final_grad_scale': scaler.get_scale(),
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'mean_wall_step_ms': elapsed * 1000 / args.steps,
        'samples_per_second': args.steps * args.shape[0] / elapsed,
        'final_loss': last_loss.item() if finite_loss else None,
    }
    if cuda:
        result['peak_allocated_MiB'] = peak_allocated / 2**20
        result['peak_reserved_MiB'] = peak_reserved / 2**20
        phases = [('zero_grad', 0, 1), ('forward', 1, 2), ('loss', 2, 3),
                  ('backward', 3, 4), ('optimizer', 4, 5), ('full_step', 0, 5)]
        result['median_gpu_ms'] = {
            name: statistics.median(e[a].elapsed_time(e[b]) for e in records)
            for name, a, b in phases
        }
        times = [e[0].elapsed_time(e[5]) for e in records]
        result['gpu_step_ms'] = times
        result['gpu_step_stdev_ms'] = statistics.stdev(times) if len(times) > 1 else 0
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    if not valid:
        raise SystemExit('Invalid timing: nonfinite final loss/gradients or skipped optimizer updates. See summary.json.')
    if not args.profile_steps:
        return
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    # Profiling is separate from throughput measurement because instrumentation is expensive.
    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        for _ in range(args.profile_steps):
            step(annotate=True)
            prof.step()
        if cuda:
            torch.cuda.synchronize()
    table = prof.key_averages(group_by_input_shape=True).table(
        sort_by='self_cuda_time_total' if cuda else 'self_cpu_time_total', row_limit=40)
    (args.output / 'operators.txt').write_text(table)
    prof.export_chrome_trace(str(args.output / 'trace.json'))
    print(table)


if __name__ == '__main__':
    main()
