"""Summarize measured runs and attribute each GPU kernel once via trace External id.

Kernel totals come from instrumented traces, not the uninstrumented step timer.
GPU annotation ranges are deliberately excluded to avoid double counting.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def checkpoint_style_from_settings(settings):
    """Read new policy metadata while retaining old profiler results."""
    return settings.get(
        'effective_checkpoint_style',
        'block' if settings.get('checkpoint') else 'none',
    )


def checkpoint_levels_from_settings(settings):
    """Return hashable level metadata for current and legacy results."""
    levels = settings.get('effective_checkpoint_levels')
    if levels is None:
        return 'all'
    return ','.join(str(level) for level in levels)


def summarize(directory):
    summary = json.loads((directory / 'summary.json').read_text())
    settings = summary['settings']
    effective_checkpoint_style = checkpoint_style_from_settings(settings)
    effective_checkpoint_levels = checkpoint_levels_from_settings(settings)
    row = {'run': directory.name, 'gpu': summary['device'],
           'torch': summary['torch'], 'cuda': summary['cuda_build'], 'cudnn': summary['cudnn'],
           'hostname': summary.get('hostname'), 'driver_inventory': summary.get('driver_inventory'),
           'model_sha256': summary.get('model_sha256'),
           **{k: settings.get(k) for k in ['variant', 'classes', 'precision', 'checkpoint',
               'deep_supervision', 'channels_last', 'compile', 'cudnn_benchmark', 'filters',
               'kernel_size', 'lr', 'seed', 'warmup', 'steps']},
           'effective_checkpoint_style': effective_checkpoint_style,
           'effective_checkpoint_levels': effective_checkpoint_levels,
           'shape': str(settings['shape']),
           'cudnn_allow_tf32': summary['cudnn_allow_tf32'],
           'matmul_allow_tf32': summary['matmul_allow_tf32'],
           'valid_training_timing': summary.get('valid_training_timing', 'not_checked'),
           'step_ms': summary['mean_wall_step_ms'],
           'samples_per_second': summary['samples_per_second'],
           'peak_allocated_MiB': summary.get('peak_allocated_MiB'),
           **{f'{name}_ms': ms for name, ms in summary.get('median_gpu_ms', {}).items()}}
    status_file = directory / 'status.json'
    row['pointwise_gemm'] = settings.get('pointwise_gemm', False)
    row['pointwise_gemm_sha256'] = summary.get('pointwise_gemm_sha256')
    row['depthwise_split'] = settings.get('depthwise_split', False)
    row['depthwise_split_sha256'] = summary.get('depthwise_split_sha256')
    row['status'] = json.loads(status_file.read_text())['status'] if status_file.exists() else 'standalone'
    if not (directory / 'trace.json').exists():
        return row
    events = json.loads((directory / 'trace.json').read_text())['traceEvents']
    ops = {e['args']['External id']: e for e in events
           if e.get('cat') == 'cpu_op' and 'External id' in e.get('args', {})}
    grouped = defaultdict(float)
    kernels = defaultdict(float)
    count = summary['settings']['profile_steps']
    for event in events:
        if event.get('cat') != 'kernel':
            continue
        op = ops.get(event.get('args', {}).get('External id'), {})
        key = (op.get('name', 'unattributed'),
               json.dumps(op.get('args', {}).get('Input Dims', [])))
        grouped[key] += event['dur'] / (1000 * count)
        kernels[event['name']] += event['dur'] / (1000 * count)
    total = sum(kernels.values())
    breakdown = {
        'note': 'Instrumented kernel durations per step; excludes annotation ranges, memcpy, and idle time.',
        'total_kernel_ms_per_step': total,
        'operators_by_shape': [
            {'operator': name, 'input_shapes': json.loads(shapes),
             'kernel_ms_per_step': ms, 'percent_kernel_time': 100 * ms / total}
            for (name, shapes), ms in sorted(grouped.items(), key=lambda x: -x[1])
        ],
        'kernels': [{'name': name, 'ms_per_step': ms, 'percent_kernel_time': 100 * ms / total}
                    for name, ms in sorted(kernels.items(), key=lambda x: -x[1])],
    }
    # Name matching is explicitly an attribution proxy for compiled fused kernels.
    norm_ms = sum(ms for name, ms in kernels.items() if 'group_norm' in name)
    breakdown['compiled_groupnorm_name_matched'] = {
        'ms_per_step': norm_ms,
        'percent_kernel_time': 100 * norm_ms / total if total else 0,
        'note': 'Includes neighboring fused work; not isolated normalization cost. '
                'Kernel naming may differ between compiler versions. For eager use operator attribution.'}
    (directory / 'kernel_breakdown.json').write_text(json.dumps(breakdown, indent=2) + '\n')
    return row


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    keys = ['pointwise_gemm', 'pointwise_gemm_sha256', 'depthwise_split',
            'depthwise_split_sha256', 'gpu', 'hostname', 'driver_inventory',
            'torch', 'cuda', 'cudnn', 'model_sha256',
            'variant', 'shape', 'classes', 'precision', 'checkpoint',
            'effective_checkpoint_style', 'effective_checkpoint_levels',
            'deep_supervision',
            'channels_last', 'compile', 'cudnn_benchmark', 'filters', 'kernel_size',
            'lr', 'seed', 'warmup', 'steps', 'cudnn_allow_tf32', 'matmul_allow_tf32']
    groups = defaultdict(list)
    for row in rows:
        if row['valid_training_timing'] is not True or row['status'] not in ('ok', 'standalone'):
            continue
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, group in groups.items():
        times = [r['step_ms'] for r in group]
        memory = [r['peak_allocated_MiB'] for r in group if r['peak_allocated_MiB'] is not None]
        result.append({**dict(zip(keys, key)), 'repeats': len(times),
                       'median_step_ms': statistics.median(times),
                       'min_step_ms': min(times), 'max_step_ms': max(times),
                       'stdev_step_ms': statistics.stdev(times) if len(times) > 1 else 0,
                       'max_peak_allocated_MiB': max(memory) if memory else None})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    rows = [summarize(p.parent) for p in sorted(args.root.rglob('summary.json'))]
    if not rows:
        parser.error('No run summaries found')
    write_csv(args.root / 'comparison.csv', rows)
    write_csv(args.root / 'aggregate.csv', aggregate(rows))
    for row in rows:
        print(f"{row['run']}: {row['step_ms']:.2f} ms/step, "
              f"{row['peak_allocated_MiB']} MiB allocated")


if __name__ == '__main__':
    main()
