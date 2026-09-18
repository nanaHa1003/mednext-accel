"""Measure native stride-two depthwise Conv3d backward components.

This benchmark does not install a custom kernel.  It separates dX, dW, dB,
and the combined backward call so that a custom-kernel experiment has a measured
upper bound before implementation.
"""
import argparse
import json
from pathlib import Path

import torch
import triton

from depthwise_split import (depthwise_stride2_input_grad,
                             depthwise_transpose_weight_grad)


def component_call(grad_output, x, weight, channels, stride, transposed, mask):
    return torch.ops.aten.convolution_backward(
        grad_output, x, weight, [channels], [stride] * 3, [1] * 3,
        [1] * 3, transposed, [0] * 3, channels, mask)


def benchmark_case(kind, channels, input_size, dtype, warmup, rep):
    transposed = kind == 'transpose'
    x = torch.randn((1, channels, input_size, input_size, input_size),
                    device='cuda', dtype=dtype)
    weight = torch.randn((channels, 1, 3, 3, 3), device='cuda', dtype=dtype)
    if transposed:
        y = torch.nn.functional.conv_transpose3d(
            x, weight, stride=2, padding=1, groups=channels)
    else:
        y = torch.nn.functional.conv3d(
            x, weight, stride=2, padding=1, groups=channels)
    grad_output = torch.randn_like(y)
    calls = {
        'dX': lambda: component_call(grad_output, x, weight, channels, 2,
                                     transposed, [True, False, False]),
        'dW': lambda: component_call(grad_output, x, weight, channels, 2,
                                     transposed, [False, True, False]),
        'dB': lambda: component_call(grad_output, x, weight, channels, 2,
                                     transposed, [False, False, True]),
        'combined': lambda: component_call(grad_output, x, weight, channels, 2,
                                           transposed, [True, True, True]),
    }
    # Execute once before timing, including the empty-mask paths returned by ATen.
    for call in calls.values():
        call()
    timings = {
        name: triton.testing.do_bench(call, warmup=warmup, rep=rep)
        for name, call in calls.items()
    }
    custom = {}
    if transposed:
        custom_fn = lambda: depthwise_transpose_weight_grad(
            x, grad_output, splits=64, block=1024)
        custom['dW'] = triton.testing.do_bench(custom_fn, warmup=warmup, rep=rep)
    else:
        custom_fn = lambda: depthwise_stride2_input_grad(
            grad_output, weight, (input_size, input_size, input_size), block=256)
        custom['dX'] = triton.testing.do_bench(custom_fn, warmup=warmup, rep=rep)
    separate = timings['dX'] + timings['dW'] + timings['dB']
    return {
        'kind': kind,
        'transposed': transposed,
        'channels': channels,
        'input_shape': list(x.shape),
        'output_shape': list(y.shape),
        'kernel_size': 3,
        'stride': 2,
        'padding': 1,
        'ms': timings,
        'custom_ms': custom,
        'custom_speedup_vs_native': {
            name: timings[name] / value for name, value in custom.items()
        },
        'dW_share_of_combined_percent': 100 * timings['dW'] / timings['combined'],
        'dX_share_of_combined_percent': 100 * timings['dX'] / timings['combined'],
        'separate_over_combined_ratio': separate / timings['combined'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=['downsample', 'transpose', 'all'],
                        default='all')
    parser.add_argument('--channels', type=int)
    parser.add_argument('--input-size', type=int)
    parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='bf16')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--rep', type=int, default=200)
    args = parser.parse_args()
    if args.channels is not None and args.channels < 1:
        parser.error('--channels must be positive')
    if args.input_size is not None and args.input_size < 1:
        parser.error('--input-size must be positive')
    dtype = {'fp32': torch.float32, 'fp16': torch.float16,
             'bf16': torch.bfloat16}[args.dtype]
    defaults = {'downsample': (32, 128), 'transpose': (64, 64)}
    kinds = ['downsample', 'transpose'] if args.kind == 'all' else [args.kind]
    cases = []
    for kind in kinds:
        default_channels, default_size = defaults[kind]
        cases.append(benchmark_case(
            kind, args.channels or default_channels,
            args.input_size or default_size, dtype, args.warmup, args.rep))
    report = {
        'torch': torch.__version__,
        'triton': triton.__version__,
        'device': torch.cuda.get_device_name(),
        'dtype': args.dtype,
        'warmup': args.warmup,
        'rep': args.rep,
        'cases': cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
