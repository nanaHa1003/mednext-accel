"""Tune the split-reduction dW kernel against ATen for the measured hotspot."""
import argparse
import json
from pathlib import Path

import torch
import triton

from depthwise_split import depthwise_weight_grad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='bf16')
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--spatial-size', type=int, default=128)
    args = parser.parse_args()
    if args.kernel_size < 1 or args.kernel_size % 2 == 0:
        parser.error('--kernel-size must be a positive odd integer')
    if args.channels < 1 or args.spatial_size < 1:
        parser.error('--channels and --spatial-size must be positive')
    dtype = {'fp32': torch.float32, 'fp16': torch.float16,
             'bf16': torch.bfloat16}[args.dtype]
    torch.manual_seed(23)
    channels = args.channels
    shape = (1, channels, args.spatial_size, args.spatial_size,
             args.spatial_size)
    x = torch.randn(shape, device='cuda', dtype=dtype)
    grad_output = torch.randn_like(x)
    kernel_size = args.kernel_size
    padding = kernel_size // 2
    weight = torch.randn(channels, 1, kernel_size, kernel_size, kernel_size,
                         device='cuda', dtype=dtype)

    def native():
        return torch.ops.aten.convolution_backward(
            grad_output, x, weight, None, [1, 1, 1], [padding] * 3,
            [1, 1, 1], False, [0, 0, 0], channels,
            [False, True, False])[1]

    reference = native()
    results = [{'implementation': 'aten', 'splits': None, 'block': None,
                'ms': triton.testing.do_bench(native, warmup=100, rep=300)}]
    for splits in (4, 8, 16, 32, 64, 128):
        for block in (128, 256, 512, 1024):
            fn = lambda: depthwise_weight_grad(
                x, grad_output, splits, block, kernel_size=kernel_size)
            output = fn()
            rel = ((output.float() - reference.float()).norm() /
                   reference.float().norm()).item()
            assert rel < (.01 if dtype != torch.float32 else 1e-4), rel
            results.append({'implementation': 'triton', 'splits': splits,
                            'block': block,
                            'ms': triton.testing.do_bench(fn, warmup=100, rep=300),
                            'relative_l2_vs_aten': rel})
            print(results[-1], flush=True)
    best = min(results[1:], key=lambda row: row['ms'])
    report = {'torch': torch.__version__, 'triton': triton.__version__,
              'device': torch.cuda.get_device_name(), 'dtype': args.dtype,
              'shape': shape, 'kernel_size': kernel_size,
              'results': results, 'best': best,
              'speedup_vs_aten': results[0]['ms'] / best['ms']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
