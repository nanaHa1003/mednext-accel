"""Per-shape selection of MedNeXt convolution implementations.

The selector benchmarks one representative layer for every eligible shape,
then returns a policy that the existing replacement helpers can apply. It is
intended for one-time startup tuning on a fixed GPU/software/dtype combination.
"""
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def _bench(fn, warmup, repetitions):
    import triton
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=repetitions))


def _pointwise_bench(in_channels, out_channels, spatial, dtype, warmup, repetitions):
    x = torch.randn((1, in_channels, *spatial), device='cuda', dtype=dtype)
    weight = torch.randn((out_channels, in_channels, 1, 1, 1), device='cuda', dtype=dtype)
    bias = torch.randn((out_channels,), device='cuda', dtype=dtype)
    grad = torch.randn((1, out_channels, *spatial), device='cuda', dtype=dtype)

    def native():
        F.conv3d(x, weight, bias)
        torch.ops.aten.convolution_backward(
            grad, x, weight, [out_channels], [1] * 3, [0] * 3, [1] * 3,
            False, [0] * 3, 1, [True, True, True])

    def gemm():
        flat = x.flatten(2)[0]
        matrix = weight.flatten(1)
        matrix.t().mm(grad.flatten(2)[0])
        grad.flatten(2)[0].mm(flat.t())
        grad.flatten(2)[0].sum(dim=1)
        torch.addmm(bias[:, None], matrix, flat)

    native_ms = _bench(native, warmup, repetitions)
    gemm_ms = _bench(gemm, warmup, repetitions)
    return {'native_ms': native_ms, 'gemm_ms': gemm_ms,
            'gemm_speedup': native_ms / gemm_ms,
            'use_gemm': gemm_ms < native_ms}


def _depthwise_bench(kind, channels, spatial, dtype, warmup, repetitions):
    from depthwise_split import (depthwise_input_grad,
                                 depthwise_stride2_input_grad,
                                 depthwise_transpose_weight_grad,
                                 depthwise_weight_grad)
    x = torch.randn((1, channels, spatial, spatial, spatial), device='cuda', dtype=dtype)
    weight = torch.randn((channels, 1, 3, 3, 3), device='cuda', dtype=dtype)
    bias = torch.randn((channels,), device='cuda', dtype=dtype)
    if kind == 'regular':
        grad_shape = (spatial, spatial, spatial)
        forward = lambda: F.conv3d(x, weight, bias, padding=1, groups=channels)
    elif kind == 'transpose':
        grad_shape = (2 * spatial - 1,) * 3
        forward = lambda: F.conv_transpose3d(
            x, weight, bias, stride=2, padding=1, groups=channels)
    elif kind == 'downsample':
        grad_shape = ((spatial + 1) // 2,) * 3
        forward = lambda: F.conv3d(
            x, weight, bias, stride=2, padding=1, groups=channels)
    else:
        raise ValueError(kind)
    grad = torch.randn((1, channels, *grad_shape), device='cuda', dtype=dtype)

    def native():
        forward()
        torch.ops.aten.convolution_backward(
            grad, x, weight, [channels], [2 if kind != 'regular' else 1] * 3,
            [1] * 3, [1] * 3, kind == 'transpose', [0] * 3, channels,
            [True, True, True])

    if kind == 'regular':
        config = _REGULAR_CONFIGS[(channels, spatial)]

        def custom():
            forward()
            depthwise_input_grad(grad, weight, block=config[2])
            depthwise_weight_grad(x, grad, splits=config[0], block=config[1])
            grad.sum(dim=(0, 2, 3, 4))
    elif kind == 'transpose':
        config = _TRANSPOSE_CONFIGS[(channels, spatial)]

        def custom():
            forward()
            torch.ops.aten.convolution_backward(
                grad, x, weight, [channels], [2] * 3, [1] * 3, [1] * 3,
                True, [0] * 3, channels, [True, False, True])
            depthwise_transpose_weight_grad(x, grad, splits=config[0], block=config[1])
    else:
        def custom():
            forward()
            depthwise_stride2_input_grad(grad, weight, (spatial,) * 3, block=256)
            torch.ops.aten.convolution_backward(
                grad, x, weight, [channels], [2] * 3, [1] * 3, [1] * 3,
                False, [0] * 3, channels, [False, True, True])

    native_ms = _bench(native, warmup, repetitions)
    custom_ms = _bench(custom, warmup, repetitions)
    return {'native_ms': native_ms, 'custom_ms': custom_ms,
            'custom_speedup': native_ms / custom_ms,
            'use_custom': custom_ms < native_ms}


def _capture_shapes(model, input_shape, dtype):
    records = []
    handles = []

    def hook(name):
        def record(module, inputs):
            x = inputs[0]
            records.append((name, module, tuple(x.shape[2:])))
        return record

    for name, module in model.named_modules():
        if type(module) in (nn.Conv3d, nn.ConvTranspose3d):
            handles.append(module.register_forward_pre_hook(hook(name)))
    states = {module: module.training for module in model.modules()}
    model.eval()
    x = torch.randn(input_shape, device='cuda', dtype=dtype)
    with torch.inference_mode(), torch.autocast('cuda', dtype=dtype):
        model(x)
    for handle in handles:
        handle.remove()
    for module, training in states.items():
        module.train(training)
    return records


def _model_signature(records):
    signature = []
    for name, module, spatial in records:
        signature.append((name, type(module).__name__, module.in_channels,
                          module.out_channels, tuple(module.kernel_size),
                          tuple(module.stride), spatial))
    return signature


def benchmark_kernel_choices(model, input_shape, dtype=torch.bfloat16,
                             warmup=10, repetitions=50,
                             include_stride2_input_grad=False, cache_path=None):
    """Benchmark eligible implementation choices and return an application policy."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for kernel selection')
    try:
        from depthwise_split import _REGULAR_CONFIGS, _TRANSPOSE_CONFIGS
    except ImportError as exc:
        raise RuntimeError(
            'Depthwise selection requires a PyTorch build exposing '
            'torch.library.triton_op; install a current PyTorch release') from exc
    if next(model.parameters()).device.type != 'cuda':
        raise ValueError('model must already be on CUDA')
    if len(input_shape) != 5 or input_shape[0] != 1:
        raise ValueError('per-shape selection currently requires a batch-one 3D input')
    records = _capture_shapes(model, tuple(input_shape), dtype)
    signature = _model_signature(records)
    cache_key = json.dumps({
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(), 'dtype': str(dtype),
        'input_shape': tuple(input_shape), 'signature': signature,
        'source': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, sort_keys=True)
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            cache = json.loads(cache_path.read_text())
            if cache.get('cache_key') == cache_key:
                cache['cached'] = True
                return cache

    policy = {'pointwise_gemm': set(), 'depthwise_regular': set(),
              'depthwise_transpose': set(), 'depthwise_downsample': set()}
    measurements = []
    seen = set()
    for _, module, spatial in records:
        if type(module) is nn.Conv3d and module.kernel_size == (1, 1, 1) and module.groups == 1:
            key = ('pointwise', module.in_channels, module.out_channels, spatial)
            if key not in seen:
                seen.add(key)
                result = _pointwise_bench(module.in_channels, module.out_channels,
                                          spatial, dtype, warmup, repetitions)
                measurements.append({'kind': 'pointwise', 'shape': key[1:], **result})
                if result['use_gemm']:
                    policy['pointwise_gemm'].add((module.in_channels, module.out_channels, *spatial))
        elif type(module) is nn.Conv3d and module.groups == module.in_channels == module.out_channels:
            if module.stride == (1, 1, 1) and (module.in_channels, spatial[0]) in _REGULAR_CONFIGS:
                kind = 'regular'
            elif module.stride == (2, 2, 2) and include_stride2_input_grad and (module.in_channels, spatial[0]) in _REGULAR_CONFIGS:
                kind = 'downsample'
            else:
                continue
            key = (kind, module.in_channels, spatial[0])
            if key not in seen:
                seen.add(key)
                result = _depthwise_bench(kind, module.in_channels, spatial[0],
                                          dtype, warmup, repetitions)
                measurements.append({'kind': kind, 'shape': key[1:], **result})
                if result['use_custom']:
                    policy['depthwise_' + kind].add((module.in_channels, spatial[0]))
        elif type(module) is nn.ConvTranspose3d and module.groups == module.in_channels == module.out_channels:
            key = ('transpose', module.in_channels, spatial[0])
            if key not in seen and (module.in_channels, spatial[0]) in _TRANSPOSE_CONFIGS:
                seen.add(key)
                result = _depthwise_bench('transpose', module.in_channels, spatial[0],
                                          dtype, warmup, repetitions)
                measurements.append({'kind': 'transpose', 'shape': key[1:], **result})
                if result['use_custom']:
                    policy['depthwise_transpose'].add((module.in_channels, spatial[0]))
    policy = {key: sorted(value) for key, value in policy.items()}
    report = {'cache_key': cache_key, 'cached': False, 'policy': policy,
              'measurements': measurements, 'input_shape': list(input_shape),
              'dtype': str(dtype), 'warmup': warmup, 'repetitions': repetitions}
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(report, indent=2) + '\n')
    return report


def apply_kernel_policy(model, report, include_stride2_input_grad=False):
    """Apply a report from benchmark_kernel_choices to an existing model."""
    from depthwise_split import replace_highres_depthwise_convs
    from pointwise_gemm import replace_pointwise_convs
    policy = report['policy'] if 'policy' in report else report
    pointwise = set(tuple(item) for item in policy.get('pointwise_gemm', []))
    replace_pointwise_convs(model, selected_shapes=pointwise)
    replace_highres_depthwise_convs(
        model, include_stride2_input_grad=include_stride2_input_grad,
        policy={key: set(tuple(item) for item in value)
                for key, value in policy.items() if key.startswith('depthwise_')})
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='JSON policy/cache output path')
    parser.add_argument('--variant', choices=['small', 'base', 'medium', 'large'],
                        default='base')
    parser.add_argument('--shape', type=int, nargs=5,
                        default=[1, 1, 128, 128, 128])
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--filters', type=int, default=32)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--dtype', choices=['fp32', 'bf16', 'fp16'], default='bf16')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repetitions', type=int, default=50)
    parser.add_argument('--include-stride2-dx', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('CUDA is required; run this command on a GPU host')
    dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16,
             'fp16': torch.float16}[args.dtype]
    import mednext
    model = getattr(mednext, f'mednext_{args.variant}')(
        spatial_dims=3, in_channels=args.shape[1], out_channels=args.classes,
        filters=args.filters, kernel_size=args.kernel_size,
        deep_supervision=True).cuda()
    report = benchmark_kernel_choices(
        model, tuple(args.shape), dtype=dtype, warmup=args.warmup,
        repetitions=args.repetitions,
        include_stride2_input_grad=args.include_stride2_dx,
        cache_path=args.output)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
