"""Compare CUDA outputs and all gradients to Conv3d; save numerical errors."""
import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from pointwise_gemm import replace_pointwise_convs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = []
    for dtype in (torch.float32, torch.bfloat16):
        for shape, cout in [((1, 32, 128, 128, 128), 64),
                            ((1, 64, 17, 17, 17), 32),
                            ((2, 32, 9, 11, 13), 64)]:
            for compiled in (False, True):
                ref = nn.Sequential(nn.Conv3d(shape[1], cout, 1)).cuda()
                candidate = copy.deepcopy(ref)
                assert replace_pointwise_convs(candidate) == 1
                params = tuple(candidate.parameters())
                if compiled:
                    candidate = torch.compile(candidate)
                x = torch.randn(shape, device='cuda', requires_grad=True)
                z = x.detach().clone().requires_grad_()
                with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
                    y, q = ref(x), candidate(z)
                assert y.dtype == q.dtype
                g = torch.randn_like(y)
                a = (y, *torch.autograd.grad(y, (x, *ref.parameters()), g))
                b = (q, *torch.autograd.grad(q, (z, *params), g))
                metrics = {}
                for name, u, v in zip(('output', 'dx', 'dw', 'db'), a, b):
                    u, v = u.detach().float(), v.detach().float()
                    diff = u - v
                    rel = (diff.norm() / u.norm().clamp_min(1e-12)).item()
                    metrics[name] = {'relative_l2': rel, 'max_abs': diff.abs().max().item()}
                    assert torch.isfinite(v).all() and rel < (0.01 if dtype == torch.bfloat16 else 1e-4), metrics
                results.append({'dtype': str(dtype), 'shape': shape, 'cout': cout,
                                'compiled': compiled, 'errors': metrics})
                print(results[-1], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
