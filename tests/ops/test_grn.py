from __future__ import annotations

import pytest
import torch

from mednext_accel.ops.grn import GlobalResponseNorm3d, global_response_norm3d_reference


def oracle(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    norm = torch.linalg.vector_norm(work, dim=(2, 3, 4), keepdim=True)
    ratio = norm / (norm.sum(dim=1, keepdim=True) + eps)
    return (work + gamma.to(work.dtype) * work * ratio + beta.to(work.dtype)).to(x.dtype)


def test_reference_grn_matches_independent_formula() -> None:
    module = GlobalResponseNorm3d(3).double()
    module.gamma.data.normal_()
    module.beta.data.normal_()
    x = torch.randn(2, 3, 2, 3, 4, dtype=torch.double)

    torch.testing.assert_close(module(x), oracle(x, module.gamma, module.beta))


def test_reference_grn_uses_channel_sum_and_zero_subgradient() -> None:
    module = GlobalResponseNorm3d(4).double()
    x = torch.zeros(2, 4, 3, 5, 7, dtype=torch.double, requires_grad=True)

    torch.testing.assert_close(module(x), oracle(x, module.gamma, module.beta))
    module(x).sum().backward()

    assert torch.isfinite(x.grad).all()


def test_reference_grn_gradcheck() -> None:
    module = GlobalResponseNorm3d(3).double()
    module.gamma.data.normal_()
    module.beta.data.normal_()
    x = torch.randn(2, 3, 2, 3, 4, dtype=torch.double, requires_grad=True)

    assert torch.autograd.gradcheck(module, (x,))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reference_grn_uses_fp32_work_for_low_precision_inputs(dtype: torch.dtype) -> None:
    module = GlobalResponseNorm3d(3)
    module.gamma.data.normal_()
    module.beta.data.normal_()
    x = torch.randn(2, 3, 2, 3, 4, dtype=dtype)

    output = module(x)

    assert output.dtype is dtype
    torch.testing.assert_close(output, oracle(x, module.gamma, module.beta))


@pytest.mark.parametrize("shape", [(3, 2, 3, 4), (2, 3, 2, 3)])
def test_reference_grn_rejects_non_ncdhw_tensors(shape: tuple[int, ...]) -> None:
    x = torch.randn(shape)
    gamma = torch.zeros(1, 3, 1, 1, 1)
    beta = torch.zeros(1, 3, 1, 1, 1)

    with pytest.raises(ValueError, match="NCDHW"):
        global_response_norm3d_reference(x, gamma, beta)


@pytest.mark.parametrize("channels", [0, -1])
def test_grn_rejects_nonpositive_channels(channels: int) -> None:
    with pytest.raises(ValueError, match="channels"):
        GlobalResponseNorm3d(channels)


@pytest.mark.parametrize("eps", [0.0, -1e-6])
def test_grn_rejects_nonpositive_epsilon(eps: float) -> None:
    with pytest.raises(ValueError, match="eps"):
        GlobalResponseNorm3d(3, eps=eps)


def test_grn_parameters_are_zero_initialized_with_broadcast_shape() -> None:
    module = GlobalResponseNorm3d(5)

    assert tuple(module.gamma.shape) == (1, 5, 1, 1, 1)
    assert tuple(module.beta.shape) == (1, 5, 1, 1, 1)
    assert torch.count_nonzero(module.gamma) == 0
    assert torch.count_nonzero(module.beta) == 0
