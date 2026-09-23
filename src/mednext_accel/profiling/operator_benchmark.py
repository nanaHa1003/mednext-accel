"""Measure native and adaptive training operators through the same module interface."""

from __future__ import annotations

import copy

from .matrix import KernelCase


def benchmark_operator(
    case: KernelCase,
    *,
    compile_mode: str,
    seed: int,
    gradient_mask: tuple[bool, bool, bool],
    probe=None,
) -> dict[str, object]:
    """Validate and time the requested training mask, retaining only scalar evidence.

    The caller owns failure recording and subprocess isolation. All component
    comparisons use the timing mask, including when only one returned gradient
    is retained for validation. Other phases remain native in the candidate's
    isolated policy, so the result measures this case's dispatch decision.
    """
    import torch
    from torch import nn

    from ..ops.adaptive import (
        AdaptiveDepthwise3d,
        AdaptivePointwise3d,
        ModelOptimizationContext,
        _context_from_tensor,
    )
    from ..optimization.policy import parse_policy
    from ..optimization.policy_resolver import PolicyResolver
    from .runner import _ProbeRecord, _time_pair, _validate_pair

    if probe is None:
        probe = _ProbeRecord("operator.setup", {"seed": seed})

    if len(gradient_mask) != 3 or any(type(flag) is not bool for flag in gradient_mask):
        raise ValueError("gradient_mask must contain input, weight and bias booleans")
    key = case.key
    if key.dtype != "bfloat16":
        raise ValueError("integrated operator benchmarks require bfloat16")
    pointwise = key.family == "pointwise_conv3d"
    if key.kernel_size != (1 if pointwise else 3):
        raise ValueError("integrated operator benchmark is outside the validated kernel domain")
    if not pointwise and len(set(key.spatial_shape)) != 1:
        raise ValueError("integrated depthwise benchmarks require cubic spatial shapes")
    torch.manual_seed(seed)
    probe.result.update(
        benchmark_kind="integrated_operator",
        compile_mode=compile_mode,
        gradient_mask=tuple(gradient_mask),
        implementation=key.implementation,
        parameters=key.parameters,
    )
    probe.stage = "operator.initialization"
    constructor = nn.ConvTranspose3d if key.direction == "transpose" else nn.Conv3d
    native = constructor(
        key.in_channels,
        key.out_channels,
        key.kernel_size,
        stride=1 if key.direction == "regular" else 2,
        padding=key.kernel_size // 2,
        groups=1 if pointwise else key.in_channels,
        device="cuda",
        dtype=torch.float32,
    ).train()
    policy = parse_policy(
        {
            "version": 2,
            "kind": "mednext-accel-policy",
            "name": "integrated-candidate",
            "target": {"vendor": "nvidia"},
            "rules": [
                {
                    "id": "candidate",
                    "when": {"family": key.family, "direction": key.direction},
                    "use": {
                        key.phase: {
                            "implementation": key.implementation,
                            **({"parameters": dict(key.parameters)} if key.parameters else {}),
                        }
                    },
                    "confidence": "measured-exact-context",
                }
            ],
        }
    )
    wrapper = AdaptivePointwise3d if pointwise else AdaptiveDepthwise3d
    candidate = wrapper(
        copy.deepcopy(native),
        PolicyResolver(external=policy),
        ModelOptimizationContext("mednext_v1", "base", "none"),
    )
    for module in (native, candidate):
        module.weight.requires_grad_(gradient_mask[1])
        module.bias.requires_grad_(gradient_mask[2])
    probe.stage = "operator.allocation.input"
    sample = torch.randn(
        key.batch,
        key.in_channels,
        *key.spatial_shape,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=gradient_mask[0],
    )
    probe.stage = "operator.selection"
    decision = candidate.resolver.resolve(
        candidate.descriptor,
        _context_from_tensor(sample, candidate.model_context, training=True),
        key.phase,
    )
    if decision.implementation != key.implementation:
        raise ValueError(
            f"adaptive policy did not select {key.implementation!r}: "
            f"{decision.implementation!r} ({decision.guard_reason or decision.rule})"
        )
    output_shape = tuple(
        (size + 1) // 2
        if key.direction == "downsample"
        else 2 * size - 1
        if key.direction == "transpose"
        else size
        for size in key.spatial_shape
    )
    probe.stage = "operator.allocation.gradient"
    gradient = torch.randn(
        key.batch, key.out_channels, *output_shape, device="cuda", dtype=torch.bfloat16
    )
    names = tuple(
        name for name, needed in zip(("dX", "dW", "dB"), gradient_mask, strict=True) if needed
    )

    def forward(module):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return module(sample)

    probe.stage = "operator.compile"
    if compile_mode != "eager":
        # A group may contain more shapes than Dynamo's per-code cache limit.
        # Each experiment owns its compiled graphs; none are reused across cases.
        torch.compiler.reset()
        forward = torch.compile(forward, mode=compile_mode, fullgraph=True)

    def training(module):
        output = forward(module)
        targets = tuple(
            value for value in (sample, module.weight, module.bias) if value.requires_grad
        )
        gradients = torch.autograd.grad(output, targets, gradient) if targets else ()
        return output, gradients

    def component(module, name):
        # Grad mode remains enabled for forward validation: no-grad would bypass
        # the adaptive path. No graph or other gradient escapes this frame.
        if name == "output":
            return forward(module).detach().clone()
        _, gradients = training(module)
        # CUDA Graph output storage may be reused by the other module's graph.
        # Own only this component outside compilation until metrics finish.
        return gradients[names.index(name)].detach().clone()

    for name in ("output", *names):
        _validate_pair(
            probe,
            name,
            lambda name=name: component(native, name),
            lambda name=name: component(candidate, name),
        )
    probe.result["valid"] = probe.result["rejection_reason"] is None
    _time_pair(probe, lambda: training(native), lambda: training(candidate))
    return {**probe.result, "status": "ok"}
