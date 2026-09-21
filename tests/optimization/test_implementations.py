import pytest

from mednext_accel.optimization.implementations import (
    ImplementationRegistry,
    ImplementationSpec,
    default_implementation_registry,
)


def test_registry_rejects_duplicate_identifiers() -> None:
    registry = ImplementationRegistry()
    spec = ImplementationSpec(
        identifier="reference", version=1, families=("pointwise_conv3d",),
        phases=("training", "inference", "export"), equivalence="exact",
        export_safe=True,
    )
    registry.register(spec)
    with pytest.raises(ValueError, match="reference"):
        registry.register(spec)


def test_approximate_implementation_requires_opt_in() -> None:
    spec = ImplementationSpec(
        identifier="gelu_tanh", version=1, families=("gelu",),
        phases=("inference",), equivalence="approximate", export_safe=True,
    )
    assert not spec.is_allowed(allow_approximate=False)
    assert spec.is_allowed(allow_approximate=True)


def test_registry_accepts_a_future_mednext_v2_operator_family() -> None:
    registry = ImplementationRegistry()
    registry.register(ImplementationSpec(
        identifier="reference_grn", version=1,
        families=("global_response_norm",),
        phases=("training", "inference", "export"), equivalence="exact",
        export_safe=True,
    ))
    assert registry.resolve_metadata("reference_grn").families == (
        "global_response_norm",
    )


def test_default_registry_has_no_backend_import_side_effects() -> None:
    assert default_implementation_registry().resolve_metadata(
        "triton_split_dw"
    ).identifier == "triton_split_dw"


def test_unknown_implementation_names_the_identifier() -> None:
    with pytest.raises(ValueError, match="missing_backend"):
        ImplementationRegistry().resolve_metadata("missing_backend")
