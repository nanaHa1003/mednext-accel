import json

import pytest
import torch

from mednext_accel.profiling import validation


def test_component_validation_accepts_reduction_noise_near_zero():
    # A harmless local cancellation difference fails elementwise allclose.
    expected = {"output": torch.tensor([1.0, 2.0]), "dW": torch.tensor([100.0, 0.0])}
    actual = {"output": expected["output"].clone(), "dW": torch.tensor([100.0, 0.03])}
    assert not torch.allclose(actual["dW"], expected["dW"], rtol=0.02, atol=0.02)

    result = validation.validate_components(actual, expected)

    assert result["valid"] is True
    assert result["validator"] == "component-relative-l2-v1"
    assert result["rejection_reason"] is None
    assert result["validation_metrics"]["output"] == {
        "finite": True,
        "relative_l2": 0.0,
        "max_absolute": 0.0,
    }
    assert result["validation_metrics"]["dW"]["relative_l2"] == pytest.approx(0.0003)
    assert result["validation_metrics"]["dW"]["max_absolute"] == pytest.approx(0.03)


@pytest.mark.parametrize("component", ["output", "dX", "dW", "dB"])
def test_component_validation_rejects_corruption_independently(component):
    expected = {name: torch.ones(10) for name in ("output", "dX", "dW", "dB")}
    actual = {name: value.clone() for name, value in expected.items()}
    actual[component] *= 1.05

    result = validation.validate_components(actual, expected)

    assert result["valid"] is False
    assert result["rejection_reason"].startswith(f"{component}:")
    assert result["validation_metrics"][component]["relative_l2"] == pytest.approx(0.05)
    assert len(result["validation_metrics"]) == 4


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("side", ["actual", "expected"])
def test_component_validation_rejects_nonfinite_and_emits_strict_json(bad_value, side):
    expected = {"dW": torch.ones(2)}
    actual = {"dW": torch.ones(2)}
    (actual if side == "actual" else expected)["dW"][0] = bad_value

    result = validation.validate_components(actual, expected)

    assert result["valid"] is False
    assert result["rejection_reason"].startswith("dW:")
    assert "non-finite" in result["rejection_reason"]
    assert result["validation_metrics"]["dW"] == {
        "finite": False,
        "relative_l2": None,
        "max_absolute": None,
    }
    json.dumps(result, allow_nan=False)


def test_component_validation_handles_zero_reference_and_deterministic_reason():
    expected = {"dX": torch.zeros(2), "dB": torch.ones(2)}
    actual = {"dX": torch.ones(2), "dB": torch.zeros(2)}

    result = validation.validate_components(actual, expected)
    reordered = validation.validate_components(dict(reversed(actual.items())), expected)

    assert result == reordered
    assert result["valid"] is False
    assert result["rejection_reason"].startswith("dB:")
    assert result["validation_metrics"]["dX"]["relative_l2"] > 1e12
    assert validation.validate_components({"dX": torch.zeros(2)}, {"dX": torch.zeros(2)})["valid"]


def test_component_validation_preserves_strict_two_percent_boundary():
    expected = {"dW": torch.tensor([100.0])}
    assert validation.validate_components({"dW": torch.tensor([101.0])}, expected)["valid"]
    assert not validation.validate_components({"dW": torch.tensor([103.0])}, expected)["valid"]
