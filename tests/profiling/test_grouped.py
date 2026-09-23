from dataclasses import replace

import pytest

from mednext_accel.profiling.grouped import case_payload, run_group_with_bisection
from mednext_accel.profiling.matrix import KernelCase, KernelCaseKey, KernelGroup


def make_case(out_channels: int) -> KernelCase:
    return KernelCase(
        KernelCaseKey(
            family="pointwise_conv3d",
            direction="regular",
            phase="training",
            batch=1,
            spatial_shape=(16, 16, 16),
            in_channels=8,
            out_channels=out_channels,
            kernel_size=1,
            dtype="bfloat16",
            implementation="pointwise_gemm_per_sample",
            parameters=(),
        )
    )


def ok_results(payload):
    return {
        "status": "ok",
        "results": [
            {"case_id": item["case_id"], "status": "ok", "valid": True} for item in payload["cases"]
        ],
    }


def test_successful_group_uses_one_child() -> None:
    group = KernelGroup("pointwise", 1, tuple(make_case(value) for value in (8, 16, 32)))
    calls = []

    def invoke(payload):
        calls.append(payload)
        return ok_results(payload)

    results = run_group_with_bisection(group, invoke)
    assert len(calls) == 1
    assert len(results) == 3


def test_failed_group_bisects_until_the_bad_case_is_isolated() -> None:
    cases = tuple(make_case(value) for value in (8, 16, 32))
    group = KernelGroup("pointwise", 1, cases)
    bad_id = cases[1].identifier
    calls = []
    attempt_events = []

    def invoke(payload):
        calls.append(tuple(item["case_id"] for item in payload["cases"]))
        if any(item["case_id"] == bad_id for item in payload["cases"]):
            return {"status": "infrastructure_error", "message": "child crashed"}
        return ok_results(payload)

    results = run_group_with_bisection(
        group,
        invoke,
        on_attempt=lambda event, physical, resolved: attempt_events.append(
            (event, physical, resolved)
        ),
    )
    assert results[bad_id]["status"] == "infrastructure_error"
    assert results[cases[0].identifier]["status"] == "ok"
    assert results[cases[2].identifier]["status"] == "ok"
    assert len(calls) == 5
    assert sum(item[1] for item in attempt_events if item[0] == "scheduled") == 4
    assert sum(item[1] for item in attempt_events if item[0] == "completed") == 5
    assert sum(item[2] for item in attempt_events if item[0] == "completed") == 3
    assert attempt_events == [
        ("scheduled", 2, 0),
        ("completed", 1, 0),
        ("completed", 1, 1),
        ("scheduled", 2, 0),
        ("completed", 1, 0),
        ("completed", 1, 1),
        ("completed", 1, 1),
    ]


@pytest.mark.parametrize("malformation", ["missing", "extra", "duplicate"])
def test_malformed_success_response_is_bisected(malformation: str) -> None:
    cases = tuple(make_case(value) for value in (8, 16))
    group = KernelGroup("pointwise", 1, cases)
    calls = []

    def invoke(payload):
        calls.append(tuple(item["case_id"] for item in payload["cases"]))
        if len(payload["cases"]) == 1:
            return ok_results(payload)
        result = ok_results(payload)
        if malformation == "missing":
            result["results"] = result["results"][:1]
        elif malformation == "extra":
            result["results"].append({"case_id": "unexpected", "status": "ok", "valid": True})
        else:
            result["results"][1] = result["results"][0]
        return result

    results = run_group_with_bisection(group, invoke)

    assert calls == [
        (cases[0].identifier, cases[1].identifier),
        (cases[0].identifier,),
        (cases[1].identifier,),
    ]
    assert set(results) == {cases[0].identifier, cases[1].identifier}
    assert all(item["status"] == "ok" for item in results.values())


def test_unhashable_case_identifier_is_bisected_and_accounted_for() -> None:
    cases = tuple(make_case(value) for value in (8, 16, 32))
    group = KernelGroup("pointwise", 1, cases)
    bad_id = cases[1].identifier
    calls = []
    attempt_events = []

    def invoke(payload):
        calls.append(tuple(item["case_id"] for item in payload["cases"]))
        result = ok_results(payload)
        for item in result["results"]:
            if item["case_id"] == bad_id:
                item["case_id"] = []
        return result

    results = run_group_with_bisection(
        group,
        invoke,
        on_attempt=lambda event, physical, resolved: attempt_events.append(
            (event, physical, resolved)
        ),
    )

    assert results[cases[0].identifier]["status"] == "ok"
    assert results[bad_id]["status"] == "infrastructure_error"
    assert results[cases[2].identifier]["status"] == "ok"
    assert len(calls) == 5
    assert sum(item[1] for item in attempt_events if item[0] == "scheduled") == 4
    assert sum(item[1] for item in attempt_events if item[0] == "completed") == 5
    assert sum(item[2] for item in attempt_events if item[0] == "completed") == 3


def test_case_seeds_are_stable_across_order_bisection_and_failures():
    cases = tuple(make_case(value) for value in (8, 16, 32))
    observed = {}

    def invoke(payload):
        for item in payload["cases"]:
            observed.setdefault(item["case_id"], set()).add(item["seed"])
        if len(payload["cases"]) > 1:
            return {"status": "timeout"}
        if payload["cases"][0]["case_id"] == cases[1].identifier:
            return {"status": "error", "message": "isolated failure"}
        return ok_results(payload)

    first = run_group_with_bisection(KernelGroup("pointwise", 1, cases), invoke, seed=37)
    reverse = run_group_with_bisection(
        KernelGroup("pointwise", 1, tuple(reversed(cases))), invoke, seed=37
    )
    seeds = [first[case.identifier]["seed"] for case in cases]
    assert len(set(seeds)) == 3, "different case identities need different random inputs"
    assert all(0 <= seed < 2**63 for seed in seeds)
    assert all(len(values) == 1 for values in observed.values())
    assert first == reverse
    assert first[cases[1].identifier]["status"] == "error"
    assert all(
        case_payload(case, seed=38)["seed"] != seed for case, seed in zip(cases, seeds, strict=True)
    )


@pytest.mark.parametrize(
    "alternative", [{"implementation": "reference"}, {"parameters": (("tile", 32),)}]
)
def test_candidate_alternatives_share_inputs_but_keep_distinct_response_ids(alternative):
    case = make_case(16)
    other = KernelCase(replace(case.key, **alternative))
    assert case_payload(case, seed=37)["seed"] == case_payload(other, seed=37)["seed"]
    assert case.comparison_identifier == other.comparison_identifier
    assert case.identifier != other.identifier
    results = run_group_with_bisection(
        KernelGroup("pointwise", 1, (case, other)), ok_results, seed=37
    )
    assert set(results) == {case.identifier, other.identifier}


@pytest.mark.parametrize(
    "context",
    [
        {"spatial_shape": (8, 16, 16)},
        {"in_channels": 16},
        {"out_channels": 32},
        {"kernel_size": 3},
        {"phase": "backward_input"},
        {"dtype": "float32"},
        {"batch": 2},
        {"family": "depthwise_conv3d"},
        {"direction": "downsample"},
    ],
)
def test_different_comparison_contexts_receive_distinct_inputs(context):
    case = make_case(16)
    other = KernelCase(replace(case.key, **context))
    assert case_payload(case, seed=37)["seed"] != case_payload(other, seed=37)["seed"]
    assert case.comparison_identifier != other.comparison_identifier
