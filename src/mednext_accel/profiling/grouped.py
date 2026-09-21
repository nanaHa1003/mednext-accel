"""Grouped kernel execution with parent-side failure isolation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from typing import Literal

from .matrix import KernelCase, KernelGroup

Invoke = Callable[[dict[str, object]], dict[str, object]]
AttemptCallback = Callable[[Literal["scheduled", "completed"], int, int], None]


def case_payload(case: KernelCase) -> dict[str, object]:
    """Serialize one kernel case for the child protocol."""

    return {"case_id": case.identifier, **asdict(case.key)}


def response_matches(group: KernelGroup, result: Mapping[str, object]) -> bool:
    """Return whether a successful response resolves exactly the requested cases."""

    items = result.get("results")
    if not isinstance(items, list):
        return False
    identifiers = [item.get("case_id") for item in items if isinstance(item, dict)]
    if len(identifiers) != len(items) or not all(
        isinstance(identifier, str) for identifier in identifiers
    ):
        return False
    expected = [case.identifier for case in group.cases]
    return len(identifiers) == len(set(identifiers)) and set(identifiers) == set(expected)


def run_group_with_bisection(
    group: KernelGroup,
    invoke: Invoke,
    on_attempt: AttemptCallback | None = None,
) -> dict[str, dict[str, object]]:
    """Run a group, bisecting infrastructure failures down to singleton cases."""

    def execute(current: KernelGroup) -> dict[str, dict[str, object]]:
        result = invoke(
            {"kind": "kernel_group", "cases": [case_payload(case) for case in current.cases]}
        )
        if result.get("status") == "ok" and not response_matches(current, result):
            result = {
                "status": "infrastructure_error",
                "message": "group response has missing or duplicate case identifiers",
            }
        terminal = result.get("status") == "ok" or len(current.cases) == 1
        if on_attempt is not None:
            if not terminal:
                on_attempt("scheduled", 2, 0)
            on_attempt("completed", 1, len(current.cases) if terminal else 0)
        if result.get("status") == "ok":
            items = result["results"]
            assert isinstance(items, list)
            return {str(item["case_id"]): item for item in items}
        if len(current.cases) == 1:
            case = current.cases[0]
            return {case.identifier: {"case_id": case.identifier, **result}}
        midpoint = len(current.cases) // 2
        left = replace(current, cases=current.cases[:midpoint])
        right = replace(current, cases=current.cases[midpoint:])
        return execute(left) | execute(right)

    return execute(group)
