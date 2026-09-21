import sys

from mednext_accel.profiling.benchmark import run_json_subprocess


def test_subprocess_records_success_and_structured_failure(tmp_path) -> None:
    success = run_json_subprocess(
        [sys.executable, "-c", 'import json; print(json.dumps({"status":"ok","x":1}))'],
        timeout=5,
    )
    assert success.status == "ok"
    assert success.payload["x"] == 1

    failed = run_json_subprocess(
        [sys.executable, "-c", 'import sys; print("boom", file=sys.stderr); sys.exit(2)'],
        timeout=5,
    )
    assert failed.status == "infrastructure_error"
    assert "boom" in failed.message
