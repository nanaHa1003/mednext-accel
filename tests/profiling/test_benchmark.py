import json
import sys

from mednext_accel.profiling import runner
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


def test_runner_round_trips_large_request_through_real_subprocess(monkeypatch) -> None:
    # Keep the actual child parser and subprocess transport, replacing only the GPU probe.
    child = (
        "from mednext_accel.profiling import runner; "
        "runner._child = lambda payload: {'status': 'ok', 'received': payload}; "
        "raise SystemExit(runner.main())"
    )

    def invoke_child(command, **kwargs):
        return run_json_subprocess([sys.executable, "-c", child, *command[3:]], **kwargs)

    monkeypatch.setattr(runner, "run_json_subprocess", invoke_child)
    payload = {"kind": "model", "optimization": {"measurements": ["x" * 262_144]}}

    result = runner._invoke(payload, timeout=30)

    assert result == {"status": "ok", "received": payload}


def test_runner_child_still_accepts_legacy_json_argument(monkeypatch, capsys) -> None:
    payload = {"kind": "model", "batch": 3}
    monkeypatch.setattr(runner, "_child", lambda value: {"status": "ok", "received": value})

    assert runner.main(["--child", json.dumps(payload)]) == 0

    assert json.loads(capsys.readouterr().out) == {"status": "ok", "received": payload}
