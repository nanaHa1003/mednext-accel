"""Publishing must never expose a policy pointing to mutable or missing evidence."""

import hashlib
import json
from dataclasses import replace

import pytest
import yaml

from mednext_accel.optimization.policy_io import load_policy
from mednext_accel.profiling import api
from mednext_accel.profiling.campaign import campaign_to_primitive, load_campaign
from mednext_accel.profiling.evidence import (
    CampaignEvidence,
    EnvironmentEvidence,
    ProfilingEvidence,
)
from mednext_accel.profiling.synthesize import Measurement, synthesize_profile


def records():
    campaign = load_campaign({"workloads": [{"variant": "small", "spatial": [32] * 3}]})
    measurement = Measurement(
        "pointwise_conv3d",
        "regular",
        "training",
        "pointwise_gemm_per_sample",
        1,
        (32, 32, 32),
        4,
        8,
        "bfloat16",
        "none",
        2.0,
        1.0,
        200,
        100,
        True,
        benchmark_kind="integrated_operator",
        memory_measured=True,
    )
    evidence = ProfilingEvidence(
        EnvironmentEvidence({"gpu": {"sm": [12, 0]}, "software": {}}),
        CampaignEvidence(campaign_to_primitive(campaign)),
        kernel_measurements=(measurement,),
    )
    policy = synthesize_profile(
        (measurement,), name="sm120-local", sm=(12, 0), objective="balanced"
    )
    return policy, evidence


def publish(policy, evidence, path, **kwargs):
    return api.write_profiling_artifacts(policy, evidence, policy_path=path, **kwargs)


def test_persisted_evidence_hash_and_policy_reference_cover_exact_utf8_bytes(tmp_path):
    policy, evidence = records()
    artifacts = publish(policy, evidence, tmp_path / "sm120-local.policy.yaml")
    raw = artifacts.evidence_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert artifacts.evidence_path.name.startswith("sm120-local.")
    assert artifacts.evidence_path.name.endswith(f"-{digest[:12]}.evidence.json")
    assert json.loads(raw) == evidence.to_primitive()
    assert ProfilingEvidence.from_primitive(json.loads(raw)) == evidence
    saved = load_policy(artifacts.policy_path)
    assert saved.rules == policy.rules
    assert saved.evidence[-1].sha256 == digest
    assert saved.evidence[-1].file == artifacts.evidence_path.name
    assert "defaults" not in yaml.safe_load(artifacts.policy_path.read_text())


def test_repeated_save_keeps_previous_evidence_immutable(tmp_path):
    policy, evidence = records()
    first = publish(policy, evidence, tmp_path / "local.policy.yaml")
    original = first.evidence_path.read_bytes()
    second = publish(policy, evidence, first.policy_path)
    assert first.evidence_path != second.evidence_path
    assert first.evidence_path.read_bytes() == original == second.evidence_path.read_bytes()
    assert load_policy(second.policy_path).evidence[-1].file == second.evidence_path.name


def test_policy_replacement_failure_keeps_old_pair_and_only_orphans_new_evidence(
    tmp_path, monkeypatch
):
    policy, evidence = records()
    old = publish(policy, evidence, tmp_path / "local.policy.yaml")
    old_policy = old.policy_path.read_bytes()
    old_evidence = old.evidence_path.read_bytes()
    replacement = replace(evidence, execution={"kernel_case_count": 1})

    def reject_replace(source, destination):
        assert len(list(tmp_path.glob("*.evidence.json"))) == 2
        assert old.policy_path.read_bytes() == old_policy
        raise OSError("injected failure between evidence publication and policy replacement")

    monkeypatch.setattr(api.os, "replace", reject_replace)
    with pytest.raises(OSError, match="injected failure"):
        publish(policy, replacement, old.policy_path)
    assert old.policy_path.read_bytes() == old_policy
    assert old.evidence_path.read_bytes() == old_evidence
    assert len(list(tmp_path.iterdir())) == 3


def test_result_save_writes_both_artifacts_and_supports_separate_evidence_directory(tmp_path):
    policy, evidence = records()
    initial = publish(policy, evidence, tmp_path / "first.policy.yaml")
    result = api.ProfilingResult(policy, evidence, initial, evidence.environment.data)
    saved = result.save(
        policy_path=tmp_path / "second.policy.yaml", evidence_directory=tmp_path / "raw"
    )
    assert saved.policy_path == tmp_path / "second.policy.yaml"
    assert saved.evidence_path.parent == tmp_path / "raw"
    assert saved.evidence_path.exists()
    assert (
        load_policy(saved.policy_path).evidence[-1].sha256
        == hashlib.sha256(saved.evidence_path.read_bytes()).hexdigest()
    )
    assert result.artifacts == initial
    with pytest.raises(ValueError, match="evidence.*policy.*YAML"):
        load_policy(saved.evidence_path)


@pytest.mark.parametrize("accepted", [True, False])
def test_profile_publishes_only_accepted_positives_and_records_untested_fallthrough(
    tmp_path, monkeypatch, accepted
):
    from mednext_accel.profiling.runner import CampaignRun, ExecutionStatistics
    from mednext_accel.profiling.whole_model import compare_model_results

    policy, source = records()
    winner = source.kernel_measurements[0]
    loser = replace(winner, in_channels=8, candidate_ms=3.0)
    comparison = compare_model_results(
        "balanced",
        {"status": "ok", "step_ms": 10, "peak_bytes": 100},
        {"status": "ok", "step_ms": 8 if accepted else 12, "peak_bytes": 100},
        workload={},
        batch=1,
        seed=0,
        effective_policy=({"name": "sm120-local", "sha256": "a" * 64},),
    )
    run = CampaignRun(
        (winner, loser), ExecutionStatistics(2, 2, 1, 1), model_comparisons=(comparison,)
    )
    monkeypatch.setattr(api, "collect_environment", lambda: source.to_primitive()["environment"])
    monkeypatch.setattr(api, "execute_campaign", lambda campaign, progress=None: run)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    result = api.profile({"workloads": [{"variant": "small", "spatial": [32] * 3}]})
    implementations = [rule.use["training"].implementation for rule in result.policy.rules]
    assert implementations.count("reference") == 1
    assert implementations.count("pointwise_gemm_per_sample") == int(accepted)
    publication = result.evidence.to_primitive()["publication"]
    assert publication["policy_accepted"] is accepted
    assert publication["matches_tested_policy"] is accepted
    assert publication["bundled_fallthrough"] == (
        "tested" if accepted else "not-validated-by-model-comparison"
    )
    assert result.evidence.kernel_measurements == (winner, loser)
    assert result.evidence.model_comparisons[0].policy_accepted is accepted
    assert load_policy(result.artifacts.policy_path) == result.policy
    assert (
        ProfilingEvidence.from_primitive(json.loads(result.artifacts.evidence_path.read_bytes()))
        == result.evidence
    )


def test_no_model_comparison_does_not_publish_unchecked_positive_rules(tmp_path, monkeypatch):
    from mednext_accel.profiling.runner import CampaignRun, ExecutionStatistics

    _, source = records()
    run = CampaignRun(source.kernel_measurements, ExecutionStatistics(1, 1, 1, 0))
    monkeypatch.setattr(api, "collect_environment", lambda: source.to_primitive()["environment"])
    monkeypatch.setattr(api, "execute_campaign", lambda campaign, progress=None: run)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    result = api.profile()
    assert result.policy.rules == ()
    assert result.evidence.publication.policy_accepted is None
    assert result.evidence.publication.bundled_fallthrough == "not-validated-by-model-comparison"


def test_failed_evidence_write_never_replaces_existing_policy(tmp_path, monkeypatch):
    policy, evidence = records()
    initial = publish(policy, evidence, tmp_path / "local.policy.yaml")
    original = initial.policy_path.read_bytes()

    def fail_publish(source, destination):
        raise OSError("disk failure publishing evidence")

    monkeypatch.setattr(api.os, "link", fail_publish)
    with pytest.raises(OSError, match="disk failure"):
        publish(policy, evidence, initial.policy_path)
    assert initial.policy_path.read_bytes() == original
    assert len(list(tmp_path.iterdir())) == 2


def test_published_evidence_cannot_claim_tested_fallthrough_after_rejection(tmp_path, monkeypatch):
    from mednext_accel.profiling.evidence import PolicyPublicationEvidence
    from mednext_accel.profiling.whole_model import compare_model_results

    _, source = records()
    comparison = compare_model_results(
        "balanced",
        {"status": "ok", "step_ms": 10, "peak_bytes": 100},
        {"status": "ok", "step_ms": 12, "peak_bytes": 100},
        workload={},
        batch=1,
        seed=0,
    )
    evidence = replace(
        source,
        model_comparisons=(comparison,),
        publication=PolicyPublicationEvidence.from_comparisons((comparison,)),
    )
    data = evidence.to_primitive()
    data["publication"]["policy_accepted"] = True
    data["publication"]["matches_tested_policy"] = True
    data["publication"]["bundled_fallthrough"] = "tested"
    data["publication"]["reason"] = "accepted"
    with pytest.raises(ValueError, match="publication contradicts"):
        ProfilingEvidence.from_primitive(data)


def test_concurrent_policy_replacement_does_not_mix_result_with_other_runs_evidence(
    tmp_path, monkeypatch
):
    from mednext_accel.profiling.runner import CampaignRun, ExecutionStatistics

    policy, source = records()
    run = CampaignRun(source.kernel_measurements, ExecutionStatistics(1, 1, 1, 0))
    monkeypatch.setattr(api, "collect_environment", lambda: source.to_primitive()["environment"])
    monkeypatch.setattr(api, "execute_campaign", lambda campaign, progress=None: run)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    original_replace = api.os.replace
    competing = []

    def interleave(source_path, destination):
        original_replace(source_path, destination)
        if not competing:
            competing.append(True)
            competing.append(
                publish(policy, replace(source, execution={"other_run": 1}), destination)
            )

    monkeypatch.setattr(api.os, "replace", interleave)
    result = api.profile()
    assert result.policy.evidence[-1].file == result.artifacts.evidence_path.name
    assert (
        result.policy.evidence[-1].sha256
        == hashlib.sha256(result.artifacts.evidence_path.read_bytes()).hexdigest()
    )
    assert result.evidence.execution != {"other_run": 1}
    assert (
        load_policy(result.artifacts.policy_path).evidence[-1].file
        == competing[1].evidence_path.name
    )


def test_frozen_clock_advances_past_existing_candidates_without_overwriting(tmp_path, monkeypatch):
    policy, evidence = records()
    monkeypatch.setattr(api.time, "time_ns", lambda: 123)
    initial = publish(policy, evidence, tmp_path / "sm120-local.policy.yaml")
    raw = initial.evidence_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    blocked = [
        tmp_path / f"sm120-local.{stamp}-{digest[:12]}.evidence.json" for stamp in range(124, 128)
    ]
    for path in blocked:
        path.write_bytes(b"preexisting evidence must remain untouched")
    real_link = api.os.link
    attempts = []

    def checked_link(source, destination):
        # Fail immediately on the original infinite retry, without waiting for a timeout.
        assert destination not in attempts, "publication retried the same occupied candidate"
        attempts.append(destination)
        real_link(source, destination)

    monkeypatch.setattr(api.os, "link", checked_link)
    second = publish(policy, evidence, initial.policy_path)
    assert second.evidence_path.name == f"sm120-local.128-{digest[:12]}.evidence.json"
    assert len(attempts) == 6
    assert second.evidence_path.read_bytes() == raw == initial.evidence_path.read_bytes()
    assert all(
        path.read_bytes() == b"preexisting evidence must remain untouched" for path in blocked
    )
    assert load_policy(second.policy_path).evidence[-1].file == second.evidence_path.name
    assert not list(tmp_path.glob(".*"))


def test_concurrent_publishers_with_frozen_clock_use_distinct_complete_evidence(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, local

    policy, evidence = records()
    monkeypatch.setattr(api.time, "time_ns", lambda: 123)
    barrier = Barrier(4)
    state = local()
    real_link = api.os.link

    def synchronized_link(source, destination):
        if not hasattr(state, "attempts"):
            state.attempts = set()
            barrier.wait(timeout=10)
        assert destination not in state.attempts, "publication retried an occupied candidate"
        state.attempts.add(destination)
        real_link(source, destination)

    monkeypatch.setattr(api.os, "link", synchronized_link)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(publish, policy, evidence, tmp_path / "sm120-local.policy.yaml")
            for _ in range(4)
        ]
        results = [future.result(timeout=10) for future in futures]
    paths = {result.evidence_path for result in results}
    assert len(paths) == 4
    raw = next(iter(paths)).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert {path.name for path in paths} == {
        f"sm120-local.{stamp}-{digest[:12]}.evidence.json" for stamp in range(123, 127)
    }
    assert all(path.read_bytes() == raw for path in paths)
    assert json.loads(raw) == evidence.to_primitive()
    stored = load_policy(results[0].policy_path)
    assert stored.evidence[-1].sha256 == digest
    assert stored.evidence[-1].file in {path.name for path in paths}
    assert len(list(tmp_path.iterdir())) == 5


@pytest.mark.parametrize("suffix", [".yaml", ".yml", ".YAML", ".YML", ".yAmL", ".yMl"])
def test_supported_policy_suffixes_preserve_evidence_filename_and_loading(tmp_path, suffix):
    policy, evidence = records()
    artifacts = publish(policy, evidence, tmp_path / f"sm120-local.policy{suffix}")
    assert artifacts.evidence_path.name.startswith("sm120-local.")
    assert ".policy." not in artifacts.evidence_path.name
    assert load_policy(artifacts.policy_path).evidence[-1].file == artifacts.evidence_path.name
