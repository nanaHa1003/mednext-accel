import copy
import json
from dataclasses import FrozenInstanceError

import pytest
import yaml

from mednext_accel.optimization.policy import parse_policy as parse


def document(**updates):
    value = {
        "version": 2,
        "kind": "mednext-accel-policy",
        "name": "test-policy",
        "target": {"vendor": "nvidia", "sm": [8, 9]},
        "scope": {"dtype": "bfloat16", "batch": {"min": 1, "max": 12}},
        "rules": [
            {
                "id": "regular-dw",
                "when": {"family": "depthwise_conv3d", "direction": "regular"},
                "use": {
                    "backward_weight": {"implementation": "triton_split_dw", "parameters": "auto"}
                },
                "confidence": "inferred-same-sm",
            }
        ],
    }
    value.update(updates)
    return value


def test_scope_normalization_and_immutability():
    source = document()
    source["rules"][0]["when"].update(batch={"min": 4}, channels=[128, 128])
    policy = parse(source)
    source["rules"][0]["when"]["channels"][0] = 7
    rule = policy.rules[0]
    assert rule.when["dtype"] == "bfloat16"
    assert rule.when["batch"].minimum == 4
    assert rule.when["batch"].maximum == 12
    assert rule.when["channels"] == (128, 128)
    assert policy.target_sm == (8, 9)
    with pytest.raises(TypeError):
        rule.when["dtype"] = "float32"
    with pytest.raises(FrozenInstanceError):
        policy.name = "mutated"


@pytest.mark.parametrize(
    "condition,value",
    [
        ("work", 1572864),
        ("in_channels", 48),
        ("out_channels", {"min": 32, "max": 128}),
        ("reduction_work", {"min": 4096}),
        ("spatial_volume", {"min": 512, "max": 262144}),
        ("total_vram_gib", {"min": 31.5, "max": 96}),
    ],
)
def test_numeric_conditions_support_exact_equality_and_ranges(condition, value):
    source = document()
    source["rules"][0]["when"][condition] = value
    result = parse(source).rules[0].when[condition]
    if isinstance(value, int):
        assert result.minimum == result.maximum == value
    else:
        assert result.minimum == value.get("min")
        assert result.maximum == value.get("max")


@pytest.mark.parametrize(
    "when",
    [
        {"dtype": "float32"},
        {"batch": {"min": 13}},
        {"batch": {"min": 0}},
        {"batch": {"max": 14}},
    ],
)
def test_rule_cannot_contradict_or_widen_scope(when):
    source = document()
    source["rules"][0]["when"].update(when)
    with pytest.raises(ValueError, match="scope|positive"):
        parse(source)


@pytest.mark.parametrize(
    "location",
    ["document", "target", "scope", "rule", "when", "use", "selection", "range", "evidence"],
)
def test_unknown_fields_are_rejected_at_every_schema_level(location):
    source = document()
    rule = source["rules"][0]
    target = {
        "document": source,
        "target": source["target"],
        "scope": source["scope"],
        "rule": rule,
        "when": rule["when"],
        "use": rule["use"],
        "selection": rule["use"]["backward_weight"],
        "range": source["scope"]["batch"],
    }
    if location == "evidence":
        source["evidence"] = [{"id": "campaign", "sha256": "a" * 64, "typo": True}]
    else:
        target[location]["typo"] = True
    with pytest.raises(ValueError, match="typo"):
        parse(source)


@pytest.mark.parametrize(
    "condition,value",
    [
        ("batch", True),
        ("batch", 1.5),
        ("batch", {"min": 4, "max": 2}),
        ("work", {"min": float("nan")}),
        ("total_vram_gib", float("inf")),
        ("spatial_shape", [2, 3]),
        ("spatial_shape", [2, 3, True]),
        ("kernel_size", [3, 0, 3]),
        ("stride", [1.0, 1, 1]),
        ("channels", [0, 32]),
        ("role", 1),
        ("batch", {}),
    ],
)
def test_invalid_match_values_are_rejected(condition, value):
    source = document(scope={})
    source["rules"][0]["when"][condition] = value
    with pytest.raises(ValueError, match=condition):
        parse(source)


@pytest.mark.parametrize(
    "selection,match",
    [
        ({"implementation": "missing"}, "unknown implementation"),
        ({"implementation": "reference", "parameters": "auto"}, "recipe"),
        (
            {
                "implementation": "triton_split_dw",
                "parameters": {"dw_splits": {"formula": "scale_work"}},
            },
            "parameters",
        ),
        ({"implementation": "triton_split_dw", "parameters": {"dw_block": 0}}, "dw_block"),
        ({"implementation": "triton_split_dw", "parameters": {"dw_splits": True}}, "dw_splits"),
        ({"implementation": "triton_split_dw", "parameters": {"dw_splits": 513}}, "dw_splits"),
        ({"implementation": "triton_split_dw", "parameters": {"dw_block": 300}}, "dw_block"),
        ({"implementation": "triton_split_dw", "parameters": {"typo": 1}}, "typo"),
    ],
)
def test_invalid_implementations_and_parameters_fail_before_execution(selection, match):
    source = document()
    source["rules"][0]["use"]["backward_weight"] = selection
    with pytest.raises(ValueError, match=match):
        parse(source)


def test_omitted_parameters_differ_from_an_explicit_empty_mapping():
    source = document()
    rule = source["rules"][0]
    rule["use"]["backward_weight"].pop("parameters")
    omitted = parse(source).rules[0].use["backward_weight"]
    rule["use"]["backward_weight"]["parameters"] = {}
    explicit = parse(source).rules[0].use["backward_weight"]
    assert omitted.parameters is None
    assert explicit.parameters == {}


@pytest.mark.parametrize(
    "source,match",
    [
        ({"schema_version": 1}, "schema.v1.*policy v2"),
        ({"version": 1, "kind": "mednext-accel-evidence"}, "evidence.*runtime policy"),
        (document(version=True), "version"),
        (document(kind="other"), "kind"),
        (document(name=""), "name"),
        (document(target={"vendor": "nvidia", "sm": [8.9, 0]}), "sm"),
        (document(rules={}), "rules"),
    ],
)
def test_document_migration_and_validation_errors(source, match):
    with pytest.raises(ValueError, match=match):
        parse(source)


def test_invalid_confidence_duplicate_ids_and_empty_use_are_rejected():
    for change, match in (({"confidence": "measured"}, "confidence"), ({"use": {}}, "use")):
        source = document()
        source["rules"][0].update(change)
        with pytest.raises(ValueError, match=match):
            parse(source)
    source = document()
    source["rules"].append(copy.deepcopy(source["rules"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        parse(source)


@pytest.mark.parametrize(
    "file", ["/absolute/evidence.json", "../escape.json", "nested/evidence.json"]
)
def test_evidence_reference_must_be_an_immutable_filename(file):
    with pytest.raises(ValueError, match="file"):
        parse(document(evidence=[{"id": "campaign", "sha256": "a" * 64, "file": file}]))


def test_policy_round_trip_preserves_scope_selections_and_provenance():
    source = document(
        evidence=[{"id": "campaign", "sha256": "a" * 64, "file": "unavailable.evidence.json"}]
    )
    policy = parse(source)
    from mednext_accel.optimization.policy import policy_to_primitive

    assert policy_to_primitive(policy) == source
    assert parse(policy_to_primitive(policy)) == policy


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_load_policy_accepts_files_and_mapping_without_evidence_file(tmp_path, suffix):
    parse(document())
    from mednext_accel.optimization.policy_io import load_policy

    source = document(evidence=[{"id": "run", "sha256": "b" * 64, "file": "missing.evidence.json"}])
    path = tmp_path / f"test{suffix}"
    path.write_text(json.dumps(source) if suffix == ".json" else yaml.safe_dump(source))
    assert load_policy(path) == load_policy(source)


def test_loader_rejects_nonmapping_documents_and_factory_modes(tmp_path):
    parse(document())
    from mednext_accel.optimization.policy_io import load_policy

    path = tmp_path / "test.yaml"
    path.write_text("- not-a-policy\n")
    with pytest.raises(ValueError, match="mapping"):
        load_policy(path)
    for mode in ("auto", "reference"):
        with pytest.raises(ValueError, match="factory mode"):
            load_policy(mode)


def test_numeric_channels_narrow_scope_and_coexist_with_exact_tombstones():
    from mednext_accel.optimization.descriptors import ExecutionContext, OperatorDescriptor
    from mednext_accel.optimization.policy import policy_to_primitive
    from mednext_accel.optimization.policy_resolver import PolicyResolver

    source = document(scope={"in_channels": {"min": 32, "max": 128}, "out_channels": 64})
    positive = source["rules"][0]
    positive["when"] = {"in_channels": {"max": 96}}
    positive["use"] = {"training": {"implementation": "pointwise_gemm_per_sample"}}
    source["rules"].insert(
        0,
        {
            "id": "negative",
            "when": {"channels": [48, 64]},
            "use": {"training": {"implementation": "reference"}},
            "confidence": "measured-exact-context",
        },
    )
    policy = parse(source)
    assert parse(policy_to_primitive(policy)) == policy
    assert policy.rules[1].when["in_channels"].minimum == 32
    resolver = PolicyResolver(external=policy)
    context = ExecutionContext(
        "training",
        "cuda",
        (8, 9),
        48 * 2**30,
        "bfloat16",
        3,
        (32,) * 3,
        "mednext_v1",
        "base",
        "none",
    )
    for channels, expected in [
        (31, "reference"),
        (32, "pointwise_gemm_per_sample"),
        (48, "reference"),
        (96, "pointwise_gemm_per_sample"),
        (97, "reference"),
    ]:
        op = OperatorDescriptor(
            "pointwise_conv3d",
            "regular",
            channels,
            64,
            (1,) * 3,
            (1,) * 3,
            (0,) * 3,
            (1,) * 3,
            1,
        )
        assert resolver.resolve(op, context, "training").implementation == expected


@pytest.mark.parametrize(
    "scope,when",
    [
        ({"in_channels": {"min": 32, "max": 64}}, {"in_channels": {"max": 96}}),
        ({"out_channels": {"min": 32}}, {"out_channels": {"max": 16}}),
        ({"in_channels": {"min": 32}}, {"channels": [16, 64]}),
        ({"channels": [32, 64]}, {"out_channels": 32}),
        ({}, {"channels": [32, 64], "in_channels": 64}),
    ],
)
def test_channel_constraints_cannot_widen_scope_or_contradict_exact_channels(scope, when):
    source = document(scope=scope)
    source["rules"][0]["when"] = when
    with pytest.raises(ValueError, match="scope|contradict"):
        parse(source)


def test_legacy_model_and_checkpoint_conditions_remain_readable():
    source = document(scope={"model_family": "mednext_v1"})
    source["rules"][0]["when"].update(variant="base", checkpointing="whole-block")
    assert parse(source).rules[0].when["checkpointing"] == "whole-block"
