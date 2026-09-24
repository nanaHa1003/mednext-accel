"""Keep copyable public YAML aligned with the shipped policy/campaign parsers."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from mednext_accel.optimization.policy import parse_policy
from mednext_accel.profiling.campaign import load_campaign

_REPOSITORY = Path(__file__).parents[1]
_GUIDES = (
    Path("README.md"),
    Path("docs/optimization.md"),
    Path("docs/development/package-design.md"),
    Path("docs/benchmarks/sm86-profile.md"),
    Path("docs/benchmarks/sm89-profile.md"),
    Path("docs/benchmarks/sm120-profile.md"),
    Path("docs/benchmarks/depthwise-kernels.md"),
    Path("docs/checkpoints.md"),
    Path("docs/export.md"),
)


@pytest.mark.parametrize("path", _GUIDES, ids=str)
def test_public_yaml_examples_load_with_current_parsers(path: Path) -> None:
    content = (_REPOSITORY / path).read_text()
    for snippet in re.findall(r"```yaml\n(.*?)\n```", content, re.DOTALL):
        data = yaml.safe_load(snippet)
        if isinstance(data, list):
            policy = parse_policy(
                {
                    "version": 2,
                    "kind": "mednext-accel-policy",
                    "name": "documented-rule-fragment",
                    "target": {"vendor": "nvidia", "sm": [8, 9]},
                    "rules": data,
                }
            )
            assert policy.rules
        elif data.get("kind") == "mednext-accel-policy":
            assert parse_policy(data).rules
        else:
            assert load_campaign(data).workloads
    assert not re.search(r"^\s*phases\s*:", content, re.MULTILINE)
    assert not re.search(r'optimization\s*=\s*["\'][^"\']*-local\.json["\']', content)


@pytest.mark.parametrize("path", _GUIDES, ids=str)
def test_public_guides_have_resolvable_local_links(path: Path) -> None:
    document = _REPOSITORY / path
    for target in re.findall(r"\]\(([^)]+)\)", document.read_text()):
        if "://" in target or target.startswith("#"):
            continue
        destination = target.split("#", 1)[0]
        assert (document.parent / destination).exists(), f"{path}: {target}"


def test_readme_documents_the_v2_public_contract() -> None:
    content = (_REPOSITORY / "README.md").read_text()

    for public_name in ("mednext_v2_base", "mednext_v2_wide"):
        assert public_name in content
    assert 'optimization="reference"' in content
    assert '"style": "expansion"' in content
    assert "user-supplied policy" in content
    assert "No MedNeXt v2 performance result is published yet" in content


def test_grn_documentation_uses_the_channel_sum_formula() -> None:
    content = (_REPOSITORY / "docs/benchmarks/activation-memory.md").read_text()

    assert "sum_channel(g[n,:]) + eps" in content
    assert "mean_channel(g[n,:]) + eps" not in content
