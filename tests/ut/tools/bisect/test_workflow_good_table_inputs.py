import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

AOP_TEMPLATES = {
    "single_node": "_e2e_nightly_single_node.yaml",
    "multi_node": "_e2e_nightly_multi_node.yaml",
    "single_node_560t": "_e2e_nightly_single_node_560t.yaml",
    "multi_node_560t": "_e2e_nightly_multi_node_560t.yaml",
}

SCHEDULE_FILES = [
    "schedule_nightly_test_a2.yaml",
    "schedule_nightly_test_a3.yaml",
    "schedule_nightly_test_a3_560t.yaml",
    "schedule_weekly_test_310p.yaml",
    "schedule_weekly_test_a2.yaml",
    "schedule_weekly_test_a3.yaml",
]

WEEKLY_BISECT_FILES = [
    "schedule_weekly_test_310p.yaml",
    "schedule_weekly_test_a3.yaml",
]

_USES_RE = re.compile(r"uses:\s*\./\.github/workflows/(_e2e_nightly_(?:single|multi)_node(?:_560t)?\.yaml)")
_SOC_INPUT_RE = re.compile(r"^\s{6}soc_version:\s*\n((?:\s{8}.*\n)*)", re.MULTILINE)


def _read(name: str) -> str:
    return (WORKFLOW_DIR / name).read_text(encoding="utf-8")


def _aop_calls(name: str) -> list[tuple[str, str]]:
    """Return (template, with-block text) for every AOP reusable call."""
    lines = _read(name).splitlines()
    calls = []
    for i, line in enumerate(lines):
        match = _USES_RE.search(line)
        if not match:
            continue
        block = [line]
        for j in range(i + 1, len(lines)):
            following = lines[j]
            if following and not following[0].isspace():
                break
            if following.strip() == "secrets:":
                break
            block.append(following)
        calls.append((match.group(1), "\n".join(block)))
    return calls


@pytest.mark.parametrize("name", SCHEDULE_FILES)
def test_every_aop_template_call_provides_real_soc_version(name: str):
    calls = _aop_calls(name)
    for template, block in calls:
        match = re.search(r"soc_version:\s*(\S+)", block)
        assert match, f"{name}: call to {template} is missing soc_version"
        assert match.group(1) != "unknown", f"{name}: call to {template} uses soc_version: unknown"


@pytest.mark.parametrize(
    "name",
    [
        "schedule_nightly_test_a2.yaml",
        "schedule_nightly_test_a3.yaml",
        "schedule_nightly_test_a3_560t.yaml",
        "schedule_weekly_test_310p.yaml",
        "schedule_weekly_test_a3.yaml",
    ],
)
def test_bisect_schedule_workflows_have_aop_calls(name: str):
    assert _aop_calls(name), f"{name}: expected at least one AOP reusable workflow call"


@pytest.mark.parametrize("name", WEEKLY_BISECT_FILES)
def test_weekly_calls_propagate_test_frequency(name: str):
    for template, block in _aop_calls(name):
        match = re.search(r"test_frequency:\s*(\S+)", block)
        assert match and match.group(1) == "weekly", f"{name}: call to {template} must pass test_frequency: weekly"


@pytest.mark.parametrize("name", sorted(AOP_TEMPLATES))
def test_reusable_templates_require_soc_version(name: str):
    text = _read(AOP_TEMPLATES[name])
    match = _SOC_INPUT_RE.search(text)
    assert match, f"{name}: soc_version input definition not found"
    block = match.group(1)
    assert "required: true" in block
    assert "default:" not in block
    assert "unknown" not in block


@pytest.mark.parametrize("name", ["single_node", "single_node_560t"])
def test_single_node_templates_write_scene_explicitly(name: str):
    text = _read(AOP_TEMPLATES[name])
    assert '--scene "single_node"' in text
    assert "--soc" in text


@pytest.mark.parametrize("name", ["multi_node", "multi_node_560t"])
def test_multi_node_templates_write_scene_explicitly(name: str):
    text = _read(AOP_TEMPLATES[name])
    assert '--scene "multi_node"' in text
    assert "--soc" in text


def test_no_unknown_soc_defaults_or_empty_soc_writes():
    for name in sorted(AOP_TEMPLATES) + SCHEDULE_FILES:
        text = _read(AOP_TEMPLATES.get(name, name))
        assert "soc_version: unknown" not in text
        assert "--soc \"\"" not in text


def test_shell_fail_closed_validation_present():
    scripts = REPO_ROOT / "tests" / "e2e" / "nightly" / "scripts"
    aop_process = (scripts / "aop_process.sh").read_text(encoding="utf-8")
    assert "valid soc is required" in aop_process
    assert "invalid scene" in aop_process
    age = (scripts / "aop_commit_age.sh").read_text(encoding="utf-8")
    assert "valid soc is required" in age
    assert "invalid scene" in age
    run_sh = (REPO_ROOT / "tests" / "e2e" / "nightly" / "multi_node" / "scripts" / "run.sh").read_text(
        encoding="utf-8"
    )
    assert "missing or invalid SOC" in run_sh
