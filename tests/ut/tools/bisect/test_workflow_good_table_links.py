from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

# Reusable templates that read good_table through AOP (age gate / bisect).
AOP_TEMPLATES = {
    "single_node": "_e2e_nightly_single_node.yaml",
    "multi_node": "_e2e_nightly_multi_node.yaml",
    "single_node_560t": "_e2e_nightly_single_node_560t.yaml",
    "multi_node_560t": "_e2e_nightly_multi_node_560t.yaml",
}

MODELS_TEMPLATE = "_e2e_nightly_single_node_models.yaml"

SCHEDULE_WORKFLOWS = [
    "schedule_nightly_test_a2.yaml",
    "schedule_nightly_test_a3.yaml",
    "schedule_nightly_test_a3_560t.yaml",
    "schedule_weekly_test_310p.yaml",
    "schedule_weekly_test_a2.yaml",
    "schedule_weekly_test_a3.yaml",
]

WEEKLY_WORKFLOWS = [
    "schedule_weekly_test_310p.yaml",
    "schedule_weekly_test_a2.yaml",
    "schedule_weekly_test_a3.yaml",
]


def _read(name: str) -> str:
    return (WORKFLOW_DIR / name).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "name",
    sorted(AOP_TEMPLATES.values()) + [MODELS_TEMPLATE] + SCHEDULE_WORKFLOWS,
)
def test_workflow_yaml_syntax(name: str):
    yaml.safe_load(_read(name))


@pytest.mark.parametrize("name", sorted(AOP_TEMPLATES))
def test_aop_template_has_good_table_success_write_path(name: str):
    """Every AOP-enabled reusable template must write the baseline it reads."""
    text = _read(AOP_TEMPLATES[name])

    assert "update_good_table.py" in text
    assert "inputs.request_id == ''" in text  # PR runs must not write baselines
    assert '--soc "${{ inputs.soc_version }}"' in text


@pytest.mark.parametrize("name", sorted(AOP_TEMPLATES))
def test_aop_template_read_and_write_share_frequency_path(name: str):
    """Read and write must use the same frequency-specific table path."""
    text = _read(AOP_TEMPLATES[name])
    frequency_path = "${{ inputs.test_frequency }}/good_table.csv"
    assert text.count(frequency_path) >= 2  # one read path + one write path


@pytest.mark.parametrize(
    "name, branch_path",
    [
        ("single_node", "vllm_ascend_branch }}/${{ inputs.test_frequency }}/good_table.csv"),
        ("single_node_560t", "vllm_ascend_branch }}/${{ inputs.test_frequency }}/good_table.csv"),
        ("multi_node", "vllm_ascend_branch || 'main' }}/${{ inputs.test_frequency }}/good_table.csv"),
        ("multi_node_560t", "vllm_ascend_branch || 'main' }}/${{ inputs.test_frequency }}/good_table.csv"),
    ],
)
def test_aop_template_read_and_write_share_branch_resolution(name: str, branch_path: str):
    text = _read(AOP_TEMPLATES[name])
    assert text.count(branch_path) >= 2


@pytest.mark.parametrize("name", ["multi_node", "multi_node_560t"])
def test_multi_node_template_writes_multi_node_scene(name: str):
    assert '--scene "multi_node"' in _read(AOP_TEMPLATES[name])


@pytest.mark.parametrize("name", ["single_node", "single_node_560t"])
def test_single_node_template_passes_scene_to_age_gate(name: str):
    text = _read(AOP_TEMPLATES[name])
    assert '            "single_node"' in text  # 5th arg of aop_commit_age.sh


def test_models_template_has_no_good_table_or_aop():
    """A2 accuracy flow is unsupported-by-design: no AOP, no good table."""
    text = _read(MODELS_TEMPLATE)
    assert "aop_commit_age" not in text
    assert "update_good_table" not in text
    assert "good_table" not in text


@pytest.mark.parametrize("name", WEEKLY_WORKFLOWS)
def test_weekly_workflows_keep_dispatch_only_trigger_semantics(name: str):
    """Weekly is dispatched by an external scheduler; no schedule-event logic."""
    text = _read(name)
    assert "eventName === 'schedule'" not in text
    assert "github.event_name == 'schedule'" not in text


@pytest.mark.parametrize(
    "name",
    ["schedule_weekly_test_310p.yaml", "schedule_weekly_test_a3.yaml"],
)
def test_weekly_bisect_workflows_propagate_good_table_dimensions(name: str):
    text = _read(name)
    assert "test_frequency: weekly" in text
    assert "soc_version:" in text
    assert "request_id" in text


def test_age_gate_filters_scene_like_python_lookup():
    """aop_commit_age.sh must not let a different scene's row satisfy the gate."""
    age_script = (REPO_ROOT / "tests" / "e2e" / "nightly" / "scripts" / "aop_commit_age.sh").read_text(
        encoding="utf-8"
    )
    assert 'scene == "" || NF < 9 || $8 == scene' in age_script


def test_multi_node_run_sh_filters_scene():
    run_script = (REPO_ROOT / "tests" / "e2e" / "nightly" / "multi_node" / "scripts" / "run.sh").read_text(
        encoding="utf-8"
    )
    assert 'BISECT_SCENE="${3:-multi_node}"' in run_script
    assert 'scene == "" || NF < 9 || $8 == scene' in run_script
