from pathlib import Path

from tools.bisect.config import BisectInput, BisectOptions
from tools.bisect.runner import MultiNodeRunner, SingleNodeRunner, _safe_name


def test_safe_name_replaces_path_and_space_separators():
    assert _safe_name("configs/my case.yaml") == "configs_my_case.yaml"


def test_base_env_includes_case_and_config_base(tmp_path: Path):
    inp = BisectInput(
        scene="single_node",
        config_yaml="case.yaml",
        bad_commit="bad",
        config_base_path="configs",
    )
    opt = BisectOptions(repo_dir=tmp_path)
    runner = SingleNodeRunner(inp, opt, builder=None)  # type: ignore[arg-type]

    env = runner._base_env()

    assert env["CONFIG_YAML_PATH"] == "case.yaml"
    assert env["CONFIG_BASE_PATH"] == "configs"


def test_multi_node_runner_selects_external_dp_test_path(tmp_path: Path):
    inp = BisectInput(
        scene="multi_node",
        config_yaml="case.yaml",
        bad_commit="bad",
        config_base_path="tests/e2e/nightly/multi_node/external_dp/config",
    )
    opt = BisectOptions(repo_dir=tmp_path)
    runner = MultiNodeRunner(inp, opt, builder=None, coordinator=None)  # type: ignore[arg-type]

    assert runner._test_path().endswith("external_dp/scripts/test_external_dp.py")
