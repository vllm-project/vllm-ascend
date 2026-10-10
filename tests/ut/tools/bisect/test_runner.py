from pathlib import Path

from tools.bisect.config import SINGLE_NODE_TEST_PATH, BisectInput, BisectOptions
from tools.bisect.runner import (
    MultiNodeRunner,
    SingleNodeRunner,
    _multi_node_test_path,
    _safe_name,
    _single_node_test_path,
)


def test_safe_name_replaces_path_and_space_separators():
    assert _safe_name("configs/my case.yaml") == "configs_my_case.yaml"


def test_base_env_includes_case_and_config_base(tmp_path: Path):
    inp = BisectInput(
        scene="single_node",
        config_yaml="case.yaml",
        bad_commit="bad",
        soc="a2",
        config_base_path="configs",
    )
    opt = BisectOptions(repo_dir=tmp_path)
    runner = SingleNodeRunner(inp, opt, builder=None)  # type: ignore[arg-type]

    env = runner._base_env()

    assert env["CONFIG_YAML_PATH"] == "case.yaml"
    assert env["CONFIG_BASE_PATH"] == "configs"


def test_bisect_selects_current_common_test_entries(tmp_path: Path):
    single_path = tmp_path / SINGLE_NODE_TEST_PATH
    single_path.parent.mkdir(parents=True)
    single_path.touch()

    multi_path = tmp_path / "tests/e2e/common/multi_node/test_multi_node.py"
    multi_path.parent.mkdir(parents=True)
    multi_path.touch()
    inp = BisectInput(
        scene="multi_node",
        config_yaml="case.yaml",
        bad_commit="bad",
        soc="a3",
    )

    assert _single_node_test_path(tmp_path) == SINGLE_NODE_TEST_PATH
    assert _multi_node_test_path(tmp_path, inp) == "tests/e2e/common/multi_node/test_multi_node.py"


def test_multi_node_runner_selects_external_dp_test_path(tmp_path: Path):
    inp = BisectInput(
        scene="multi_node",
        config_yaml="case.yaml",
        bad_commit="bad",
        soc="a3",
        config_base_path="tests/e2e/nightly/multi_node/external_dp/config",
    )
    opt = BisectOptions(repo_dir=tmp_path)
    runner = MultiNodeRunner(inp, opt, builder=None, coordinator=None)  # type: ignore[arg-type]

    assert runner._test_path().endswith("external_dp/scripts/test_external_dp.py")
