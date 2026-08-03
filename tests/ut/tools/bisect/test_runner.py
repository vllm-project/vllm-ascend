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
        soc="a2",
        config_base_path="configs",
    )
    opt = BisectOptions(repo_dir=tmp_path)
    runner = SingleNodeRunner(inp, opt, builder=None)  # type: ignore[arg-type]

    env = runner._base_env()

    assert env["CONFIG_YAML_PATH"] == "case.yaml"
    assert env["CONFIG_BASE_PATH"] == "configs"


def test_single_node_runner_replays_pytest_driven_path(tmp_path: Path):
    inp = BisectInput(
        scene="single_node",
        config_yaml=None,
        test_path="tests/e2e/weekly/single_node/models/test_case.py",
        bad_commit="bad",
    )
    runner = SingleNodeRunner(inp, BisectOptions(repo_dir=tmp_path), builder=None)  # type: ignore[arg-type]

    assert runner._test_command() == [
        "python",
        "-m",
        "pytest",
        "-sv",
        "tests/e2e/weekly/single_node/models/test_case.py",
    ]
    assert "CONFIG_YAML_PATH" not in runner._base_env()


def test_single_node_runner_selects_accuracy_test_from_model_type(tmp_path: Path):
    config_dir = tmp_path / "tests/e2e/models/configs"
    config_dir.mkdir(parents=True)
    (config_dir / "model.yaml").write_text("model_type: vllm-asr\n", encoding="utf-8")
    inp = BisectInput(
        scene="single_node",
        config_yaml="model.yaml",
        bad_commit="bad",
        config_base_path="tests/e2e/models/configs",
    )
    runner = SingleNodeRunner(inp, BisectOptions(repo_dir=tmp_path), builder=None)  # type: ignore[arg-type]

    command = runner._test_command()

    assert "tests/e2e/models/test_asr_eval_correctness.py" in command
    assert command[-2:] == ["--config", str(config_dir / "model.yaml")]


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
