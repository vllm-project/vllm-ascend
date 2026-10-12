import json
import shlex
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from tools import aisbench


@pytest.mark.parametrize("reasoning_effort", [None, "low"])
def test_request_config_reasoning_effort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reasoning_effort: str | None):
    request_conf = tmp_path / "vllm_api_general_chat.py"
    request_conf.write_text(
        "model='test',\n"
        "host_port=8000,\n"
        "host_ip='localhost',\n"
        "max_out_len=1024,\n"
        "batch_size=1,\n"
        "trust_remote_code=True,\n"
        "generation_kwargs=dict(\n"
        "    temperature=0,\n"
        "    ignore_eos=False,\n"
        "),\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(aisbench, "REQUEST_CONF_DIR", str(tmp_path))
    runner = aisbench.AisbenchRunner.__new__(aisbench.AisbenchRunner)
    runner.__dict__.update(
        model="test-model",
        port=8001,
        host_ip="localhost",
        max_out_len=65536,
        batch_size=32,
        trust_remote_code=True,
        request_conf="vllm_api_general_chat",
        top_p=None,
        top_k=None,
        seed=None,
        min_p=None,
        presence_penalty=None,
        repetition_penalty=None,
        thinking=True,
        reasoning_effort=reasoning_effort,
        task_type="accuracy",
        temperature=None,
        no_pred=False,
    )

    runner._init_request_conf()

    content = (tmp_path / "vllm_api_general_chat_custom.py").read_text(encoding="utf-8")
    assert 'chat_template_kwargs={"thinking": True}' in content
    if reasoning_effort is None:
        assert "reasoning_effort=" not in content
    else:
        assert 'reasoning_effort="low"' in content


@pytest.fixture
def calibration_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_JOB_NAME", "test-job")
    config = {
        "case_type": "accuracy",
        "case_name": "acc_gpqa",
        "dataset_path_local": "local-dataset",
        "model_path": "local-model",
        "request_conf": "vllm_api_general_chat",
        "dataset_conf": "gpqa/test",
        "max_out_len": 65536,
        "batch_size": 32,
        "work_dir": str(tmp_path / "${BENCHMARK_JOB_NAME}"),
        "calibration_only": True,
    }
    runner = aisbench.AisbenchRunner
    for method in ("_init_dataset_conf", "_init_request_conf", "_wait_for_task"):
        monkeypatch.setattr(runner, method, lambda self: None)
    monkeypatch.setattr(
        runner, "_run_aisbench_task", lambda self: Path(self.work_dir).mkdir(parents=True, exist_ok=True)
    )
    monkeypatch.setattr(runner, "_get_result_accuracy", lambda self: setattr(self, "result", 80.0))

    def performance_result(self):
        self.result_json = {"Output Token Throughput": {"total": "42 token/s"}}
        self.result = [None, self.result_json]

    monkeypatch.setattr(runner, "_get_result_performance", performance_result)
    return config


@pytest.mark.parametrize("case_type", ["accuracy", "performance"])
def test_calibration_persists_measurement_without_verifying(calibration_setup, monkeypatch, case_type):
    config = {**calibration_setup, "case_type": case_type}
    verify = Mock(side_effect=AssertionError("must not use default regression gates"))
    monkeypatch.setattr(aisbench.AisbenchRunner, f"_{case_type}_verify", verify)
    runner = aisbench.AisbenchRunner("test-model", 8000, config)
    output = json.loads(Path(runner.work_dir, "measurement.json").read_text(encoding="utf-8"))
    assert output["validation_status"] == "uncalibrated"
    assert output["case_type"] == case_type
    assert output["measurement"] == (80.0 if case_type == "accuracy" else runner.result_json)
    assert "${BENCHMARK_JOB_NAME}" not in runner.work_dir
    verify.assert_not_called()


@pytest.mark.parametrize(
    "override",
    [
        {"work_dir": None},
        {"case_type": "spec_decode"},
        {"case_type": "curl_cmd"},
        {"calibration_only": "false"},
        {"baseline": 0},
        {"threshold": 0.97},
        {"tpot_threshold": 90},
        {"input_throughput_threshold": 1},
    ],
)
def test_calibration_rejects_invalid_configuration(calibration_setup, override):
    with pytest.raises(ValueError, match="calibration_only"):
        aisbench.AisbenchRunner("test-model", 8000, {**calibration_setup, **override})


@pytest.mark.parametrize("case_type, threshold", [("accuracy", 1), ("performance", 0.97)])
def test_existing_default_verification_is_unchanged(calibration_setup, monkeypatch, case_type, threshold):
    verify = Mock()
    monkeypatch.setattr(aisbench.AisbenchRunner, f"_{case_type}_verify", verify)
    runner = aisbench.AisbenchRunner(
        "test-model", 8000, {**calibration_setup, "case_type": case_type, "calibration_only": False}
    )
    assert runner.baseline == 1
    assert runner.threshold == threshold
    verify.assert_called_once()


def test_all_calibration_cases_are_collected_before_nonpassing_verdict(calibration_setup, tmp_path):
    cases = [
        {**calibration_setup, "work_dir": str(tmp_path / "accuracy")},
        {**calibration_setup, "case_type": "performance", "work_dir": str(tmp_path / "performance")},
    ]
    with pytest.raises(AssertionError, match="Calibration-only measurements collected"):
        aisbench.run_aisbench_cases("test-model", 8000, cases)
    for case in cases:
        assert Path(case["work_dir"], "measurement.json").is_file()


def test_calibration_runtime_error_is_not_hidden(calibration_setup, monkeypatch):
    def fail(self):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(aisbench.AisbenchRunner, "_wait_for_task", fail)
    with pytest.raises(AssertionError, match="some aisbench cases failed"):
        aisbench.run_aisbench_cases("test-model", 8000, [calibration_setup])
    work_dir = Path(aisbench.os.path.expandvars(calibration_setup["work_dir"]))
    assert not (work_dir / "measurement.json").exists()


def test_work_dir_and_stdout_are_shell_quoted(tmp_path, monkeypatch):
    runner = aisbench.AisbenchRunner.__new__(aisbench.AisbenchRunner)
    runner.__dict__.update(
        task_type="accuracy",
        dataset_conf="gpqa/test",
        request_conf="request",
        num_prompts=None,
        work_dir=str(tmp_path / "results with spaces;literal"),
    )
    popen = Mock()
    monkeypatch.setattr(aisbench.subprocess, "Popen", popen)
    runner._run_aisbench_task()
    args = shlex.split(popen.call_args.args[0])
    assert args[args.index("--work-dir") + 1] == runner.work_dir
    assert args[args.index(">") + 1] == runner.stdout_file
    assert Path(runner.work_dir).is_dir()


def test_kimi_a5_internal_dp_and_calibration_config():
    root = Path(aisbench.__file__).resolve().parents[1]
    config_path = root / "tests/e2e/cases/models/configs/Kimi/Kimi-K3-W4A8-MXFP-QuaRot-A5.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["dp_load_balancing"] == "internal"
    assert config["num_nodes"] == len(config["deployment"]) == 4
    assert config["npu_per_node"] == 8
    for rank, deployment in enumerate(config["deployment"]):
        args = shlex.split(deployment["server_cmd"])
        assert args[args.index("--tensor-parallel-size") + 1] == "8"
        assert args[args.index("--data-parallel-size") + 1] == "4"
        assert args[args.index("--data-parallel-size-local") + 1] == "1"
        assert "--enable-expert-parallel" in args
        assert ("--headless" in args) == (rank > 0)
        if rank == 0:
            assert "--data-parallel-start-rank" not in args
        else:
            assert args[args.index("--data-parallel-start-rank") + 1] == str(rank)
            assert args[args.index("--data-parallel-address") + 1] == "$MASTER_IP"
    paths = []
    for case in config["benchmarks"].values():
        assert case["calibration_only"] is True
        assert not any(key in case for key in ("baseline", "threshold", "tpot_threshold"))
        paths.append(case["work_dir"])
    assert len(paths) == len(set(paths))
