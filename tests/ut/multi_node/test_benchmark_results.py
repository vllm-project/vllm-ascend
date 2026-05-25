import json

from tests.e2e.nightly.multi_node.scripts.benchmark_results import (
    build_task_entry,
    extract_hardware,
    filter_environment,
    task_passed,
    write_results_json,
)


def test_extract_hardware_normalizes_runner_label():
    assert extract_hardware("linux-aarch64-a3-0") == "A3"
    assert extract_hardware("linux-aarch64-a2-0") == "A2"
    assert extract_hardware("custom") == "custom"


def test_accuracy_task_entry():
    case = {
        "case_type": "accuracy",
        "dataset_path": "vllm-ascend/GSM8K",
        "baseline": 0.8,
        "threshold": 0.05,
        "num_prompts": 4,
    }

    entry = build_task_entry("acc", case, 0.82)

    assert entry["name"] == "GSM8K"
    assert entry["metrics"] == {"accuracy": 0.82}
    assert entry["test_input"] == {"num_prompts": 4}
    assert entry["pass_fail"] == "pass"


def test_performance_task_entry():
    case = {
        "case_type": "performance",
        "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_str_perf",
        "baseline": 10,
        "threshold": 0.5,
        "request_rate": 1,
    }
    result = [
        "csv",
        {
            "Output Token Throughput": {"total": "8 token/s"},
            "Benchmark Duration": {"total": "2.5s"},
        },
    ]

    entry = build_task_entry("perf", case, result)

    assert entry["name"] == "gsm8k"
    assert entry["metrics"]["Output_Token_Throughput(OTT)"] == 8
    assert entry["metrics"]["Benchmark_Duration(BD)"] == 2.5
    assert entry["pass_fail"] == "pass"


def test_empty_result_fails():
    assert not task_passed({"case_type": "accuracy"}, "")
    assert build_task_entry("case", {"case_type": "accuracy"}, "")["pass_fail"] == "fail"


def test_filter_environment_removes_runtime_keys():
    env = {
        "SERVER_PORT": "7100",
        "LOCAL_IP": "10.0.0.1",
        "VLLM_USE_MODELSCOPE": "true",
    }

    assert filter_environment(env) == {"VLLM_USE_MODELSCOPE": "true"}


def test_write_results_json(tmp_path):
    output_path = write_results_json({"model_name": "model"}, job_name="job", output_dir=tmp_path)

    assert output_path == tmp_path / "job.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"model_name": "model"}
