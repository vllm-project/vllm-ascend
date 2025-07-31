import os
from dataclasses import dataclass

import lm_eval
import numpy as np
import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

RTOL = 0.03
TEST_DIR = os.path.dirname(__file__)


@dataclass
class EnvConfig:
    vllm_version: str
    vllm_commit: str
    vllm_ascend_version: str
    vllm_ascend_commit: str
    cann_version: str
    torch_version: str
    torch_npu_version: str


@pytest.fixture
def env_config() -> EnvConfig:
    return EnvConfig(vllm_version=os.getenv('VLLM_VERSION', 'unknown'),
                     vllm_commit=os.getenv('VLLM_COMMIT', 'unknown'),
                     vllm_ascend_version=os.getenv('VLLM_ASCEND_VERSION',
                                                   'unknown'),
                     vllm_ascend_commit=os.getenv('VLLM_ASCEND_COMMIT',
                                                  'unknown'),
                     cann_version=os.getenv('CANN_VERSION', 'unknown'),
                     torch_version=os.getenv('TORCH_VERSION', 'unknown'),
                     torch_npu_version=os.getenv('TORCH_NPU_VERSION',
                                                 'unknown'))


def build_model_args(eval_config, tp_size):
    trust_remote_code = eval_config.get("trust_remote_code", False)
    max_model_len = eval_config.get("max_model_len", 4096)
    model_args = {
        "pretrained": eval_config["model_name"],
        "tensor_parallel_size": tp_size,
        "dtype": "auto",
        "trust_remote_code": trust_remote_code,
        "max_model_len": max_model_len,
    }
    for s in [
            "max_images", "gpu_memory_utilization", "enable_expert_parallel",
            "tensor_parallel_size"
    ]:
        val = eval_config.get(s, None)
        if val is not None:
            model_args[s] = val

    print("Model Parameters:")
    print(model_args)

    return model_args


def generate_report(tp_size, eval_config, report_data, report_output,
                    env_config):
    env = Environment(loader=FileSystemLoader(TEST_DIR))
    template = env.get_template("report_template.md")
    model_args = build_model_args(eval_config, tp_size)

    report_content = template.render(
        vllm_version=env_config.vllm_version,
        vllm_commit=env_config.vllm_commit,
        vllm_ascend_version=env_config.vllm_ascend_version,
        vllm_ascend_commit=env_config.vllm_ascend_commit,
        cann_version=env_config.cann_version,
        torch_version=env_config.torch_version,
        torch_npu_version=env_config.torch_npu_version,
        model_name=eval_config["model_name"],
        model_args=f"'{','.join(f'{k}={v}' for k, v in model_args.items())}'",
        model_type=eval_config.get("model", "vllm"),
        datasets=",".join([task["name"] for task in eval_config["tasks"]]),
        apply_chat_template=eval_config.get("apply_chat_template", True),
        fewshot_as_multiturn=eval_config.get("fewshot_as_multiturn", True),
        limit=eval_config.get("limit", None),
        batch_size="auto",
        num_fewshot=eval_config.get("num_fewshot", "N/A"),
        rows=report_data["rows"])

    os.makedirs(os.path.dirname(report_output), exist_ok=True)
    with open(report_output, 'w', encoding='utf-8') as f:
        f.write(report_content)


def test_lm_eval_correctness_param(config_filename, tp_size, report_output,
                                   env_config):
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))
    model_args = build_model_args(eval_config, tp_size)
    success = True
    report_data: dict[str, list[dict]] = {"rows": []}

    eval_params = {
        "model": eval_config.get("model", "vllm"),
        "model_args": model_args,
        "tasks": [task["name"] for task in eval_config["tasks"]],
        "apply_chat_template": eval_config.get("apply_chat_template", True),
        "fewshot_as_multiturn": eval_config.get("fewshot_as_multiturn", True),
        "limit": eval_config.get("limit", None),
        "batch_size": "auto",
    }
    for s in ["num_fewshot", "fewshot_as_multiturn", "apply_chat_template"]:
        val = eval_config.get(s, None)
        if val is not None:
            eval_params[s] = val

    print("Eval Parameters:")
    print(eval_params)

    results = lm_eval.simple_evaluate(**eval_params)

    for task in eval_config["tasks"]:
        task_name = task["name"]
        task_result = results["results"][task_name]
        for metric in task["metrics"]:
            metric_name = metric["name"]
            ground_truth = metric["value"]
            measured_value = task_result[metric_name]
            task_success = bool(
                np.isclose(ground_truth, measured_value, rtol=RTOL))
            success = success and task_success

            print(f"{task_name} | {metric_name}: "
                  f"ground_truth={ground_truth} | measured={measured_value} | "
                  f"success={'✅' if task_success else '❌'}")

            report_data["rows"].append({
                "task":
                task_name,
                "metric":
                metric_name,
                "value":
                f"✅{measured_value}" if success else f"❌{measured_value}",
                "stderr":
                task_result[
                    metric_name.replace(',', '_stderr,') if metric_name ==
                    "acc,none" else metric_name.replace(',', '_stderr,')]
            })
    generate_report(tp_size, eval_config, report_data, report_output,
                    env_config)
    assert success
