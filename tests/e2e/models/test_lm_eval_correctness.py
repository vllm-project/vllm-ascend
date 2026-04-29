import os
import warnings
from dataclasses import dataclass

import lm_eval
import numpy as np
import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

warnings.filterwarnings(
    "ignore",
    message=r"builtin type (SwigPyPacked|SwigPyObject|swigvarlink) has no __module__ attribute",
    category=DeprecationWarning,
)

DEFAULT_RTOL = 0.05
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
    return EnvConfig(
        vllm_version=os.getenv("VLLM_VERSION", "unknown"),
        vllm_commit=os.getenv("VLLM_COMMIT", "unknown"),
        vllm_ascend_version=os.getenv("VLLM_ASCEND_VERSION", "unknown"),
        vllm_ascend_commit=os.getenv("VLLM_ASCEND_COMMIT", "unknown"),
        cann_version=os.getenv("CANN_VERSION", "unknown"),
        torch_version=os.getenv("TORCH_VERSION", "unknown"),
        torch_npu_version=os.getenv("TORCH_NPU_VERSION", "unknown"),
    )


def build_model_args(eval_config, tp_size):
    serve_cfg = eval_config.get("serve", {})
    trust_remote_code = serve_cfg.get("trust_remote_code", False)
    max_model_len = serve_cfg.get("max_model_len", 4096)
    dtype = serve_cfg.get("dtype", "auto")
    model_args = {
        "pretrained": eval_config["model_name"],
        "tensor_parallel_size": tp_size,
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "max_model_len": max_model_len,
    }
    for s in [
        "max_images",
        "gpu_memory_utilization",
        "enable_expert_parallel",
        "tensor_parallel_size",
        "enforce_eager",
        "enable_thinking",
        "quantization",
    ]:
        val = serve_cfg.get(s, None)
        if val is not None:
            model_args[s] = val

    print("Model Parameters:")
    print(model_args)

    return model_args


def generate_report(tp_size, eval_config, report_data, report_dir, env_config):
    env = Environment(loader=FileSystemLoader(TEST_DIR))
    template = env.get_template("report_template.md")
    model_args = build_model_args(eval_config, tp_size)

    parallel_mode = f"TP{model_args.get('tensor_parallel_size', 1)}"
    if model_args.get("enable_expert_parallel", False):
        parallel_mode += " + EP"

    execution_model = f"{'Eager' if model_args.get('enforce_eager', False) else 'ACLGraph'}"

    report_content = template.render(
        vllm_version=env_config.vllm_version,
        vllm_commit=env_config.vllm_commit,
        vllm_ascend_version=env_config.vllm_ascend_version,
        vllm_ascend_commit=env_config.vllm_ascend_commit,
        cann_version=env_config.cann_version,
        torch_version=env_config.torch_version,
        torch_npu_version=env_config.torch_npu_version,
        hardware=eval_config.get("hardware", "unknown"),
        model_name=eval_config["model_name"],
        model_args=f"'{','.join(f'{k}={v}' for k, v in model_args.items())}'",
        model_type=eval_config.get("model_type", "vllm"),
        datasets=",".join([task["name"] for task in eval_config["tasks"]]),
        apply_chat_template=eval_config.get("apply_chat_template", True),
        fewshot_as_multiturn=eval_config.get("fewshot_as_multiturn", True),
        limit=eval_config.get("limit", "N/A"),
        batch_size=eval_config.get("batch_size", "auto"),
        num_fewshot=eval_config.get("num_fewshot", "N/A"),
        rows=report_data["rows"],
        parallel_mode=parallel_mode,
        execution_model=execution_model,
    )

    report_output = os.path.join(report_dir, f"{os.path.basename(eval_config['model_name'])}.md")
    os.makedirs(os.path.dirname(report_output), exist_ok=True)
    with open(report_output, "w", encoding="utf-8") as f:
        f.write(report_content)


def resolve_metric_rtol(eval_config: dict, task: dict, metric: dict) -> float:
    """Resolve relative tolerance with priority: metric > task > eval config > default."""
    metric_rtol = metric.get("rtol")
    if metric_rtol is not None:
        return float(metric_rtol)
    task_rtol = task.get("rtol")
    if task_rtol is not None:
        return float(task_rtol)
    eval_rtol = eval_config.get("rtol")
    if eval_rtol is not None:
        return float(eval_rtol)
    return DEFAULT_RTOL


def test_lm_eval_correctness_param(config_filename, tp_size, report_dir, env_config):
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))

    if eval_config.get("model_type", "vllm") == "vllm-asr":
        pytest.skip("Skipping ASR config, use test_asr_eval.py instead")

    model_args = build_model_args(eval_config, tp_size)
    success = True
    report_data: dict[str, list[dict]] = {"rows": []}
    failure_details: list[str] = []

    eval_params = {
        "model": eval_config.get("model_type", "vllm"),
        "model_args": model_args,
        "tasks": [task["name"] for task in eval_config["tasks"]],
        "apply_chat_template": eval_config.get("apply_chat_template", True),
        "fewshot_as_multiturn": eval_config.get("fewshot_as_multiturn", True),
        "limit": eval_config.get("limit", None),
        "batch_size": eval_config.get("batch_size", "auto"),
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
            measured_value = round(task_result[metric_name], 4)
            rtol = resolve_metric_rtol(eval_config, task, metric)
            task_success = bool(np.isclose(ground_truth, measured_value, rtol=rtol))
            success = success and task_success

            if not task_success:
                lower_bound = round(ground_truth * (1 - rtol), 4)
                upper_bound = round(ground_truth * (1 + rtol), 4)
                failure_details.append(
                    f"{task_name}/{metric_name}: expected={ground_truth}, measured={measured_value}, "
                    f"allowed=[{lower_bound}, {upper_bound}], rtol={rtol}"
                )

            print(
                f"{task_name} | {metric_name}: "
                f"ground_truth={ground_truth} | measured={measured_value} | "
                f"success={'✅' if task_success else '❌'}"
            )

            report_data["rows"].append(
                {
                    "task": task_name,
                    "metric": metric_name,
                    "value": f"✅{measured_value}" if task_success else f"❌{measured_value}",
                    "stderr": task_result[
                        metric_name.replace(",", "_stderr,")
                        if metric_name == "acc,none"
                        else metric_name.replace(",", "_stderr,")
                    ],
                }
            )
    generate_report(tp_size, eval_config, report_data, report_dir, env_config)
    assert success, "lm_eval correctness check failed:\n" + "\n".join(failure_details)
