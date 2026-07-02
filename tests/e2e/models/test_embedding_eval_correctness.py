import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from jinja2 import Environment, FileSystemLoader

from tests.e2e.conftest import VllmRunner

RTOL = 0.05
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


def build_runner_kwargs(eval_config: dict[str, Any], tp_size: str) -> dict[str, Any]:
    serve_cfg = eval_config.get("serve", {})
    effective_tp = int(tp_size) if (tp_size and tp_size != "1") else int(serve_cfg.get("tensor_parallel_size", 1))

    runner_kwargs: dict[str, Any] = {
        k: v
        for k, v in {
            "runner": serve_cfg.get("runner", "pooling"),
            "dtype": serve_cfg.get("dtype", "auto"),
            "tensor_parallel_size": effective_tp,
            "max_model_len": serve_cfg.get("max_model_len"),
            "gpu_memory_utilization": serve_cfg.get("gpu_memory_utilization"),
            "enforce_eager": serve_cfg.get("enforce_eager", False),
            "enable_prefix_caching": serve_cfg.get("enable_prefix_caching"),
            "cudagraph_capture_sizes": serve_cfg.get("cudagraph_capture_sizes"),
        }.items()
        if v is not None
    }
    return runner_kwargs


def resolve_model_name(eval_config: dict[str, Any]) -> str:
    model_name = eval_config["model_name"]
    local_path_env = eval_config.get("local_model_path_env")
    if local_path_env:
        model_name = os.getenv(local_path_env, model_name)
    return model_name


def format_query(instruction: str, query: str) -> str:
    return f"Instruct: {instruction}\nQuery: {query}"


def generate_embedding_report(
    eval_config: dict[str, Any],
    report_data: dict[str, list[dict[str, Any]]],
    report_dir: str,
    env_config: EnvConfig,
    runner_kwargs: dict[str, Any],
) -> None:
    jinja_env = Environment(loader=FileSystemLoader(TEST_DIR))
    template = jinja_env.get_template("report_template.md")

    tp_size = runner_kwargs.get("tensor_parallel_size", 1)
    execution_model = "Eager" if runner_kwargs.get("enforce_eager", False) else "ACLGraph"
    model_args_str = ",".join(f"{k}={v}" for k, v in runner_kwargs.items())

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
        model_args=f"'{model_args_str}'",
        model_type=eval_config.get("model_type", "vllm-embedding"),
        datasets=",".join(t["name"] for t in eval_config["tasks"]),
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        limit=eval_config.get("limit", "N/A"),
        batch_size=eval_config.get("batch_size", 4),
        num_fewshot="N/A",
        rows=report_data["rows"],
        parallel_mode=f"TP{tp_size}",
        execution_model=execution_model,
        show_command=False,
    )

    report_path = os.path.join(report_dir, f"{os.path.basename(eval_config['model_name'])}.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)


def test_embedding_eval_param(config_filename, tp_size, report_dir, env_config):
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))

    if eval_config.get("model_type", "vllm") != "vllm-embedding":
        pytest.skip(f"Skipping non-embedding config (model_type={eval_config.get('model_type', 'vllm')})")

    model_name = resolve_model_name(eval_config)
    runner_kwargs = build_runner_kwargs(eval_config, tp_size)
    expected_norm = float(eval_config.get("expected_l2_norm", 1.0))
    norm_atol = float(eval_config.get("norm_atol", 1e-3))

    print(f"\nLoading embedding model: {model_name}")
    print(f"  VllmRunner kwargs: {runner_kwargs}")

    success = True
    report_data: dict[str, list[dict[str, Any]]] = {"rows": []}

    with VllmRunner(model_name, **runner_kwargs) as vllm_model:
        for task in eval_config["tasks"]:
            task_name = task["name"]
            instruction = task["instruction"]
            documents = task["documents"]
            queries = task["queries"]

            input_texts = [format_query(instruction, item["text"]) for item in queries] + documents
            embeddings = torch.tensor(vllm_model.embed(input_texts), dtype=torch.float32)
            query_embeddings = embeddings[: len(queries)]
            document_embeddings = embeddings[len(queries) :]
            scores = query_embeddings @ document_embeddings.T

            measured_dimension = int(embeddings.shape[-1])
            norms = embeddings.norm(dim=1)
            mean_norm = round(float(norms.mean().item()), 4)

            top1_correct = 0
            margins: list[float] = []
            for query_idx, query in enumerate(queries):
                relevant_doc = int(query["relevant_doc"])
                query_scores = scores[query_idx]
                ranking = torch.argsort(query_scores, descending=True).tolist()
                if ranking[0] == relevant_doc:
                    top1_correct += 1
                negative_scores = [
                    float(score) for doc_idx, score in enumerate(query_scores) if doc_idx != relevant_doc
                ]
                margin = float(query_scores[relevant_doc]) - max(negative_scores)
                margins.append(margin)
                print(
                    f"{task_name} | query={query_idx} | relevant_doc={relevant_doc} | "
                    f"scores={[round(float(s), 4) for s in query_scores]} | margin={margin:.4f}"
                )

            top1_accuracy = round(top1_correct / len(queries), 4)
            min_margin = round(min(margins), 4)

            metric_values = {
                "top1_accuracy": top1_accuracy,
                "embedding_dimension": measured_dimension,
                "mean_l2_norm": mean_norm,
                "min_score_margin": min_margin,
            }

            for metric in task["metrics"]:
                metric_name = metric["name"]
                expected_value = metric["value"]
                measured_value = metric_values[metric_name]

                if metric_name == "embedding_dimension":
                    task_success = measured_value == expected_value
                elif metric_name == "mean_l2_norm":
                    task_success = bool(np.isclose(expected_value, measured_value, atol=norm_atol))
                else:
                    task_success = measured_value >= expected_value * (1 - RTOL)

                success = success and task_success
                status = "PASS" if task_success else "FAIL"
                print(f"{task_name} | {metric_name}: expected={expected_value} | measured={measured_value} | {status}")

                report_data["rows"].append(
                    {
                        "task": task_name,
                        "metric": metric_name,
                        "value": f"{status}:{measured_value}",
                        "stderr": 0.0,
                    }
                )

            norm_success = bool(
                torch.allclose(
                    norms,
                    torch.full_like(norms, expected_norm),
                    atol=norm_atol,
                    rtol=0.0,
                )
            )
            success = success and norm_success
            assert norm_success, f"Embedding L2 norms are not close to {expected_norm}: {norms.tolist()}"

    generate_embedding_report(eval_config, report_data, report_dir, env_config, runner_kwargs)
    assert success, "One or more embedding checks did not meet the configured threshold."
