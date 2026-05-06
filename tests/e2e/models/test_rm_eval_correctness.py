import os
from dataclasses import dataclass

import pytest
import regex as re
import yaml
from jinja2 import Environment, FileSystemLoader
from modelscope.msdatasets import MsDataset  # type: ignore[import-untyped]

from tests.e2e.conftest import VllmRunner

# Allow up to 5 % relative degradation from the declared ground-truth accuracy.
RTOL = 0.05

TEST_DIR = os.path.dirname(__file__)

# Default system prompt for Qwen2.5-Math-RM style models.
_DEFAULT_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


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


def format_rm_input(system_prompt: str, problem: str, solution: str) -> str:
    """Format a (problem, solution) pair using the Qwen chat template."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n{solution}<|im_end|>"
    )


def perturb_answer(solution: str) -> str:
    """Create an obviously wrong solution for a GSM8K-style answer string.

    GSM8K answers end with ``#### <number>``.  We replace that number with
    ``correct * 3 + 137`` so the final answer is clearly incorrect while the
    reasoning chain looks plausible.
    """
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", solution)
    if match:
        num_str = match.group(1).replace(",", "")
        try:
            correct_num = float(num_str)
            wrong_num = int(correct_num * 3 + 137)
            return solution[: match.start()] + f"#### {wrong_num}"
        except ValueError:
            pass
    # Fallback: append an unmistakably wrong sentinel answer.
    return solution + "\n#### -999999"


def extract_reward_score(reward_output) -> float:
    """Extract a scalar score from VllmRunner.reward() output for one sample.

    VllmRunner.reward() returns list[list[float]] or list[Tensor]; for a reward
    model with a single output the inner list has one element.  For a token-level
    reward model the output is a 2-D tensor [seq_len, 1]; in both cases we take
    the last element (final-step score).
    """
    if isinstance(reward_output, (list, tuple)):
        return float(reward_output[-1])
    # Tensor (e.g. shape [seq_len, 1] from a token-level reward model)
    return float(reward_output.flatten()[-1].item())


def generate_rm_report(
    eval_config: dict,
    report_data: dict,
    report_dir: str,
    env_config: EnvConfig,
) -> None:
    """Write a Markdown accuracy report using the shared Jinja2 template."""
    jinja_env = Environment(loader=FileSystemLoader(TEST_DIR))
    template = jinja_env.get_template("report_template.md")

    serve_cfg = eval_config.get("serve", {})
    tp_size = serve_cfg.get("tensor_parallel_size", 1)
    ep_enabled = serve_cfg.get("enable_expert_parallel", False)
    enforce_eager = serve_cfg.get("enforce_eager", False)

    parallel_mode = f"TP{tp_size}"
    if ep_enabled:
        parallel_mode += " + EP"
    execution_model = "Eager" if enforce_eager else "ACLGraph"
    model_args_str = ",".join(f"{k}={v}" for k, v in serve_cfg.items())

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
        model_type=eval_config.get("model_type", "vllm-rm"),
        datasets=",".join(t["name"] for t in eval_config["tasks"]),
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        limit=eval_config.get("limit", "N/A"),
        batch_size=eval_config.get("batch_size", 4),
        num_fewshot="N/A",
        rows=report_data["rows"],
        parallel_mode=parallel_mode,
        execution_model=execution_model,
    )

    report_path = os.path.join(report_dir, f"{os.path.basename(eval_config['model_name'])}.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)


def test_rm_eval_param(config_filename, tp_size, report_dir, env_config):
    """Parametrised reward-model accuracy test driven by a YAML config file.

    Skips automatically when the config's model_type is not "vllm-rm".
    """
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))

    if eval_config.get("model_type", "vllm") != "vllm-rm":
        pytest.skip(f"Skipping non-RM config (model_type={eval_config.get('model_type', 'vllm')})")

    model_name: str = eval_config["model_name"]
    limit: int | None = eval_config.get("limit", None)
    batch_size: int = eval_config.get("batch_size", 4)
    system_prompt: str = eval_config.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)
    serve_cfg: dict = eval_config.get("serve", {})

    # CLI --tp-size takes precedence over the YAML tensor_parallel_size.
    effective_tp = int(tp_size) if (tp_size and tp_size != "1") else int(serve_cfg.get("tensor_parallel_size", 1))

    runner_kwargs: dict = {
        k: v
        for k, v in {
            "runner": "pooling",
            "dtype": serve_cfg.get("dtype", "auto"),
            "tensor_parallel_size": effective_tp,
            "enforce_eager": serve_cfg.get("enforce_eager", False),
            "max_model_len": serve_cfg.get("max_model_len"),
            "gpu_memory_utilization": serve_cfg.get("gpu_memory_utilization"),
        }.items()
        if v is not None
    }

    print(f"\nLoading reward model: {model_name}")
    print(f"  VllmRunner kwargs: {runner_kwargs}")

    success = True
    report_data: dict[str, list[dict]] = {"rows": []}

    with VllmRunner(model_name, **runner_kwargs) as vllm_model:
        for task in eval_config["tasks"]:
            task_name: str = task["name"]
            dataset_name: str = task["dataset"]
            split: str = task["split"]
            dataset_config_name: str | None = task.get("dataset_config")
            task_type: str = task.get("task_type", "correctness")

            # Column names for "correctness" tasks (e.g. GSM8K).
            problem_col: str = task.get("problem_column", "question")
            solution_col: str = task.get("solution_column", "answer")

            # Column names for "pairwise" tasks (e.g. reward-bench).
            prompt_col: str = task.get("prompt_column", "prompt")
            chosen_col: str = task.get("chosen_column", "chosen")
            rejected_col: str = task.get("rejected_column", "rejected")

            split_expr = f"{split}[:{limit}]" if limit is not None else split
            print(f"\nLoading dataset via ModelScope: {dataset_name} / {dataset_config_name} ({split_expr})")

            # MsDataset may bypass the HF_HUB_OFFLINE lock; patch temporarily.
            ds = MsDataset.load(
                dataset_name,
                subset_name=dataset_config_name,
                split=split_expr,
            )
            print(f"  {len(ds)} samples to evaluate (task_type={task_type})")

            correct_count = 0
            total_count = 0

            for batch_start in range(0, len(ds), batch_size):
                batch = ds.select(range(batch_start, min(batch_start + batch_size, len(ds))))

                if task_type == "pairwise":
                    positive_texts = [format_rm_input(system_prompt, s[prompt_col], s[chosen_col]) for s in batch]
                    negative_texts = [format_rm_input(system_prompt, s[prompt_col], s[rejected_col]) for s in batch]
                else:
                    positive_texts = [format_rm_input(system_prompt, s[problem_col], s[solution_col]) for s in batch]
                    negative_texts = [
                        format_rm_input(system_prompt, s[problem_col], perturb_answer(s[solution_col])) for s in batch
                    ]

                pos_rewards = vllm_model.reward(positive_texts)
                neg_rewards = vllm_model.reward(negative_texts)

                for pos_r, neg_r in zip(pos_rewards, neg_rewards):
                    if extract_reward_score(pos_r) > extract_reward_score(neg_r):
                        correct_count += 1
                    total_count += 1

                if (batch_start // batch_size + 1) % 5 == 0:
                    print(f"  processed {batch_start + len(batch)}/{len(ds)} samples …")

            measured_accuracy = round(correct_count / total_count, 4) if total_count > 0 else 0.0
            print(f"\n{task_name} accuracy = {measured_accuracy:.4f}")

            for metric in task["metrics"]:
                if metric["name"] != "accuracy":
                    continue
                ground_truth = metric["value"]
                # Pass if measured accuracy meets or exceeds the threshold
                # (allow up to RTOL relative degradation).
                task_success = measured_accuracy >= ground_truth * (1 - RTOL)
                success = success and task_success

                status = "✅" if task_success else "❌"
                print(f"{task_name} | accuracy: ground_truth={ground_truth} | measured={measured_accuracy} | {status}")

                report_data["rows"].append(
                    {
                        "task": task_name,
                        "metric": "accuracy",
                        "value": f"{status}{measured_accuracy}",
                        "stderr": "N/A",
                    }
                )

    generate_rm_report(eval_config, report_data, report_dir, env_config)
    assert success, "One or more RM tasks did not meet the accuracy threshold. See output above."
