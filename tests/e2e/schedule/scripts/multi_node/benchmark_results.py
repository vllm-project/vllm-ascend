import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PORT_ENV_KEYS = {"SERVER_PORT", "ENCODE_PORT", "PD_PORT", "PROXY_PORT"}
INFRA_ENV_KEYS = {
    "HCCL_IF_IP",
    "HCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "TP_SOCKET_IFNAME",
    "LOCAL_IP",
    "NIC_NAME",
    "MASTER_IP",
    "DISAGGREGATED_PREFILL_PROXY_SCRIPT",
}
PERF_METRIC_RENAME: dict[str, str] = {
    "Benchmark Duration": "Benchmark_Duration(BD)",
    "Prefill Token Throughput": "Prefill_Token_Throughput(PTT)",
    "Input Token Throughput": "Input_Token_Throughput(ITT)",
    "Output Token Throughput": "Output_Token_Throughput(OTT)",
    "Total Token Throughput": "Total_Token_Throughput(TTT)",
}


def extract_hardware(runner: str) -> str:
    runner_lower = runner.lower()
    for label in ("a3", "a2"):
        if label in runner_lower:
            return label.upper()
    return runner


def get_vllm_version() -> str:
    try:
        import vllm

        return vllm.__version__
    except Exception:
        return ""


def task_passed(case_config: dict[str, Any], result: Any) -> bool:
    if result == "":
        return False
    case_type = case_config.get("case_type")
    baseline = case_config.get("baseline")
    threshold = case_config.get("threshold")
    if baseline is None or threshold is None:
        return True
    if case_type == "accuracy" and isinstance(result, (int, float)):
        return abs(float(result) - float(baseline)) <= float(threshold)
    if case_type == "performance" and isinstance(result, list) and len(result) == 2:
        _, result_json = result
        throughput_str = result_json.get("Output Token Throughput", {}).get("total", "")
        try:
            throughput_val = float(throughput_str.replace("token/s", "").strip())
            return throughput_val >= float(threshold) * float(baseline)
        except (ValueError, AttributeError):
            return False
    return True


def build_task_entry(case_key: str, case_config: dict[str, Any], result: Any) -> dict[str, Any]:
    dataset_path = case_config.get("dataset_path", "")
    dataset_conf = case_config.get("dataset_conf", "")
    if dataset_path:
        task_name = dataset_path.split("/", 1)[-1]
    elif dataset_conf:
        task_name = dataset_conf.split("/")[0]
    else:
        task_name = case_key

    case_type = case_config.get("case_type", "unknown")
    metrics: dict[str, float] = {}
    if result == "":
        pass
    elif case_type == "accuracy" and isinstance(result, (int, float)):
        metrics["accuracy"] = round(float(result), 4)
    elif case_type == "performance" and isinstance(result, list) and len(result) == 2:
        _, result_json = result
        for metric_name, metric_data in result_json.items():
            if not isinstance(metric_data, dict):
                continue
            total_str = metric_data.get("total", "")
            try:
                value = float(total_str.replace("token/s", "").replace("ms", "").replace("s", "").strip())
                metrics[PERF_METRIC_RENAME.get(metric_name, metric_name)] = round(value, 4)
            except (ValueError, AttributeError):
                pass

    test_input_keys = ("num_prompts", "max_out_len", "batch_size", "request_rate")
    test_input = {key: case_config[key] for key in test_input_keys if key in case_config}

    target: dict[str, Any] = {}
    if case_config.get("baseline") is not None:
        target["baseline"] = case_config["baseline"]
    if case_config.get("threshold") is not None:
        target["threshold"] = case_config["threshold"]

    entry: dict[str, Any] = {"name": task_name, "metrics": metrics, "test_input": test_input}
    if target:
        entry["target"] = target
    entry["pass_fail"] = "pass" if task_passed(case_config, result) else "fail"
    return entry


def filter_environment(envs: dict[str, Any]) -> dict[str, Any]:
    exclude = PORT_ENV_KEYS | INFRA_ENV_KEYS
    return {key: value for key, value in envs.items() if key not in exclude}


def write_results_json(
    output: dict[str, Any],
    *,
    job_name: str,
    output_dir: Path | None = None,
) -> Path:
    if output_dir is None:
        output_dir = Path("/root/.cache/benchmark_results") / job_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{job_name}.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Benchmark results saved to PVC at %s", output_path)
    print(f"Benchmark results saved to PVC at {output_path}")
    return output_path
