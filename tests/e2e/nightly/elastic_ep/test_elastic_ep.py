"""
End-to-end Elastic EP scaling tests for vllm-ascend.

Launches a vLLM serve instance with Elastic EP enabled, performs scale-up and
scale-down operations, and validates that inference quality is preserved using
GSM8K accuracy evaluation (via aisbench).
"""

import os
import socket
import subprocess
import time
from dataclasses import dataclass, field

import pytest
import requests

# ---------------------------------------------------------------------------
# Server / model constants
# ---------------------------------------------------------------------------

QWEN3_30B_A3B_MODEL = "Qwen/Qwen3-30B-A3B"
QWEN3_30B_A3B_W8A8_MODEL = "vllm-ascend/Qwen3-30B-A3B-W8A8"
QWEN3_235B_A22B_MODEL = "Qwen/Qwen3-235B-A22B"
DATASET_NAME = "/vllm-ascend/gsm8k-lite"
"""HuggingFace / ModelScope identifier of the MoE model used for testing."""

MAX_MODEL_LEN = 16384
MAX_NUM_SEQS = 16

# How long (seconds) to wait after a scale request before evaluating,
# giving the Elastic EP state machine time to finish reconfiguration.
_SCALE_DELAY_SECONDS = 30

# GSM8K accuracy baseline and tolerance.
# The model is expected to achieve accuracy within this range after scaling.
GSM8K_BASELINE = 95.0
GSM8K_THRESHOLD = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def cleanup_ray_between_tests():
    """Force-stop any lingering Ray processes between tests."""
    subprocess.run(["ray", "stop", "--force"], timeout=30, capture_output=True)
    time.sleep(5)

    env_dict = _make_env_dict()
    for key, value in env_dict.items():
        os.environ[key] = value

    subprocess.run(["ray", "start", "--head"], timeout=30, capture_output=True)
    time.sleep(5)
    yield


def _send_scale_command(server, new_dp_size: int) -> bool:
    """POST a scale request to the server's Elastic EP endpoint.

    Returns ``True`` on HTTP 200, ``False`` otherwise.
    """
    url = server.url_for("scale_elastic_ep")
    payload = {"new_data_parallel_size": new_dp_size}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        code = response.status_code
        if code != 200:
            print(f"[scale] HTTP {code}: {response.text}")
        return code == 200
    except requests.exceptions.RequestException as exc:
        print(f"[scale] Request failed: {exc}")
        return False


def _run_gsm8k_eval(server, model_name: str, stage: str) -> float:
    """Run GSM8K accuracy evaluation using aisbench.

    Returns the measured accuracy percentage.
    """
    from tools.aisbench import AisbenchRunner

    aisbench_case = {
        "case_type": "accuracy",
        "dataset_path": DATASET_NAME,
        "request_conf": "vllm_api_general_chat",
        "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_chat_prompt",
        "max_out_len": 4096,
        "batch_size": 32,
        "baseline": GSM8K_BASELINE,
        "threshold": GSM8K_THRESHOLD,
    }

    with AisbenchRunner(
        model=model_name,
        port=server.port,
        aisbench_config=aisbench_case,
        verify=True,
    ) as aisbench:
        accuracy = aisbench.result
        print(f"[{stage}] GSM8K accuracy: {accuracy:.2f}")
        return accuracy


def _make_env_dict() -> dict[str, str]:
    """Return the common environment-variable overrides for the server."""
    env = {
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "BENCHMARK_HOME": "./benchmark",
        "HCCL_BUFFSIZE": "1024",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
    }
    if os.environ.get("VLLM_USE_MODELSCOPE", "").lower() in ("", "0", "false"):
        pass
    else:
        env["VLLM_USE_MODELSCOPE"] = "true"
    return env


# ---------------------------------------------------------------------------
# Configuration dataclasses and test runner
# ---------------------------------------------------------------------------


@dataclass
class ScaleSequence:
    """Defines a sequence of scaling operations."""

    name: str
    steps: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class ElasticEPTestConfig:
    """Configuration for an Elastic EP test."""

    name: str
    data_parallel_size: int
    data_parallel_size_local: int
    tensor_parallel_size: int
    compilation_config: str | None = None
    additional_config: str | None = None
    quant: bool = False
    scale_sequence: ScaleSequence = field(
        default_factory=lambda: ScaleSequence(
            name="default",
            steps=[
                (7, "Scale down (dp=8 -> dp=7)"),
                (4, "Scale down (dp=7 -> dp=4)"),
                (7, "Scale up (dp=4 -> dp=7)"),
                (8, "Scale up (dp=7 -> dp=8)"),
            ],
        )
    )


# Define common additional_config
COMMON_ADDITIONAL_CONFIG = '{"eplb_config": {"dynamic_eplb": false, "num_redundant_experts": 128}}'

# Define test configurations
TEST_CONFIGS = [
    ElasticEPTestConfig(
        name="Qwen3-30B-3B, Default Graph",
        data_parallel_size=8,
        data_parallel_size_local=8,
        tensor_parallel_size=1,
        additional_config=COMMON_ADDITIONAL_CONFIG,
    ),
    ElasticEPTestConfig(
        name="Qwen3-30B-3B, FULL Graph",
        data_parallel_size=8,
        data_parallel_size_local=8,
        tensor_parallel_size=1,
        compilation_config='{"cudagraph_mode": "FULL"}',
        additional_config=COMMON_ADDITIONAL_CONFIG,
    ),
    ElasticEPTestConfig(
        name="Qwen3-30B-3B, PIECEWISE Graph",
        data_parallel_size=8,
        data_parallel_size_local=8,
        tensor_parallel_size=1,
        compilation_config='{"cudagraph_mode": "PIECEWISE"}',
        additional_config=COMMON_ADDITIONAL_CONFIG,
    ),
    ElasticEPTestConfig(
        name="Qwen3-30B-3B, FULL DECODE ONLY Graph",
        data_parallel_size=8,
        data_parallel_size_local=8,
        tensor_parallel_size=1,
        compilation_config='{"cudagraph_mode": "FULL_DECODE_ONLY"}',
        additional_config=COMMON_ADDITIONAL_CONFIG,
    ),
    ElasticEPTestConfig(
        name="Qwen3-30B-3B, TP=4, Default, FC1, FC2",
        data_parallel_size=4,
        data_parallel_size_local=4,
        tensor_parallel_size=4,
        additional_config=(
            '{"eplb_config": {"dynamic_eplb": false,'
            ' "num_redundant_experts": 128},'
            ' "enable_flashcomm1": true,'
            ' "enable_flashcomm2_parallel_size": 2}'
        ),
        scale_sequence=ScaleSequence(
            name="tp4_scaling",
            steps=[
                (3, "Scale down (dp=4 -> dp=3)"),
                (2, "Scale down (dp=3 -> dp=2)"),
                (3, "Scale up (dp=2 -> dp=3)"),
                (4, "Scale up (dp=3 -> dp=4)"),
            ],
        ),
    ),
    ElasticEPTestConfig(
        name="Qwen3-30B-3B-W8A8, Default Graph",
        data_parallel_size=8,
        data_parallel_size_local=8,
        tensor_parallel_size=1,
        additional_config=COMMON_ADDITIONAL_CONFIG,
        quant=True,
    ),
    ElasticEPTestConfig(
        name="Qwen3-30B-3B-W8A8, TP=4, Default, FC1, FC2",
        data_parallel_size=4,
        data_parallel_size_local=4,
        tensor_parallel_size=4,
        additional_config=(
            '{"eplb_config": {"dynamic_eplb": false,'
            ' "num_redundant_experts": 128},'
            ' "enable_flashcomm1": true,'
            ' "enable_flashcomm2_parallel_size": 2}'
        ),
        quant=True,
        scale_sequence=ScaleSequence(
            name="tp4_scaling",
            steps=[
                (3, "Scale down (dp=4 -> dp=3)"),
                (2, "Scale down (dp=3 -> dp=2)"),
                (3, "Scale up (dp=2 -> dp=3)"),
                (4, "Scale up (dp=3 -> dp=4)"),
            ],
        ),
    ),
    ElasticEPTestConfig(
        name="Qwen3-235B-A22B, TP=4, Default, FC1, FC2",
        data_parallel_size=8,
        data_parallel_size_local=8,
        tensor_parallel_size=2,
        additional_config=(
            '{"eplb_config": {"dynamic_eplb": false, "num_redundant_experts": 32}, "enable_flashcomm1": true}'
        ),
        scale_sequence=ScaleSequence(
            name="tp2_scaling",
            steps=[
                (7, "Scale down (dp=8 -> dp=7)"),
                (8, "Scale up (dp=7 -> dp=8)"),
            ],
        ),
    ),
]


def _build_vllm_args(config: ElasticEPTestConfig) -> list[str]:
    """Build vLLM server arguments from configuration."""
    args = [
        "--host",
        "0.0.0.0",
        "--port",
        str(get_free_port()),
        "--trust-remote-code",
        "--data-parallel-size",
        str(config.data_parallel_size),
        "--data-parallel-size-local",
        str(config.data_parallel_size_local),
        "--data-parallel-backend",
        "ray",
        "--enable-expert-parallel",
        "--enable-elastic-ep",
        "--enable-eplb",
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--gpu-memory-utilization",
        "0.9",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
    ]

    if config.quant:
        args.extend(["--quantization", "ascend"])

    if config.compilation_config:
        args.extend(["--compilation-config", config.compilation_config])

    if config.additional_config:
        args.extend(["--additional_config", config.additional_config])

    return args


def _run_elastic_ep_test(config: ElasticEPTestConfig, model_name: str) -> None:
    """Run a complete Elastic EP test with the given configuration."""
    from tests.e2e.conftest import RemoteOpenAIServer

    vllm_serve_args = _build_vllm_args(config)
    env_dict = _make_env_dict()

    # Extract port from args (last port specification)
    port_index = vllm_serve_args.index("--port") + 1
    server_port = int(vllm_serve_args[port_index])

    with RemoteOpenAIServer(
        model_name,
        vllm_serve_args,
        server_host="127.0.0.1",
        server_port=server_port,
        env_dict=env_dict,
        auto_port=False,
        max_wait_seconds=1800,
    ) as server:
        print(f"Server started on port {server.port}")

        # Store all accuracies for summary
        accuracies: dict[str, float] = {}

        # Run initial baseline evaluation
        initial_stage = f"Initial (dp={config.data_parallel_size})"
        accuracies[initial_stage] = _run_gsm8k_eval(server, model_name, initial_stage)
        print(f"  Initial accuracy: {accuracies[initial_stage]:.2f}")

        # Run scaling steps
        for new_dp_size, stage_description in config.scale_sequence.steps:
            assert _send_scale_command(server, new_dp_size), f"{stage_description} failed"
            time.sleep(_SCALE_DELAY_SECONDS)
            accuracies[stage_description] = _run_gsm8k_eval(server, model_name, stage_description)
            print(f"  {stage_description} accuracy: {accuracies[stage_description]:.2f}")

        # Print summary
        print(f"nElastic EP Accuracy Summary ({config.name}):")
        for stage, acc in accuracies.items():
            print(f"  {stage}: {acc:.2f}")
        print(f"  Baseline: {GSM8K_BASELINE:.2f} +/- {GSM8K_THRESHOLD:.2f}")

        # Assert all accuracies are within range
        for stage, acc in accuracies.items():
            lower_bound = GSM8K_BASELINE - GSM8K_THRESHOLD
            upper_bound = GSM8K_BASELINE + GSM8K_THRESHOLD
            assert lower_bound <= acc <= upper_bound, (
                f"{stage} GSM8K accuracy {acc:.2f} is outside expected range [{lower_bound}, {upper_bound}]"
            )


# ---------------------------------------------------------------------------
# Test functions - one for each configuration
# ---------------------------------------------------------------------------


def test_elastic_ep_scaling_qwen3_30b() -> None:
    """Scale dp 8 -> 7 -> 4 -> 7 -> 8 (tp=1, 8 NPUs) with Default Graph"""
    _run_elastic_ep_test(TEST_CONFIGS[0], QWEN3_30B_A3B_MODEL)


def test_elastic_ep_scaling_qwen3_30b_with_full_graph() -> None:
    """Scale dp 8 -> 7 -> 4 -> 7 -> 8 (tp=1, 8 NPUs) with FULL Graph"""
    _run_elastic_ep_test(TEST_CONFIGS[1], QWEN3_30B_A3B_MODEL)


def test_elastic_ep_scaling_qwen3_30b_with_piecewise_graph() -> None:
    """Scale dp 8 -> 7 -> 4 -> 7 -> 8 (tp=1, 8 NPUs) with PIECEWISE Graph"""
    _run_elastic_ep_test(TEST_CONFIGS[2], QWEN3_30B_A3B_MODEL)


def test_elastic_ep_scaling_qwen3_30b_with_full_decode_only_graph() -> None:
    """Scale dp 8 -> 7 -> 4 -> 7 -> 8 (tp=1, 8 NPUs) with FULL DECODE ONLY"""
    _run_elastic_ep_test(TEST_CONFIGS[3], QWEN3_30B_A3B_MODEL)


def test_elastic_ep_scaling_qwen3_30b_with_tp4() -> None:
    """Scale dp 4 -> 3 -> 2 -> 3 -> 4 (tp=4, 16 NPUs) with FC1, FC2"""
    _run_elastic_ep_test(TEST_CONFIGS[4], QWEN3_30B_A3B_MODEL)


def test_elastic_ep_scaling_qwen3_30b_w8w8() -> None:
    """Scale dp 8 -> 7 -> 4 -> 7 -> 8 (tp=1, 8 NPUs) W8A8 Default Graph"""
    _run_elastic_ep_test(TEST_CONFIGS[5], QWEN3_30B_A3B_W8A8_MODEL)


def test_elastic_ep_scaling_qwen3_30b_w8w8_with_tp4() -> None:
    """Scale dp 4 -> 3 -> 2 -> 3 -> 4 (tp=4, 16 NPUs) W8A8 with FC1, FC2"""
    _run_elastic_ep_test(TEST_CONFIGS[6], QWEN3_30B_A3B_W8A8_MODEL)


def test_elastic_ep_scaling_qwen3_235b_with_tp2() -> None:
    """Scale dp 8 -> 7 -> 8 (tp=2, 16 NPUs) 235B with Default, FC1"""
    _run_elastic_ep_test(TEST_CONFIGS[7], QWEN3_235B_A22B_MODEL)
