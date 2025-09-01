import socket
import subprocess
from typing import Optional, Tuple

import psutil
import pytest
from modelscope import snapshot_download

from tests.e2e.conftest import RemoteOpenAIServer


def get_net_interface(ip: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """
    Returns specified IP and its network interface.
    If no IP is provided, uses the first from hostname -I.
    """
    if ip is None:
        ips = subprocess.check_output(["hostname",
                                       "-I"]).decode().strip().split()
        if not ips:
            return None
        ip = ips[0]

    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and addr.address == ip:
                return ip, iface
    return None


def get_default_envs() -> dict[str, str]:
    """Returns default network and system environment variables."""
    result = get_net_interface()
    if result is None:
        raise RuntimeError("Failed to get default network IP and interface")
    ip, nic_name = result

    return {
        "HCCL_IF_IP": ip,
        "GLOO_SOCKET_IFNAME": nic_name,
        "TP_SOCKET_IFNAME": nic_name,
        "HCCL_SOCKET_IFNAME": nic_name,
        "OMP_PROC_BIND": "false",
        "OMP_NUM_THREADS": "100",
        "VLLM_USE_V1": "1",
        "HCCL_BUFFSIZE": "1024",
        "VLLM_USE_MODELSCOPE": "true",
    }


MODELS = ['vllm-ascend/DeepSeek-V3-W8A8']
PORMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
] * 40


@pytest.mark.parametrize("model", MODELS)
def test_multi_dp(model: str) -> None:
    env_dict = get_default_envs()

    default_args = [
        "--host",
        "0.0.0.0",
        "--data-parallel-size",
        "4",
        "--tensor-parallel-size",
        "4",
        "--enable-expert-parallel",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "16",
        "--quantization",
        "ascend",
        "--trust-remote-code",
        "--gpu-memory-utilization",
        "0.9",
        "--additional-config",
        '{"ascend_scheduler_config":{"enabled":false},"torchair_graph_config":{"enabled":true}}',
    ]

    model_path = snapshot_download(repo_id=model)

    with RemoteOpenAIServer(
            model_path,
            default_args,
            env_dict=env_dict,
            seed=1024,
            max_wait_seconds=1000,
    ) as remote_server:
        base_url = remote_server.url_root
        cmd = [
            "vllm",
            "bench",
            "serve",
            "--model",
            model_path,
            "--dataset-name",
            "random",
            "--random-input-len",
            "128",
            "--random-output-len",
            "128",
            "--num-prompts",
            "200",
            "--trust-remote-code",
            "--base-url",
            base_url,
            "--request-rate",
            "10",
        ]
        subprocess.run(cmd, check=True)
