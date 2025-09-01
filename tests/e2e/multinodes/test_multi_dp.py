import json
import socket
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple

import psutil
import pytest
from modelscope import snapshot_download

from tests.e2e.conftest import RemoteOpenAIServer


@dataclass
class ModelMetadata:
    model_name: str = None
    is_quant: bool = False
    dp_size: int = 1
    tp_size: int = 1
    enable_ep: bool = True
    additional_config: Optional[dict] = None


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


deepseek_model = ModelMetadata(
    model_name="vllm-ascend/DeepSeek-V3-W8A8",
    is_quant=True,
    dp_size=4,
    tp_size=4,
    enable_ep=True,
    additional_config={
        "ascend_scheduler_config": {
            "enabled": False
        },
        "torchair_graph_config": {
            "enabled": True
        },
    },
)

MODELS = [deepseek_model]


@pytest.mark.parametrize("model", MODELS)
def test_multi_dp(model: ModelMetadata) -> None:
    env_dict = get_default_envs()

    model_name = model.model_name
    tp_size = model.tp_size
    dp_size = model.dp_size
    enable_ep = model.enable_ep
    is_quant = model.is_quant
    assert model_name is not None, "Model name must be specified"

    default_args = [
        "--host",
        "0.0.0.0",
        "--data-parallel-size",
        str(dp_size),
        "--tensor-parallel-size",
        str(tp_size),
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "16",
        "--trust-remote-code",
        "--gpu-memory-utilization",
        "0.9",
    ]

    if enable_ep:
        default_args += ["--enable-expert-parallel"]
    if is_quant:
        default_args += ["--quantization", "ascend"]
    if model.additional_config is not None:
        default_args += [
            "--additional-config",
            json.dumps(model.additional_config),
        ]

    model_path = snapshot_download(repo_id=model_name)

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
