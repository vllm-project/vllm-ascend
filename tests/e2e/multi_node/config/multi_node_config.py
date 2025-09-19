# python

import json
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tests.e2e.multi_node.config.utils import get_leader_ip, get_net_interface

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIG_PATH = Path("tests/e2e/multi_node/config/config.json")


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    model: str = "vllm-ascend/DeepSeek-V3-W8A8"
    trust_remote_code: bool = True
    enable_expert_parallel: bool = True
    gpu_memory_utilization: float = 0.9
    headless: bool = False
    quantization: Optional[str] = None
    tensor_parallel_size: int = 8
    data_parallel_size: int = 2
    data_parallel_size_local: int = 1
    data_parallel_start_rank: int = 0
    data_parallel_rpc_port: int = 13389
    data_parallel_address: Optional[str] = None
    kv_transfer_config: Optional[Dict[str, Any]] = None
    additional_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, server_param: Dict[str, Any]) -> "ServerConfig":
        server_cfg = cls()

        for field_name in server_cfg.__dataclass_fields__:
            if field_name in server_param:
                field_type = server_cfg.__dataclass_fields__[field_name].type
                raw_value = server_param[field_name]
                value = raw_value
                if field_type is bool:
                    value = _to_bool(raw_value)
                if field_type is int:
                    value = int(raw_value)
                if field_type is float:
                    value = float(raw_value)
                setattr(server_cfg, field_name, value)
        return server_cfg

    def init_dp_param(
        self,
        is_leader: bool = True,
        is_disaggregate_prefill: bool = False,
        dp_size: int = 4,
        world_size: int = 2,
    ) -> None:
        self.data_parallel_address = get_net_interface()[0]
        if not is_disaggregate_prefill and not is_leader:
            self.headless = True
            self.data_parallel_start_rank = dp_size // world_size
            self.data_parallel_address = get_leader_ip()
        if is_disaggregate_prefill:
            self.data_parallel_start_rank = 0

    def to_list(self) -> list[str]:
        args: list[str] = []
        for f in fields(self):
            key = f.name.replace("_", "-")
            value = getattr(self, f.name)

            if value is None or not value:
                continue
            if value == "model":
                continue

            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            elif isinstance(value, dict):
                if value:
                    args.append(f"--{key}")
                    args.append(json.dumps(value))
            else:
                args.append(f"--{key}")
                args.append(str(value))
        return args


@dataclass
class PerfConfig:
    batch_size: int = 1
    seq_len: int = 1
    iterations: int = 1

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PerfConfig":
        if not d:
            return cls()
        return cls(
            batch_size=int(d.get("batch_size", 1)),
            seq_len=int(d.get("seq_len", 1)),
            iterations=int(d.get("iterations", 1)),
        )


@dataclass
class AccuracyConfig:
    prompt: str
    expected_output: str


@dataclass
class MultiNodeConfig:
    test_name: str = "Unnamed Test"
    disaggregate_prefill: bool = False
    world_size: int = 0
    server_host: str = "0.0.0.0"
    server_port: int = 8888
    perf_config: Optional[PerfConfig] = None
    accuracy_config: Optional[AccuracyConfig] = None
    server_config: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MultiNodeConfig":
        world_size = d.get("num_nodes", 2)
        is_disaggregate_prefill = d.get("disaggregate_prefill", False)
        node_index = os.getenv("LWS_WORKER_INDEX", 0)

        server_param = d.get("server_parameters", {}) or {}
        if not server_param:
            raise ValueError("server_parameters is required")
        client_param = d.get("client_parameters", {}) or {}

        is_leader = int(node_index) == 0
        if is_leader:
            server_param = server_param.get("leader_config", {})
        else:
            server_param = server_param.get("worker_config", {})
        cur_cfg = ServerConfig.from_dict(server_param)

        # Init dp relevant parameters
        cur_cfg.init_dp_param(is_leader=is_leader,
                              is_disaggregate_prefill=is_disaggregate_prefill,
                              dp_size=cur_cfg.data_parallel_size,
                              world_size=world_size)

        perf_cfg = PerfConfig.from_dict(client_param) if client_param else None
        server_host = get_leader_ip()
        server_port = get_avaliable_port()
        if not is_disaggregate_prefill:
            # For regular multi-node, all workers connect to the leader server.
            # For disaggregate prefill, the endpoint should be a unique proxy to avoid conflict.
            leader_param = d.get("server_parameters",
                                 {}).get("leader_config", {})
            server_host = leader_param.get("host", 8000)
        return cls(
            test_name=str(d.get("test_name", "Unnamed Test")),
            disaggregate_prefill=is_disaggregate_prefill,
            world_size=world_size,
            server_config=cur_cfg,
            perf_config=perf_cfg,
            server_host=server_host,
            server_port=server_port,
        )


def get_avaliable_port(start_port: int = 6000, end_port: int = 7000) -> int:
    import socket
    for port in range(start_port, end_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available port found")


def load_configs(path: Union[str, Path] = None):
    path = Path(path or CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {path}")
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("Configuration file must contain a list or an object")
    configs: List[MultiNodeConfig] = []
    for idx, item in enumerate(raw):
        try:
            configs.append(MultiNodeConfig.from_dict(item))
        except Exception as e:
            LOG.error("Failed to parse config index %d: %s", idx, e)
            raise
    return configs


if __name__ == "__main__":
    config = load_configs()
    for cfg in config:
        print(cfg)
        print(cfg.server_config.to_list())
