import json
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from tests.e2e.multi_node.config.utils import (get_avaliable_port,
                                               get_leader_ip,
                                               get_net_interface)

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIG_PATH = Path("tests/e2e/multi_node/config/config.json")

T = TypeVar("T", bound="BaseConfig")


# =========================
# Base Config
# =========================
@dataclass
class BaseConfig:
    model: str = "vllm-ascend/DeepSeek-V3-W8A8"
    _extra_fields: Optional[Dict[str, Any]] = None

    @classmethod
    def from_config(cls: Type[T], data: dict[str, Any]) -> T:
        """Create config instance from dict, keeping unknown fields."""
        field_names = {f.name for f in fields(cls)}
        valid_fields = {k: v for k, v in data.items() if k in field_names}
        extra_fields = {k: v for k, v in data.items() if k not in field_names}
        obj = cls(**valid_fields)
        obj._extra_fields = extra_fields or {}
        return obj

    def to_list(self) -> List[str]:
        """Convert all fields (including _extra_fields) to CLI arguments."""
        args: List[str] = []
        all_items = {**vars(self), **(self._extra_fields or {})}

        for key, value in all_items.items():
            if key in ("model", "_extra_fields") or value in (None, "", [],
                                                              {}):
                continue
            key = key.replace("_", "-")

            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            elif isinstance(value, dict):
                args += [f"--{key}", json.dumps(value, ensure_ascii=False)]
            else:
                args += [f"--{key}", str(value)]
        return args


# =========================
# Server Config
# =========================
@dataclass
class ServerConfig(BaseConfig):
    host: str = "0.0.0.0"
    port: int = 8080
    trust_remote_code: bool = True
    enable_expert_parallel: bool = True
    gpu_memory_utilization: float = 0.9
    headless: bool = False
    quantization: Optional[str] = None
    tensor_parallel_size: int = 8
    max_model_len: int = 8192
    max_num_batched_token: int = 8192
    data_parallel_size: int = 4
    data_parallel_size_local: int = 2
    data_parallel_start_rank: int = 0
    data_parallel_rpc_port: int = 13389
    data_parallel_address: Optional[str] = None
    kv_transfer_config: Optional[Dict[str, Any]] = None
    additional_config: Optional[Dict[str, Any]] = None

    def init_dp_param(
        self,
        is_leader: bool,
        is_disaggregate_prefill: bool,
        dp_size: int,
        world_size: int,
    ) -> None:
        """Initialize distributed parallel parameters."""
        iface = get_net_interface()
        if iface is None:
            raise RuntimeError("No available network interface found")
        self.data_parallel_address = iface[0]

        if is_disaggregate_prefill:
            self.data_parallel_start_rank = 0
            return

        if not is_leader:
            self.headless = True
            self.data_parallel_start_rank = dp_size // world_size
            self.data_parallel_address = get_leader_ip()


@dataclass
class PerfConfig(BaseConfig):
    pass


@dataclass
class AccuracyConfig:
    prompt: str
    expected_output: str


# =========================
# MultiNode Config
# =========================
@dataclass
class MultiNodeConfig:
    test_name: str = "Unnamed Test"
    disaggregate_prefill: bool = False
    enable_multithread_load: bool = True
    world_size: int = 2
    server_host: str = "0.0.0.0"
    server_port: int = 8888
    server_config: ServerConfig = field(default_factory=ServerConfig)
    perf_config: Optional[PerfConfig] = None
    accuracy_config: Optional[AccuracyConfig] = None

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "MultiNodeConfig":
        """Create a MultiNodeConfig from raw dict."""
        num_nodes = cfg.get("num_nodes", 2)
        is_disaggregate_prefill = cfg.get("disaggregate_prefill", False)
        node_index = int(os.getenv("LWS_WORKER_INDEX", 0))
        is_leader = node_index == 0

        # server config
        server_cfg_data = cfg.get("server_parameters", {})
        if not server_cfg_data:
            raise ValueError("Missing required key: 'server_parameters'")

        role_key = "leader_config" if is_leader else "worker_config"
        server_cfg_dict = server_cfg_data.get(role_key, {})
        server_cfg: ServerConfig = ServerConfig.from_config(server_cfg_dict)

        if cfg.get("enable_multithread_load"):
            server_cfg.model_loader_extra_config = { # type: ignore[attr-defined]
                "enable_multithread_load": True,
                "num_threads": 8,
            }

        # distributed param init
        server_cfg.init_dp_param(
            is_leader=is_leader,
            is_disaggregate_prefill=is_disaggregate_prefill,
            dp_size=server_cfg.data_parallel_size,
            world_size=num_nodes,
        )

        perf_cfg: Optional[PerfConfig] = (PerfConfig.from_config(
            cfg.get("client_parameters", {})) if cfg.get("client_parameters")
                                          else None)

        # network info
        leader_cfg = server_cfg_data.get("leader_config", {})
        server_host = get_leader_ip()
        server_port = (get_avaliable_port() if is_disaggregate_prefill else
                       leader_cfg.get("port", 8080))

        return cls(
            test_name=str(cfg.get("test_name", "Unnamed Test")),
            disaggregate_prefill=is_disaggregate_prefill,
            enable_multithread_load=cfg.get("enable_multithread_load", False),
            world_size=num_nodes,
            server_config=server_cfg,
            perf_config=perf_cfg,
            server_host=server_host,
            server_port=server_port,
        )


# =========================
# Loader
# =========================
def load_configs(
        path: Union[str, Path] = CONFIG_PATH) -> List[MultiNodeConfig]:
    """Load one or multiple configs from JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    raw = json.loads(path.read_text())
    configs_data = raw if isinstance(raw, list) else [raw]

    configs = []
    for idx, item in enumerate(configs_data):
        try:
            configs.append(MultiNodeConfig.from_config(item))
        except Exception as e:
            LOG.exception(f"Failed to parse config #{idx}: {e}")
            raise
    return configs
