import logging
import os
from dataclasses import dataclass, field
from typing import Any

import regex as re

from tests.e2e.common.kv_pool.config import (
    KVPoolConfig,
    parse_kv_pool_config,
    validate_kv_pool_config,
)
from tests.e2e.common.multi_node.utils import (
    load_yaml_mapping,
    resolve_cluster_ips,
)
from tests.e2e.common.multi_node.utils import (
    resolve_current_node_index as resolve_node_index,
)

logger = logging.getLogger(__name__)

ROUTING_DISAGGREGATED_PREFILL = "disaggregated_prefill"
PROXY_SCRIPT_BY_ROUTING_TYPE = {
    ROUTING_DISAGGREGATED_PREFILL: "examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py",
}

CLUSTER_PLACEHOLDER_RE = re.compile(r"\$\{(NODE_(\d+)_IP|LOCAL_IP|MASTER_IP|LWS_WORKER_INDEX)\}")


@dataclass(frozen=True)
class RoutingConfig:
    """Proxy routing metadata shared by all external DP ranks."""

    type: str
    proxy_node_index: int
    proxy_host: str
    proxy_port: int
    proxy_script: str
    groups: dict[str, list[int]]


@dataclass(frozen=True)
class NodeInfo:
    """One server topology loaded from a config entry.

    Multiple entries may target the same ``node_index``; each entry then owns a
    disjoint NPU range starting at ``device_start``. This is what allows a
    single node to host both a prefiller and a decoder instance.
    """

    ip: str
    port_start: int
    dp_rpc_port: int
    dp_size: int
    dp_size_local: int
    dp_rank_start: int
    tp_size: int
    dp_address: str
    cp_size: int = 1
    sp_size: int = 1
    pp_size: int = 1
    pp_layer_partition: str = ""
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = ""
    master_port: int = 7060
    distributed_executor_backend: str = ""
    is_headless: bool = False
    node_index: int = 0
    device_start: int = 0

    @property
    def devices_per_rank(self) -> int:
        base = self.tp_size * self.cp_size * self.sp_size
        if self.nnodes > 1:
            return base
        return base * self.pp_size

    @property
    def devices_per_node(self) -> int:
        return self.dp_size_local * self.devices_per_rank


@dataclass(frozen=True)
class NodeTemplate:
    """Per-node env and argument template for launching vLLM servers."""

    envs: dict[str, Any]
    server_cmd_template: list[str]


@dataclass(frozen=True)
class RankInfo:
    """One concrete vLLM server rank expanded from a config entry."""

    config_index: int
    node_index: int
    role: str
    local_rank: int
    dp_rank: int
    host: str
    port: int
    visible_devices: str
    dp_size: int
    dp_size_local: int
    tp_size: int
    cp_size: int
    sp_size: int
    pp_size: int
    dp_address: str
    dp_rpc_port: int
    port_start: int
    pp_layer_partition: str
    nnodes: int
    node_rank: int
    master_addr: str
    master_port: int
    distributed_executor_backend: str
    is_headless: bool


@dataclass(frozen=True)
class ExternalDPConfig:
    """Top-level external DP test config after YAML anchors are merged."""

    test_name: str
    model: str
    num_nodes: int
    npu_per_node: int
    cluster_hosts: list[str] | None
    cluster_ips: list[str]
    routing: RoutingConfig
    nodes: list[NodeInfo]
    launch_templates: list[NodeTemplate]
    benchmark_cases: list[dict[str, Any]] = field(default_factory=list)
    special_dependencies: dict[str, str] = field(default_factory=dict)
    kv_pool: KVPoolConfig | None = None
    test_content: list[str] = field(default_factory=list)
    chat_prompts: list[Any] = field(default_factory=list)
    acceptance_rate: dict[str, Any] = field(default_factory=dict)
    api_keyword_args: dict[str, Any] | list | None = None
    expected_response: dict[str, Any] = field(default_factory=dict)

    @property
    def is_disaggregated_prefill(self) -> bool:
        return self.routing.type == ROUTING_DISAGGREGATED_PREFILL


def replace_cluster_placeholders(
    value: Any,
    *,
    cluster_ips: list[str],
    local_ip: str | None = None,
    current_node_index: int | None = None,
) -> Any:
    if isinstance(value, dict):
        return {
            key: replace_cluster_placeholders(
                val,
                cluster_ips=cluster_ips,
                local_ip=local_ip,
                current_node_index=current_node_index,
            )
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [
            replace_cluster_placeholders(
                item,
                cluster_ips=cluster_ips,
                local_ip=local_ip,
                current_node_index=current_node_index,
            )
            for item in value
        ]
    if not isinstance(value, str):
        return value

    def repl(match: re.Match[str]) -> str:
        token = match.group(1)
        node_index = match.group(2)
        if node_index is not None:
            idx = int(node_index)
            if idx >= len(cluster_ips):
                raise ValueError(f"Cluster placeholder ${{{token}}} is out of range")
            return cluster_ips[idx]
        if token == "MASTER_IP":
            return cluster_ips[0]
        if token == "LOCAL_IP":
            if local_ip is None:
                return match.group(0)
            return local_ip
        if token == "LWS_WORKER_INDEX":
            if current_node_index is None:
                return os.environ.get("LWS_WORKER_INDEX", match.group(0))
            return str(current_node_index)
        return match.group(0)

    return CLUSTER_PLACEHOLDER_RE.sub(repl, value)


def resolve_current_node_index(config: ExternalDPConfig) -> int:
    return resolve_node_index(config.cluster_ips)


class ExternalDPConfigLoader:
    """Load, normalize, and validate external DP YAML files."""

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | None = None,
        *,
        cluster_ips: list[str] | None = None,
    ) -> ExternalDPConfig:
        raw_config = cls._load_yaml(yaml_path)
        cls._validate_root(raw_config)

        num_nodes = int(raw_config["num_nodes"])
        resolved_cluster_ips = cls._resolve_cluster_ips(raw_config, num_nodes, cluster_ips)

        model = str(raw_config["model"])
        routing = cls._parse_routing(raw_config["routing"], resolved_cluster_ips)
        nodes = cls._parse_nodes(raw_config, resolved_cluster_ips)
        launch_templates = cls._parse_templates(raw_config)
        benchmark_cases = cls._parse_benchmarks(raw_config)
        kv_pool = cls._parse_kv_pool(raw_config)

        config = ExternalDPConfig(
            test_name=str(raw_config.get("test_name", "external_dp_test")),
            model=model,
            num_nodes=num_nodes,
            npu_per_node=int(raw_config["npu_per_node"]),
            cluster_hosts=raw_config.get("cluster_hosts"),
            cluster_ips=resolved_cluster_ips,
            routing=routing,
            nodes=nodes,
            launch_templates=launch_templates,
            benchmark_cases=benchmark_cases,
            special_dependencies=dict(raw_config.get("special_dependencies", {})),
            kv_pool=kv_pool,
            test_content=list(raw_config.get("test_content", [])),
            chat_prompts=list(raw_config.get("chat_prompts", [])),
            acceptance_rate=dict(raw_config.get("acceptance_rate", {})),
            api_keyword_args=raw_config.get("api_keyword_args"),
            expected_response=dict(raw_config.get("expected_response", {})),
        )
        cls._validate_config(config)
        return config

    @staticmethod
    def _load_yaml(yaml_path: str | None) -> dict[str, Any]:
        default_config_name = "GLM5_1-W8A8-EP-external.yaml"
        default_config_base_path = "tests/SIT/cases/models/configs/GLM/"
        return load_yaml_mapping(
            yaml_path,
            default_name=default_config_name,
            default_base_path=default_config_base_path,
            description="external DP config",
        )

    @staticmethod
    def _validate_root(config: dict[str, Any]) -> None:
        required = ["model", "num_nodes", "npu_per_node", "routing", "config", "templates", "benchmarks"]
        missing = [key for key in required if key not in config]
        if missing:
            raise KeyError(f"Missing required external DP config fields: {missing}")
        if int(config["num_nodes"]) <= 0:
            raise ValueError("num_nodes must be greater than 0")

    @staticmethod
    def _resolve_cluster_ips(
        raw_config: dict[str, Any],
        num_nodes: int,
        cluster_ips: list[str] | None,
    ) -> list[str]:
        return resolve_cluster_ips(
            raw_config,
            num_nodes,
            cluster_ips,
            dns_log_message="Resolving external DP cluster IPs via LWS DNS",
        )

    @staticmethod
    def _parse_routing(raw_routing: dict[str, Any], cluster_ips: list[str]) -> RoutingConfig:
        routing_type = str(raw_routing["type"])
        if routing_type not in PROXY_SCRIPT_BY_ROUTING_TYPE:
            raise ValueError(f"Unsupported routing.type: {routing_type}")

        proxy_node_index = 0
        proxy_port = 1999
        if proxy_node_index >= len(cluster_ips) or proxy_node_index < 0:
            raise ValueError("routing.proxy_node_index out of range")
        local_ip = cluster_ips[proxy_node_index]
        routing = replace_cluster_placeholders(
            raw_routing,
            cluster_ips=cluster_ips,
            local_ip=local_ip,
            current_node_index=proxy_node_index,
        )
        proxy_script = routing.get("proxy_script", PROXY_SCRIPT_BY_ROUTING_TYPE[routing_type])
        if not isinstance(proxy_script, str) or not proxy_script.strip():
            raise ValueError("routing.proxy_script must be a non-empty string")
        return RoutingConfig(
            type=routing_type,
            proxy_node_index=proxy_node_index,
            proxy_host=local_ip,
            proxy_port=proxy_port,
            proxy_script=proxy_script,
            groups={
                str(name): [int(index) for index in indices] for name, indices in routing.get("groups", {}).items()
            },
        )

    @staticmethod
    def _parse_nodes(raw_config: dict[str, Any], cluster_ips: list[str]) -> list[NodeInfo]:
        nodes: list[NodeInfo] = []
        for index, raw_node in enumerate(raw_config["config"]):
            node_index = int(raw_node.get("node_index", index))
            if node_index < 0 or node_index >= len(cluster_ips):
                raise ValueError(f"config[{index}].node_index out of range: {node_index}")
            node = replace_cluster_placeholders(
                raw_node,
                cluster_ips=cluster_ips,
                local_ip=cluster_ips[node_index],
                current_node_index=node_index,
            )
            nodes.append(
                NodeInfo(
                    ip=cluster_ips[node_index],
                    port_start=int(node["port_start"]),
                    dp_rpc_port=int(node["dp_rpc_port"]),
                    dp_size=int(node.get("dp_size", 1)),
                    dp_size_local=int(node.get("dp_size_local", 1)),
                    dp_rank_start=int(node.get("dp_rank_start", 0)),
                    tp_size=int(node.get("tp_size", 1)),
                    cp_size=int(node.get("cp_size", 1)),
                    sp_size=int(node.get("sp_size", 1)),
                    dp_address=str(node["dp_address"]),
                    pp_size=int(node.get("pp_size", 1)),
                    pp_layer_partition=str(node.get("pp_layer_partition", "")),
                    nnodes=int(node.get("nnodes", 1)),
                    node_rank=int(node.get("node_rank", 0)),
                    master_addr=str(node.get("master_addr", "")),
                    master_port=int(node.get("master_port", 7060)),
                    distributed_executor_backend=str(node.get("distributed_executor_backend", "")),
                    is_headless=bool(node.get("is_headless", False)),
                    node_index=node_index,
                    device_start=int(node.get("device_start", 0)),
                )
            )
        return nodes

    @staticmethod
    def _parse_templates(raw_config: dict[str, Any]) -> list[NodeTemplate]:
        templates: list[NodeTemplate] = []
        for index, raw_template in enumerate(raw_config["templates"]):
            envs = raw_template.get("envs")
            server_cmd_template = raw_template.get("server_cmd_template")
            if envs is None or server_cmd_template is None:
                raise KeyError(f"templates[{index}] must contain envs and server_cmd_template")
            if not isinstance(server_cmd_template, list):
                raise TypeError(f"templates[{index}].server_cmd_template must be a list")
            templates.append(
                NodeTemplate(
                    envs=dict(envs),
                    server_cmd_template=[str(arg) for arg in server_cmd_template],
                )
            )
        return templates

    @staticmethod
    def _parse_benchmarks(raw_config: dict[str, Any]) -> list[dict[str, Any]]:
        benchmark_cases: list[dict[str, Any]] = []
        for name, case in (raw_config.get("benchmarks") or {}).items():
            case_with_name = dict(case)
            case_with_name["benchmark_name"] = name
            case_with_name["benchmark_artifact_name"] = f"{raw_config.get('test_name', 'external_dp_test')}__{name}"
            benchmark_cases.append(case_with_name)
        return benchmark_cases

    @staticmethod
    def _parse_kv_pool(raw_config: dict[str, Any]) -> KVPoolConfig | None:
        return parse_kv_pool_config(raw_config.get("kv_pool"))

    @classmethod
    def _validate_config(cls, config: ExternalDPConfig) -> None:
        cls._validate_config_sizes(config)
        cls._validate_routing(config)
        cls._validate_node_parallel_config(config)
        cls._validate_kv_pool(config)

    @staticmethod
    def _validate_config_sizes(config: ExternalDPConfig) -> None:
        expected_node_indices = set(range(config.num_nodes))
        actual_node_indices = {node.node_index for node in config.nodes}
        if actual_node_indices != expected_node_indices:
            raise AssertionError(
                f"config node_index coverage {sorted(actual_node_indices)} != [0, {config.num_nodes - 1}]"
            )
        if len(config.launch_templates) != len(config.nodes):
            raise AssertionError(
                f"templates size ({len(config.launch_templates)}) != config size ({len(config.nodes)})"
            )
        if config.cluster_hosts and len(config.cluster_hosts) != config.num_nodes:
            raise AssertionError("cluster_hosts size mismatch")

    @staticmethod
    def _validate_routing(config: ExternalDPConfig) -> None:
        if config.routing.type not in PROXY_SCRIPT_BY_ROUTING_TYPE:
            raise ValueError(f"Unsupported routing.type: {config.routing.type}")

        groups = config.routing.groups
        if config.routing.type == ROUTING_DISAGGREGATED_PREFILL and (
            not groups.get("prefiller") or not groups.get("decoder")
        ):
            raise ValueError("disaggregated_prefill routing requires prefiller and decoder groups")

        seen_group_indices: dict[int, str] = {}
        for group_name, indices in groups.items():
            for index in indices:
                if index < 0 or index >= len(config.nodes):
                    raise ValueError(f"routing.groups.{group_name} index out of range: {index}")
                if index in seen_group_indices:
                    raise ValueError(
                        f"config index {index} appears in both {seen_group_indices[index]} and {group_name}"
                    )
                seen_group_indices[index] = group_name

        missing = [index for index in range(len(config.nodes)) if index not in seen_group_indices]
        if missing:
            raise ValueError(f"routing.groups does not assign a role for config indices: {missing}")

        if config.routing.proxy_node_index < 0 or config.routing.proxy_node_index >= config.num_nodes:
            raise ValueError("routing.proxy_node_index out of range")

    @staticmethod
    def _validate_node_parallel_config(config: ExternalDPConfig) -> None:
        device_ranges: dict[int, list[tuple[int, int, int]]] = {}
        port_ranges: dict[int, list[tuple[int, int, int]]] = {}
        rpc_ports: dict[int, dict[int, int]] = {}
        for config_index, node in enumerate(config.nodes):
            parallel_sizes = {
                "dp_size": node.dp_size,
                "dp_size_local": node.dp_size_local,
                "tp_size": node.tp_size,
                "cp_size": node.cp_size,
                "sp_size": node.sp_size,
                "pp_size": node.pp_size,
            }
            invalid_sizes = {name: value for name, value in parallel_sizes.items() if value < 1}
            if invalid_sizes:
                raise ValueError(f"config[{config_index}] parallel sizes must be >= 1: {invalid_sizes}")
            if node.dp_rank_start < 0:
                raise ValueError(f"config[{config_index}] dp_rank_start must be >= 0")
            if node.device_start < 0:
                raise ValueError(f"config[{config_index}] device_start must be >= 0")
            device_start = node.device_start
            device_end = device_start + node.devices_per_node
            if device_end > config.npu_per_node:
                raise ValueError(
                    f"config[{config_index}] on node {node.node_index} uses NPUs "
                    f"[{device_start}, {device_end}), but npu_per_node is {config.npu_per_node}"
                )
            for other_start, other_end, other_config_index in device_ranges.setdefault(node.node_index, []):
                if device_start < other_end and other_start < device_end:
                    raise ValueError(
                        f"config[{config_index}] NPU range [{device_start}, {device_end}) overlaps "
                        f"config[{other_config_index}] [{other_start}, {other_end}) on node {node.node_index}"
                    )
            device_ranges[node.node_index].append((device_start, device_end, config_index))
            port_start = node.port_start
            port_end = port_start + node.dp_size_local
            for other_start, other_end, other_config_index in port_ranges.setdefault(node.node_index, []):
                if port_start < other_end and other_start < port_end:
                    raise ValueError(
                        f"config[{config_index}] API port range [{port_start}, {port_end}) overlaps "
                        f"config[{other_config_index}] [{other_start}, {other_end}) on node {node.node_index}"
                    )
            port_ranges[node.node_index].append((port_start, port_end, config_index))
            node_rpc_ports = rpc_ports.setdefault(node.node_index, {})
            if node.dp_rpc_port in node_rpc_ports:
                raise ValueError(
                    f"config[{config_index}] dp_rpc_port {node.dp_rpc_port} conflicts with "
                    f"config[{node_rpc_ports[node.dp_rpc_port]}] on node {node.node_index}"
                )
            node_rpc_ports[node.dp_rpc_port] = config_index
            if node.dp_rank_start + node.dp_size_local > node.dp_size:
                raise ValueError(f"config[{config_index}] dp rank range exceeds dp_size")
            if node.nnodes > 1 and node.node_rank > 0 and not node.is_headless:
                raise ValueError(f"config[{config_index}] is PP worker but not headless")
            if node.nnodes > 1 and not node.master_addr:
                raise ValueError(f"config[{config_index}] uses multi-node PP but master_addr is empty")

    @staticmethod
    def _validate_kv_pool(config: ExternalDPConfig) -> None:
        kv_pool = config.kv_pool
        if kv_pool is None:
            return
        validate_kv_pool_config(kv_pool)


class RankResolver:
    """Expand config entries into concrete vLLM server ranks."""

    def __init__(self, config: ExternalDPConfig):
        self.config = config

    def resolve(self) -> list[RankInfo]:
        role_by_config_index = self._role_by_config_index()
        ranks: list[RankInfo] = []
        for config_index, node_info in enumerate(self.config.nodes):
            role = role_by_config_index[config_index]
            ranks.extend(self._expand_config(config_index, role, node_info))
        return ranks

    def _role_by_config_index(self) -> dict[int, str]:
        role_by_index: dict[int, str] = {}
        for role, config_indices in self.config.routing.groups.items():
            for index in config_indices:
                role_by_index[index] = role

        missing = [index for index in range(len(self.config.nodes)) if index not in role_by_index]
        if missing:
            raise ValueError(f"routing.groups does not assign role for config indices: {missing}")
        return role_by_index

    @staticmethod
    def _expand_config(config_index: int, role: str, node_info: NodeInfo) -> list[RankInfo]:
        ranks: list[RankInfo] = []
        for local_rank in range(node_info.dp_size_local):
            dp_rank = node_info.dp_rank_start + local_rank
            port = node_info.port_start + local_rank
            first_device = node_info.device_start + local_rank * node_info.devices_per_rank
            device_range = range(first_device, first_device + node_info.devices_per_rank)
            visible_devices = ",".join(str(device) for device in device_range)
            ranks.append(
                RankInfo(
                    config_index=config_index,
                    node_index=node_info.node_index,
                    role=role,
                    local_rank=local_rank,
                    dp_rank=dp_rank,
                    host=node_info.ip,
                    port=port,
                    visible_devices=visible_devices,
                    dp_size=node_info.dp_size,
                    dp_size_local=node_info.dp_size_local,
                    tp_size=node_info.tp_size,
                    cp_size=node_info.cp_size,
                    sp_size=node_info.sp_size,
                    pp_size=node_info.pp_size,
                    dp_address=node_info.dp_address,
                    dp_rpc_port=node_info.dp_rpc_port,
                    port_start=node_info.port_start,
                    pp_layer_partition=node_info.pp_layer_partition,
                    nnodes=node_info.nnodes,
                    node_rank=node_info.node_rank,
                    master_addr=node_info.master_addr,
                    master_port=node_info.master_port,
                    distributed_executor_backend=node_info.distributed_executor_backend,
                    is_headless=node_info.is_headless,
                )
            )
        return ranks
