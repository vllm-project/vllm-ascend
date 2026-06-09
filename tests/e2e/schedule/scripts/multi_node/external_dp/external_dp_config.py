import logging
import os
from dataclasses import dataclass, field
from typing import Any

import regex as re

from tests.e2e.schedule.scripts.multi_node.utils import (
    load_yaml_mapping,
    resolve_cluster_ips,
)
from tests.e2e.schedule.scripts.multi_node.utils import (
    resolve_current_node_index as resolve_node_index,
)

logger = logging.getLogger(__name__)

ROUTING_GENERIC_DP = "generic_dp"
ROUTING_DISAGGREGATED_PREFILL = "disaggregated_prefill"
PROXY_SCRIPT_BY_ROUTING_TYPE = {
    ROUTING_GENERIC_DP: "examples/external_online_dp/dp_load_balance_proxy_server.py",
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
    """Per-node external DP server topology loaded from one config entry."""

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

    @property
    def devices_per_rank(self) -> int:
        return self.tp_size * self.cp_size * self.sp_size * self.pp_size

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
    """One concrete vLLM server rank expanded from a node config."""

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
        )
        cls._validate_config(config)
        return config

    @staticmethod
    def _load_yaml(yaml_path: str | None) -> dict[str, Any]:
        default_config_name = "GLM5_1-W8A8-EP-external.yaml"
        default_config_base_path = "tests/e2e/schedule/scripts/multi_node/external_dp/config/"
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
        return RoutingConfig(
            type=routing_type,
            proxy_node_index=proxy_node_index,
            proxy_host=local_ip,
            proxy_port=proxy_port,
            proxy_script=PROXY_SCRIPT_BY_ROUTING_TYPE[routing_type],
            groups={
                str(name): [int(index) for index in indices] for name, indices in routing.get("groups", {}).items()
            },
        )

    @staticmethod
    def _parse_nodes(raw_config: dict[str, Any], cluster_ips: list[str]) -> list[NodeInfo]:
        nodes: list[NodeInfo] = []
        for index, raw_node in enumerate(raw_config["config"]):
            raw_node_index = raw_node.get("node_index")
            if raw_node_index is not None and int(raw_node_index) != index:
                raise ValueError(f"config[{index}].node_index must equal {index}")
            node = replace_cluster_placeholders(
                raw_node,
                cluster_ips=cluster_ips,
                local_ip=cluster_ips[index],
                current_node_index=index,
            )
            nodes.append(
                NodeInfo(
                    ip=cluster_ips[index],
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
            case_with_name["case_name"] = name
            benchmark_cases.append(case_with_name)
        return benchmark_cases

    @classmethod
    def _validate_config(cls, config: ExternalDPConfig) -> None:
        cls._validate_config_sizes(config)
        cls._validate_routing(config)
        cls._validate_node_parallel_config(config)

    @staticmethod
    def _validate_config_sizes(config: ExternalDPConfig) -> None:
        if len(config.nodes) != config.num_nodes:
            raise AssertionError(f"config size ({len(config.nodes)}) != num_nodes ({config.num_nodes})")
        if len(config.launch_templates) != config.num_nodes:
            raise AssertionError(f"templates size ({len(config.launch_templates)}) != num_nodes ({config.num_nodes})")
        if config.cluster_hosts and len(config.cluster_hosts) != config.num_nodes:
            raise AssertionError("cluster_hosts size mismatch")

    @staticmethod
    def _validate_routing(config: ExternalDPConfig) -> None:
        if config.routing.type not in PROXY_SCRIPT_BY_ROUTING_TYPE:
            raise ValueError(f"Unsupported routing.type: {config.routing.type}")

        groups = config.routing.groups
        if config.routing.type == ROUTING_GENERIC_DP and not groups.get("worker"):
            raise ValueError("generic_dp routing requires routing.groups.worker")
        if config.routing.type == ROUTING_DISAGGREGATED_PREFILL and (
            not groups.get("prefiller") or not groups.get("decoder")
        ):
            raise ValueError("disaggregated_prefill routing requires prefiller and decoder groups")

        seen_group_indices: dict[int, str] = {}
        for group_name, indices in groups.items():
            for index in indices:
                if index < 0 or index >= config.num_nodes:
                    raise ValueError(f"routing.groups.{group_name} index out of range: {index}")
                if index in seen_group_indices:
                    raise ValueError(f"node index {index} appears in both {seen_group_indices[index]} and {group_name}")
                seen_group_indices[index] = group_name

        if config.routing.proxy_node_index < 0 or config.routing.proxy_node_index >= config.num_nodes:
            raise ValueError("routing.proxy_node_index out of range")

    @staticmethod
    def _validate_node_parallel_config(config: ExternalDPConfig) -> None:
        for node_index, node in enumerate(config.nodes):
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
                raise ValueError(f"node {node_index} parallel sizes must be >= 1: {invalid_sizes}")
            if node.dp_rank_start < 0:
                raise ValueError(f"node {node_index} dp_rank_start must be >= 0")
            if node.devices_per_node > config.npu_per_node:
                raise ValueError(
                    f"node {node_index} uses {node.devices_per_node} NPUs, but npu_per_node is {config.npu_per_node}"
                )
            if node.dp_rank_start + node.dp_size_local > node.dp_size:
                raise ValueError(f"node {node_index} dp rank range exceeds dp_size")


class RankResolver:
    """Expand node-level configs into concrete vLLM server ranks."""

    def __init__(self, config: ExternalDPConfig):
        self.config = config

    def resolve(self) -> list[RankInfo]:
        role_by_node_index = self._role_by_node_index()
        ranks: list[RankInfo] = []
        for node_index, node_info in enumerate(self.config.nodes):
            role = role_by_node_index[node_index]
            ranks.extend(self._expand_node(node_index, role, node_info))
        return ranks

    def _role_by_node_index(self) -> dict[int, str]:
        role_by_index: dict[int, str] = {}
        for role, node_indices in self.config.routing.groups.items():
            for index in node_indices:
                role_by_index[index] = role

        missing = [index for index in range(self.config.num_nodes) if index not in role_by_index]
        if missing:
            raise ValueError(f"routing.groups does not assign role for node indices: {missing}")
        return role_by_index

    @staticmethod
    def _expand_node(node_index: int, role: str, node_info: NodeInfo) -> list[RankInfo]:
        ranks: list[RankInfo] = []
        for local_rank in range(node_info.dp_size_local):
            dp_rank = node_info.dp_rank_start + local_rank
            port = node_info.port_start + local_rank
            device_range = range(
                local_rank * node_info.devices_per_rank,
                (local_rank + 1) * node_info.devices_per_rank,
            )
            visible_devices = ",".join(str(device) for device in device_range)
            ranks.append(
                RankInfo(
                    node_index=node_index,
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
                )
            )
        return ranks
