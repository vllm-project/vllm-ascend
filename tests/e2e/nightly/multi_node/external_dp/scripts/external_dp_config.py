import logging
import os
import re
import socket
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_BASE_PATH = "tests/e2e/nightly/multi_node/external_dp/config/"
DEFAULT_CONFIG_NAME = "generic_dp_smoke.yaml"
BACKEND_HEALTHCHECK_PATH = "/health"
BACKEND_READY_TIMEOUT = 3600
ROUTING_GENERIC_DP = "generic_dp"
ROUTING_PD = "disaggregated_prefill"
SUPPORTED_ROUTING_TYPES = {ROUTING_GENERIC_DP, ROUTING_PD}
CLUSTER_PLACEHOLDER_RE = re.compile(r"\$\{(NODE_(\d+)_IP|LOCAL_IP|MASTER_IP|LWS_WORKER_INDEX)\}")


@dataclass(frozen=True)
class RoutingConfig:
    """Proxy routing metadata shared by all external DP endpoints."""

    type: str
    proxy_node_index: int
    proxy_host: str
    proxy_port: int
    proxy_script: str
    groups: dict[str, list[int]]


@dataclass(frozen=True)
class NodeConfig:
    """Per-node backend topology loaded from one config entry."""

    node_ip: str
    host: str
    port_start: int
    dp_rpc_port: int
    dp_group: str
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
    """Per-node env and argument template for launching vLLM backends."""

    envs: dict[str, Any]
    server_cmd_template: list[str]


@dataclass(frozen=True)
class ExternalDPEndpoint:
    """One concrete backend process expanded from a node config."""

    config_index: int
    role: str
    dp_group: str
    local_rank: int
    dp_rank: int
    host: str
    bind_host: str
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
    node_configs: list[NodeConfig]
    templates: list[NodeTemplate]
    benchmarks: dict[str, Any] = field(default_factory=dict)

    @property
    def is_disaggregated_prefill(self) -> bool:
        return self.routing.type == ROUTING_PD

    def benchmark_cases(self) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        for name, case in self.benchmarks.items():
            case_with_name = deepcopy(case)
            case_with_name["case_name"] = name
            cases.append(case_with_name)
        return cases


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
    worker_index = os.environ.get("LWS_WORKER_INDEX")
    if worker_index:
        return int(worker_index)

    local_ips = set(_get_all_ipv4())
    for index, ip in enumerate(config.cluster_ips):
        if ip in local_ips:
            return index
    raise RuntimeError("Unable to determine current node index")


def _dns_resolver(retries: int = 240, base_delay: float = 0.5):
    def resolve(dns: str) -> str:
        delay = base_delay
        for attempt in range(retries):
            try:
                return socket.gethostbyname(dns)
            except socket.gaierror:
                if attempt == retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 1.5, 5)
        raise RuntimeError(f"Unable to resolve DNS: {dns}")

    return resolve


def _get_cluster_dns_list(world_size: int) -> list[str]:
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")

    leader_dns = os.getenv("LWS_LEADER_ADDRESS")
    if not leader_dns:
        raise RuntimeError("environment variable LWS_LEADER_ADDRESS is not set")

    parts = leader_dns.split(".")
    if len(parts) < 3:
        raise ValueError(f"invalid leader DNS format: {leader_dns}")

    leader_name, group_name, namespace = parts[0], parts[1], parts[2]
    worker_dns_list = [f"{leader_name}-{idx}.{group_name}.{namespace}" for idx in range(1, world_size)]
    return [leader_dns, *worker_dns_list]


def _get_cluster_ips(world_size: int) -> list[str]:
    resolver = _dns_resolver()
    return [resolver(dns) for dns in _get_cluster_dns_list(world_size)]


def _get_all_ipv4() -> list[str]:
    ipv4s = {"127.0.0.1"}
    hostname = socket.gethostname()
    for info in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
        ipv4s.add(info[4][0])
    return list(ipv4s)


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
        node_configs = cls._parse_node_configs(raw_config, resolved_cluster_ips)
        templates = cls._parse_templates(raw_config)

        config = ExternalDPConfig(
            test_name=str(raw_config.get("test_name", "external_dp_test")),
            model=model,
            num_nodes=num_nodes,
            npu_per_node=int(raw_config["npu_per_node"]),
            cluster_hosts=raw_config.get("cluster_hosts"),
            cluster_ips=resolved_cluster_ips,
            routing=routing,
            node_configs=node_configs,
            templates=templates,
            benchmarks=raw_config.get("benchmarks", {}),
        )
        cls._validate_config(config)
        return config

    @staticmethod
    def _load_yaml(yaml_path: str | None) -> dict[str, Any]:
        if not yaml_path:
            yaml_path = os.getenv("CONFIG_YAML_PATH", DEFAULT_CONFIG_NAME)

        path = Path(yaml_path)
        if not path.is_absolute() and not path.exists():
            base_path = os.getenv("CONFIG_BASE_PATH") or DEFAULT_CONFIG_BASE_PATH
            path = Path(base_path) / yaml_path

        logger.info("Loading external DP config yaml: %s", path)
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise TypeError(f"External DP config must be a mapping: {path}")
        return data

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
        if cluster_ips is not None:
            if len(cluster_ips) != num_nodes:
                raise AssertionError("cluster_ips size mismatch")
            return cluster_ips

        cluster_hosts = raw_config.get("cluster_hosts")
        if cluster_hosts:
            if len(cluster_hosts) != num_nodes:
                raise AssertionError("cluster_hosts size mismatch")
            return list(cluster_hosts)

        logger.info("Resolving external DP cluster IPs via LWS DNS")
        return _get_cluster_ips(num_nodes)

    @staticmethod
    def _parse_routing(raw_routing: dict[str, Any], cluster_ips: list[str]) -> RoutingConfig:
        proxy_node_index = int(raw_routing.get("proxy_node_index", 0))
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
            type=str(routing["type"]),
            proxy_node_index=proxy_node_index,
            proxy_host=str(routing.get("proxy_host", cluster_ips[proxy_node_index])),
            proxy_port=int(routing["proxy_port"]),
            proxy_script=str(routing["proxy_script"]),
            groups={
                str(name): [int(index) for index in indices]
                for name, indices in routing.get("groups", {}).items()
            },
        )

    @staticmethod
    def _parse_node_configs(raw_config: dict[str, Any], cluster_ips: list[str]) -> list[NodeConfig]:
        node_configs: list[NodeConfig] = []
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
            node_configs.append(
                NodeConfig(
                    node_ip=cluster_ips[index],
                    host=str(node["host"]),
                    port_start=int(node["port_start"]),
                    dp_rpc_port=int(node["dp_rpc_port"]),
                    dp_group=str(node.get("dp_group", "default")),
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
        return node_configs

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
    def _validate_config(config: ExternalDPConfig) -> None:
        if len(config.node_configs) != config.num_nodes:
            raise AssertionError(f"config size ({len(config.node_configs)}) != num_nodes ({config.num_nodes})")
        if len(config.templates) != config.num_nodes:
            raise AssertionError(f"templates size ({len(config.templates)}) != num_nodes ({config.num_nodes})")

        if config.routing.type not in SUPPORTED_ROUTING_TYPES:
            raise ValueError(f"Unsupported routing.type: {config.routing.type}")
        groups = config.routing.groups
        if config.routing.type == ROUTING_GENERIC_DP and not groups.get("worker"):
            raise ValueError("generic_dp routing requires routing.groups.worker")
        if config.routing.type == ROUTING_PD and (not groups.get("prefiller") or not groups.get("decoder")):
            raise ValueError("disaggregated_prefill routing requires prefiller and decoder groups")

        seen_group_indices: dict[int, str] = {}
        for group_name, indices in groups.items():
            for index in indices:
                if index < 0 or index >= config.num_nodes:
                    raise ValueError(f"routing.groups.{group_name} index out of range: {index}")
                if index in seen_group_indices:
                    raise ValueError(
                        f"config index {index} appears in both {seen_group_indices[index]} and {group_name}"
                    )
                seen_group_indices[index] = group_name

        if config.routing.proxy_node_index < 0 or config.routing.proxy_node_index >= config.num_nodes:
            raise ValueError("routing.proxy_node_index out of range")
        if config.cluster_hosts and len(config.cluster_hosts) != config.num_nodes:
            raise AssertionError("cluster_hosts size mismatch")

        for node_index, node in enumerate(config.node_configs):
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
                    f"node {node_index} uses {node.devices_per_node} NPUs, "
                    f"but npu_per_node is {config.npu_per_node}"
                )
            if node.dp_rank_start + node.dp_size_local > node.dp_size:
                raise ValueError(f"node {node_index} dp rank range exceeds dp_size")
            ports = [node.port_start + local_rank for local_rank in range(node.dp_size_local)]
            if len(set(ports)) != len(ports):
                raise ValueError(f"node {node_index} has duplicate endpoint ports")

            used_devices: set[int] = set()
            for local_rank in range(node.dp_size_local):
                devices = range(
                    local_rank * node.devices_per_rank,
                    (local_rank + 1) * node.devices_per_rank,
                )
                overlap = used_devices.intersection(devices)
                if overlap:
                    raise ValueError(f"node {node_index} visible_devices overlap: {sorted(overlap)}")
                used_devices.update(devices)


class EndpointResolver:
    """Expand node-level configs into backend endpoints."""

    def __init__(self, config: ExternalDPConfig):
        self.config = config

    def resolve(self) -> list[ExternalDPEndpoint]:
        role_by_config_index = self._role_by_config_index()
        endpoints: list[ExternalDPEndpoint] = []
        for config_index, node_config in enumerate(self.config.node_configs):
            role = role_by_config_index[config_index]
            endpoints.extend(self._expand_node(config_index, role, node_config))
        return endpoints

    def _role_by_config_index(self) -> dict[int, str]:
        role_by_index: dict[int, str] = {}
        for role, config_indices in self.config.routing.groups.items():
            for index in config_indices:
                role_by_index[index] = role

        missing = [index for index in range(self.config.num_nodes) if index not in role_by_index]
        if missing:
            raise ValueError(f"routing.groups does not assign role for config indices: {missing}")
        return role_by_index

    @staticmethod
    def _expand_node(node_index: int, role: str, node_config: NodeConfig) -> list[ExternalDPEndpoint]:
        endpoints: list[ExternalDPEndpoint] = []
        for local_rank in range(node_config.dp_size_local):
            dp_rank = node_config.dp_rank_start + local_rank
            port = node_config.port_start + local_rank
            device_range = range(
                local_rank * node_config.devices_per_rank,
                (local_rank + 1) * node_config.devices_per_rank,
            )
            visible_devices = ",".join(str(device) for device in device_range)
            endpoints.append(
                ExternalDPEndpoint(
                    config_index=node_index,
                    role=role,
                    dp_group=node_config.dp_group,
                    local_rank=local_rank,
                    dp_rank=dp_rank,
                    host=node_config.node_ip,
                    bind_host=node_config.host,
                    port=port,
                    visible_devices=visible_devices,
                    dp_size=node_config.dp_size,
                    dp_size_local=node_config.dp_size_local,
                    tp_size=node_config.tp_size,
                    cp_size=node_config.cp_size,
                    sp_size=node_config.sp_size,
                    pp_size=node_config.pp_size,
                    dp_address=node_config.dp_address,
                    dp_rpc_port=node_config.dp_rpc_port,
                    port_start=node_config.port_start,
                )
            )
        return endpoints
