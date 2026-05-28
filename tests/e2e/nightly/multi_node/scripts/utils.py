import logging
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@contextmanager
def temp_env(env_dict: dict[str, Any]):
    old_env = {}
    for key, value in env_dict.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_yaml_mapping(
    yaml_path: str | None,
    *,
    default_name: str,
    default_base_path: str,
    description: str,
) -> dict[str, Any]:
    if not yaml_path:
        yaml_path = os.getenv("CONFIG_YAML_PATH", default_name)

    path = Path(yaml_path)
    if not path.is_absolute() and not path.exists():
        base_path = os.getenv("CONFIG_BASE_PATH") or default_base_path
        path = Path(base_path) / yaml_path

    logger.info("Loading %s yaml: %s", description, path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{description} must be a mapping: {path}")
    return data


def dns_resolver(retries: int = 240, base_delay: float = 0.5):
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


def get_cluster_dns_list(world_size: int) -> list[str]:
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


def get_cluster_ips(world_size: int = 2) -> list[str]:
    resolver = dns_resolver()
    return [resolver(dns) for dns in get_cluster_dns_list(world_size)]


def resolve_cluster_ips(
    raw_config: dict[str, Any],
    num_nodes: int,
    explicit_cluster_ips: list[str] | None = None,
    *,
    cluster_hosts_log_message: str | None = None,
    dns_log_message: str = "Resolving cluster IPs via DNS...",
) -> list[str]:
    if explicit_cluster_ips is not None:
        if len(explicit_cluster_ips) != num_nodes:
            raise AssertionError("cluster_ips size mismatch")
        return explicit_cluster_ips

    cluster_hosts = raw_config.get("cluster_hosts")
    if cluster_hosts:
        if cluster_hosts_log_message:
            logger.info(cluster_hosts_log_message)
        if len(cluster_hosts) != num_nodes:
            raise AssertionError("cluster_hosts size mismatch")
        return list(cluster_hosts)

    logger.info(dns_log_message)
    return get_cluster_ips(num_nodes)


def get_available_port(start_port: int = 6000, end_port: int = 7000) -> int:
    for port in range(start_port, end_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available port found")


def get_cur_ip(retries: int = 20, base_delay: float = 0.5) -> str:
    delay = base_delay
    for attempt in range(retries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            try:
                return socket.gethostbyname(socket.gethostname())
            except Exception:
                if attempt == retries - 1:
                    raise RuntimeError("Failed to determine local IP address")
                time.sleep(delay)
                delay = min(delay * 1.5, 5)
    raise RuntimeError("Failed to determine local IP address")


def get_net_interface(ip: str | None = None) -> str:
    import psutil

    if ip is None:
        ip = get_cur_ip()

    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and addr.address == ip:
                return iface
    raise RuntimeError(f"No network interface found for IP {ip}")


def get_all_ipv4() -> list[str]:
    ipv4s = {"127.0.0.1"}
    hostname = socket.gethostname()
    for info in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
        ipv4s.add(info[4][0])
    return list(ipv4s)


def resolve_current_node_index(cluster_ips: list[str]) -> int:
    worker_index = os.environ.get("LWS_WORKER_INDEX")
    if worker_index:
        return int(worker_index)

    local_ips = set(get_all_ipv4())
    for index, ip in enumerate(cluster_ips):
        if ip in local_ips:
            return index
    raise RuntimeError("Unable to determine current node index")
