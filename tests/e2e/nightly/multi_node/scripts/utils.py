import logging
import os
import socket
import time
from contextlib import contextmanager
from typing import List, Optional

import psutil

DISAGGEGATED_PREFILL_PORT = 5333
CONFIG_BASE_PATH = "tests/e2e/nightly/multi_node/config/"
DEFAULT_SERVER_PORT = 8080


@contextmanager
def temp_env(env_dict):
    old_env = {}
    for k, v in env_dict.items():
        old_env[k] = os.environ.get(k)
        os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def dns_resolver(retries: int = 240, base_delay: float = 0.5):
    # We should resolve DNS with retries to avoid transient network issues.
    # When the pod is just started, DNS resolution may fail.
    def resolve(dns: str):
        delay = base_delay
        for attempt in range(retries):
            try:
                return socket.gethostbyname(dns)
            except socket.gaierror:
                if attempt == retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 1.5, 5)

    return resolve


def get_cluster_dns_list(world_size: int) -> List[str]:
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")

    leader_dns = os.getenv("LWS_LEADER_ADDRESS")
    if not leader_dns:
        raise RuntimeError(
            "environment variable LWS_LEADER_ADDRESS is not set")

    # Expected format:
    # <leader-name>.<group-name>.<namespace>
    parts = leader_dns.split(".")
    if len(parts) < 3:
        raise ValueError(f"invalid leader DNS format: {leader_dns}")

    leader_name, group_name, namespace = parts[0], parts[1], parts[2]

    worker_dns_list = [
        f"{leader_name}-{idx}.{group_name}.{namespace}"
        for idx in range(1, world_size)
    ]

    return [leader_dns, *worker_dns_list]


def get_cluster_ips(word_size: int = 2) -> list[str]:
    resolver = dns_resolver()
    return [resolver(dns) for dns in get_cluster_dns_list(word_size)]


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


def get_cur_ip(retries: int = 20, base_delay: float = 0.5):
    """
    Returns the pod/machine's primary IP address with retry.
    This is necessary because network interfaces may not be ready
    immediately after container startup.
    """
    delay = base_delay

    for attempt in range(retries):
        try:
            # Best method: UDP trick (doesn't actually send packets)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            # fallback: hostname resolution
            try:
                return socket.gethostbyname(socket.gethostname())
            except Exception:
                if attempt == retries - 1:
                    raise RuntimeError("Failed to determine local IP address")
                time.sleep(delay)
                delay = min(delay * 1.5, 5)


def get_net_interface(ip: Optional[str] = None) -> str:
    """
    Returns specified IP's inetwork interface.
    If no IP is provided, uses the first from hostname -I.
    """
    if ip is None:
        ip = get_cur_ip()

    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and addr.address == ip:
                return iface
    raise RuntimeError(f"No network interface found for IP {ip}")


def get_all_ipv4():
    """get all the ipv4 address for current node"""
    ipv4s = set()
    hostname = socket.gethostname()

    for info in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
        ipv4s.add(info[4][0])

    ipv4s.add("127.0.0.1")

    return list(ipv4s)


def setup_logger():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
