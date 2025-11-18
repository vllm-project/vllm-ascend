import logging
import os
import socket
from contextlib import contextmanager
from typing import Optional

import psutil

# import torch.distributed as dist


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


# @contextmanager
# def dist_group(backend="gloo"):
#     if dist.is_initialized():
#         yield
#         return

#     dist.init_process_group(backend=backend)
#     try:
#         yield
#     finally:
#         dist.destroy_process_group()


def get_cluster_ips(word_size: int = 2) -> list[str]:
    """
    Returns the IP addresses of all nodes in the cluster.
    0: leader
    1~N-1: workers
    """
    leader_dns = os.getenv("LWS_LEADER_ADDRESS")
    if not leader_dns:
        raise RuntimeError("LWS_LEADER_ADDRESS is not set")
    cluster_dns = [leader_dns]
    for i in range(1, word_size):
        cur_dns = f"vllm-0-{i}.vllm.vllm-project"
        cluster_dns.append(cur_dns)
    return [socket.gethostbyname(dns) for dns in cluster_dns]


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


def get_cur_ip() -> str:
    """Returns the current machine's IP address."""
    return socket.gethostbyname_ex(socket.gethostname())[2][0]


def get_net_interface(ip: Optional[str] = None) -> Optional[str]:
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
    return None


def setup_logger():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
