import os
import socket
import subprocess
from typing import Optional, Tuple

import psutil


def get_leader_ip():
    leader_dns = os.getenv("LWS_LEADER_ADDRESS")
    assert leader_dns is not None, "cannot find leader address"
    return socket.gethostbyname(leader_dns)


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
        "NUMEXPR_MAX_THREADS": "100",
    }


def generate_ranktable():
    pass
