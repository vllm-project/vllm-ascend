import logging
import os
import socket
from typing import Optional

import psutil


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


def generate_ranktable(word_size: int = 2, npus_per_node: int = 16):
    # gen_ranktable.sh --ips 172.19.32.175 172.19.241.49 \
    # --npus-per-node 16 --network-card-name eth0 --prefill-device-cnt 16 --decode-device-cnt 16
    # ips = get_cluster_ips(word_size)
    # iface = get_net_interface(ips[0])
    # if iface is None:
    #     raise RuntimeError("Failed to get network interface")
    # cmd = [
    #     "bash", "gen_ranktable.sh", "--ips", *ips, "--npus-per-node",
    #     str(npus_per_node), "--network-card-name", iface,
    #     "--prefill-device-cnt",
    #     str(npus_per_node), "--decode-device-cnt",
    #     str(npus_per_node)
    # ]
    pass


def setup_logger():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
