import os

from tests.e2e.nightly.multi_node.scripts.utils import (
    get_all_ipv4,
    get_available_port,
    get_cluster_ips,
    get_net_interface,
    setup_logger,
    temp_env,
)

DISAGGEGATED_PREFILL_PORT = 5333
DEFAULT_CONFIG_BASE_PATH = "tests/e2e/nightly/multi_node/internal_dp/config/"
CONFIG_BASE_PATH = os.getenv("CONFIG_BASE_PATH") or DEFAULT_CONFIG_BASE_PATH
DEFAULT_SERVER_PORT = 8080

__all__ = [
    "CONFIG_BASE_PATH",
    "DEFAULT_CONFIG_BASE_PATH",
    "DEFAULT_SERVER_PORT",
    "DISAGGEGATED_PREFILL_PORT",
    "get_all_ipv4",
    "get_available_port",
    "get_cluster_ips",
    "get_net_interface",
    "setup_logger",
    "temp_env",
]
