import logging
import os
import socket
import subprocess

from tests.e2e.multi_node.config.common import (ASCEND_ENV_PATH,
                                                DISAGGEGATED_PREFILL_PORT,
                                                LIB_PATH,
                                                LOAD_BALANCER_PROXY_SCRIPT,
                                                RANKTABLE_GEN_PATH,
                                                RANKTABLE_PATH)
from tests.e2e.multi_node.config.multi_node_config import MultiNodeConfig
from tests.e2e.multi_node.config.utils import (get_cluster_ips, get_cur_ip,
                                               get_net_interface, setup_logger)

setup_logger()
logger = logging.getLogger(__name__)


class DisaggegatedPrefill:

    def __init__(self, config: MultiNodeConfig = None):
        self.world_size = config.world_size
        self.npus_per_node = int(os.getenv("NPU_PER_NODE", "16"))
        self.ips = get_cluster_ips(self.world_size)
        cur_ip = get_cur_ip()
        self.nic_name = get_net_interface(cur_ip)
        if self.nic_name is None:
            raise RuntimeError("Failed to get network interface")
        if config is not None:
            server_config = config.server_config
            self.prefill_device_cnt = server_config.data_parallel_size_local * server_config.tensor_parallel_size
            self.decode_device_cnt = server_config.data_parallel_size_local * server_config.tensor_parallel_size

    def setup_and_run_ranktable(self):
        """Generate ranktable.json for multi-node setup."""
        if os.path.exists(RANKTABLE_PATH):
            logger.info(f"RANKTABLE_PATH is already set: {RANKTABLE_PATH}")
            return
        self._set_env()

        local_ips = socket.gethostbyname_ex(socket.gethostname())[2]
        local_host = "127.0.0.1"
        master_addr = self.ips[0]
        master_port = DISAGGEGATED_PREFILL_PORT
        nnodes = len(self.ips)
        node_rank = None

        for i, ip in enumerate(self.ips):
            if ip in local_ips:
                local_host = ip
                node_rank = i
                break

        if node_rank is None:
            logger.error(
                '"NODE_RANK" must be defined — local host not in provided IP list'
            )
            raise ValueError("Local host IP not found in ips list")

        world_size = self.npus_per_node * nnodes
        rank_start = self.npus_per_node * node_rank

        logger.info("\n========> Parameters:\n"
                    f"LOCAL_HOST: {local_host}\n"
                    f"WORLD_SIZE: {world_size}\n"
                    f"RANKSTART: {rank_start}\n"
                    f"NNODES: {nnodes}\n"
                    f"NODE_RANK: {node_rank}\n"
                    "=====================")

        cmd = [
            "torchrun",
            "--nproc_per_node",
            "1",
            "--nnodes",
            str(nnodes),
            "--node_rank",
            str(node_rank),
            "--master_addr",
            master_addr,
            "--master_port",
            str(master_port),
            RANKTABLE_GEN_PATH,
            "--ranktable-path",
            RANKTABLE_PATH,
            "--local-host",
            local_host,
            "--prefill-device-cnt",
            str(self.prefill_device_cnt),
            "--decode-device-cnt",
            str(self.decode_device_cnt),
        ]

        env = os.environ.copy()
        env["GLOO_SOCKET_IFNAME"] = self.nic_name

        logger.info("Running command:")
        logger.info(" ".join(cmd))

        subprocess.run(cmd, env=env, check=True)

    def launch_server_proxy(self):
        """Launch the proxy server for disaggregated prefill."""
        # python toy_proxy_server.py --host 172.19.32.175 --port 1025 --prefiller-hosts 172.19.241.49
        # --prefiller-port 20002 --decoder-hosts 172.19.123.51 --decoder-ports 20002

        cmd = [
            "python", LOAD_BALANCER_PROXY_SCRIPT, "--host", self.ips[0],
            "--port", "1025", "--prefiller-hosts", ",".join(self.ips),
            "--prefiller-port", "20002", "--decoder-hosts", ",".join(self.ips),
            "--decoder-ports", "20002"
        ]
        env = os.environ.copy()
        env["DISAGGREGATED_PREFILL_RANK_TABLE_PATH"] = RANKTABLE_PATH
        env["VLLM_ASCEND_LLMDD_RPC_PORT"] = str(DISAGGEGATED_PREFILL_PORT)

        logger.info("Launching proxy server with command:")
        logger.info(" ".join(cmd))

        subprocess.Popen(cmd, env=env)

    @classmethod
    def _set_env(cls):
        ascend_env = ASCEND_ENV_PATH
        if os.path.exists(ascend_env):
            subprocess.run(["bash", "-c", f"source {ascend_env}"], check=False)
        else:
            logger.warning(f"Ascend env file not found: {ascend_env}")

        lib_path = LIB_PATH
        os.environ[
            "LD_LIBRARY_PATH"] = f"{lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"


if __name__ == "__main__":
    from tests.e2e.multi_node.config.common import CONFIG_PATH
    from tests.e2e.multi_node.config.multi_node_config import load_configs
    configs = load_configs(CONFIG_PATH)
    pd_config = configs[1]
    dp = DisaggegatedPrefill(pd_config)
    dp.setup_and_run_ranktable()
