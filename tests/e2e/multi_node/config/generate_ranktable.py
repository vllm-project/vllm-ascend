import logging
import os
import socket
import subprocess

from tests.e2e.multi_node.config.common import (ASCEND_ENV_PATH,
                                                DECODER_START_PORT,
                                                DISAGGEGATED_PREFILL_PORT,
                                                LIB_PATH,
                                                LOAD_BALANCER_PROXY_SCRIPT,
                                                PREFILLER_START_PORT,
                                                RANKTABLE_GEN_PATH,
                                                RANKTABLE_PATH)
from tests.e2e.multi_node.config.multi_node_config import MultiNodeConfig
from tests.e2e.multi_node.config.utils import (get_cluster_ips, get_cur_ip,
                                               get_net_interface, setup_logger)

setup_logger()
logger = logging.getLogger(__name__)


class DisaggegatedPrefill:

    def __init__(self, config: MultiNodeConfig = None):
        """
        Initialize DisaggregatedPrefill with configuration.
        world_size: total number of nodes
        num_prefillers: total number of prefillers
        num_prefiller_nodes: number of nodes dedicated to prefillers
        num_decoders: total number of decoders
        num_decoder_nodes: number of nodes dedicated to decoders
        npus_per_node: number of NPUs per node (default 16)
        ips: list of IPs for all nodes, len should be equal to world_size
        """
        self.world_size = config.world_size
        self.num_prefillers = config.num_prefillers
        self.num_prefiller_nodes = config.num_prefiller_nodes
        self.num_decoders = config.num_decoders
        self.num_decoder_nodes = config.num_decoder_nodes
        # for A3 cluster, we assume 16 NPUs per node
        self.npus_per_node = int(os.getenv("NPU_PER_NODE", "16"))
        self.ips = get_cluster_ips(self.world_size)
        self.cur_ip = get_cur_ip()
        self.with_prefill = os.getenv("WITH_PREFILL", "1") == "1"
        self.is_leader = self.cur_ip == self.ips[0]
        if self.is_leader:
            assert self.with_prefill, "Leader node must have prefill enabled"

        self.nic_name = get_net_interface(self.cur_ip)
        if self.nic_name is None:
            raise RuntimeError("Failed to get network interface")

        self.prefill_device_cnt = self.num_prefillers * self.num_prefiller_nodes * self.npus_per_node
        self.decode_device_cnt = self.num_decoders * self.num_decoder_nodes * self.npus_per_node

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
        """
        Launch the proxy server for disaggregated prefill.
        Currently, we assume prefillers and decoders share all devices equally
        eg: we have 2 nodes (16 NPUs each), and prefillers and decoders 
        Occupy all devices of these two nodes in turn
        """
        if not self.is_leader:
            logger.info(
                "Skipping proxy server launch, proxy only launch on leader node"
            )
            return
        assert self.world_size == len(
            self.ips), "World size and IPs length mismatch"
        assert self.world_size >= self.num_decoder_nodes + self.num_prefiller_nodes, \
            "Not enough nodes for the specified number of prefillers and decoders"

        # prefillers and decoders ips should be equal to num_prefiller and num_decoder
        prefill_ips = self.ips[
            :self.num_prefillers,
        ]

        # Assign ports for each prefillers and decoders
        prefiller_ports = [
            PREFILLER_START_PORT + i for i in range(len(prefill_ips))
        ]
        decoder_ports = [
            DECODER_START_PORT + i for i in range(len(decode_ips))
        ]

        proxy_cmd = [
            "python", LOAD_BALANCER_PROXY_SCRIPT, "--host", self.ips[0],
            "--port", "1025", "--prefiller-hosts", ",".join(self.prefill_ips),
            "--prefiller-port", ",".join(prefiller_ports), "--decoder-hosts",
            ",".join(self.decode_ips), "--decoder-ports",
            ",".join(decoder_ports)
        ]

        logger.info("Launching proxy server with command:")
        logger.info(" ".join(proxy_cmd))

        subprocess.Popen(proxy_cmd, env=env)

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
    print(pd_config.server_config.to_list())
    dp = DisaggegatedPrefill(pd_config)
    dp.setup_and_run_ranktable()
