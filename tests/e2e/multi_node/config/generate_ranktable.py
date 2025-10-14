import logging
import os
import socket
import subprocess

import torch.distributed as dist

from tests.e2e.multi_node.config.common import (ASCEND_ENV_PATH,
                                                DECODER_START_PORT,
                                                DISAGGEGATED_PREFILL_PORT,
                                                LIB_PATH,
                                                LOAD_BALANCER_PROXY_SCRIPT,
                                                PREFILLER_START_PORT,
                                                RANKTABLE_GEN_PATH,
                                                RANKTABLE_PATH)
from tests.e2e.multi_node.config.multi_node_config import MultiNodeConfig
from tests.e2e.multi_node.config.utils import (dist_group, get_cluster_ips,
                                               get_cur_ip, get_net_interface,
                                               setup_logger, temp_env)

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
        self.world_size: int = config.world_size
        self.num_prefillers: int = config.num_prefillers
        self.num_prefiller_nodes: int = config.num_prefiller_nodes
        self.num_decoders: int = config.num_decoders
        self.num_decoder_nodes: int = config.num_decoder_nodes
        # for A3 cluster, we assume 16 NPUs per node
        self.npus_per_node: int = int(os.getenv("NPU_PER_NODE", "16"))
        self.ips: list[str] = get_cluster_ips(self.world_size)
        self.cur_ip: str = get_cur_ip()
        self.with_prefill: bool = os.getenv("WITH_PREFILL", "1") == "1"
        # headless means that there is no API server on this node
        self.headless: bool = config.server_config.headless
        self.server_port: int = config.server_port
        self.is_leader: bool = self.cur_ip == self.ips[0]
        if self.is_leader:
            assert self.with_prefill, "Leader node must have prefill enabled"
            assert not self.headless, "Leader node cannot be headless"

        self.nic_name: str = get_net_interface(self.cur_ip)
        if self.nic_name is None:
            raise RuntimeError("Failed to get network interface")

        self.prefill_device_cnt: int = self.num_prefillers * self.num_prefiller_nodes * self.npus_per_node
        self.decode_device_cnt: int = self.num_decoders * self.num_decoder_nodes * self.npus_per_node

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
        nnodes = self.world_size
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

    def sync_node_roles(self, local_ip: str, with_prefill: bool,
                        headless: bool) -> list[tuple[bool, bool]]:
        """
        sync (local_ip, with_prefill, headless) info across all nodes
        """
        world_size = dist.get_world_size()
        local_info = (local_ip, with_prefill, headless)
        gathered_info = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_info, local_info)
        return gathered_info

    def launch_server_proxy(self):
        """
        Launch the proxy server for the disaggregated prefill architecture.

        In this setup, prefillers and decoders are distributed across multiple nodes.
        For example:
            - If there are 2 nodes (16 NPUs each)
            - Prefillers and decoders will each occupy all NPUs of one node

        This method:
            1. Synchronizes role information (prefill/decoder) across nodes
            2. Collects IPs for prefillers and decoders
            3. Assigns communication ports
            4. Launches the proxy load balancer
        """
        assert self.world_size == len(
            self.ips), "World size does not match the number of IPs"
        assert self.world_size >= self.num_decoder_nodes + self.num_prefiller_nodes, (
            "Insufficient nodes for the configured number of prefillers and decoders"
        )

        prefill_ips, decode_ips = [], []

        # === Step 1. Synchronize node roles using distributed communication ===
        with temp_env({
                "MASTER_ADDR": self.ips[0],
                "MASTER_PORT": str(DISAGGEGATED_PREFILL_PORT),
                "WORLD_SIZE": str(self.world_size),
                "RANK": str(self.ips.index(self.cur_ip)),
        }):
            with dist_group(backend="gloo"):
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                logger.info(
                    f"[Rank {rank}] Connected to process group (world_size={world_size})"
                )

                all_roles = self.sync_node_roles(self.cur_ip,
                                                 self.with_prefill,
                                                 self.headless)

        assert len(
            all_roles
        ) == self.world_size, "Mismatch between world size and gathered roles"

        # === Step 2. Split IPs by role ===
        for local_ip, with_prefill, headless in all_roles:
            if headless:
                continue
            if with_prefill:
                prefill_ips.append(local_ip)
            else:
                decode_ips.append(local_ip)

        # === Step 3. Assign ports ===
        prefill_ports = [
            PREFILLER_START_PORT + i for i in range(len(prefill_ips))
        ]
        decode_ports = [DECODER_START_PORT + i for i in range(len(decode_ips))]

        if self.is_leader:
            # === Step 4. Build and launch proxy command ===
            proxy_cmd = [
                "python",
                LOAD_BALANCER_PROXY_SCRIPT,
                "--host",
                self.ips[0],
                "--port",
                self.server_port,
                "--prefiller-hosts",
                " ".join(prefill_ips),
                "--prefiller-ports",
                " ".join(map(str, prefill_ports)),
                "--decoder-hosts",
                " ".join(decode_ips),
                "--decoder-ports",
                " ".join(map(str, decode_ports)),
            ]

            logger.info("Launching proxy server with the following command:")
            logger.info(" ".join(proxy_cmd))

            subprocess.Popen(proxy_cmd, env=os.environ.copy())

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
    dp.launch_server_proxy()
