import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

import regex as re
import yaml

from tests.e2e.nightly.multi_node.config.utils import (get_avaliable_port,
                                                       get_cluster_ips,
                                                       get_cur_ip,
                                                       get_net_interface,
                                                       setup_logger)

setup_logger()
logger = logging.getLogger(__name__)
DISAGGREGATED_PREFILL_PROXY_SCRIPT = "examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py"
DISAGGEGATED_PREFILL_PORT = 5333


@dataclass
class NodeInfo:
    index: int
    ip: str
    server_cmd: str
    headless: bool
    server_port: int


class MultiNodeConfig:

    def __init__(self,
                 model: str,
                 test_name: str,
                 npu_per_node: int = 16,
                 server_port: int = 8080,
                 disaggregated_prefill: Optional[dict] = None,
                 envs: Optional[dict] = None,
                 nodes_info: Optional[list[NodeInfo]] = None,
                 perf_cmd: Optional[str] = None,
                 acc_cmd: Optional[str] = None):
        self.test_name = test_name
        self.model = model
        self.nodes_info = nodes_info or []
        self.num_nodes = len(self.nodes_info)
        self.npu_per_node = npu_per_node
        self.server_port = server_port
        self.envs = envs if envs is not None else {}
        self.proxy_port = get_avaliable_port()
        self.perf_cmd = perf_cmd
        self.acc_cmd = acc_cmd
        assert perf_cmd is not None, "perf_cmd must be provided"
        assert acc_cmd is not None, "acc_cmd must be provided"

        self.cur_index = int(os.getenv("LWS_WORKER_INDEX", 0))
        self.cur_ip = get_cur_ip()
        self.nic_name = get_net_interface(self.cur_ip)
        self.cluster_ips = get_cluster_ips(self.num_nodes)
        self.cur_node_info: NodeInfo = self.nodes_info[self.cur_index]
        self.disaggregated_prefill = disaggregated_prefill
        self._init_disaggregated_prefill()

        self._init_dist_env()
        self.server_cmd = self._expand_env_vars(self.cur_node_info.server_cmd,
                                                self.envs)

    def _init_disaggregated_prefill(self):
        if self.disaggregated_prefill:
            decode_host_index = self.disaggregated_prefill.get(
                "decoder_host_index")
            if not decode_host_index:
                raise RuntimeError("got empty decode_host_index")
            self.decode_start_index: int = decode_host_index[0]
            self.num_prefillers = self.decode_start_index
            self.num_decoders = self.num_nodes - self.num_prefillers
            if self.disaggregated_prefill.get(
                    "ranktable_gen_path") is not None:
                self._gen_ranktable()

    def _init_dist_env(self):
        self.envs["HCCL_IF_IP"] = self.cur_ip
        self.envs["GLOO_SOCKET_IFNAME"] = self.nic_name
        self.envs["TP_SOCKET_IFNAME"] = self.nic_name
        self.envs["HCCL_SOCKET_IFNAME"] = self.nic_name
        self.envs["LOCAL_IP"] = self.cur_ip
        self.envs["NIC_NAME"] = self.nic_name

        if self.disaggregated_prefill:
            self.envs[
                "DISAGGREGATED_PREFILL_RANK_TABLE_PATH"] = self.disaggregated_prefill.get(
                    "ranktable_path")
            if self.cur_index < self.decode_start_index:
                self.envs["MASTER_IP"] = self.cluster_ips[0]
            else:
                self.envs["MASTER_IP"] = self.cluster_ips[
                    self.decode_start_index]

        ascend_path = "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages"
        self.envs[
            "LD_LIBRARY_PATH"] = f"{ascend_path}:{self.envs.get('LD_LIBRARY_PATH', os.environ.get('LD_LIBRARY_PATH', ''))}"

        # keep the envs keys and values as strings
        str_envs = {k: str(v) for k, v in self.envs.items()}
        self.envs.clear()
        self.envs.update(str_envs)

    @staticmethod
    def _expand_env_vars(cmd: str, env: dict) -> str:
        """Expand environment variables in the command string."""
        cmd = str(cmd)
        pattern = re.compile(r"\$(\w+)|\$\{(\w+)\}")

        def replace_var(match):
            var_name = match.group(1) or match.group(2)
            return str(env.get(var_name, match.group(0)))

        return pattern.sub(replace_var, cmd)

    class _ProxyContext:

        def __init__(self, outer, proxy_script):
            self.outer = outer
            self.proxy_script = proxy_script
            self.process = None

        def __enter__(self):
            o = self.outer
            if not o.disaggregated_prefill or not o.is_master:
                logger.info(
                    "Disaggregated prefill not enabled or not master node, skipping proxy launch."
                )
                return self

            prefiller_indices = o.disaggregated_prefill["prefiller_host_index"]
            decoder_indices = o.disaggregated_prefill["decoder_host_index"]

            common_indices = set(prefiller_indices) & set(decoder_indices)
            assert not common_indices, f"Common indices found: {common_indices}"
            assert o.proxy_port is not None, "proxy_port must be set"

            prefiller_ips = [o.cluster_ips[i] for i in prefiller_indices]
            decoder_ips = [o.cluster_ips[i] for i in decoder_indices]
            prefiller_ports_list = [str(o.server_port)] * len(prefiller_ips)
            decoder_ports_list = [str(o.server_port)] * len(decoder_ips)

            proxy_cmd = [
                "python",
                self.proxy_script,
                "--host",
                o.cur_ip,
                "--port",
                str(o.proxy_port),
                "--prefiller-hosts",
                *prefiller_ips,
                "--prefiller-ports",
                *prefiller_ports_list,
                "--decoder-hosts",
                *decoder_ips,
                "--decoder-ports",
                *decoder_ports_list,
            ]

            env = os.environ.copy()
            env.update(o.envs)
            logger.info(f"Launching proxy: {' '.join(proxy_cmd)}")

            self.process = subprocess.Popen(proxy_cmd, env=env)
            o.proxy_process = self.process
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.process:
                logger.info("Terminating proxy server process...")
                try:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Proxy process did not terminate, killing it...")
                    self.process.kill()
                logger.info("Proxy server process terminated.")

    def launch_server_proxy(self, proxy_script: str):
        """Return a context manager that launches the proxy server if disaggregated prefill is enabled."""
        return self._ProxyContext(self, proxy_script)

    @classmethod
    def from_yaml(cls, yaml_path: Optional[str] = None):
        if not yaml_path:
            yaml_path = os.getenv(
                "CONFIG_YAML_PATH",
                "tests/e2e/nightly/multi_node/config/models/DeepSeek-V3.yaml")
        with open(yaml_path, 'r') as file:
            config_data = yaml.safe_load(file)
        test_name = config_data.get("test_name", "default_test")
        model = config_data.get("model", "default_model")
        envs = config_data.get("env_common", {})
        num_nodes = config_data.get("num_nodes", 2)
        npu_per_node = config_data.get("npu_per_node", 16)
        disaggregated_prefill = config_data.get("disaggregated_prefill")
        # If disaggregated_prefill is set, override server_port to an available port for proxy running
        server_port = config_data.get("server_port", 8080)

        deployments = config_data.get("deployment", [])
        assert len(deployments) == num_nodes, \
            f"Number of deployments ({len(deployments)}) must match num_nodes ({num_nodes})"

        cluster_ips = get_cluster_ips(num_nodes)
        nodes_info = []

        for index, deployment in enumerate(deployments):
            # after assert len(deployments) == num_nodes, we can assume that this will must have a match
            server_cmd = deployment.get("server_cmd", "")
            headless = "--headless" in server_cmd
            nodes_info.append(
                NodeInfo(ip=cluster_ips[index],
                         index=index,
                         headless=headless,
                         server_port=server_port,
                         server_cmd=server_cmd))

        benchmarks = config_data.get("benchmarks", {})
        assert benchmarks is not None, "benchmarks must be provided"
        perf_cmd = benchmarks["perf"]
        acc_cmd = benchmarks["acc"]

        return cls(model=model,
                   test_name=test_name,
                   npu_per_node=npu_per_node,
                   envs=envs,
                   server_port=server_port,
                   disaggregated_prefill=disaggregated_prefill,
                   nodes_info=nodes_info,
                   perf_cmd=perf_cmd,
                   acc_cmd=acc_cmd)

    @property
    def world_size(self):
        return self.num_nodes * self.npu_per_node

    @property
    def is_master(self):
        return self.cur_index == 0

    def _gen_ranktable(self):
        cluster_ip = self.cluster_ips
        assert len(cluster_ip) > 0
        nnodes = self.num_nodes
        node_rank = self.cur_index
        master_addr = cluster_ip[0]
        master_port = DISAGGEGATED_PREFILL_PORT
        ranktable_gen_path = self.disaggregated_prefill.get(
            "ranktable_gen_path")
        ranktable_path = self.disaggregated_prefill.get("ranktable_path")
        if os.path.exists(ranktable_path):
            return

        local_host = self.cur_ip

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
            ranktable_gen_path,
            "--ranktable-path",
            ranktable_path,
            "--local-host",
            local_host,
            "--prefill-device-cnt",
            str(self.npu_per_node * self.num_prefillers),
            "--decode-device-cnt",
            str(self.npu_per_node * self.num_decoders),
        ]

        env = os.environ.copy()
        env["GLOO_SOCKET_IFNAME"] = self.nic_name

        print(cmd)
        subprocess.run(cmd, env=env, check=True)
        assert os.path.exists(ranktable_path), "failed generate ranktable.json"
