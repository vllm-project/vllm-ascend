import logging
import os
import subprocess
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


class MultiNodeConfig:

    def __init__(self,
                 model: str,
                 test_name: str,
                 num_nodes: int = 2,
                 npu_per_node: int = 16,
                 server_port: int = 8080,
                 headless: bool = False,
                 disaggregated_prefill: Optional[dict] = None,
                 envs: Optional[dict] = None,
                 server_cmd: str = "",
                 perf_cmd: Optional[str] = None,
                 acc_cmd: Optional[str] = None):
        self.test_name = test_name
        self.model = model
        self.num_nodes = num_nodes
        self.npu_per_node = npu_per_node
        self.envs = envs if envs is not None else {}
        self.server_port = server_port
        if disaggregated_prefill:
            self.proxy_port = get_avaliable_port()
        self.headless = headless
        self.server_cmd = server_cmd
        self.perf_cmd = perf_cmd
        self.acc_cmd = acc_cmd
        assert perf_cmd is not None, "perf_cmd must be provided"
        assert acc_cmd is not None, "acc_cmd must be provided"
        assert server_cmd is not None, "server_cmd must be provided"

        self.cur_index = os.getenv("LWS_WORKER_INDEX", 0)
        self.cur_ip = get_cur_ip()
        self.nic_name = get_net_interface(self.cur_ip)
        self.cluster_ips = get_cluster_ips(num_nodes)
        self.disaggregated_prefill = disaggregated_prefill
        self._init_dist_env()
        self.server_cmd = self._expand_env_vars(self.server_cmd, self.envs)

    def _init_dist_env(self):
        self.envs["HCCL_IF_IP"] = self.cur_ip
        self.envs["GLOO_SOCKET_IFNAME"] = self.nic_name
        self.envs["TP_SOCKET_IFNAME"] = self.nic_name
        self.envs["HCCL_SOCKET_IFNAME"] = self.nic_name
        self.envs["LOCAL_IP"] = self.cur_ip
        self.envs["NIC_NAME"] = self.nic_name
        self.envs["MASTER_IP"] = self.cluster_ips[0]
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
        for deployment in deployments:
            if deployment.get("local_index") == int(
                    os.getenv("LWS_WORKER_INDEX", 0)):
                envs_extend = deployment.get("env_extend", {})
                if envs_extend:
                    envs.update(envs_extend)
                server_cmd = deployment.get("server_cmd")
                headless = deployment.get("headless", False)
                break
        benchmarks = config_data.get("benchmarks", {})
        assert benchmarks is not None, "benchmarks must be provided"
        perf_cmd = benchmarks["perf"]
        acc_cmd = benchmarks["acc"]

        return cls(model=model,
                   test_name=test_name,
                   num_nodes=num_nodes,
                   npu_per_node=npu_per_node,
                   envs=envs,
                   server_port=server_port,
                   headless=headless,
                   disaggregated_prefill=disaggregated_prefill,
                   server_cmd=server_cmd,
                   perf_cmd=perf_cmd,
                   acc_cmd=acc_cmd)

    @property
    def world_size(self):
        return self.num_nodes * self.npu_per_node

    @property
    def is_master(self):
        return int(self.cur_index) == 0
