import os
import subprocess

import regex as re
import yaml

from tests.e2e.nightly.multi_node.config.utils import (get_avaliable_port,
                                                       get_cluster_ips,
                                                       get_cur_ip,
                                                       get_net_interface)

DISAGGREGATED_PREFILL_PROXY_SCRIPT = "examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py"


class MultiNodeConfig:

    def __init__(self,
                 model: str,
                 test_name: str,
                 num_nodes: int = 2,
                 npu_per_node: int = 16,
                 server_port: int = 8080,
                 headless: bool = False,
                 disaggregated_prefill: dict = None,
                 envs: dict = None,
                 server_cmd: str = "",
                 perf_cmd: dict = None,
                 acc_cmd: dict = None):
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

    def launch_server_proxy(self):
        if not self.disaggregated_prefill or not self.is_master:
            return
        prefiller_indices = self.disaggregated_prefill["prefiller_host_index"]
        decoder_indices = self.disaggregated_prefill["decoder_host_index"]

        common_indices = set(prefiller_indices) & set(decoder_indices)
        assert len(common_indices) == 0, \
            f"prefiller_host_index and decoder_host_index must not share common indices. Common indices: {common_indices}"

        # Launch the proxy server only on the master node
        assert self.proxy_port is not None, "proxy_port must be set for disaggregated prefill"

        prefiller_ips, decoder_ips = [], []
        for index, ip in enumerate(self.cluster_ips):
            if index in prefiller_indices:
                prefiller_ips.append(ip)
            if index in decoder_indices:
                decoder_ips.append(ip)

        assert len(prefiller_ips) == len(prefiller_indices), \
            f"Missing prefiller IPs. Expected {len(prefiller_indices)}, found {len(prefiller_ips)}"
        assert len(decoder_ips) == len(decoder_indices), \
            f"Missing decoder IPs. Expected {len(decoder_indices)}, found {len(decoder_ips)}"

        prefiller_ips_str = " ".join(prefiller_ips)
        decoder_ips_str = " ".join(decoder_ips)
        prefiller_ports = " ".join([str(self.server_port)] *
                                   len(prefiller_ips))
        decoder_ports = " ".join([str(self.server_port)] * len(decoder_ips))

        proxy_cmd = [
            "python", DISAGGREGATED_PREFILL_PROXY_SCRIPT, "--host",
            self.cur_ip, "--port",
            str(self.proxy_port), "--prefiller-hosts", prefiller_ips_str,
            "--prefiller-ports", prefiller_ports, "--decoder-hosts",
            decoder_ips_str, "--decoder-ports", decoder_ports
        ]

        env = os.environ.copy()
        env.update(self.envs or {})
        subprocess.Popen(proxy_cmd, env=env)

    @classmethod
    def from_yaml(cls, yaml_path: str = None):
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
                    os.getenv("LWS_WORKER_INDEX")):
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


if __name__ == '__main__':
    config = MultiNodeConfig.from_yaml()
    print(config.envs)
    print(config.server_cmd)
    print(config.perf_cmd)
    print(config.acc_cmd)
    config.launch_server_proxy()
