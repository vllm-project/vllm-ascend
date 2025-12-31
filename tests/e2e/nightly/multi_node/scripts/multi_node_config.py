import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

import regex as re
import yaml

# isort: off
from tests.e2e.nightly.multi_node.scripts.utils import (
    CONFIG_BASE_PATH, DEFAULT_SERVER_PORT, get_all_ipv4, get_cluster_ips,
    get_net_interface, setup_logger, get_avaliable_port)
# isort: on
setup_logger()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeInfo:
    index: int
    ip: str
    server_cmd: str
    envs: dict | None = None
    headless: bool = False

    def __post_init__(self):
        if not self.ip:
            raise ValueError("NodeInfo.ip must not be empty")

    def __str__(self) -> str:
        return ("NodeInfo(\n"
                f"  index={self.index},\n"
                f"  ip={self.ip},\n"
                f"  headless={self.headless},\n"
                ")")


class DisaggregatedPrefillCfg:

    def __init__(self, raw_cfg: dict, num_nodes: int):
        self.prefiller_indices: list[int] = raw_cfg.get(
            "prefiller_host_index", [])
        self.decoder_indices: list[int] = raw_cfg.get("decoder_host_index", [])

        if not self.decoder_indices:
            raise RuntimeError("decoder_host_index must be provided")

        self._validate(num_nodes)

        self.decode_start_index = self.decoder_indices[0]
        self.num_prefillers = len(self.prefiller_indices)
        self.num_decoders = len(self.decoder_indices)

    def _validate(self, num_nodes: int):
        overlap = set(self.prefiller_indices) & set(self.decoder_indices)
        if overlap:
            raise AssertionError(f"Prefiller and decoder overlap: {overlap}")

        all_indices = self.prefiller_indices + self.decoder_indices
        if any(i >= num_nodes for i in all_indices):
            raise ValueError("Disaggregated prefill index out of range")

    def is_prefiller(self, index: int) -> bool:
        return index in self.prefiller_indices

    def is_decoder(self, index: int) -> bool:
        return index in self.decoder_indices

    def master_ip_for_node(self, index: int, nodes: list[NodeInfo]) -> str:
        if self.is_prefiller(index):
            return nodes[0].ip
        return nodes[self.decode_start_index].ip


class DistEnvBuilder:

    def __init__(
        self,
        *,
        cur_node: NodeInfo,
        master_ip: str,
        common_envs: dict,
    ):
        self.cur_ip = cur_node.ip
        self.nic_name = get_net_interface(self.cur_ip)
        self.master_ip = master_ip

        # envs
        common_envs = common_envs
        current_envs = cur_node.envs or {}
        # Node-specific envs override common envs
        self.base_envs = {**common_envs, **current_envs}

    def build(self) -> dict:
        envs = dict(self.base_envs)

        envs.update({
            "HCCL_IF_IP": self.cur_ip,
            "HCCL_SOCKET_IFNAME": self.nic_name,
            "GLOO_SOCKET_IFNAME": self.nic_name,
            "TP_SOCKET_IFNAME": self.nic_name,
            "LOCAL_IP": self.cur_ip,
            "NIC_NAME": self.nic_name,
            "MASTER_IP": self.master_ip,
        })

        return {k: str(v) for k, v in envs.items()}


class ProxyLauncher:

    def __init__(
        self,
        *,
        nodes: list[NodeInfo],
        envs: dict,
        proxy_port: int,
        cur_index: int,
        disagg_cfg: DisaggregatedPrefillCfg | None = None,
    ):
        self.nodes = nodes
        self.cfg = disagg_cfg
        self.server_port = envs.get("SERVER_PORT", DEFAULT_SERVER_PORT)
        self.proxy_port = proxy_port
        self.proxy_script = envs.get(
            "DISAGGREGATED_PREFILL_PROXY_SCRIPT",
            'examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py'
        )
        self.envs = envs
        self.is_master = cur_index == 0
        self.cur_ip = nodes[cur_index].ip
        self.process: Optional[subprocess.Popen[bytes]] = None

    def __enter__(self):
        if not self.is_master or self.cfg is None:
            logger.info("Not launching proxy on non-master node")
            return self
        prefiller_ips = [self.nodes[i].ip for i in self.cfg.prefiller_indices]
        decoder_ips = [self.nodes[i].ip for i in self.cfg.decoder_indices]

        cmd = [
            "python",
            self.proxy_script,
            "--host",
            self.cur_ip,
            "--port",
            str(self.proxy_port),
            "--prefiller-hosts",
            *prefiller_ips,
            "--prefiller-ports",
            *[str(self.server_port)] * len(prefiller_ips),
            "--decoder-hosts",
            *decoder_ips,
            "--decoder-ports",
            *[str(self.server_port)] * len(decoder_ips),
        ]

        logger.info("Launching proxy: %s", " ".join(cmd))
        self.process = subprocess.Popen(cmd, env={**os.environ, **self.envs})
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.process:
            return
        logger.info("Stopping proxy server...")
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()


class MultiNodeConfig:

    def __init__(
        self,
        *,
        model: str,
        test_name: str,
        nodes: list[NodeInfo],
        npu_per_node: int,
        envs: dict,
        disaggregated_prefill: dict | None,
        perf_cmd: str | None,
        acc_cmd: str | None,
    ):
        self.model = model
        self.test_name = test_name
        self.nodes = nodes
        self.npu_per_node = npu_per_node
        self.perf_cmd = perf_cmd
        self.acc_cmd = acc_cmd

        self.cur_index = self._resolve_cur_index()
        self.cur_node = self.nodes[self.cur_index]

        self.disagg_cfg = (DisaggregatedPrefillCfg(disaggregated_prefill,
                                                   len(nodes))
                           if disaggregated_prefill else None)

        master_ip = (self.disagg_cfg.master_ip_for_node(
            self.cur_index, self.nodes)
                     if self.disagg_cfg else self.nodes[0].ip)
        self.proxy_port = get_avaliable_port()

        self.envs = DistEnvBuilder(
            cur_node=self.cur_node,
            master_ip=master_ip,
            common_envs=envs,
        ).build()
        logger.info("Node %d envs: %s", self.cur_index, self.envs)

        self.server_cmd = self._expand_env(self.cur_node.server_cmd)

    def _resolve_cur_index(self) -> int:
        if (idx := os.environ.get("LWS_WORKER_INDEX")):
            return int(idx)

        local_ips = get_all_ipv4()
        for i, node in enumerate(self.nodes):
            if node.ip in local_ips:
                return i

        raise RuntimeError("Unable to determine current node index")

    def _expand_env(self, cmd: str) -> str:
        pattern = re.compile(r"\$(\w+)|\$\{(\w+)\}")

        def repl(m):
            key = m.group(1) or m.group(2)
            return self.envs.get(key, m.group(0))

        return pattern.sub(repl, cmd)

    @property
    def world_size(self) -> int:
        return len(self.nodes) * self.npu_per_node

    @property
    def is_master(self) -> bool:
        return self.cur_index == 0

    @property
    def server_port(self) -> int:
        return self.envs.get("SERVER_PORT", DEFAULT_SERVER_PORT)

    @property
    def master_ip(self) -> str:
        return self.nodes[0].ip

    @property
    def benchmark_endpoint(self) -> tuple[str, int]:
        """
        Endpoint used by benchmark clients.
        """
        master_ip = self.nodes[0].ip
        server_port = self.envs.get("SERVER_PORT", DEFAULT_SERVER_PORT)
        if self.disagg_cfg:
            return master_ip, self.proxy_port
        return master_ip, server_port


class MultiNodeConfigLoader:
    """Load MultiNodeConfig from yaml file."""

    DEFAULT_CONFIG_NAME = "DeepSeek-V3.yaml"

    @classmethod
    def from_yaml(cls, yaml_path: Optional[str] = None) -> MultiNodeConfig:
        config = cls._load_yaml(yaml_path)
        cls._validate_root(config)

        nodes = cls._parse_nodes(config)
        benchmarks = cls._parse_benchmarks(config)

        return MultiNodeConfig(
            model=config["model"],
            test_name=config.get("test_name", "untitled_test"),
            nodes=nodes,
            npu_per_node=config.get("npu_per_node", 16),
            envs=config.get("env_common", {}),
            disaggregated_prefill=config.get("disaggregated_prefill"),
            perf_cmd=benchmarks.get("perf"),
            acc_cmd=benchmarks.get("acc"),
        )

    @classmethod
    def _load_yaml(cls, yaml_path: Optional[str]) -> dict:
        if not yaml_path:
            yaml_path = os.getenv("CONFIG_YAML_PATH", cls.DEFAULT_CONFIG_NAME)

        full_path = os.path.join(CONFIG_BASE_PATH, yaml_path)
        logger.info("Loading config yaml: %s", full_path)

        with open(full_path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _validate_root(cfg: dict):
        required = [
            "model", "deployment", "num_nodes", "npu_per_node", "env_common",
            "benchmarks"
        ]
        missing = [k for k in required if k not in cfg]
        if missing:
            raise KeyError(f"Missing required config fields: {missing}")

    @classmethod
    def _parse_nodes(cls, cfg: dict) -> list[NodeInfo]:
        num_nodes = cfg["num_nodes"]
        deployments = cfg["deployment"]

        if len(deployments) != num_nodes:
            raise AssertionError(
                f"deployment size ({len(deployments)}) != num_nodes ({num_nodes})"
            )

        cluster_ips = cls._resolve_cluster_ips(cfg, num_nodes)

        nodes: list[NodeInfo] = []
        for idx, deploy in enumerate(deployments):
            cmd = deploy.get("server_cmd", "")
            envs = deploy.get("envs", {})
            nodes.append(
                NodeInfo(
                    index=idx,
                    ip=cluster_ips[idx],
                    server_cmd=cmd,
                    envs=envs,
                    headless="--headless" in cmd,
                ))
        return nodes

    @staticmethod
    def _parse_benchmarks(cfg: dict) -> dict:
        benchmarks = cfg.get("benchmarks") or {}
        return benchmarks

    @staticmethod
    def _resolve_cluster_ips(cfg: dict, num_nodes: int) -> list[str]:
        if "cluster_hosts" in cfg and cfg["cluster_hosts"]:
            logger.info(
                "Using cluster_hosts from config. This typically indicates that your current environment is a non-Kubernetes environment."
            )
            ips = cfg["cluster_hosts"]
            if len(ips) != num_nodes:
                raise AssertionError("cluster_hosts size mismatch")
            return ips

        logger.info("Resolving cluster IPs via DNS...")
        return get_cluster_ips(num_nodes)
