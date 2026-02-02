import logging
import os
import re
from typing import Any

import yaml

CONFIG_BASE_PATH = "tests/e2e/nightly/single_node/config"
DEFAULT_SERVER_PORT = 8080

logger = logging.getLogger(__name__)


class SingleNodeConfig:

    def __init__(
        self,
        *,
        model: str,
        envs: dict[str, str],
        server_cmd: str,
        perf_cmd: dict[str, Any] | None,
        acc_cmd: dict[str, Any] | None,
    ):
        self.model = model
        self.perf_cmd = perf_cmd
        self.acc_cmd = acc_cmd
        self.envs = envs
        self.server_cmd = self._expand_env(server_cmd)

    def _expand_env(self, cmd: str) -> str:
        pattern = re.compile(r"\$(\w+)|\$\{(\w+)\}")

        def repl(m):
            key = m.group(1) or m.group(2)
            return self.envs.get(key, m.group(0))
        
        return pattern.sub(repl, cmd)
    
    @property
    def server_port(self) -> int:
        return self.envs.get("SERVER_PORT", DEFAULT_SERVER_PORT)


class SingleNodeConfigLoader:
    """Load SingleNodeConfig from yaml file."""

    DEFAULT_CONFIG_NAME = "Qwen3-32B.yaml"

    @classmethod
    def from_yaml(cls, yaml_path: str | None = None) -> SingleNodeConfig:
        config = cls._load_yaml(yaml_path)
        cls._validate_root(config)
        benchmarks = cls._parse_benchmarks(config)

        return SingleNodeConfig(
            model=config["model"],
            envs=config.get("env_common", {}),
            server_cmd=config["server_cmd"],
            perf_cmd=benchmarks.get("perf"),
            acc_cmd=benchmarks.get("acc"),
        )

    @classmethod
    def _load_yaml(cls, yaml_path: str | None) -> dict:
        if not yaml_path:
            yaml_path = os.getenv("CONFIG_YAML_PATH", cls.DEFAULT_CONFIG_NAME)

        full_path = os.path.join(CONFIG_BASE_PATH, yaml_path)
        logger.info("Loading config yaml: %s", full_path)

        with open(full_path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def _validate_root(config: dict):
        required = [
            "model", "env_common", "server_cmd", "benchmarks"
        ]
        missing = [k for k in required if k not in config]
        if missing:
            raise KeyError(f"Missing required config fields: {missing}")


    @staticmethod
    def _parse_benchmarks(cfg: dict) -> dict:
        benchmarks = cfg.get("benchmarks") or {}
        return benchmarks
