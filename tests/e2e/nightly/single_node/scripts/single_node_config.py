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
        envs: dict[str, Any] | None = None,
        prompts: list[str] | None = None,
        api_keyword_args: dict[str, Any] | None = None,
        benchmarks: dict[str, Any] | None = None,
        server_cmd: list[str],
    ):
        self.model = model
        self.envs = envs or {}
        self.prompts = prompts
        self.api_keyword_args = api_keyword_args
        self.benchmarks = benchmarks
        self.server_cmd = self._expand_env(server_cmd)

    def _expand_env(self, cmd: list[str]) -> list[str]:
        pattern = re.compile(r"\$(\w+)|\$\{(\w+)\}")

        def repl(m):
            key = m.group(1) or m.group(2)
            return str(self.envs.get(key, m.group(0)))

        return [pattern.sub(repl, str(arg)) for arg in cmd]

    @property
    def server_port(self) -> int:
        return int(self.envs.get("SERVER_PORT", DEFAULT_SERVER_PORT))


class SingleNodeConfigLoader:
    """Load SingleNodeConfig from yaml file."""

    DEFAULT_CONFIG_NAME = "Qwen3-32B.yaml"

    @classmethod
    def from_yaml_cases(cls, yaml_path: str | None = None) -> list[SingleNodeConfig]:
        config = cls._load_yaml(yaml_path)

        if "test_cases" not in config:
            raise KeyError("test_cases field is required in config yaml")

        cases = config.get("test_cases")
        cls._validate_para(cases)

        return cls._parse_test_cases(cases)

    @classmethod
    def _load_yaml(cls, yaml_path: str | None) -> dict:
        if not yaml_path:
            yaml_path = os.getenv("CONFIG_YAML_PATH", cls.DEFAULT_CONFIG_NAME)

        full_path = os.path.join(CONFIG_BASE_PATH, yaml_path)
        logger.info("Loading config yaml: %s", full_path)

        with open(full_path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def _validate_para(cases: dict):
        if not cases:
            raise ValueError("test_cases is empty")
        required = ["model", "envs", "cmd_base", "benchmarks"]
        for case in cases:
            missing = [k for k in required if k not in case]
            if missing:
                raise KeyError(f"Missing required config fields: {missing}")

    @classmethod
    def _parse_test_cases(cls, cases: dict[str, Any]) -> list[SingleNodeConfig]:
        result: list[SingleNodeConfig] = []
        for case in cases:
            cmd_base = case.get("cmd_base", [])
            cmd_extra = case.get("cmd_extra", [])
            full_cmd = list(cmd_base) + list(cmd_extra)
            result.append(
                SingleNodeConfig(
                    model=case["model"],
                    envs=case.get("envs", {}),
                    server_cmd=full_cmd,
                    benchmarks=case.get("benchmarks", {}),
                    prompts=case.get("prompts"),
                    api_keyword_args=case.get("api_keyword_args"),
                )
            )
        return result
