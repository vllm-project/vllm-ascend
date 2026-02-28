import logging
import os
import re
from typing import Any

import yaml
from vllm.utils.network_utils import get_open_port

CONFIG_BASE_PATH = "tests/e2e/nightly/single_node/config"
DEFAULT_SERVER_PORT = get_open_port()

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
        server_cmd: list[str] | None = None,
        test_content: list[str] | None = None,
        service_mode: str = "openai",
        **kwargs: Any,
    ):
        self.model = model
        self.envs = envs or {}
        self.prompts = prompts
        self.test_content = test_content or ["completion"]
        self.service_mode = service_mode
        self.api_keyword_args = api_keyword_args
        self.benchmarks = benchmarks or {}

        if self.envs.get("SERVER_PORT") == "DEFAULT_PORT":
            self.envs["SERVER_PORT"] = str(DEFAULT_SERVER_PORT)
        self.server_cmd = self._expand_env(server_cmd or [])

        self.extra_config = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _expand_env(self, cmd: list[str]) -> list[str]:
        pattern = re.compile(r"\$(\w+)|\$\{(\w+)\}")

        def repl(m):
            key = m.group(1) or m.group(2)
            return str(self.envs.get(key, m.group(0)))

        return [pattern.sub(repl, str(arg)) for arg in cmd]

    @property
    def server_port(self) -> int:
        if self.envs.get("SERVER_PORT") == "DEFAULT_PORT":
            return int(DEFAULT_SERVER_PORT)
        return int(self.envs.get("SERVER_PORT"))


class SingleNodeConfigLoader:
    """Load SingleNodeConfig from yaml file."""

    DEFAULT_CONFIG_NAME = "Kimi-K2-Thinking.yaml"
    STANDARD_CASE_FIELDS = {
        "name",
        "model",
        "envs",
        "prompts",
        "api_keyword_args",
        "benchmarks",
        "service_mode",
        "server_cmd",
        "server_cmd_extra",
        "test_content",
        "epd_server_cmds",
        "epd_proxy_args",
    }

    @classmethod
    def from_yaml_cases(cls, yaml_path: str | None = None) -> list[SingleNodeConfig]:
        config = cls._load_yaml(yaml_path)

        if "test_cases" not in config:
            raise KeyError("test_cases field is required in config yaml")

        cases = config.get("test_cases")
        if not isinstance(cases, list):
            raise TypeError("test_cases must be a list")
        cls._validate_para(cases)

        return cls._parse_test_cases(cases)

    @classmethod
    def _load_yaml(cls, yaml_path: str | None) -> dict[str, Any]:
        if not yaml_path:
            yaml_path = os.getenv("CONFIG_YAML_PATH", cls.DEFAULT_CONFIG_NAME)

        full_path = os.path.join(CONFIG_BASE_PATH, yaml_path)
        logger.info("Loading config yaml: %s", full_path)

        with open(full_path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def _validate_para(cases: list[dict[str, Any]]) -> None:
        if not cases:
            raise ValueError("test_cases is empty")
        for case in cases:
            mode = case.get("service_mode", "openai")
            required = ["model", "envs", "benchmarks"]
            if mode == "epd":
                required.extend(["epd_server_cmds", "epd_proxy_args"])
            else:
                required.append("server_cmd")
            missing = [k for k in required if k not in case]
            if missing:
                raise KeyError(f"Missing required config fields: {missing}")

    @classmethod
    def _parse_test_cases(cls, cases: list[dict[str, Any]]) -> list[SingleNodeConfig]:
        result: list[SingleNodeConfig] = []
        for case in cases:
            server_cmd = case.get("server_cmd", [])
            server_cmd_extra = case.get("server_cmd_extra", [])
            full_cmd = list(server_cmd) + list(server_cmd_extra)
            extra_case_fields = {key: value for key, value in case.items() if key not in cls.STANDARD_CASE_FIELDS}
            result.append(
                SingleNodeConfig(
                    model=case["model"],
                    envs=case.get("envs", {}),
                    server_cmd=full_cmd,
                    benchmarks=case.get("benchmarks", {}),
                    prompts=case.get("prompts"),
                    api_keyword_args=case.get("api_keyword_args"),
                    test_content=case.get("test_content"),
                    service_mode=case.get("service_mode", "openai"),
                    **extra_case_fields,
                )
            )
        return result
