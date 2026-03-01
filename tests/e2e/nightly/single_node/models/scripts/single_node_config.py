import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import yaml
from vllm.utils.network_utils import get_open_port

CONFIG_BASE_PATH = "tests/e2e/nightly/single_node/models/models_yaml"
DEFAULT_SERVER_PORT = get_open_port()

logger = logging.getLogger(__name__)

# Default prompts and API args fallback
PROMPTS = [
    "San Francisco is a",
]

API_KEYWORD_ARGS = {
    "max_tokens": 10,
}


def expand_values(values: list[str], envs: dict[str, Any]) -> list[str]:
    """Helper clearly extracting var interpolation out of config processing.
    We convert the dictionary values to str since re.sub takes string replacements."""
    pattern = re.compile(r"\$(\w+)|\$\{(\w+)\}")

    def repl(m: re.Match[str]) -> str:
        key = m.group(1) or m.group(2)
        return str(envs.get(key, m.group(0)))

    return [pattern.sub(repl, str(arg)) for arg in values]


@dataclass
class SingleNodeConfig:
    model: str
    envs: dict[str, Any] = field(default_factory=dict)
    prompts: list[str] = field(default_factory=lambda: PROMPTS)
    api_keyword_args: dict[str, Any] = field(default_factory=lambda: API_KEYWORD_ARGS)
    benchmarks: dict[str, Any] = field(default_factory=dict)
    server_cmd: list[str] = field(default_factory=list)
    test_content: list[str] = field(default_factory=lambda: ["completion"])
    service_mode: str = "openai"
    epd_server_cmds: list[list[str]] = field(default_factory=list)
    epd_proxy_args: list[str] = field(default_factory=list)
    extra_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.envs.get("SERVER_PORT") == "DEFAULT_PORT":
            self.envs["SERVER_PORT"] = str(DEFAULT_SERVER_PORT)

        # fallback to defaults if explicit None was passed during dict-parsing
        if self.prompts is None:
            self.prompts = PROMPTS
        if self.api_keyword_args is None:
            self.api_keyword_args = API_KEYWORD_ARGS
        if self.benchmarks is None:
            self.benchmarks = {}
        if self.test_content is None:
            self.test_content = ["completion"]

        self.server_cmd = expand_values(self.server_cmd or [], self.envs)

        for key, value in self.extra_config.items():
            setattr(self, key, value)

    @property
    def server_port(self) -> int:
        if self.envs.get("SERVER_PORT") == "DEFAULT_PORT":
            return int(DEFAULT_SERVER_PORT)
        return int(self.envs.get("SERVER_PORT", DEFAULT_SERVER_PORT))


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
            required = ["model", "envs"]
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

            # Safe parsing mapping
            result.append(
                SingleNodeConfig(
                    model=case["model"],
                    envs=case.get("envs", {}),
                    server_cmd=full_cmd,
                    epd_server_cmds=case.get("epd_server_cmds", []),
                    epd_proxy_args=case.get("epd_proxy_args", []),
                    benchmarks=case.get("benchmarks", {}),
                    prompts=case.get("prompts", PROMPTS),
                    api_keyword_args=case.get("api_keyword_args", API_KEYWORD_ARGS),
                    test_content=case.get("test_content", ["completion"]),
                    service_mode=case.get("service_mode", "openai"),
                    extra_config=extra_case_fields,
                )
            )
        return result
