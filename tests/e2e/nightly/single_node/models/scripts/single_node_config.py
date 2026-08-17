import logging
import os
from dataclasses import dataclass, field
from typing import Any

import regex as re
import yaml
from vllm import envs as envs_vllm
from vllm.utils.network_utils import get_open_port

CONFIG_BASE_PATH = os.getenv("CONFIG_BASE_PATH") or "tests/e2e/nightly/single_node/models/configs"

logger = logging.getLogger(__name__)

# Default prompts and API args fallback
PROMPTS = [
    "San Francisco is a",
]

API_KEYWORD_ARGS = {
    "max_tokens": 10,
}


@dataclass
class SingleNodeConfig:
    name: str
    model: str
    envs: dict[str, Any] = field(default_factory=dict)
    special_dependencies: dict[str, Any] = field(default_factory=dict)
    prompts: list[str] = field(default_factory=lambda: PROMPTS)
    api_keyword_args: dict[str, Any] = field(default_factory=lambda: API_KEYWORD_ARGS)
    benchmarks: dict[str, Any] = field(default_factory=dict)
    server_cmd: list[str] = field(default_factory=list)
    test_content: list[str] = field(default_factory=lambda: ["completion"])
    service_mode: str = "openai"
    epd_server_cmds: list[list[str]] = field(default_factory=list)
    epd_proxy_args: list[str] = field(default_factory=list)
    mm_request: dict[str, Any] = field(default_factory=dict)
    expected_response: dict[str, Any] = field(default_factory=dict)
    extra_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        port_keys = ["SERVER_PORT", "ENCODE_PORT", "PD_PORT", "PROXY_PORT"]
        for env_key in port_keys:
            if self.envs.get(env_key) in ["DEFAULT_PORT", None]:
                self.envs[env_key] = str(get_open_port())

        if self.prompts is None:
            self.prompts = PROMPTS
        if self.api_keyword_args is None:
            self.api_keyword_args = API_KEYWORD_ARGS
        if self.benchmarks is None:
            self.benchmarks = {}
        if self.special_dependencies is None:
            self.special_dependencies = {}
        if self.test_content is None:
            self.test_content = []
        if self.mm_request is None:
            self.mm_request = {}
        if self.expected_response is None:
            self.expected_response = {}
        self.server_cmd = self._expand_values(self.server_cmd or [], self.envs)
        self.epd_server_cmds = [self._expand_values(cmd, self.envs) for cmd in self.epd_server_cmds]
        self.epd_proxy_args = self._expand_values(self.epd_proxy_args or [], self.envs)

        for key, value in self.extra_config.items():
            setattr(self, key, value)

    @staticmethod
    def _expand_values(values: list[str], envs: dict[str, Any]) -> list[str]:
        """Interpolate $VAR/${VAR} placeholders with provided env values."""
        pattern = re.compile(r"\$(\w+)|\$\{(\w+)\}")

        def repl(m: re.Match[str]) -> str:
            key = m.group(1) or m.group(2)
            return str(envs.get(key, m.group(0)))

        return [pattern.sub(repl, str(arg)) for arg in values]

    def _get_required_port(self, key: str) -> int:
        value = self.envs.get(key)
        if value is None:
            raise ValueError(f"Missing required port env: {key}")
        return int(value)

    @property
    def server_port(self) -> int:
        return self._get_required_port("SERVER_PORT")

    @property
    def encode_port(self) -> int:
        return self._get_required_port("ENCODE_PORT")

    @property
    def pd_port(self) -> int:
        return self._get_required_port("PD_PORT")

    @property
    def proxy_port(self) -> int:
        return self._get_required_port("PROXY_PORT")


class SingleNodeConfigLoader:
    """Load SingleNodeConfig from yaml file."""

    DEFAULT_CONFIG_NAME = "Kimi-K2-Thinking.yaml"
    STANDARD_CASE_FIELDS = {
        "name",
        "model",
        "envs",
        "special_dependencies",
        "prompts",
        "api_keyword_args",
        "benchmarks",
        "service_mode",
        "server_cmd",
        "server_cmd_extra",
        "test_content",
        "epd_server_cmds",
        "epd_proxy_args",
        "expected_response",
        "mm_request",
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
            required = ["name", "model", "envs"]
            if mode == "epd":
                required.extend(["epd_server_cmds", "epd_proxy_args"])
            else:
                required.append("server_cmd")
            missing = [k for k in required if k not in case]
            if missing:
                raise KeyError(f"Missing required config fields: {missing}")

            if not isinstance(case["name"], str) or not case["name"].strip():
                raise ValueError("test case field 'name' must be a non-empty string")

    @staticmethod
    def _resolve_runner_case(case: dict[str, Any]) -> dict[str, Any]:
        """Merge the selected model-runner overrides into one test case."""
        runner_configs = case.get("runner_configs")
        if runner_configs is None:
            return case
        if not isinstance(runner_configs, dict):
            raise TypeError("test case runner_configs must be a mapping")

        runner_name = "v2" if bool(envs_vllm.VLLM_USE_V2_MODEL_RUNNER) else "v1"
        if runner_name not in runner_configs:
            raise KeyError(f"test case runner_configs.{runner_name} is required")
        runner_config = runner_configs[runner_name]
        if not isinstance(runner_config, dict):
            raise TypeError(f"test case runner_configs.{runner_name} must be a mapping")

        runner_envs = runner_config.get("envs", {})
        if not isinstance(runner_envs, dict):
            raise TypeError(f"test case runner_configs.{runner_name}.envs must be a mapping")
        runner_server_cmd_extra = runner_config.get("server_cmd_extra", [])
        if not isinstance(runner_server_cmd_extra, list):
            raise TypeError(f"test case runner_configs.{runner_name}.server_cmd_extra must be a list")

        resolved_case = dict(case)
        resolved_case.pop("runner_configs")
        envs = dict(case.get("envs", {}))
        envs.update(runner_envs)
        resolved_case["envs"] = envs
        resolved_case["server_cmd_extra"] = [
            *case.get("server_cmd_extra", []),
            *runner_server_cmd_extra,
        ]
        return resolved_case

    @classmethod
    def _parse_test_cases(cls, cases: list[dict[str, Any]]) -> list[SingleNodeConfig]:
        result: list[SingleNodeConfig] = []
        for case in cases:
            case = cls._resolve_runner_case(case)
            server_cmd = case.get("server_cmd", [])
            server_cmd_extra = case.get("server_cmd_extra", [])
            full_cmd = list(server_cmd) + list(server_cmd_extra)
            extra_case_fields = {key: value for key, value in case.items() if key not in cls.STANDARD_CASE_FIELDS}

            # Safe parsing mapping
            result.append(
                SingleNodeConfig(
                    name=case["name"],
                    model=case["model"],
                    envs=case.get("envs", {}),
                    special_dependencies=case.get("special_dependencies", {}),
                    server_cmd=full_cmd,
                    epd_server_cmds=case.get("epd_server_cmds", []),
                    epd_proxy_args=case.get("epd_proxy_args", []),
                    benchmarks=case.get("benchmarks", {}),
                    prompts=case.get("prompts", PROMPTS),
                    api_keyword_args=case.get("api_keyword_args", API_KEYWORD_ARGS),
                    test_content=case.get("test_content", ["completion"]),
                    service_mode=case.get("service_mode", "openai"),
                    extra_config=extra_case_fields,
                    expected_response=case.get("expected_response", {}),
                    mm_request=case.get("mm_request", {}),
                )
            )
        return result
