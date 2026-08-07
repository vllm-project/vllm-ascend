import os

import pytest
import yaml

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.weekly.single_node.engine_func_test_robot.utility.http_client import (
    HTTPClient,
)

env_dict: dict = {}

server_args: list = [
    "--served-model-name",
    "auto",
    "--max-model-len",
    "65536",
    "--tensor-parallel-size",
    "2",
    "--enable-expert-parallel",
    "--allowed-local-media-path",
    "/",
    "--limit-mm-per-prompt.video",
    "1",
    "--limit-mm-per-prompt.image",
    "5",
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "hermes",
    "--safetensors-load-strategy",
    "prefetch",
]


@pytest.fixture(scope="session")
def api_client(request):
    model = "Qwen/Qwen3-VL-30B-A3B-Instruct"

    with RemoteOpenAIServer(model, server_args, server_port=8000, env_dict=env_dict, auto_port=False) as server:
        yield HTTPClient(base_url=server.url_root)


def _load_yaml_config():
    """Load yaml configuration from file."""
    config_base_path = os.getenv("CONFIG_BASE_PATH") or "tests/e2e/weekly/single_node/configs"
    yaml_path = os.getenv("CONFIG_YAML_PATH", "Qwen3.5-27B-w8a8-A3.yaml")
    full_path = os.path.join(config_base_path, yaml_path)

    with open(full_path) as f:
        data = yaml.safe_load(f)

    test_cases = data.get("test_cases", [])
    if not test_cases:
        raise ValueError(f"No test_cases found in {full_path}")

    return test_cases[0]


def _expand_env_vars(values, envs):
    """Expand $VAR and ${VAR} placeholders in values."""
    import re

    pattern = re.compile(r"\$(\w+)|\$\{(\w+)\}")

    def repl(m):
        key = m.group(1) or m.group(2)
        return str(envs.get(key, m.group(0)))

    return [pattern.sub(repl, str(arg)) for arg in values]


@pytest.fixture(scope="session")
def yaml_api_client(request):
    """Fixture that loads server configuration from yaml file."""
    config = _load_yaml_config()

    model = config["model"]
    envs = config.get("envs", {})

    # Assign default port if not set
    if envs.get("SERVER_PORT") in ["DEFAULT_PORT", None]:
        from vllm.utils.network_utils import get_open_port

        envs["SERVER_PORT"] = str(get_open_port())

    server_cmd = _expand_env_vars(config.get("server_cmd", []), envs)
    port = int(envs["SERVER_PORT"])

    with RemoteOpenAIServer(model, server_cmd, server_port=port, env_dict=envs, auto_port=False) as server:
        yield HTTPClient(base_url=server.url_root)


def pytest_addoption(parser):
    parser.addoption("--thinkTagOutput", action="store", type=str, default="false", required=False)
    parser.addoption("--engineArchitecture", action="store", default="single", choices=["pd", "single"])
    parser.addoption("--maxModelLength", action="store", default="65536")
    parser.addoption("--model", action="store", default="qwen")
    parser.addoption("--imageNum", action="store", type=int, default=1)
    parser.addoption("--videoNum", action="store", type=int, default=1)
    parser.addoption("--audioNum", action="store", type=int, default=1)
