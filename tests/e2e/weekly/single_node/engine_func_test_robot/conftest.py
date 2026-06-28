import pytest

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


def pytest_addoption(parser):
    parser.addoption("--thinkTagOutput", action="store", type=str, default="false", required=False)
    parser.addoption("--engineArchitecture", action="store", default="single", choices=["pd", "single"])
    parser.addoption("--maxModelLength", action="store", default="128")
    parser.addoption("--model", action="store", default="qwen")
    parser.addoption("--imageNum", action="store", type=int, default=1)
    parser.addoption("--videoNum", action="store", type=int, default=1)
    parser.addoption("--audioNum", action="store", type=int, default=1)
