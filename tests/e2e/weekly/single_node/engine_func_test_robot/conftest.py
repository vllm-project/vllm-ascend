import pytest
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.weekly.single_node.engine_func_test_robot.utility.http_client import (
    HTTPClient,
)

env_dict = {}

server_model_name = "auto"

server_args = [
    "--served-model-name",
    server_model_name,
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
    port = get_open_port()

    model = "Qwen/Qwen3-VL-30B-A3B-Instruct"

    with (RemoteOpenAIServer(model, server_args, server_port=port, env_dict=env_dict, auto_port=False) as server,):
        yield HTTPClient(base_url=server.url_root)
