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
    host = "0.0.0.0"
    port = 8000

    # model = "/mnt/share/weights/deepseekv4-flash-w8a8-mtp"
    # model = "/mnt/share/weight/Qwen3-VL-30B-A3B-Instruct"
    model = "Qwen/Qwen3-VL-30B-A3B-Instruct"

    with (
        RemoteOpenAIServer(
            model, server_args, server_port=port, env_dict=env_dict, auto_port=False
        ) as server,
    ):
        yield HTTPClient(base_url=server.url_root)
    # yield HTTPClient(base_url=f"http://{host}:{port}")


def pytest_addoption(parser):
    parser.addoption(
        "--modelServiceIP", required=False, help="设置模型服务ip，必传字段"
    )
    parser.addoption(
        "--modelServicePort",
        action="store",
        default="7001",
        help="设置模型服务port，默认端口7001",
    )
    parser.addoption(
        "--thinkTagOutput",
        action="store",
        type=str,
        required=False,
        help="设置模型服务是否需要输出think标签",
    )
    parser.addoption(
        "--engineType",
        action="store",
        default="vllm",
        help="设置引擎类型（如vllm/sglang）",
    )
    parser.addoption(
        "--engineArchitecture",
        action="store",
        default="pd",
        choices=["pd", "single"],
        help="设置引擎架构（如pd/single），default: pd",
    )
    parser.addoption(
        "--maxModelLength",
        action="store",
        default="128",
        help="设置模型最大上下文长度，单位：k（1024）",
    )
    parser.addoption(
        "--model",
        action="store",
        default="auto",
        help="实际挂载的后端模型名称，默认auto",
    )
    parser.addoption(
        "--modelService",
        action="store",
        default="auto",
        help='模型服务标识，默认auto。如果不是auto，则是aicloud:<thetaops服务名>，示例"aicloud:thetaopsModelServiceName"',
    )
    parser.addoption(
        "--imageNum", action="store", type=int, default=1, help="设置图片数量，默认1"
    )
    parser.addoption(
        "--videoNum", action="store", type=int, default=1, help="设置视频数量，默认1"
    )
    parser.addoption(
        "--audioNum", action="store", type=int, default=1, help="设置音频数量，默认1"
    )
