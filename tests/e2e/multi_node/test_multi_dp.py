import subprocess

import pytest

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.multi_node.config.multi_node_config import (MultiNodeConfig,
                                                           load_configs)
from tests.e2e.multi_node.config.utils import get_default_envs

configs = load_configs()


def get_benchmark_cmd(model: str, base_url: str, args: list[str]):
    """vllm bench serve <model> --base-url <url> ..."""
    return [
        "vllm", "bench", "serve", "--model", model, "--base-url", base_url
    ] + args


@pytest.mark.parametrize("config", configs)
def test_multi_dp(config: MultiNodeConfig) -> None:
    env_dict = get_default_envs()

    server_config = config.server_config
    perf_config = config.perf_config
    model_name = server_config.model
    assert model_name is not None, "Model name must be specified"

    server_args = server_config.to_list()

    with RemoteOpenAIServer(
            model_name,
            server_args,
            server_host=config.server_host,
            server_port=config.server_port,
            env_dict=env_dict,
            auto_port=False,
            seed=1024,
            max_wait_seconds=1000,
    ) as remote_server:
        base_url = remote_server.url_root
        assert perf_config is not None, "Perf config must be specified for perf tests"
        perf_cmd = get_benchmark_cmd(server_config.model, base_url,
                                     perf_config.to_list())
        if server_config.headless:
            remote_server.hang_until_terminated()
        else:
            # run perf benchmark
            subprocess.run(perf_cmd, check=True)
