import subprocess

import pytest

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.multi_node.config.multi_node_config import (MultiNodeConfig,
                                                           load_configs)
from tests.e2e.multi_node.config.utils import get_default_envs

configs = load_configs()


@pytest.mark.parametrize("config", configs)
def test_multi_dp(config: MultiNodeConfig) -> None:
    env_dict = get_default_envs()

    server_config = config.server_config
    model_name = server_config.model
    assert model_name is not None, "Model name must be specified"

    server_args = server_config.to_list()

    with RemoteOpenAIServer(
            model_name,
            config.server_host,
            config.server_port,
            server_args,
            env_dict=env_dict,
            auto_port=False,
            seed=1024,
            max_wait_seconds=1000,
    ) as remote_server:
        base_url = remote_server.url_root
        cmd = [
            "vllm",
            "bench",
            "serve",
            "--model",
            model_name,
            "--dataset-name",
            "random",
            "--random-input-len",
            "128",
            "--random-output-len",
            "128",
            "--num-prompts",
            "200",
            "--trust-remote-code",
            "--base-url",
            base_url,
            "--request-rate",
            "10",
        ]
        subprocess.run(cmd, check=True)
