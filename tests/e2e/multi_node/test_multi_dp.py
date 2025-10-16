import subprocess
from pathlib import Path

import pytest

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.multi_node.config.common import RANKTABLE_PATH, get_world_size, get_npu_per_node
from tests.e2e.multi_node.config.multi_node_config import (MultiNodeConfig,
                                                           load_configs)
from tests.e2e.multi_node.config.utils import get_default_envs, get_cluster_ips
from tests.e2e.multi_node.config.generate_ranktable import DisaggegatedPrefill

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
    if config.disaggregate_prefill:
        disaggerated_prefill = DisaggegatedPrefill(config)

        # generate ranktable.json
        disaggerated_prefill.setup_and_run_ranktable()
        # run proxy
        disaggerated_prefill.launch_server_proxy()

        env_dict["DISAGGREGATED_PREFILL_RANK_TABLE_PATH"] = RANKTABLE_PATH
        #env_dict["VLLM_ASCEND_LLMDD_RPC_PORT"] = "5559"

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
