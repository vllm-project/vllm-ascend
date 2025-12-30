import pytest

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.nightly.multi_node.scripts.multi_node_config import (
    MultiNodeConfigLoader, ProxyLauncher)
from tools.aisbench import run_aisbench_cases


@pytest.mark.asyncio
async def test_multi_node() -> None:
    config = MultiNodeConfigLoader.from_yaml()

    with ProxyLauncher(
            nodes=config.nodes,
            disagg_cfg=config.disagg_cfg,
            envs=config.envs,
            proxy_port=config.proxy_port,
            cur_index=config.cur_index,
    ) as proxy:

        with RemoteOpenAIServer(
                model=config.model,
                vllm_serve_args=config.server_cmd,
                server_port=config.server_port,
                server_host=config.master_ip,
                env_dict=config.envs,
                auto_port=False,
                proxy_port=proxy.proxy_port,
                disaggregated_prefill=config.disagg_cfg,
                nodes_info=config.nodes,
                max_wait_seconds=2800,
        ) as server:

            host, port = config.benchmark_endpoint

            if config.is_master:
                run_aisbench_cases(
                    model=config.model,
                    port=port,
                    aisbench_cases=[config.acc_cmd, config.perf_cmd],
                    host_ip=host,
                )
            else:
                # We should keep listening on the master node's server url determining when to exit.
                server.hang_until_terminated(
                    f"http://{host}:{config.server_port}/health")
