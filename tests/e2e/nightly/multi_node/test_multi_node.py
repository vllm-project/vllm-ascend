from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.nightly.multi_node.config.multi_node_config import (
    DISAGGREGATED_PREFILL_PROXY_SCRIPT, MultiNodeConfig)


def test_multi_node() -> None:
    config = MultiNodeConfig.from_yaml()
    env_dict = config.envs
    # perf_cmd = config.perf_cmd
    # acc_cmd = config.acc_cmd
    server_port = config.server_port
    server_host = config.cluster_ips[0]
    with config.launch_server_proxy(DISAGGREGATED_PREFILL_PROXY_SCRIPT):
        with RemoteOpenAIServer(
                model=config.model,
                vllm_serve_args=config.server_cmd,
                disaggregated_prefill=config.disaggregated_prefill,
                cluster_ips=config.cluster_ips,
                server_port=server_port,
                server_host=server_host,
                env_dict=env_dict,
                auto_port=False,
                max_wait_seconds=2000,
        ) as remote_server:
            # base_url = remote_server.url_root
            if config.is_master:
                pass
                # TODO: enable perf and acc test
                # subprocess.run(perf_cmd, check=True)
                # subprocess.run(acc_cmd, check=True)
            else:
                remote_server.hang_until_terminated()
