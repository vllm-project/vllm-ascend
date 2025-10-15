from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.nightly.multi_node.config.multi_node_config import \
    MultiNodeConfig


def test_multi_node() -> None:
    config = MultiNodeConfig.from_yaml()
    env_dict = config.envs
    perf_cmd = config.perf_cmd
    acc_cmd = config.acc_cmd
    if config.disaggregated_prefill:
        config.launch_server_proxy()

    with RemoteOpenAIServer(
            model=config.model,
            vllm_serve_args=config.server_cmd,
            server_port=config.server_port,
            env_dict=env_dict,
            auto_port=False,
            max_wait_seconds=1000,
    ) as remote_server:
        base_url = remote_server.url_root
        if config.is_master:
            pass
            # subprocess.run(perf_cmd, check=True)
            # subprocess.run(acc_cmd, check=True)
        else:
            remote_server.hang_until_terminated()
            # TODO: enable perf and acc test
