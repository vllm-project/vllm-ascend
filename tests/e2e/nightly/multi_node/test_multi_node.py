import time

import openai
import pytest
from modelscope import snapshot_download  # type: ignore
from requests.exceptions import ConnectionError, HTTPError, Timeout

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.nightly.multi_node.config.multi_node_config import (
    DISAGGREGATED_PREFILL_PROXY_SCRIPT, MultiNodeConfig)
from tools.aisbench import run_aisbench_cases

prompts = [
    "San Francisco is a",
]

api_keyword_args = {
    "max_tokens": 10,
}


def get_local_model_path_with_retry(
    model: str,
    revision: str = "master",
    max_retries: int = 5,
    delay: int = 5,
) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            local_model_path = snapshot_download(
                model_id=model,
                revision=revision,
                resume_download=True,
            )
            return local_model_path

        except HTTPError as e:
            status = getattr(e.response, "status_code", None)
            url = getattr(e.response, "url", None)

        except (ConnectionError, Timeout):
            continue

        if attempt < max_retries:
            time.sleep(delay)
    return None


@pytest.mark.asyncio
async def test_multi_node() -> None:
    config = MultiNodeConfig.from_yaml()
    env_dict = config.envs
    perf_cmd = config.perf_cmd
    acc_cmd = config.acc_cmd
    nodes_info = config.nodes_info
    disaggregated_prefill = config.disaggregated_prefill
    server_port = config.server_port
    proxy_port = config.proxy_port
    server_host = config.cluster_ips[0]
    with config.launch_server_proxy(DISAGGREGATED_PREFILL_PROXY_SCRIPT):
        with RemoteOpenAIServer(
                model=config.model,
                vllm_serve_args=config.server_cmd,
                server_port=server_port,
                server_host=server_host,
                env_dict=env_dict,
                auto_port=False,
                proxy_port=proxy_port,
                disaggregated_prefill=disaggregated_prefill,
                nodes_info=nodes_info,
                max_wait_seconds=2000,
        ) as remote_server:
            if config.is_master:
                port = proxy_port if disaggregated_prefill else server_port
                base_url = f"http://localhost:{port}/v1/completions"
                client = openai.AsyncOpenAI(base_url=base_url,
                                            api_key="token-abc123",
                                            max_retries=0,
                                            **{"timeout": 600})
                batch = await client.completions.create(
                    model=config.model,
                    prompt=prompts,
                    **api_keyword_args,
                )
                choices: list[openai.types.CompletionChoice] = batch.choices
                assert choices[0].text, "empty response"
                # aisbench test
                if acc_cmd:
                    run_aisbench_cases(config.model, port, acc_cmd)
                if perf_cmd:
                    run_aisbench_cases(config.model, port, perf_cmd)
            else:
                remote_server.hang_until_terminated()
