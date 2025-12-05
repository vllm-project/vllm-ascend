import time
from typing import Any, List, Optional, Union

import httpx
import pytest
from modelscope import snapshot_download  # type: ignore
from requests.exceptions import ConnectionError, HTTPError, Timeout

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.nightly.multi_node.config.multi_node_config import \
    MultiNodeConfig
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
) -> Optional[str]:
    for attempt in range(1, max_retries + 1):
        try:
            local_model_path = snapshot_download(
                model_id=model,
                revision=revision,
            )
            return local_model_path

        except HTTPError:
            continue

        except (ConnectionError, Timeout):
            continue

        if attempt < max_retries:
            time.sleep(delay)
    return None


async def get_completions(url: str, model: str, prompts: Union[str, List[str]],
                          **api_kwargs: Any) -> List[str]:
    """
    Asynchronously send HTTP requests to endpoint.

    Args:
        url: Full endpoint URL, e.g. "http://localhost:1025/v1/completions"
        model: Model name or local model path
        prompts: A single prompt string or a list of prompts
        **api_kwargs: Additional parameters (e.g., max_tokens, temperature)

    Returns:
        List[str]: A list of generated texts corresponding to each prompt
    """
    headers = {"Content-Type": "application/json"}

    if isinstance(prompts, str):
        prompts = [prompts]

    results = []
    async with httpx.AsyncClient(timeout=600.0) as client:
        for prompt in prompts:
            payload = {"model": model, "prompt": prompt, **api_kwargs}

            response = await client.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Request failed with status {response.status_code}: {response.text}"
                )

            resp_json = response.json()
            choices = resp_json.get("choices", [])
            if not choices or not choices[0].get("text"):
                raise ValueError("Empty response from server")

            results.append(choices[0]["text"])

    return results


@pytest.mark.asyncio
async def test_multi_node() -> None:
    config = MultiNodeConfig.from_yaml()
    # To avoid modelscope 400 HttpError, we should download the model with retry
    local_model_path = get_local_model_path_with_retry(config.model)
    config.server_cmd = config.server_cmd.replace(config.model,
                                                  local_model_path)
    assert local_model_path is not None, "can not find any local weight for test"
    env_dict = config.envs
    perf_cmd = config.perf_cmd
    acc_cmd = config.acc_cmd
    nodes_info = config.nodes_info
    disaggregated_prefill = config.disaggregated_prefill
    server_port = config.server_port
    proxy_port = config.proxy_port
    server_host = config.master_ip
    proxy_script = config.envs.get("DISAGGREGATED_PREFILL_PROXY_SCRIPT", \
        'examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py')
    with config.launch_server_proxy(proxy_script):
        with RemoteOpenAIServer(
                model=local_model_path,
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
                # aisbench test
                aisbench_cases = [acc_cmd, perf_cmd]
                run_aisbench_cases(local_model_path,
                                   port,
                                   aisbench_cases,
                                   host_ip=config.master_ip)
            else:
                # for the nodes except master, should hang until the task complete
                master_url = f"http://{config.master_ip}:{server_port}/health"
                remote_server.hang_until_terminated(master_url)
