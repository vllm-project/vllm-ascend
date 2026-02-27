from typing import Any

import openai
import pytest

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.nightly.single_node.scripts.single_node_config import SingleNodeConfigLoader
from tools.aisbench import run_aisbench_cases

PROMPTS = [
    "San Francisco is a",
]

API_KEYWORD_ARGS = {
    "max_tokens": 10,
}

configs = SingleNodeConfigLoader.from_yaml_cases()


@pytest.mark.asyncio
@pytest.mark.parametrize("config", configs)
async def test_single_node(config) -> None:
    prompts = config.prompts or PROMPTS
    api_keyword_args = config.api_keyword_args or API_KEYWORD_ARGS
    request_keyword_args: dict[str, Any] = {
        **api_keyword_args,
    }
    with RemoteOpenAIServer(
        model=config.model,
        vllm_serve_args=config.server_cmd,
        server_port=config.server_port,
        env_dict=config.envs,
        auto_port=False,
    ) as server:
        if "completion" in config.test_content:
            client = server.get_async_client()
            batch = await client.completions.create(
                model=config.model,
                prompt=prompts,
                **request_keyword_args,
            )
            choices: list[openai.types.CompletionChoice] = batch.choices
            assert choices[0].text, "empty response"
            print(choices)

        if "image" in config.test_content:
            from tools.send_mm_request import send_image_request

            send_image_request(config.model, server)

        if "chat_completion" in config.test_content:
            from tools.send_request import send_v1_chat_completions

            send_v1_chat_completions(
                prompts[0],
                model=config.model,
                server=server,
                request_args=api_keyword_args,
            )

        aisbench_cases = [v for v in config.benchmarks.values() if v]
        if aisbench_cases:
            result = run_aisbench_cases(
                model=config.model,
                port=config.server_port,
                aisbench_cases=aisbench_cases,
            )
            if "TTFT_comparison" in config.test_content:
                from tools.aisbench import get_TTFT

                TTFT = get_TTFT(result)
                TTFT0 = TTFT[1]
                TTFT75 = TTFT[2]
                assert TTFT75 < 0.8 * TTFT0, (
                    f"The TTFT for prefix75 {TTFT75} is not less than 0.8*TTFT for prefix0 {TTFT0}."
                )
                print(f"The TTFT for prefix75 {TTFT75} is less than 0.8*TTFT for prefix0 {TTFT0}.")
