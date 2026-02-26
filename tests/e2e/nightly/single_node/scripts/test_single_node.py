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
        client = server.get_async_client()
        batch = await client.completions.create(
            model=config.model,
            prompt=prompts,
            **request_keyword_args,
        )
        choices: list[openai.types.CompletionChoice] = batch.choices
        assert choices[0].text, "empty response"
        print(choices)
        # aisbench test
        aisbench_cases = [v for v in config.benchmarks.values() if v]
        if aisbench_cases:
            run_aisbench_cases(
                model=config.model,
                port=config.server_port,
                aisbench_cases=aisbench_cases,
            )
