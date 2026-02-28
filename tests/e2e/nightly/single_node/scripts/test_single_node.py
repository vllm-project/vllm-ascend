import re
from typing import Any

import openai
import pytest
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import DisaggEpdProxy, RemoteEPDServer, RemoteOpenAIServer
from tests.e2e.nightly.single_node.scripts.single_node_config import SingleNodeConfigLoader
from tools.aisbench import run_aisbench_cases

PROMPTS = [
    "San Francisco is a",
]

API_KEYWORD_ARGS = {
    "max_tokens": 10,
}

configs = SingleNodeConfigLoader.from_yaml_cases()


def _expand_values(values: list[str], extra_envs: dict[str, str]) -> list[str]:
    pattern = re.compile(r"\$(\w+)|\$\{(\w+)\}")

    def repl(match: re.Match[str]) -> str:
        key = match.group(1) or match.group(2)
        return str(extra_envs.get(key, match.group(0)))

    return [pattern.sub(repl, str(value)) for value in values]


async def _run_test_content(config, server, prompts: list[str],
                            request_keyword_args: dict[str, Any]) -> None:
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
            request_args=request_keyword_args,
        )


def _run_benchmarks(config, port: int) -> None:
    aisbench_cases = [v for v in config.benchmarks.values() if v]
    if aisbench_cases:
        result = run_aisbench_cases(
            model=config.model,
            port=port,
            aisbench_cases=aisbench_cases,
        )
        if "TTFT_comparison" in config.test_content:
            from tools.aisbench import get_TTFT

            ttft = get_TTFT(result)
            ttft0 = ttft[1]
            ttft75 = ttft[2]
            assert ttft75 < 0.8 * ttft0, (
                f"The TTFT for prefix75 {ttft75} is not less than 0.8*TTFT for prefix0 {ttft0}."
            )
            print(f"The TTFT for prefix75 {ttft75} is less than 0.8*TTFT for prefix0 {ttft0}.")


@pytest.mark.asyncio
@pytest.mark.parametrize("config", configs)
async def test_single_node(config) -> None:
    prompts = config.prompts or PROMPTS
    api_keyword_args = config.api_keyword_args or API_KEYWORD_ARGS
    request_keyword_args: dict[str, Any] = {
        **api_keyword_args,
    }
    if config.service_mode == "epd":
        encode_port = str(get_open_port())
        decode_port = str(get_open_port())
        proxy_port = str(get_open_port())
        expand_envs = {
            **config.envs,
            "ENCODE_PORT": encode_port,
            "DECODE_PORT": decode_port,
            "PD_PORT": decode_port,
            "PROXY_PORT": proxy_port,
        }
        epd_server_cmds = [
            _expand_values(cmd, expand_envs)
            for cmd in config.epd_server_cmds
        ]
        epd_proxy_args = _expand_values(config.epd_proxy_args, expand_envs)
        with RemoteEPDServer(vllm_serve_args=epd_server_cmds,
                             env_dict=config.envs) as _, DisaggEpdProxy(
                                 proxy_args=epd_proxy_args,
                                 env_dict=config.envs) as proxy:
            await _run_test_content(config, proxy, prompts, request_keyword_args)
            _run_benchmarks(config, proxy.port)
        return

    with RemoteOpenAIServer(
        model=config.model,
        vllm_serve_args=config.server_cmd,
        server_port=config.server_port,
        env_dict=config.envs,
        auto_port=False,
    ) as server:
        await _run_test_content(config, server, prompts, request_keyword_args)
        _run_benchmarks(config, config.server_port)
