# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import pytest
import regex as re
import requests
import yaml
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer

logger = logging.getLogger(__name__)

CONFIG_BASE_PATH = os.getenv("CONFIG_BASE_PATH") or "tests/e2e/weekly/single_node/configs"
YAML_CONFIG_PATH = os.getenv("CONFIG_YAML_PATH", "Qwen3.5-27B-w8a8-A3.yaml")


@dataclass
class MessagesTestConfig:
    name: str
    model: str
    envs: dict[str, Any] = field(default_factory=dict)
    server_cmd: list[str] = field(default_factory=list)
    messages_tests: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        port_keys = ["SERVER_PORT", "ENCODE_PORT", "PD_PORT", "PROXY_PORT"]
        for env_key in port_keys:
            if self.envs.get(env_key) in ["DEFAULT_PORT", None]:
                self.envs[env_key] = str(get_open_port())

        self.server_cmd = self._expand_values(self.server_cmd or [], self.envs)

    @staticmethod
    def _expand_values(values: list[str], envs: dict[str, Any]) -> list[str]:
        pattern = re.compile(r"\$(\w+)|\$\{(\w+)\}")

        def repl(m: re.Match[str]) -> str:
            key = m.group(1) or m.group(2)
            return str(envs.get(key, m.group(0)))

        return [pattern.sub(repl, str(arg)) for arg in values]

    @property
    def server_port(self) -> int:
        value = self.envs.get("SERVER_PORT")
        if value is None:
            raise ValueError("Missing required port env: SERVER_PORT")
        return int(value)


def load_config() -> MessagesTestConfig:
    full_path = os.path.join(CONFIG_BASE_PATH, YAML_CONFIG_PATH)
    logger.info("Loading config yaml: %s", full_path)

    with open(full_path) as f:
        data = yaml.safe_load(f)

    test_cases = data.get("test_cases", [])
    if not test_cases:
        raise ValueError("No test_cases found in yaml")

    case = test_cases[0]
    return MessagesTestConfig(
        name=case["name"],
        model=case["model"],
        envs=case.get("envs", {}),
        server_cmd=case.get("server_cmd", []),
        messages_tests=case.get("messages_tests", []),
    )


async def run_messages_test(config: MessagesTestConfig, server: RemoteOpenAIServer) -> None:
    url = f"http://{server.host}:{server.port}"

    for test_case in config.messages_tests:
        test_type = test_case.get("type", "basic")
        prompt = test_case.get("prompt", "Hello!")
        max_tokens = test_case.get("max_tokens", 100)

        request_body = {
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }

        if test_type == "tool_calling":
            tools = test_case.get("tools", [])
            request_body["tools"] = tools
        elif test_type == "streaming":
            request_body["stream"] = True

        if test_type == "streaming":
            response = requests.post(
                f"{url}/v1/messages",
                headers={"Content-Type": "application/json"},
                json=request_body,
                stream=True,
            )
            assert response.status_code == 200, f"Streaming request failed: {response.text}"

            events = []
            for line in response.iter_lines():
                if line:
                    events.append(line.decode("utf-8"))

            assert len(events) > 0, f"No SSE events received for test '{test_type}'"
            assert any("message_start" in e for e in events), f"Missing 'message_start' event"
            assert any("content_block_delta" in e for e in events), f"Missing 'content_block_delta' event"
            assert any("message_stop" in e for e in events), f"Missing 'message_stop' event"
            logger.info("Streaming test passed (events=%d)", len(events))
        else:
            response = requests.post(
                f"{url}/v1/messages",
                headers={"Content-Type": "application/json"},
                json=request_body,
            )
            assert response.status_code == 200, f"Request failed: {response.text}"
            data = response.json()

            assert data["type"] == "message", f"Expected type 'message', got {data.get('type')}"
            assert data["role"] == "assistant", f"Expected role 'assistant', got {data.get('role')}"
            assert "content" in data, "Response missing 'content' field"
            assert isinstance(data["content"], list), f"Expected content to be a list"
            assert len(data["content"]) > 0, "Expected non-empty content"

            if test_type == "tool_calling":
                valid_blocks = [block for block in data["content"] if block.get("type") in ["text", "tool_use"]]
                assert len(valid_blocks) > 0, "No text or tool_use content found"
            else:
                text_content = [block for block in data["content"] if block.get("type") == "text"]
                assert len(text_content) > 0, "No text content found"
                actual_text = text_content[0].get("text", "")
                assert actual_text and actual_text.strip(), "Empty or whitespace-only text response"

            logger.info("Test '%s' passed: %s", test_type, data)


@pytest.mark.asyncio
async def test_messages_api() -> None:
    config = load_config()

    if not config.messages_tests:
        pytest.skip("No messages_tests defined in yaml config")

    with RemoteOpenAIServer(
        model=config.model,
        vllm_serve_args=config.server_cmd,
        server_port=config.server_port,
        env_dict=config.envs,
        auto_port=False,
    ) as server:
        await run_messages_test(config, server)
