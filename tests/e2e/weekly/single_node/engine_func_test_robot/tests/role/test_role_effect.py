import json

import regex as re

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


def test_role_system_instruction_follow(api_client, request):
    """Test role system instruction follow."""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "system", "content": "Reply with JSON only."},
            {"role": "user", "content": 'Return {"ok": true}.'},
        ],
        "stream": False,
        "max_tokens": 10240,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: response behavior is valid
    content = response.json()["choices"][0]["message"]["content"]
    assert content is not None, "response should be valid"
    assert len(content) > 0, "response should be valid"

    # Check: response behavior is valid
    json_content = re.sub(r"\s*<think>[\s\S]*?</think>\s*", "", content)
    json_match = re.search(r"(\{.*\})\s*(?:$|`|```)$", json_content, re.S)
    assert json_match, "response should be valid"
    if json_match:
        assert json.loads(json_match.group(1)), "response"


def test_role_long_conversation_pruning(api_client, request):
    """Test role long conversation pruning."""
    # Check: response behavior is valid
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(20):
        messages.append({"role": "user", "content": f"Conversation turn {i + 1}"})
        messages.append({"role": "assistant", "content": f"Conversation turn {i + 1}"})
    messages.append({"role": "user", "content": "Hello."})

    request_body = {"model": "auto", "messages": messages, "stream": False, "max_tokens": 50}

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: response behavior is valid
    content = response.json()["choices"][0]["message"]["content"]
    assert content is not None, "response should be valid"
    assert len(content) > 0, "response should be valid"
