import re
import pytest
from ...utility import request_helper as helper
from ...utility import assertion


def test_chat_stream_context_exceed(api_client, request):
    """Test chat stream context exceed."""
    with open("data/4k.txt", "r") as f:
        text = f.read()

    max_length = int(request.config.getoption("--maxModelLength"))
    repeat_count = max_length // 4

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": text * repeat_count
        }],
        "stream": True,
        "max_tokens": 512
    }


    # Check: response behavior is valid
    for i in range(10):
        response = helper.send_request(api_client, "/v1/chat/completions", request_body)

        # Check: response behavior is valid
        engine_type = request.config.getoption("--engineType")
        engine_arch = request.config.getoption("--engineArchitecture")

        if engine_type == "vllm" and engine_arch == "single":
            assertion.assert_status_code_400(response, f"vllm single-node streaming request #{i + 1}")
            assert not re.search(
                r'^data:\s?\[DONE\](?:\n|$)', response.text, re.M
            ), (
                f"vllm single-node streaming response #{i + 1} should not contain DONE"
            )
        else:
            assertion.assert_status_code_200(response, f"request #{i + 1}")
            assertion.assert_stream_has_done(response.text, f"streaming request #{i + 1}")

        assertion.assert_error_code_400(response, f"streaming request #{i + 1}")


def test_chat_non_stream_context_exceed(api_client, request):
    """Test chat non stream context exceed."""
    with open("data/4k.txt", "r") as f:
        text = f.read()

    max_length = int(request.config.getoption("--maxModelLength"))
    repeat_count = max_length // 4

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": text * repeat_count
        }],
        "stream": False,
        "max_tokens": 512,
        "stop": ["Input:"]
    }


    # Check: response behavior is valid
    for i in range(10):
        response = helper.send_request(api_client, "/v1/chat/completions", request_body)

        # Check: status code and error code are 400
        assertion.assert_status_code_400(response, f"non-streaming request #{i + 1}")
        assertion.assert_error_code_400(response, f"streaming request #{i + 1}")


def test_completions_stream_context_exceed(api_client, request):
    """Test completions stream context exceed."""
    with open("data/4k.txt", "r") as f:
        text = f.read()

    max_length = int(request.config.getoption("--maxModelLength"))
    repeat_count = max_length // 4

    request_body = {
        "model": "auto",
        "prompt": text * repeat_count,
        "stream": True,
        "max_tokens": 512,
        "stop": ["Input:"]
    }


    # Check: response behavior is valid
    for i in range(10):
        response = helper.send_request(api_client, "/v1/completions", request_body)

        # Check: response behavior is valid
        engine_type = request.config.getoption("--engineType")
        engine_arch = request.config.getoption("--engineArchitecture")

        if engine_type == "vllm" and engine_arch == "single":
            assertion.assert_status_code_400(response, f"vllm single-node streaming request #{i + 1}")
            assert not re.search(
                r'^data:\s?\[DONE\](?:\n|$)', response.text, re.M
            ), (
                f"vllm single-node streaming response #{i + 1} should not contain DONE"
            )
        else:
            assertion.assert_status_code_200(response, f"request #{i + 1}")
            assertion.assert_stream_has_done(response.text, f"streaming request #{i + 1}")

        assertion.assert_error_code_400(response, f"streaming request #{i + 1}")


def test_completions_non_stream_context_exceed(api_client, request):
    """Test completions non stream context exceed."""
    with open("data/4k.txt", "r") as f:
        text = f.read()

    max_length = int(request.config.getoption("--maxModelLength"))
    repeat_count = max_length // 4

    request_body = {
        "model": "auto",
        "prompt": text * repeat_count,
        "stream": False,
        "max_tokens": 512,
        "stop": ["Input:"]
    }


    # Check: response behavior is valid
    for i in range(10):
        response = helper.send_request(api_client, "/v1/completions", request_body)

        # Check: status code and error code are 400
        assertion.assert_status_code_400(response, f"non-streaming request #{i + 1}")
        assertion.assert_error_code_400(response, f"streaming request #{i + 1}")
