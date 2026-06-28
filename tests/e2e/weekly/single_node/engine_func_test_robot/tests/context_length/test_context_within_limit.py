import pytest
from ...utility import request_helper as helper
from ...utility import assertion


def test_chat_stream_context_within_limit(api_client, request):
    """Test chat stream context within limit."""
    with open("data/4k.txt", "r") as f:
        text = f.read().replace("\n", "")

    max_length = int(request.config.getoption("--maxModelLength"))
    repeat_count = max_length // 5

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": text * repeat_count
        }],
        "stream": True,
        "max_tokens": 51200000
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response, 'response')

    # Check: streaming response contains [DONE]
    assertion.assert_stream_has_done(response.text, 'response')

    # Check: finish_reason is valid
    finish_reason = assertion.assert_stream_single_finish_reason(response.text, 'response')
    assertion.assert_finish_reason_valid(finish_reason, 'response')


def test_chat_non_stream_context_within_limit(api_client, request):
    """Test chat non stream context within limit."""
    with open("data/4k.txt", "r") as f:
        text = f.read().replace("\n", "")

    max_length = int(request.config.getoption("--maxModelLength"))
    repeat_count = max_length // 5

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": text * repeat_count
        }],
        "stream": False,
        "max_tokens": 51200000
    }


    for i in range(10):
        response = helper.send_request(api_client, "/v1/chat/completions", request_body)

        # Check: status code is 200
        assertion.assert_status_code_200(response, f"request #{i + 1}")

        # Check: finish_reason is valid
        finish_reason = response.json()["choices"][0]["finish_reason"]
        assertion.assert_finish_reason_valid(finish_reason, f"request #{i + 1}")


def test_completions_stream_context_within_limit(api_client, request):
    """Test completions stream context within limit."""
    with open("data/4k.txt", "r") as f:
        text = f.read().replace("\n", "")

    max_length = int(request.config.getoption("--maxModelLength"))
    repeat_count = max_length // 5

    request_body = {
        "model": "auto",
        "prompt": text * repeat_count,
        "stream": True,
        "max_tokens": 51200000
    }

    response = helper.send_request(api_client, "/v1/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response, 'response')

    # Check: streaming response contains [DONE]
    assertion.assert_stream_has_done(response.text, 'response')

    # Check: finish_reason is valid
    finish_reason = assertion.assert_stream_single_finish_reason(response.text, 'response')
    assertion.assert_finish_reason_valid(finish_reason, 'response')


def test_completions_non_stream_context_within_limit(api_client, request):
    """Test completions non stream context within limit."""
    with open("data/4k.txt", "r") as f:
        text = f.read().replace("\n", "")

    max_length = int(request.config.getoption("--maxModelLength"))
    repeat_count = max_length // 5

    request_body = {
        "model": "auto",
        "prompt": text * repeat_count,
        "stream": False,
        "max_tokens": 51200000
    }


    for i in range(1):
        response = helper.send_request(api_client, "/v1/completions", request_body)

        # Check: status code is 200
        assertion.assert_status_code_200(response, f"request #{i + 1}")

        # Check: finish_reason is valid
        finish_reason = response.json()["choices"][0]["finish_reason"]
        assertion.assert_finish_reason_valid(finish_reason, f"request #{i + 1}")
