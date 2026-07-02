import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_null(api_client, request, stream):
    """chat_template_kwargs is null; the request should respond normally."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": None,
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check 1: status code is 200
    assertion.assert_status_code_200(response)

    # Check 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check 3: finish_reason is stop or length
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_empty_object(api_client, request, stream):
    """chat_template_kwargs is an empty object {}; the optional field should be handled normally."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {},
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check 1: status code is 200
    assertion.assert_status_code_200(response)

    # Check 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check 3: finish_reason is valid
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_with_add_generation_prompt(api_client, request, stream):
    """Set add_generation_prompt in chat_template_kwargs to control generation prompt insertion."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {"add_generation_prompt": True},
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check 1: status code is 200
    assertion.assert_status_code_200(response)

    # Check 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_with_custom_system_prompt(api_client, request, stream):
    """Set custom system-prompt-related parameters in chat_template_kwargs."""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "system", "content": "你是AI助手"},
            {"role": "user", "content": "你好"},
        ],
        "chat_template_kwargs": {"enable_system_prompt": True},
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check 1: status code is 200
    assertion.assert_status_code_200(response)

    # Check 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_date_params(api_client, request, stream):
    """chat_template_kwargs contains date-related parameters; some models support dynamic dates."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "今天是星期几"}],
        "chat_template_kwargs": {"date": "2025-04-01", "time": "10:00:00"},
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check 1: status code is 200
    assertion.assert_status_code_200(response)

    # Check 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_multiple_params(api_client, request, stream):
    """chat_template_kwargs contains multiple valid parameters."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "请简单回答"}],
        "chat_template_kwargs": {
            "add_generation_prompt": True,
            "tools_prompt": "default",
            "custom_var": "custom_value",
        },
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check 1: status code is 200
    assertion.assert_status_code_200(response)

    # Check 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_with_tools_prompt(api_client, request, stream):
    """Set tools_prompt in chat_template_kwargs to control the tool-call prompt format."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "需要查询天气"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ],
        "chat_template_kwargs": {"tools_prompt": "tool_instruction"},
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check 1: status code is 200
    assertion.assert_status_code_200(response)

    # Check 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_bos_token(api_client, request, stream):
    """Set add_special_tokens or bos_token-related parameters in chat_template_kwargs."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {"add_special_tokens": True},
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check 1: status code is 200
    assertion.assert_status_code_200(response)

    # Check 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_skip_special_tokens(api_client, request, stream):
    """Set skip_special_tokens in chat_template_kwargs."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {"skip_special_tokens": False},
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check 1: status code is 200
    assertion.assert_status_code_200(response)

    # Check 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)
