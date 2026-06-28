import pytest
from ...utility import request_helper as helper
from ...utility import assertion


def test_chat_stream_context_within_limit(api_client, request):
    """
    chat接口流式：
        1. 上下文长度在限制内，请求正常，
        2. auto truncate自动截短max_tokens（prompt_tokens(小于上下文长度) +max_tokens，总和大于模型上下文长度，auto truncate，正常响应，顺序执行N次，引擎不崩溃）
    """
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

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：请求成功（状态码200）
    assertion.assert_status_code_200(response, "流式响应")

    # 校验点2：流式响应包含[DONE]
    assertion.assert_stream_has_done(response.text, "流式响应")

    # 校验点3：流式响应finish_reason唯一且有效
    finish_reason = assertion.assert_stream_single_finish_reason(response.text, "流式响应")
    assertion.assert_finish_reason_valid(finish_reason, "流式响应")


def test_chat_non_stream_context_within_limit(api_client, request):
    """
    chat接口非流式：
        1. 上下文长度在限制内，请求正常，
        2. auto truncate自动截短max_tokens（prompt_tokens(小于上下文长度) +max_tokens，总和大于模型上下文长度，auto truncate，正常响应，顺序执行N次，引擎不崩溃）
    """
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

    helper.attach_request_body(request_body)

    for i in range(10):
        response = helper.send_request(api_client, "/v1/chat/completions", request_body)
        helper.attach_response_body(response, f"第{i+1}次响应")

        # 校验点1：请求成功（状态码200）
        assertion.assert_status_code_200(response, f"第{i+1}次请求")

        # 校验点2：finish_reason有效
        finish_reason = response.json()["choices"][0]["finish_reason"]
        assertion.assert_finish_reason_valid(finish_reason, f"第{i+1}次请求")


def test_completions_stream_context_within_limit(api_client, request):
    """
    completions接口流式：
        1. 上下文长度在限制内，请求正常，
        2. auto truncate自动截短max_tokens（prompt_tokens(小于上下文长度) +max_tokens，总和大于模型上下文长度，auto truncate，正常响应，顺序执行N次，引擎不崩溃）
    """
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

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：请求成功（状态码200）
    assertion.assert_status_code_200(response, "流式响应")

    # 校验点2：流式响应包含[DONE]
    assertion.assert_stream_has_done(response.text, "流式响应")

    # 校验点3：流式响应finish_reason唯一且有效
    finish_reason = assertion.assert_stream_single_finish_reason(response.text, "流式响应")
    assertion.assert_finish_reason_valid(finish_reason, "流式响应")


def test_completions_non_stream_context_within_limit(api_client, request):
    """
    completions接口非流式：
        1. 上下文长度在限制内，请求正常，
        2. auto truncate自动截短max_tokens（prompt_tokens(小于上下文长度) +max_tokens，总和大于模型上下文长度，auto truncate，正常响应，顺序执行N次，引擎不崩溃）
    """
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

    helper.attach_request_body(request_body)

    for i in range(1):
        response = helper.send_request(api_client, "/v1/completions", request_body)
        helper.attach_response_body(response, f"第{i+1}次响应")

        # 校验点1：请求成功（状态码200）
        assertion.assert_status_code_200(response, f"第{i+1}次请求")

        # 校验点2：finish_reason有效
        finish_reason = response.json()["choices"][0]["finish_reason"]
        assertion.assert_finish_reason_valid(finish_reason, f"第{i+1}次请求")
