import re
import pytest
import pytest_check as check
from ...utility import request_helper as helper
from ...utility import assertion


def test_chat_stream_context_exceed(api_client, request):
    """chat接口流式：上下文长度超限，请求十次，引擎不崩溃"""
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

    helper.attach_request_body(request_body)

    # 执行10次，验证引擎不崩溃
    for i in range(10):
        response = helper.send_request(api_client, "/v1/chat/completions", request_body)
        helper.attach_response_body(response, f"第{i+1}次响应")

        # 校验点：引擎架构影响响应行为
        engine_type = request.config.getoption("--engineType")
        engine_arch = request.config.getoption("--engineArchitecture")

        if engine_type == "vllm" and engine_arch == "single":
            assertion.assert_status_code_400(response, f"vllm单机流式第{i+1}次")
            check.is_false(
                re.search(r'^data:\s?\[DONE\](?:\n|$)', response.text, re.M),
                f"vllm单机流式第{i+1}次响应不应存在done"
            )
        else:
            assertion.assert_status_code_200(response, f"流式第{i+1}次")
            assertion.assert_stream_has_done(response.text, f"流式第{i+1}次")

        assertion.assert_error_code_400(response, f"流式第{i+1}次")


def test_chat_non_stream_context_exceed(api_client, request):
    """chat接口非流式：上下文长度超限，请求十次，引擎不崩溃"""
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

    helper.attach_request_body(request_body)

    # 执行10次，验证引擎不崩溃
    for i in range(10):
        response = helper.send_request(api_client, "/v1/chat/completions", request_body)
        helper.attach_response_body(response, f"第{i+1}次响应")

        # 校验点：状态码400，错误码400
        assertion.assert_status_code_400(response, f"非流式第{i+1}次")
        assertion.assert_error_code_400(response, f"非流式第{i+1}次")


def test_completions_stream_context_exceed(api_client, request):
    """completions接口流式：上下文长度超限，请求十次，引擎不崩溃"""
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

    helper.attach_request_body(request_body)

    # 执行10次，验证引擎不崩溃
    for i in range(10):
        response = helper.send_request(api_client, "/v1/completions", request_body)
        helper.attach_response_body(response, f"第{i+1}次响应")

        # 校验点：引擎架构影响响应行为
        engine_type = request.config.getoption("--engineType")
        engine_arch = request.config.getoption("--engineArchitecture")

        if engine_type == "vllm" and engine_arch == "single":
            assertion.assert_status_code_400(response, f"vllm单机流式第{i+1}次")
            check.is_false(
                re.search(r'^data:\s?\[DONE\](?:\n|$)', response.text, re.M),
                f"vllm单机流式第{i+1}次响应不应存在done"
            )
        else:
            assertion.assert_status_code_200(response, f"流式第{i+1}次")
            assertion.assert_stream_has_done(response.text, f"流式第{i+1}次")

        assertion.assert_error_code_400(response, f"流式第{i+1}次")


def test_completions_non_stream_context_exceed(api_client, request):
    """completions接口非流式：上下文长度超限，返回400错误"""
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

    helper.attach_request_body(request_body)

    # 执行10次，验证引擎不崩溃
    for i in range(10):
        response = helper.send_request(api_client, "/v1/completions", request_body)
        helper.attach_response_body(response, f"第{i+1}次响应")

        # 校验点：状态码400，错误码400
        assertion.assert_status_code_400(response, f"非流式第{i+1}次")
        assertion.assert_error_code_400(response, f"非流式第{i+1}次")