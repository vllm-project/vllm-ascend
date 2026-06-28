import pytest
from ...utility import request_helper as helper
from ...utility import assertion


def test_presence_penalty_exceed_upper_limit_non_stream(api_client, request):
    """非流式：presence_penalty超过上限(>2.0)，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "presence_penalty": 2.1,
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_presence_penalty_exceed_upper_limit_stream(api_client, request):
    """流式：presence_penalty超过上限，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "presence_penalty": 2.1,
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_presence_penalty_below_lower_limit_non_stream(api_client, request):
    """非流式：presence_penalty低于下限(<-2.0)，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "presence_penalty": -2.1,
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_presence_penalty_below_lower_limit_stream(api_client, request):
    """流式：presence_penalty低于下限，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "presence_penalty": -2.1,
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_presence_penalty_string_non_stream(api_client, request):
    """非流式：presence_penalty为字符串整型，应该正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "presence_penalty": "1",
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：finish_reason为stop或length
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_presence_penalty_string_stream(api_client, request):
    """流式：presence_penalty为字符串整型，应该正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "presence_penalty": "1",
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    assertion.assert_stream_has_done(response.text)

    # 校验点3：finish_reason为stop或length
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    assertion.assert_finish_reason_valid(finish_reason)


def test_presence_penalty_null_non_stream(api_client, request):
    """非流式：presence_penalty为null，应该正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "presence_penalty": None,
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：finish_reason为stop或length
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_presence_penalty_null_stream(api_client, request):
    """流式：presence_penalty为null，应该正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "presence_penalty": None,
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    assertion.assert_stream_has_done(response.text)

    # 校验点3：finish_reason为stop或length
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    assertion.assert_finish_reason_valid(finish_reason)


def test_presence_penalty_array_non_stream(api_client, request):
    """非流式：presence_penalty为数组类型，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "presence_penalty": [0.5, 1.0],
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_presence_penalty_array_stream(api_client, request):
    """流式：presence_penalty为数组类型，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "presence_penalty": [0.5, 1.0],
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_presence_penalty_object_non_stream(api_client, request):
    """非流式：presence_penalty为对象类型，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "presence_penalty": {"value": 1.0},
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_presence_penalty_object_stream(api_client, request):
    """流式：presence_penalty为对象类型，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "presence_penalty": {"value": 1.0},
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)