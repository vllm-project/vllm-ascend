import pytest
from ...utility import request_helper as helper
from ...utility import assertion


def test_stop_integer_non_stream(api_client, request):
    """非流式：stop为整数类型，正常响应或错误码400都算通过"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "stop": 123,
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：正常响应（finish_reason为stop或length）或错误码400
    if response.status_code == 200:
        finish_reason = response.json()["choices"][0]["finish_reason"]
        assertion.assert_finish_reason_valid(finish_reason)
    else:
        assertion.assert_error_code_400(response)


def test_stop_integer_stream(api_client, request):
    """流式：stop为整数类型，正常响应或错误码400都算通过"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "stop": 123,
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：正常响应（finish_reason为stop或length）或错误码400
    if response.status_code == 200:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
        assertion.assert_finish_reason_valid(finish_reason)
    else:
        assertion.assert_error_code_400(response)


def test_stop_null_non_stream(api_client, request):
    """非流式：stop为null，应该正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "stop": None,
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


def test_stop_null_stream(api_client, request):
    """流式：stop为null，应该正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "stop": None,
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


def test_stop_nested_array_non_stream(api_client, request):
    """非流式：stop为嵌套数组，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "stop": [["a", "b"]],
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_stop_nested_array_stream(api_client, request):
    """流式：stop为嵌套数组，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "stop": [["a", "b"]],
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_stop_object_non_stream(api_client, request):
    """非流式：stop为对象类型，正常响应或错误码400都算通过"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "stop": {"key": "value"},
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：正常响应（finish_reason为stop或length）或错误码400
    if response.status_code == 200:
        finish_reason = response.json()["choices"][0]["finish_reason"]
        assertion.assert_finish_reason_valid(finish_reason)
    else:
        assertion.assert_error_code_400(response)


def test_stop_object_stream(api_client, request):
    """流式：stop为对象类型，正常响应或错误码400都算通过"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "stop": {"key": "value"},
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：正常响应（finish_reason为stop或length）或错误码400
    if response.status_code == 200:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
        assertion.assert_finish_reason_valid(finish_reason)
    else:
        assertion.assert_error_code_400(response)


def test_stop_exceed_max_limit_non_stream(api_client, request):
    """非流式：stop序列数量超过最大限制，应该正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "stop": ["a", "b", "c", "d", "e", "f"],
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码200，finish_reason为stop或length
    assertion.assert_status_code_200(response)
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_stop_exceed_max_limit_stream(api_client, request):
    """流式：stop序列数量超过最大限制，应该正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "stop": ["a", "b", "c", "d", "e", "f"],
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码200，finish_reason为stop或length
    assertion.assert_status_code_200(response)
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    assertion.assert_finish_reason_valid(finish_reason)


def test_stop_sequence_too_long_non_stream(api_client, request):
    """非流式：单个stop序列超过最大长度（通常32字符），应该正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "stop": ["a" * 100],  # 超长序列
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


def test_stop_sequence_too_long_stream(api_client, request):
    """流式：单个stop序列超过最大长度，应该正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "stop": ["a" * 100],
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