import pytest
from ...utility import request_helper as helper
from ...utility import assertion


def test_max_tokens_zero_non_stream(api_client, request):
    """非流式：max_tokens=0为无效值，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_tokens": 0,
        "stream": False
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_max_tokens_zero_stream(api_client, request):
    """流式：max_tokens=0为无效值，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_tokens": 0,
        "stream": True
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_max_tokens_negative_non_stream(api_client, request):
    """非流式：max_tokens为负数，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_tokens": -10,
        "stream": False
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_max_tokens_negative_stream(api_client, request):
    """流式：max_tokens为负数，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_tokens": -10,
        "stream": True
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_max_tokens_float_non_stream(api_client, request):
    """非流式：max_tokens为浮点数，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_tokens": 50.5,
        "stream": False
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_max_tokens_float_stream(api_client, request):
    """流式：max_tokens为浮点数，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_tokens": 50.5,
        "stream": True
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_max_tokens_string_non_stream(api_client, request):
    """非流式：max_tokens为字符串，可能返回400错误或正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_tokens": "100",
        "stream": False
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验：error code为400错误，或者finish_reason为stop/length都算pass
    if assertion.has_error_code(response):
        # 存在error code，校验为400
        assertion.assert_error_code_400(response)
    else:
        # 没有error code，检查finish_reason为stop或length
        finish_reason = response.json()["choices"][0]["finish_reason"]
        assertion.assert_finish_reason_valid(finish_reason)


def test_max_tokens_string_stream(api_client, request):
    """流式：max_tokens为字符串，可能返回400错误或正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_tokens": "100",
        "stream": True
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验：error code为400错误，或者finish_reason为stop/length都算pass
    if assertion.has_error_code(response):
        # 存在error code，校验为400
        assertion.assert_error_code_400(response)
    else:
        # 没有error code，检查finish_reason为stop或length
        assertion.assert_stream_has_done(response.text)
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
        assertion.assert_finish_reason_valid(finish_reason)


def test_max_tokens_boolean_non_stream(api_client, request):
    """非流式：max_tokens为布尔值，可能返回400错误或正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_tokens": True,
        "stream": False
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验：error code为400错误，或者finish_reason为stop/length都算pass
    if assertion.has_error_code(response):
        # 存在error code，校验为400
        assertion.assert_error_code_400(response)
    else:
        # 没有error code，检查finish_reason为stop或length
        finish_reason = response.json()["choices"][0]["finish_reason"]
        assertion.assert_finish_reason_valid(finish_reason)


def test_max_tokens_boolean_stream(api_client, request):
    """流式：max_tokens为布尔值，可能返回400错误或正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_tokens": False,
        "stream": True
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验：error code为400错误，或者finish_reason为stop/length都算pass
    if assertion.has_error_code(response):
        # 存在error code，校验为400
        assertion.assert_error_code_400(response)
    else:
        # 没有error code，检查finish_reason为stop或length
        assertion.assert_stream_has_done(response.text)
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
        assertion.assert_finish_reason_valid(finish_reason)



