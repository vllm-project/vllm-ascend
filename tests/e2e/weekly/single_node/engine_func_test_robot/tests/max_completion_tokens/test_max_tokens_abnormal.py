import pytest
from ...utility import request_helper as helper
from ...utility import assertion


def test_max_tokens_zero_non_stream(api_client, request):
    """非流式：max_completion_tokens=0，可能返回400错误或content为空字符串"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_completion_tokens": 0,
        "stream": False
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验：error code为400错误，或者content为空字符串且finish_reason为stop/length都算pass
    if assertion.has_error_code(response):
        # 存在error code，校验为400
        assertion.assert_error_code_400(response)
    else:
        # 没有error code，检查content为空字符串和finish_reason
        content = response.json()["choices"][0]["message"]["content"]
        assert content == "", f"期望content为空字符串，实际为: {content}"
        finish_reason = response.json()["choices"][0]["finish_reason"]
        assertion.assert_finish_reason_valid(finish_reason)


def test_max_tokens_zero_stream(api_client, request):
    """流式：max_completion_tokens=0为无效值，http status code:200, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_completion_tokens": 0,
        "stream": True
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_max_tokens_negative_non_stream(api_client, request):
    """非流式：max_completion_tokens为负数，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_completion_tokens": -10,
        "stream": False
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_max_tokens_negative_stream(api_client, request):
    """流式：max_completion_tokens为负数，http status code:200, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_completion_tokens": -10,
        "stream": True
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_max_tokens_float_non_stream(api_client, request):
    """非流式：max_completion_tokens为浮点数，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_completion_tokens": 50.5,
        "stream": False
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_max_tokens_float_stream(api_client, request):
    """流式：max_completion_tokens为浮点数，http status code:200, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_completion_tokens": 50.5,
        "stream": True
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_max_tokens_string_non_stream(api_client, request):
    """非流式：max_completion_tokens为字符串数值，可能返回400错误或正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_completion_tokens": "100",
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
    """流式：max_completion_tokens为字符串数值，可能返回400错误或正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_completion_tokens": "100",
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


def test_max_tokens_null_non_stream(api_client, request):
    """非流式：max_completion_tokens为null，可以正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_completion_tokens": None,
        "stream": False
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：finish_reason为stop或length
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_max_tokens_null_stream(api_client, request):
    """流式：max_completion_tokens为null，可以正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_completion_tokens": None,
        "stream": True
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


def test_max_tokens_exceed_model_limit_stream(api_client, request):
    """流式：max_completion_tokens超过模型限制，http status code:200, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_completion_tokens": 100000,
        "stream": True
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)
