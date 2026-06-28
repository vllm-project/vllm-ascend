import pytest
from ...utility import request_helper as helper
from ...utility import assertion


def test_top_k_zero_non_stream(api_client, request):
    """非流式：top_k=0为有效值，应该正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天天气怎么样"
        }],
        "top_k": 0,
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码200
    assertion.assert_status_code_200(response)

    # 校验点：finish_reason为stop或length
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_top_k_zero_stream(api_client, request):
    """流式：top_k=0为有效值，应该正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天天气怎么样"
        }],
        "top_k": 0,
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码200
    assertion.assert_status_code_200(response)

    # 校验点：流式响应包含[DONE]
    assertion.assert_stream_has_done(response.text)

    # 校验点：finish_reason为stop或length
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    assertion.assert_finish_reason_valid(finish_reason)


def test_top_k_negative_not_minus_one_non_stream(api_client, request):
    """非流式：top_k为小于-1的负数，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天天气怎么样"
        }],
        "top_k": -5,
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_top_k_negative_not_minus_one_stream(api_client, request):
    """流式：top_k为小于-1的负数，pd架构状态码200+错误码400，single架构状态码400"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天天气怎么样"
        }],
        "top_k": -5,
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：根据引擎架构判断状态码，错误码均为400
    engine_arch = request.config.getoption("--engineArchitecture")
    if engine_arch == "pd":
        assertion.assert_status_code_200(response)
    else:  # single
        assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_top_k_exceed_vocab_size_non_stream(api_client, request):
    """非流式：top_k超过词表大小（如999999999），请求可以接受但会被钳制到词表大小"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天天气怎么样"
        }],
        "top_k": 999999999,
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码200（超大值通常会被截断到vocab_size）
    assertion.assert_status_code_200(response)

    # 验证finish_reason有效
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_top_k_exceed_vocab_size_stream(api_client, request):
    """流式：top_k超过词表大小（如999999999），请求可以接受但会被钳制到词表大小"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天天气怎么样"
        }],
        "top_k": 999999999,
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码200，流式响应包含[DONE]
    assertion.assert_status_code_200(response)
    assertion.assert_stream_has_done(response.text)

    # 验证finish_reason有效
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    assertion.assert_finish_reason_valid(finish_reason)


def test_top_k_float_non_stream(api_client, request):
    """非流式：top_k为浮点数类型，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天天气怎么样"
        }],
        "top_k": 10.5,
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_top_k_float_stream(api_client, request):
    """流式：top_k为浮点数类型，pd架构状态码200+错误码400，single架构状态码400"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天天气怎么样"
        }],
        "top_k": 10.5,
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：根据引擎架构判断状态码，错误码均为400
    engine_arch = request.config.getoption("--engineArchitecture")
    if engine_arch == "pd":
        assertion.assert_status_code_200(response)
    else:  # single
        assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_top_k_string_non_stream(api_client, request):
    """非流式：top_k为字符串数值，应该正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天天气怎么样"
        }],
        "top_k": "50",
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


def test_top_k_string_stream(api_client, request):
    """流式：top_k为字符串数值，应该正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天天气怎么样"
        }],
        "top_k": "50",
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


def test_top_k_null_non_stream(api_client, request):
    """非流式：top_k为null，应该正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天天气怎么样"
        }],
        "top_k": None,
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


def test_top_k_null_stream(api_client, request):
    """流式：top_k为null，应该正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天天气怎么样"
        }],
        "top_k": None,
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