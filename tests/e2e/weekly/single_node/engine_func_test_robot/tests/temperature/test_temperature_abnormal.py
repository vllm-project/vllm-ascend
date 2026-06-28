import pytest
from ...utility import request_helper as helper
from ...utility import assertion


def test_temperature_negative_non_stream(api_client, request):
    """非流式：temperature为负数，应该返回错误码400"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "temperature": -0.5,
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_temperature_negative_stream(api_client, request):
    """流式：temperature为负数，应该返回错误码400"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "temperature": -0.5,
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_temperature_exceed_upper_limit_non_stream(api_client, request):
    """非流式：temperature取很大的值（如100.0），作为边界测试，应正常响应
    
    说明：temperature没有上限，取很大的值仍可正常响应
    """
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "temperature": 100.0,  # 很大的值，作为边界测试
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


def test_temperature_exceed_upper_limit_stream(api_client, request):
    """流式：temperature取很大的值（如100.0），作为边界测试，应正常响应
    
    说明：temperature没有上限，取很大的值仍可正常响应
    """
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "temperature": 100.0,  # 很大的值，作为边界测试
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


def test_temperature_string_non_stream(api_client, request):
    """非流式：temperature为字符串整型，应该正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "temperature": "1",
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


def test_temperature_string_stream(api_client, request):
    """流式：temperature为字符串数值，应该正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "temperature": "0.7",
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


def test_temperature_null_non_stream(api_client, request):
    """非流式：temperature为null，应该正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "temperature": None,
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


def test_temperature_null_stream(api_client, request):
    """流式：temperature为null，应该正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "temperature": None,
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


def test_temperature_array_non_stream(api_client, request):
    """非流式：temperature为数组类型，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "temperature": [0.5, 0.7],
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_temperature_array_stream(api_client, request):
    """流式：temperature为数组类型，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "temperature": [0.5, 0.7],
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_temperature_object_non_stream(api_client, request):
    """非流式：temperature为对象类型，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "temperature": {"value": 0.7},
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_temperature_object_stream(api_client, request):
    """流式：temperature为对象类型，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "temperature": {"value": 0.7},
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)
