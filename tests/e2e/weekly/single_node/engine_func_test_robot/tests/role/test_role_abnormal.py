import pytest
from ...utility import request_helper as helper
from ...utility import assertion


def test_role_invalid_value_non_stream(api_client, request):
    """非流式：role值为无效角色（如'unknown'），响应状态码可以是400或200，错误码可以是400或者正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "unknown",
            "content": "你好"
        }],
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验：状态码400+错误码400 或 状态码200+错误码400 或 状态码200+正常响应
    if response.status_code == 400:
        assertion.assert_error_code_400(response)
    else:
        assertion.assert_status_code_200(response)
        response_json = response.json()
        error_code = response_json.get("error", {}).get("code") or response_json.get("code")
        if error_code == 400:
            assertion.assert_error_code_400(response)
        else:
            finish_reason = response_json["choices"][0]["finish_reason"]
            assertion.assert_finish_reason_valid(finish_reason)


def test_role_invalid_value_stream(api_client, request):
    """流式：role值为无效角色，响应状态码可以是400或200，错误码可以是400或者正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "unknown",
            "content": "你好"
        }],
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验：状态码400+错误码400 或 状态码200+错误码400 或 状态码200+正常响应
    if response.status_code == 400:
        assertion.assert_error_code_400(response)
    else:
        assertion.assert_status_code_200(response)
        import re
        match = re.search(r'\"code\"\s?:\s?(\d+)', response.text, re.M)
        if match and int(match.group(1)) == 400:
            assertion.assert_error_code_400(response)
        else:
            finish_reason = assertion.assert_stream_single_finish_reason(response.text)
            assertion.assert_finish_reason_valid(finish_reason)


def test_role_missing_non_stream(api_client, request):
    """非流式：message对象缺少role字段，可以是错误码400或者正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            # 缺少role字段
            "content": "你好"
        }],
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验：错误码400 或 正常响应
    response_json = response.json()
    error_code = response_json.get("error", {}).get("code") or response_json.get("code")
    if error_code == 400:
        assertion.assert_error_code_400(response)
    else:
        finish_reason = response_json["choices"][0]["finish_reason"]
        assertion.assert_finish_reason_valid(finish_reason)


def test_role_missing_stream(api_client, request):
    """流式：message对象缺少role字段，http status code:200, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            # 缺少role字段
            "content": "你好"
        }],
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_role_null_non_stream(api_client, request):
    """非流式：role为null，可以是错误码400或者正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": None,
            "content": "你好"
        }],
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验：错误码400 或 正常响应
    response_json = response.json()
    error_code = response_json.get("error", {}).get("code") or response_json.get("code")
    if error_code == 400:
        assertion.assert_error_code_400(response)
    else:
        finish_reason = response_json["choices"][0]["finish_reason"]
        assertion.assert_finish_reason_valid(finish_reason)


def test_role_null_stream(api_client, request):
    """流式：role为null，http status code:200, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": None,
            "content": "你好"
        }],
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_role_integer_non_stream(api_client, request):
    """非流式：role为整数类型，响应状态码可以是400或200，错误码可以是400或者正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": 123,
            "content": "你好"
        }],
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验：状态码400+错误码400 或 状态码200+错误码400 或 状态码200+正常响应
    if response.status_code == 400:
        assertion.assert_error_code_400(response)
    else:
        assertion.assert_status_code_200(response)
        response_json = response.json()
        error_code = response_json.get("error", {}).get("code") or response_json.get("code")
        if error_code == 400:
            assertion.assert_error_code_400(response)
        else:
            finish_reason = response_json["choices"][0]["finish_reason"]
            assertion.assert_finish_reason_valid(finish_reason)


def test_role_integer_stream(api_client, request):
    """流式：role为整数类型，响应状态码可以是400或200，错误码可以是400或者正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": 123,
            "content": "你好"
        }],
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验：状态码400+错误码400 或 状态码200+错误码400 或 状态码200+正常响应
    if response.status_code == 400:
        assertion.assert_error_code_400(response)
    else:
        assertion.assert_status_code_200(response)
        import re
        match = re.search(r'\"code\"\s?:\s?(\d+)', response.text, re.M)
        if match and int(match.group(1)) == 400:
            assertion.assert_error_code_400(response)
        else:
            finish_reason = assertion.assert_stream_single_finish_reason(response.text)
            assertion.assert_finish_reason_valid(finish_reason)


def test_role_empty_string_non_stream(api_client, request):
    """非流式：role为空字符串，响应状态码可以是400或200，错误码可以是400或者正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "",
            "content": "你好"
        }],
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验：状态码400+错误码400 或 状态码200+错误码400 或 状态码200+正常响应
    if response.status_code == 400:
        assertion.assert_error_code_400(response)
    else:
        assertion.assert_status_code_200(response)
        response_json = response.json()
        error_code = response_json.get("error", {}).get("code") or response_json.get("code")
        if error_code == 400:
            assertion.assert_error_code_400(response)
        else:
            finish_reason = response_json["choices"][0]["finish_reason"]
            assertion.assert_finish_reason_valid(finish_reason)


def test_role_empty_string_stream(api_client, request):
    """流式：role为空字符串，响应状态码可以是400或200，错误码可以是400或者正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "",
            "content": "你好"
        }],
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验：状态码400+错误码400 或 状态码200+错误码400 或 状态码200+正常响应
    if response.status_code == 400:
        assertion.assert_error_code_400(response)
    else:
        assertion.assert_status_code_200(response)
        import re
        match = re.search(r'\"code\"\s?:\s?(\d+)', response.text, re.M)
        if match and int(match.group(1)) == 400:
            assertion.assert_error_code_400(response)
        else:
            finish_reason = assertion.assert_stream_single_finish_reason(response.text)
            assertion.assert_finish_reason_valid(finish_reason)


def test_role_case_sensitive_user_non_stream(api_client, request):
    """非流式：role大小写敏感，'User'不等于'user'，响应状态码可以是400或200，错误码可以是400或者正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "User",  # 首字母大写
            "content": "你好"
        }],
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验：状态码400+错误码400 或 状态码200+错误码400 或 状态码200+正常响应
    if response.status_code == 400:
        assertion.assert_error_code_400(response)
    else:
        assertion.assert_status_code_200(response)
        response_json = response.json()
        error_code = response_json.get("error", {}).get("code") or response_json.get("code")
        if error_code == 400:
            assertion.assert_error_code_400(response)
        else:
            finish_reason = response_json["choices"][0]["finish_reason"]
            assertion.assert_finish_reason_valid(finish_reason)


def test_role_case_sensitive_user_stream(api_client, request):
    """流式：role大小写敏感，响应状态码可以是400或200，错误码可以是400或者正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "User",  # 首字母大写
            "content": "你好"
        }],
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验：状态码400+错误码400 或 状态码200+错误码400 或 状态码200+正常响应
    if response.status_code == 400:
        assertion.assert_error_code_400(response)
    else:
        assertion.assert_status_code_200(response)
        import re
        match = re.search(r'\"code\"\s?:\s?(\d+)', response.text, re.M)
        if match and int(match.group(1)) == 400:
            assertion.assert_error_code_400(response)
        else:
            finish_reason = assertion.assert_stream_single_finish_reason(response.text)
            assertion.assert_finish_reason_valid(finish_reason)


def test_role_case_sensitive_system_non_stream(api_client, request):
    """非流式：role大小写敏感，'SYSTEM'不等于'system'，响应状态码可以是400或200，错误码可以是400或者正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "SYSTEM",
            "content": "你是一个助手"
        }],
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验：状态码400+错误码400 或 状态码200+错误码400 或 状态码200+正常响应
    if response.status_code == 400:
        assertion.assert_error_code_400(response)
    else:
        assertion.assert_status_code_200(response)
        response_json = response.json()
        error_code = response_json.get("error", {}).get("code") or response_json.get("code")
        if error_code == 400:
            assertion.assert_error_code_400(response)
        else:
            finish_reason = response_json["choices"][0]["finish_reason"]
            assertion.assert_finish_reason_valid(finish_reason)


def test_role_case_sensitive_system_stream(api_client, request):
    """流式：role大小写敏感，响应状态码可以是400或200，错误码可以是400或者正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "SYSTEM",
            "content": "你是一个助手"
        }],
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验：状态码400+错误码400 或 状态码200+错误码400 或 状态码200+正常响应
    if response.status_code == 400:
        assertion.assert_error_code_400(response)
    else:
        assertion.assert_status_code_200(response)
        import re
        match = re.search(r'\"code\"\s?:\s?(\d+)', response.text, re.M)
        if match and int(match.group(1)) == 400:
            assertion.assert_error_code_400(response)
        else:
            finish_reason = assertion.assert_stream_single_finish_reason(response.text)
            assertion.assert_finish_reason_valid(finish_reason)
