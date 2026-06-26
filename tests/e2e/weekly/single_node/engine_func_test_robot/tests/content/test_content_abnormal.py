from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


def test_content_integer_non_stream(api_client, request):
    """非流式：content为整数类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": 12345}],
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_integer_stream(api_client, request):
    """流式：content为整数类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": 12345}],
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_object_non_stream(api_client, request):
    """非流式：content为对象类型（非标准多模态格式），应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": {"text": "hello", "extra": "data"}}],
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_object_stream(api_client, request):
    """流式：content为对象类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": {"text": "hello", "extra": "data"}}],
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_boolean_non_stream(api_client, request):
    """非流式：content为布尔类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": True}],
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_boolean_stream(api_client, request):
    """流式：content为布尔类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": False}],
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_array_invalid_format_non_stream(api_client, request):
    """非流式：content为数组但格式不符合OpenAI多模态规范（字符串数组），应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": ["invalid", "array", "format"]}],
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_array_invalid_format_stream(api_client, request):
    """流式：content为数组但格式不符合OpenAI多模态规范，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": ["invalid", "array", "format"]}],
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


# ==================== Content Array Abnormal Tests ====================


def test_content_array_missing_type_non_stream(api_client, request):
    """非流式：content数组对象缺少type字段，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"text": "你好"}]}],  # 缺少type字段
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_array_missing_type_stream(api_client, request):
    """流式：content数组对象缺少type字段，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"text": "你好"}]}],
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_array_missing_text_non_stream(api_client, request):
    """非流式：content数组对象type为text但缺少text字段，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "text"}]}],  # 缺少text字段
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_array_invalid_type_non_stream(api_client, request):
    """非流式：content数组对象type为无效值，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "invalid_type", "text": "你好"}]}],
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_array_invalid_type_stream(api_client, request):
    """流式：content数组对象type为无效值，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "unknown", "text": "你好"}]}],
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_array_text_field_null_non_stream(api_client, request):
    """非流式：content数组对象text字段为null，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "text", "text": None}]}],
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_array_text_field_integer_non_stream(api_client, request):
    """非流式：content数组对象text字段为整数类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "text", "text": 12345}]}],
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_content_array_text_field_integer_stream(api_client, request):
    """流式：content数组对象text字段为整数类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "text", "text": 12345}]}],
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)
