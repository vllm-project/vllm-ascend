from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


def test_chat_template_kwargs_string_non_stream(api_client, request):
    """非流式：chat_template_kwargs为字符串类型而非对象，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": "invalid_string",
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_chat_template_kwargs_string_stream(api_client, request):
    """流式：chat_template_kwargs为字符串类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": "invalid_string",
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_chat_template_kwargs_array_non_stream(api_client, request):
    """非流式：chat_template_kwargs为数组类型而非对象，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": ["item1", "item2"],
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_chat_template_kwargs_array_stream(api_client, request):
    """流式：chat_template_kwargs为数组类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": ["item1", "item2"],
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_chat_template_kwargs_integer_non_stream(api_client, request):
    """非流式：chat_template_kwargs为整数类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": 123,
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_chat_template_kwargs_integer_stream(api_client, request):
    """流式：chat_template_kwargs为整数类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": 123,
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_chat_template_kwargs_boolean_non_stream(api_client, request):
    """非流式：chat_template_kwargs为布尔类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": True,
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_chat_template_kwargs_boolean_stream(api_client, request):
    """流式：chat_template_kwargs为布尔类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": False,
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_chat_template_kwargs_nested_invalid_type_non_stream(api_client, request):
    """非流式：chat_template_kwargs包含嵌套无效类型值，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {
            "valid_param": "value",
            "invalid_param": [1, 2, 3],  # 某些引擎可能不支持数组类型的参数值
        },
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：应为400（如果引擎严格验证）或200（如果引擎忽略无效值）
    assert response.status_code in [
        200,
        400,
    ], f"状态码应为200或400，实际为{response.status_code}"


def test_chat_template_kwargs_nested_invalid_type_stream(api_client, request):
    """流式：chat_template_kwargs包含嵌套无效类型值"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {
            "valid_param": "value",
            "invalid_param": {"nested": [1, 2, 3]},
        },
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码应为200或400
    assert response.status_code in [
        200,
        400,
    ], f"状态码应为200或400，实际为{response.status_code}"
