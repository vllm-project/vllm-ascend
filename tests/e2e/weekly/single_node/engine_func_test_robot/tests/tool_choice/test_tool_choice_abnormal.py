import pytest
from ...utility import request_helper as helper
from ...utility import assertion


TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }
]


def test_tool_choice_invalid_string_non_stream(api_client, request):
    """非流式：tool_choice为无效字符串（非'none'/'auto'/'required'），应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "invalid_choice",
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_tool_choice_invalid_string_stream(api_client, request):
    """流式：tool_choice为无效字符串，http status code:200, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "unknown_option",
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_tool_choice_nonexistent_function_name_non_stream(api_client, request):
    """非流式：tool_choice指定的函数名不存在于tools列表中，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "nonexistent_function",
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_tool_choice_nonexistent_function_name_stream(api_client, request):
    """流式：tool_choice指定的函数名不存在于tools列表中，http status code:200, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "nonexistent_function",
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_tool_choice_null_non_stream(api_client, request):
    """非流式：tool_choice为null，可以正常响应
    
    说明：null在提供了 tools 时等同于 "auto"
    """
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": None,
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：finish_reason为stop或length
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_tool_choice_null_stream(api_client, request):
    """流式：tool_choice为null，请求正常响应
    
    说明：null在提供了 tools 时等同于 "auto"
    """
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": None,
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：finish_reason为stop或length
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    assertion.assert_finish_reason_valid(finish_reason)


def test_tool_choice_integer_non_stream(api_client, request):
    """非流式：tool_choice为整数类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": 123,
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_tool_choice_integer_stream(api_client, request):
    """流式：tool_choice为整数类型，http status code:200, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": 0,
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_tool_choice_empty_object_non_stream(api_client, request):
    """非流式：tool_choice为空对象{}，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": {},
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_tool_choice_empty_object_stream(api_client, request):
    """流式：tool_choice为空对象{}，http status code:200, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": {},
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_tool_choice_malformed_function_object_non_stream(api_client, request):
    """非流式：tool_choice为function对象但缺少必要字段，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": {
            "type": "function"
            # 缺少function.name字段
        },
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_tool_choice_malformed_function_object_stream(api_client, request):
    """流式：tool_choice为function对象但缺少必要字段，http status code:200, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": {
            "type": "function",
            "function": {}  # 空的function对象，缺少name
        },
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_tool_choice_wrong_type_value_non_stream(api_client, request):
    """非流式：tool_choice的type字段值不正确（非'function'），应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": {
            "type": "invalid_type",
            "function": {"name": "get_weather"}
        },
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_tool_choice_wrong_type_value_stream(api_client, request):
    """流式：tool_choice的type字段值不正确，http status code:200, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": {
            "type": "tool",
            "function": {"name": "get_weather"}
        },
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_tool_choice_array_type_non_stream(api_client, request):
    """非流式：tool_choice为数组类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": ["get_weather"],
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_tool_choice_array_type_stream(api_client, request):
    """流式：tool_choice为数组类型，http status code:200, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": ["auto"],
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_tool_choice_boolean_non_stream(api_client, request):
    """非流式：tool_choice为布尔类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": True,
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_tool_choice_boolean_stream(api_client, request):
    """流式：tool_choice为布尔类型，http status code:200, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": False,
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)
