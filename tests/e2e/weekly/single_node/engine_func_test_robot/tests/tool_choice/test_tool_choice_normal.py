import pytest
import json
from ...utility import request_helper as helper
from ...utility import assertion


# 通用工具定义，用于测试
TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    },
                    "date": {
                        "type": "string",
                        "description": "日期，格式YYYY-MM-DD"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "获取当前时间",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_none(api_client, request, stream):
    """tool_choice为'none'，模型不应调用任何工具"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "北京今天天气怎么样？"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "none",
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_auto(api_client, request, stream):
    """tool_choice为'auto'，模型应自动决定是否调用工具"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "北京今天天气怎么样？"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "auto",
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_required(api_client, request, stream):
    """tool_choice为'required'，模型必须调用工具"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "查一下时间"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "required",
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_specific_function_string(api_client, request, stream):
    """tool_choice为特定工具名称字符串（如'get_weather'），错误码400
    
    说明：Only named tools, "none", "auto" or "required" are supported.
    这里的 "named tools" 指的是对象格式的命名工具选择，即：{"type": "function", "function": {"name": "工具名"}}
    字符串格式不支持，应返回错误码400。
    """
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "北京今天天气怎么样？"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "get_weather",
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_function_object(api_client, request, stream):
    """tool_choice为function对象格式{'type': 'function', 'function': {'name': 'xxx'}}"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "北京今天天气怎么样？"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": {
            "type": "function",
            "function": {
                "name": "get_weather"
            }
        },
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_without_tools(api_client, request, stream):
    """tool_choice存在但tools未提供，应正常响应或返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tool_choice": "auto",
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200（如果引擎忽略tool_choice）或400（如果要求必须配合tools）
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_with_tools_non_function_call_query(api_client, request, stream):
    """tool_choice为'auto'但用户查询不涉及工具调用场景"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请介绍一下你自己"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "auto",
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_tool_with_null_description(api_client, request, stream):
    """tools中包含description为null的工具，tool_choice为'auto'"""
    tools_with_null_desc = [
        {
            "type": "function",
            "function": {
                "name": "simple_tool",
                "description": None,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"}
                    },
                    "required": ["input"]
                }
            }
        }
    ]

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "调用简单工具"
        }],
        "tools": tools_with_null_desc,
        "tool_choice": "auto",
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200或400
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_parallel_tools(api_client, request, stream):
    """tool_choice配合多个工具，测试并行工具调用场景（如果引擎支持）"""
    multiple_tools = [
        {
            "type": "function",
            "function": {
                "name": "tool_a",
                "description": "工具A",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tool_b",
                "description": "工具B",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "同时调用多个工具"
        }],
        "tools": multiple_tools,
        "tool_choice": "required",
        "stream": stream,
        "max_tokens": 5120
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)
