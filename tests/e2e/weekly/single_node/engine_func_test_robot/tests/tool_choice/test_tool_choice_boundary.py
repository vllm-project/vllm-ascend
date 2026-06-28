import pytest
from ...utility import request_helper as helper
from ...utility import assertion


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_empty_tools_list(api_client, request, stream):
    """tools为空列表时tool_choice为'auto'，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": [],
        "tool_choice": "auto",
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200（如果引擎忽略空tools）或400（如果要求至少一个工具）
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_case_sensitivity_none(api_client, request, stream):
    """tool_choice大小写敏感性测试：'None' vs 'none'"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "test_func",
                    "description": "测试函数",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ],
        "tool_choice": "None",  # 首字母大写，标准应为小写
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200（如果不区分大小写）或400（如果严格区分）
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_case_sensitivity_auto(api_client, request, stream):
    """tool_choice大小写敏感性测试：'AUTO' vs 'auto'"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "test_func",
                    "description": "测试函数",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ],
        "tool_choice": "AUTO",  # 全大写，标准应为小写
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200或400
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_unicode_function_name(api_client, request, stream):
    """tool_choice指定Unicode函数名，边界测试"""
    tools_with_unicode_name = [
        {
            "type": "function",
            "function": {
                "name": "获取天气",
                "description": "获取天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    ]

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": tools_with_unicode_name,
        "tool_choice": "获取天气",
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200或400
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_special_chars_function_name(api_client, request, stream):
    """tool_choice包含特殊字符函数名，边界测试"""
    tools_with_special_name = [
        {
            "type": "function",
            "function": {
                "name": "function-name_with.special",
                "description": "测试特殊字符函数名",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": tools_with_special_name,
        "tool_choice": "function-name_with.special",
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200或400
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_very_long_function_name(api_client, request, stream):
    """tool_choice指定超长函数名，边界测试"""
    long_name = "a" * 200  # 超长函数名
    tools_with_long_name = [
        {
            "type": "function",
            "function": {
                "name": long_name,
                "description": "测试超长函数名",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": tools_with_long_name,
        "tool_choice": long_name,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200或400
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_many_tools(api_client, request, stream):
    """tool_choice配合大量tools，边界测试"""
    many_tools = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": f"工具{i}的描述",
                "parameters": {"type": "object", "properties": {}}
            }
        }
        for i in range(50)  # 50个工具
    ]

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "调用tool_25"
        }],
        "tools": many_tools,
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
def test_tool_choice_type_field_case_variations(api_client, request, stream):
    """tool_choice对象的type字段大小写变体，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "test_func",
                    "description": "测试函数",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ],
        "tool_choice": {
            "type": "FUNCTION",  # 全大写
            "function": {
                "name": "test_func"
            }
        },
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200或400
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_additional_properties(api_client, request, stream):
    """tool_choice对象包含额外属性，边界测试（引擎可能忽略或报错）"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "test_func",
                    "description": "测试函数",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {
                "name": "test_func",
                "extra_field": "extra_value"  # 额外字段
            }
        },
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200（如果引擎忽略额外字段）或400（如果严格校验）
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_whitespace_in_string(api_client, request, stream):
    """tool_choice字符串值包含前后空格，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "test_func",
                    "description": "测试函数",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ],
        "tool_choice": " auto ",  # 带前后空格
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200（如果引擎trim）或400（如果严格匹配）
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_function_name_partial_match(api_client, request, stream):
    """tool_choice函数名部分匹配但不完全相同，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather_info",
                    "description": "获取天气信息",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ],
        "tool_choice": "get_weather",  # 部分匹配实际函数名
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为400（部分匹配不应成功）或200（如果引擎有特殊处理）
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_duplicate_function_names(api_client, request, stream):
    """tools列表中存在同名函数（应被过滤或报错），边界测试"""
    tools_with_duplicates = [
        {
            "type": "function",
            "function": {
                "name": "same_name",
                "description": "第一个同名函数",
                "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "same_name",  # 重复名称
                "description": "第二个同名函数",
                "parameters": {"type": "object", "properties": {"b": {"type": "number"}}}
            }
        }
    ]

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "调用same_name"
        }],
        "tools": tools_with_duplicates,
        "tool_choice": "same_name",
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200（如果引擎去重）或400（如果拒绝重复名称）
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_tool_choice_omit_tools_with_required(api_client, request, stream):
    """tool_choice为'required'但tools列表为空，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": [],
        "tool_choice": "required",
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为400（required必须配合tools）或200（如果引擎降级处理）
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"
