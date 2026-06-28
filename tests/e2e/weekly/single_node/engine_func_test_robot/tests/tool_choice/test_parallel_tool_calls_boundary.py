import pytest
import json
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


def _validate_tool_calls_count(response, stream, max_count):
    """校验tool_calls数量的函数
    
    Args:
        response: HTTP响应对象
        stream: 是否流式响应
        max_count: tool_calls的最大数量
    """
    if stream:
        # 流式响应校验
        assertion.assert_stream_has_done(response.text)
        
        tool_call_indices = set()
        finish_reason_tool_calls = False
        for line in response.text.strip().split("\n"):
            if line.startswith("data: ") and "[DONE]" not in line:
                chunk = json.loads(line[6:])
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("tool_calls"):
                        for tc in delta.get("tool_calls", []):
                            if "index" in tc:
                                tool_call_indices.add(tc.get("index"))
                    if choices[0].get("finish_reason") == "tool_calls":
                        finish_reason_tool_calls = True
        
        # 校验 tool_calls 数量
        if tool_call_indices:
            assert len(tool_call_indices) <= max_count,                 f"流式响应tool_calls数量应<={max_count}，实际为{len(tool_call_indices)}"
            assert finish_reason_tool_calls, "流式响应有tool_calls时finish_reason应为tool_calls"
    else:
        # 非流式响应校验
        response_json = response.json()
        choices = response_json.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            if message.get("tool_calls") or choice.get("finish_reason") == "tool_calls":
                tool_calls = message.get("tool_calls", [])
                assert len(tool_calls) <= max_count,                     f"非流式响应tool_calls数量应<={max_count}，实际为{len(tool_calls)}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_empty_tools_list(api_client, request, stream):
    """parallel_tool_calls=True但tools为空列表，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": [],
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200（如果引擎忽略）或400（如果要求tools不为空）
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_false_with_empty_tools_list(api_client, request, stream):
    """parallel_tool_calls=False但tools为空列表，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": [],
        "parallel_tool_calls": False,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200或400
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_many_tools(api_client, request, stream):
    """parallel_tool_calls=True配合大量工具，边界测试"""
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
            "content": "调用多个工具"
        }],
        "tools": many_tools,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200或400
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_tool_choice_required_and_false(api_client, request, stream):
    """parallel_tool_calls=False配合tool_choice='required'，边界测试
    
    说明：required要求必须调用工具，但parallel_tool_calls=False限制只能调用一个
    两者组合应该正常工作，只返回一个工具调用
    """
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "查一下北京和上海的天气"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码200
    assertion.assert_status_code_200(response)

    # 校验点：tool_calls数量最多只有1个（流式和非流式都校验）
    _validate_tool_calls_count(response, stream, max_count=1)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_duplicate_tool_names(api_client, request, stream):
    """parallel_tool_calls=True但tools中有重复函数名，边界测试"""
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
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200（如果引擎去重）或400（如果拒绝重复名称）
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_tool_having_null_description(api_client, request, stream):
    """parallel_tool_calls=True配合description为null的工具，边界测试"""
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
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200或400
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_tool_having_empty_parameters(api_client, request, stream):
    """parallel_tool_calls=True配合空参数的工具，边界测试"""
    tools_with_empty_params = [
        {
            "type": "function",
            "function": {
                "name": "no_param_tool",
                "description": "无参数工具",
                "parameters": {}  # 空参数定义
            }
        }
    ]

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "调用无参数工具"
        }],
        "tools": tools_with_empty_params,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200或400
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_unicode_tool_name(api_client, request, stream):
    """parallel_tool_calls=True配合Unicode函数名，边界测试"""
    tools_with_unicode = [
        {
            "type": "function",
            "function": {
                "name": "获取天气信息",
                "description": "获取天气信息",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "获取时间",
                "description": "获取当前时间",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "查一下天气和时间"
        }],
        "tools": tools_with_unicode,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200或400
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_special_chars_tool_name(api_client, request, stream):
    """parallel_tool_calls=True配合特殊字符函数名，边界测试"""
    tools_with_special_chars = [
        {
            "type": "function",
            "function": {
                "name": "tool-name_with.special",
                "description": "特殊字符函数名",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "调用工具"
        }],
        "tools": tools_with_special_chars,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200或400
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_very_long_tool_name(api_client, request, stream):
    """parallel_tool_calls=True配合超长函数名，边界测试"""
    long_name = "a" * 200  # 超长函数名
    tools_with_long_name = [
        {
            "type": "function",
            "function": {
                "name": long_name,
                "description": "超长函数名工具",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "调用工具"
        }],
        "tools": tools_with_long_name,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200或400
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_complex_nested_parameters(api_client, request, stream):
    """parallel_tool_calls=True配合复杂嵌套参数的工具，边界测试"""
    tools_with_complex_params = [
        {
            "type": "function",
            "function": {
                "name": "complex_tool",
                "description": "复杂参数工具",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nested": {
                            "type": "object",
                            "properties": {
                                "level1": {
                                    "type": "object",
                                    "properties": {
                                        "level2": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        },
                        "array_of_objects": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "value": {"type": "number"}
                                }
                            }
                        }
                    },
                    "required": ["nested"]
                }
            }
        }
    ]

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "调用复杂工具"
        }],
        "tools": tools_with_complex_params,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200或400
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_false_with_single_tool_multiple_times(api_client, request, stream):
    """parallel_tool_calls=False，单个工具被多次调用场景，边界测试
    
    说明：即使只有一个工具，用户请求可能需要多次调用同一工具
    parallel_tool_calls=False应限制只返回一次调用
    """
    single_tool = [
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

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "查一下北京、上海、广州三个城市的天气"
        }],
        "tools": single_tool,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码200
    assertion.assert_status_code_200(response)

    # 校验点：tool_calls数量最多只有1个（流式和非流式都校验）
    _validate_tool_calls_count(response, stream, max_count=1)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_max_tokens_limit(api_client, request, stream):
    """parallel_tool_calls=True配合较小的max_tokens，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "查一下北京、上海、广州、深圳四个城市的天气"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 10  # 很小的token限制
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200（即使token限制很小）或400
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_all_parameters_combination(api_client, request, stream):
    """parallel_tool_calls配合多种参数组合，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "system", "content": "你是一个有用的助手"},
            {"role": "user", "content": "查一下北京天气"}
        ],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
        "stream": stream
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)
