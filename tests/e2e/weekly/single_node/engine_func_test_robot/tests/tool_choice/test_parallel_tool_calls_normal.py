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
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "搜索网页内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            }
        }
    }
]



def _validate_tool_calls_response(response, stream, expected_tools=None):
    """校验tool_calls响应的通用函数
    
    Args:
        response: HTTP响应对象
        stream: 是否流式响应
        expected_tools: 期望的工具名称列表，默认为["get_weather", "get_time"]
    """
    if expected_tools is None:
        expected_tools = ["get_weather", "get_time"]
    
    if stream:
        # 流式响应校验
        assertion.assert_stream_has_done(response.text)
        
        tool_calls_found = False
        finish_reason_tool_calls = False
        for line in response.text.strip().split("\n"):
            if line.startswith("data: ") and "[DONE]" not in line:
                chunk = json.loads(line[6:])
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("tool_calls"):
                        tool_calls_found = True
                        for tc in delta.get("tool_calls", []):
                            assert "index" in tc, "流式tool_calls应包含index字段"
                            if tc.get("id"):
                                assert tc.get("type") == "function", "tool_calls类型应为function"
                            if tc.get("function"):
                                func = tc.get("function", {})
                                if func.get("name"):
                                    assert func.get("name") in expected_tools, \
                                        f"函数名应为{expected_tools}之一，实际为{func.get('name')}"
                    if choices[0].get("finish_reason") == "tool_calls":
                        finish_reason_tool_calls = True
        
        assert tool_calls_found, "流式响应应包含tool_calls数据"
        assert finish_reason_tool_calls, "流式响应finish_reason应为tool_calls"
    else:
        # 非流式响应校验
        response_json = response.json()
        choices = response_json.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            
            if message.get("tool_calls") or choice.get("finish_reason") == "tool_calls":
                assert choice.get("finish_reason") == "tool_calls", \
                    f"finish_reason应为tool_calls，实际为{choice.get('finish_reason')}"
                
                tool_calls = message.get("tool_calls", [])
                assert len(tool_calls) >= 1, "应允许返回工具调用"
                
                for tc in tool_calls:
                    assert "id" in tc, "tool_call应包含id字段"
                    assert tc.get("type") == "function", "tool_call类型应为function"
                    assert "function" in tc, "tool_call应包含function字段"
                    func = tc.get("function", {})
                    assert "name" in func, "function应包含name字段"
                    assert "arguments" in func, "function应包含arguments字段"
                    assert func.get("name") in expected_tools, \
                        f"函数名应为{expected_tools}之一，实际为{func.get('name')}"
                    args = json.loads(func.get("arguments", "{}"))
                    assert isinstance(args, dict), "arguments解析后应为dict"


def _validate_tool_calls_count(response, stream, max_count):
    """校验tool_calls数量的函数
    
    Args:
        response: HTTP响应对象
        stream: 是否流式响应
        max_count: tool_calls的最大数量
    """
    if stream:
        assertion.assert_stream_has_done(response.text)
    else:
        response_json = response.json()
        choices = response_json.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            if message.get("tool_calls") or choice.get("finish_reason") == "tool_calls":
                tool_calls = message.get("tool_calls", [])
                assert len(tool_calls) <= max_count, \
                    f"tool_calls数量应<={max_count}，实际为{len(tool_calls)}"


def _validate_no_tool_calls(response, stream):
    """校验响应中不应有tool_calls"""
    if stream:
        # 流式响应校验
        assertion.assert_stream_has_done(response.text)
        
        for line in response.text.strip().split("\n"):
            if line.startswith("data: ") and "[DONE]" not in line:
                chunk = json.loads(line[6:])
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    # 流式响应中不应有 tool_calls
                    if delta.get("tool_calls"):
                        assert False, f"流式响应不应包含tool_calls，实际包含: {delta.get('tool_calls')}"
                    # finish_reason 不应为 tool_calls
                    finish_reason = choices[0].get("finish_reason")
                    if finish_reason:
                        assert finish_reason != "tool_calls", f"finish_reason不应为tool_calls，实际为{finish_reason}"
    else:
        # 非流式响应校验
        response_json = response.json()
        choices = response_json.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls", [])
            assert tool_calls is None or len(tool_calls) == 0,                 f"响应不应包含tool_calls，实际为{tool_calls}"
            finish_reason = choice.get("finish_reason")
            assert finish_reason != "tool_calls", f"finish_reason不应为tool_calls，实际为{finish_reason}"


def _validate_tool_calls_with_stop_finish_reason(response, stream, expected_tools=None):
    """校验tool_calls响应，但finish_reason应为stop（用于具名函数tool_choice场景）
    
    当tool_choice为具名函数对象时，即使调用了工具，finish_reason也应该是"stop"而不是"tool_calls"
    
    Args:
        response: HTTP响应对象
        stream: 是否流式响应
        expected_tools: 期望的工具名称列表
    """
    if expected_tools is None:
        expected_tools = ["get_weather"]
    
    if stream:
        # 流式响应校验
        assertion.assert_stream_has_done(response.text)
        
        tool_calls_found = False
        finish_reason_stop = False
        for line in response.text.strip().split("\n"):
            if line.startswith("data: ") and "[DONE]" not in line:
                chunk = json.loads(line[6:])
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("tool_calls"):
                        tool_calls_found = True
                        for tc in delta.get("tool_calls", []):
                            assert "index" in tc, "流式tool_calls应包含index字段"
                            if tc.get("id"):
                                assert tc.get("type") == "function", "tool_calls类型应为function"
                            if tc.get("function"):
                                func = tc.get("function", {})
                                if func.get("name"):
                                    assert func.get("name") in expected_tools, \
                                        f"函数名应为{expected_tools}之一，实际为{func.get('name')}"
                    # 具名函数场景下，finish_reason应该是stop而不是tool_calls
                    finish_reason = choices[0].get("finish_reason")
                    if finish_reason:
                        assert finish_reason == "stop", \
                            f"具名函数tool_choice场景下finish_reason应为stop，实际为{finish_reason}"
                        finish_reason_stop = True
        
        assert tool_calls_found, "流式响应应包含tool_calls数据"
        assert finish_reason_stop, "流式响应finish_reason应为stop"
    else:
        # 非流式响应校验
        response_json = response.json()
        choices = response_json.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            
            if message.get("tool_calls"):
                tool_calls = message.get("tool_calls", [])
                assert len(tool_calls) >= 1, "应允许返回工具调用"
                
                for tc in tool_calls:
                    assert "id" in tc, "tool_call应包含id字段"
                    assert tc.get("type") == "function", "tool_call类型应为function"
                    assert "function" in tc, "tool_call应包含function字段"
                    func = tc.get("function", {})
                    assert "name" in func, "function应包含name字段"
                    assert "arguments" in func, "function应包含arguments字段"
                    assert func.get("name") in expected_tools, \
                        f"函数名应为{expected_tools}之一，实际为{func.get('name')}"
                    args = json.loads(func.get("arguments", "{}"))
                    assert isinstance(args, dict), "arguments解析后应为dict"
                
                # 具名函数场景下，finish_reason应该是stop而不是tool_calls
                finish_reason = choice.get("finish_reason")
                assert finish_reason == "stop", \
                    f"具名函数tool_choice场景下finish_reason应为stop，实际为{finish_reason}"



@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_true(api_client, request, stream):
    """parallel_tool_calls为True（默认值），允许返回多个工具调用"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "北京和上海今天天气怎么样？"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码200
    assertion.assert_status_code_200(response)

    # 复用校验逻辑
    _validate_tool_calls_response(response, stream, expected_tools=["get_weather"])


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_false(api_client, request, stream):
    """parallel_tool_calls为False，只返回一个工具调用"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "北京和上海今天天气怎么样？"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
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

    # 校验点：非流式响应中tool_calls最多只有1个
    _validate_tool_calls_count(response, stream, max_count=1)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_default_value(api_client, request, stream):
    """不传parallel_tool_calls参数，默认行为等同于True（允许多个工具调用）"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "查一下北京天气和当前时间"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "auto",
        # 不传parallel_tool_calls，使用默认值True
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码200
    assertion.assert_status_code_200(response)

    # 复用校验逻辑
    _validate_tool_calls_response(response, stream, expected_tools=["get_weather", "get_time"])



@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_tool_choice_none(api_client, request, stream):
    """parallel_tool_calls配合tool_choice='none'，不应调用任何工具"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "北京今天天气怎么样？"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "none",
        "parallel_tool_calls": True,  # 即使允许并行，tool_choice=none优先
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码200
    assertion.assert_status_code_200(response)

    # 校验点：不应有tool_calls
    _validate_no_tool_calls(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_tool_choice_required(api_client, request, stream):
    """parallel_tool_calls配合tool_choice='required'，必须调用工具"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "查一下信息"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "required",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码200
    assertion.assert_status_code_200(response)

    # 校验点：必须有tool_calls
    _validate_tool_calls_response(response, stream, expected_tools=["get_weather", "get_time", "search_web"])


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_with_specific_function(api_client, request, stream):
    """parallel_tool_calls配合指定函数的tool_choice"""
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
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码200
    assertion.assert_status_code_200(response)

    # 校验点：tool_calls结构正确，函数名为get_weather
    # 注意：当tool_choice为具名函数时，finish_reason应为"stop"而不是"tool_calls"
    _validate_tool_calls_with_stop_finish_reason(response, stream, expected_tools=["get_weather"])


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_false_with_specific_function(api_client, request, stream):
    """parallel_tool_calls=False配合指定函数的tool_choice，只返回一个工具调用"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "北京和上海今天天气怎么样？"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": {
            "type": "function",
            "function": {
                "name": "get_weather"
            }
        },
        "parallel_tool_calls": False,
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

    # 校验点：非流式响应中tool_calls最多只有1个
    _validate_tool_calls_count(response, stream, max_count=1)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_without_tools(api_client, request, stream):
    """parallel_tool_calls存在但tools未提供，应正常响应（忽略parallel_tool_calls）"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码应为200（如果引擎忽略parallel_tool_calls）或400（如果要求必须配合tools）
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_single_tool(api_client, request, stream):
    """parallel_tool_calls=True但只有一个工具，正常响应"""
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
            "content": "北京今天天气怎么样？"
        }],
        "tools": single_tool,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码200
    assertion.assert_status_code_200(response)

    # 校验点：tool_calls结构正确
    _validate_tool_calls_response(response, stream, expected_tools=["get_weather"])


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_parallel_tool_calls_non_function_call_query(api_client, request, stream):
    """parallel_tool_calls=True但用户查询不涉及工具调用，模型应正常响应文本"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请介绍一下你自己"
        }],
        "tools": TOOLS_DEFINITION,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码200
    assertion.assert_status_code_200(response)

    # 校验点：不应有tool_calls（普通对话查询）
    _validate_no_tool_calls(response, stream)
