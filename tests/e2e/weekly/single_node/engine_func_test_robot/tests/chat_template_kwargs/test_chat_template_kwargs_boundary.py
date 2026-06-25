import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_very_long_value(api_client, request, stream):
    """chat_template_kwargs值为超长字符串，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {"custom_param": "a" * 10000},  # 超长字符串值
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码应为200（如果引擎接受）或400（如果超出限制）
    assert response.status_code in [
        200,
        400,
    ], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_many_keys(api_client, request, stream):
    """chat_template_kwargs包含大量键值对，边界测试"""
    # 构造包含大量键的对象
    kwargs = {f"param_{i}": f"value_{i}" for i in range(100)}

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": kwargs,
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码应为200或400
    assert response.status_code in [
        200,
        400,
    ], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_unicode_keys(api_client, request, stream):
    """chat_template_kwargs包含Unicode字符的键名，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {
            "中文参数": "value",
            "日本語パラメータ": "value",
            "emoji_参数": "value",
        },
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码应为200或400（取决于引擎是否支持非ASCII键名）
    assert response.status_code in [
        200,
        400,
    ], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_special_chars_in_keys(api_client, request, stream):
    """chat_template_kwargs键名包含特殊字符，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {
            "param-with-dash": "value",
            "param_with_underscore": "value",
            "param.with.dot": "value",
            "param:with:colon": "value",
        },
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码应为200或400
    assert response.status_code in [
        200,
        400,
    ], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_numeric_string_values(api_client, request, stream):
    """chat_template_kwargs值为数字字符串，边界类型转换测试"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {
            "number_as_string": "12345",
            "float_as_string": "3.14159",
            "bool_as_string": "true",
        },
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码应为200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_mixed_types_values(api_client, request, stream):
    """chat_template_kwargs值为混合类型（数字、布尔、null），边界测试"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {
            "int_value": 42,
            "float_value": 3.14,
            "bool_value": True,
            "null_value": None,
            "string_value": "text",
        },
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码应为200或400（取决于引擎对非字符串值的处理）
    assert response.status_code in [
        200,
        400,
    ], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_reserved_words_keys(api_client, request, stream):
    """chat_template_kwargs使用保留字或内部关键字作为键名，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {
            "model": "overridden_model",  # 可能与请求参数冲突的键
            "messages": "overridden",  # 可能与请求参数冲突的键
            "stream": True,  # 可能与请求参数冲突的键
            "temperature": 2.0,  # 可能与生成参数冲突的键
        },
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码应为200（如果引擎正确处理命名空间隔离）或400（如果冲突）
    assert response.status_code in [
        200,
        400,
    ], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_empty_string_values(api_client, request, stream):
    """chat_template_kwargs值为空字符串，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {
            "empty_string": "",
            "whitespace_only": "   ",
            "null_string": "null",
        },
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_deeply_nested_object(api_client, request, stream):
    """chat_template_kwargs为嵌套多层对象，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {"level1": {"level2": {"level3": {"level4": {"level5": {"deep_value": "found"}}}}}},
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码应为200（如果引擎展平嵌套对象）或400（如果拒绝嵌套）
    assert response.status_code in [
        200,
        400,
    ], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_case_sensitive_keys(api_client, request, stream):
    """chat_template_kwargs键名大小写敏感测试，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {
            "Add_Generation_Prompt": True,  # 与标准驼峰式不同
            "ADD_GENERATION_PROMPT": True,  # 全大写
            "add_generation_prompt": True,  # 标准小写
        },
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码应为200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)
