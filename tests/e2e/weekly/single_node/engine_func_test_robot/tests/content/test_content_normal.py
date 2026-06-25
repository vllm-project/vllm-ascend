import pytest
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_simple_string(api_client, request, stream):
    """content为普通字符串，请求正常"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好，请简单介绍一下自己"}],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # 校验点3：finish_reason有效
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_empty_string(api_client, request, stream):
    """content为空字符串，请求正常"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": ""}],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # 校验点3：finish_reason为stop或length
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_null(api_client, request, stream):
    """content为null，请求正常"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": None}],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验：error code为400错误，或者finish_reason为stop/length都算pass
    if assertion.has_error_code(response):
        # 存在error code，校验为400
        assertion.assert_error_code_400(response)
    else:
        # 没有error code，检查finish_reason为stop或length
        if stream:
            assertion.assert_stream_has_done(response.text)

        if stream:
            finish_reason = assertion.assert_stream_single_finish_reason(response.text)
        else:
            finish_reason = response.json()["choices"][0]["finish_reason"]
        assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_array_empty(api_client, request, stream):
    """content为空数组[]，请求正常"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": []}],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # 校验点3：finish_reason为stop或length
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_missing(api_client, request, stream):
    """message对象缺少content字段，请求正常"""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "user"
                # 缺少content字段
            }
        ],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # 校验点3：finish_reason为stop或length
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_with_special_chars(api_client, request, stream):
    """content包含特殊字符（标点、符号等），请求正常"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "Hello! 你好~ @#$%^&*()_+-=[]{}|;':\",./<>?"}
        ],
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
def test_content_multiline_text(api_client, request, stream):
    """content包含多行文本（换行符），请求正常"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "第一行\n第二行\n\n空行后的第三行"}],
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
def test_content_with_emoji(api_client, request, stream):
    """content包含Emoji表情，请求正常"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "你好👋 很高兴见到你😊 这是一颗星星⭐"}
        ],
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
def test_content_unicode_chinese(api_client, request, stream):
    """content包含中文字符和Unicode字符，请求正常"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": " apples 中文测试 日本語テスト 한국어"}
        ],
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
def test_content_long_text(api_client, request, stream):
    """content为较长文本（约1000字符），请求正常"""
    long_content = "这是测试文本。" * 100
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": long_content}],
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
def test_content_code_snippet(api_client, request, stream):
    """content为代码片段，请求正常"""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": "```python\ndef hello():\n    print('Hello World')\n```请解释这段代码",
            }
        ],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


# ==================== Content Array Format Tests ====================


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_array_text_objects(api_client, request, stream):
    """content为多文本对象数组格式（OpenAI多模态标准格式）"""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "你好"},
                    {"type": "text", "text": "你是谁？"},
                ],
            }
        ],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # 校验点3：finish_reason有效
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_array_single_text_object(api_client, request, stream):
    """content为单文本对象数组格式"""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "请简单介绍一下自己"}],
            }
        ],
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
def test_content_array_empty_text(api_client, request, stream):
    """content为数组格式但text为空字符串"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "text", "text": ""}]}],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码应为200或400（取决于引擎实现）
    assert response.status_code in [
        200,
        400,
    ], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_array_many_text_objects(api_client, request, stream):
    """content为包含多个文本对象的数组（边界测试）"""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "第一部分内容。"},
                    {"type": "text", "text": "第二部分内容。"},
                    {"type": "text", "text": "第三部分内容。"},
                    {"type": "text", "text": "第四部分内容。"},
                ],
            }
        ],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)
