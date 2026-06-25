import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_whitespace_only(api_client, request, stream):
    """content仅包含空白字符（空格、制表符、换行），边界情况处理"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "   \t\n\n   "}],
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
def test_content_single_char(api_client, request, stream):
    """content为单个字符，最小有效内容边界"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "?"}],
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
def test_content_with_null_bytes(api_client, request, stream):
    """content包含null字节\x00，边界安全性测试"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Hello\x00World"}],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码应为200或400（取决于引擎处理null字节的方式）
    assert response.status_code in [
        200,
        400,
    ], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_json_escape_sequences(api_client, request, stream):
    """content包含JSON转义字符，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": 'Line1\nLine2\tTabbed"Quoted"\\Backslash'}],
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
def test_content_unicode_edge_cases(api_client, request, stream):
    """content包含Unicode边界字符（如零宽字符、特殊组合字符）"""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": "零宽空格:\u200b 零宽连接符:\u200d 从右向左符:\u202e 组合字符:é",
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
def test_content_rare_unicode_blocks(api_client, request, stream):
    """content包含罕见Unicode区块字符（ emoji变体、数学符号等）"""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": "数学:∀∃∈∉  表情变体:👨🏻‍💻  盲文:⠓⠑⠇⠇⠕  箭头:↳↴↵",
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
def test_content_rtl_languages(api_client, request, stream):
    """content包含从右到左语言（阿拉伯语、希伯来语等）"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "مرحبا بالعالم (Arabic) שלום עולם (Hebrew)"}],
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
def test_content_mixed_encoding_simulation(api_client, request, stream):
    """content模拟混合编码场景（已正确编码的UTF-8）"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Mixed: English中文العربية日本語🌍"}],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


# ==================== Content Array Format Boundary Tests ====================


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_array_type_field_case_sensitive(api_client, request, stream):
    """content数组格式type字段大小写敏感性边界测试"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "TEXT", "text": "你好"}]}],  # 大写TEXT
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码应为200（不敏感）或400（敏感）
    assert response.status_code in [
        200,
        400,
    ], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_array_extra_fields(api_client, request, stream):
    """content数组格式包含额外字段，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "你好", "extra_field": "extra_value"}],
            }
        ],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码应为200（如果引擎忽略额外字段）或400（如果严格校验）
    assert response.status_code in [
        200,
        400,
    ], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_array_text_whitespace_only(api_client, request, stream):
    """content数组格式text字段仅包含空白字符，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "   \t\n  "}]}],
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
def test_content_array_very_long_text(api_client, request, stream):
    """content数组格式text字段为超长文本，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "A" * 5000}]}],
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
def test_content_array_many_objects(api_client, request, stream):
    """content数组格式包含大量text对象，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"分段{i}"} for i in range(50)],
            }
        ],
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
def test_content_array_unicode_text(api_client, request, stream):
    """content数组格式text字段为Unicode字符，边界测试"""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "中文🇨🇳日本語🗾العربية🌍"}],
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
