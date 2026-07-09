import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_whitespace_only(api_client, request, stream):
    """Content contains only whitespace characters (spaces, tabs, newlines), boundary case handling"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "   \t\n\n   "}],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint: status code should be 200 or 400 (depends on engine implementation)
    assert response.status_code in [
        200,
        400,
    ], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_single_char(api_client, request, stream):
    """Content is a single character, minimum valid content boundary"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "?"}],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint 1: status code 200
    assertion.assert_status_code_200(response)

    # Checkpoint 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_with_null_bytes(api_client, request, stream):
    """Content contains null bytes \x00, boundary security test"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Hello\x00World"}],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint: status code should be 200 or 400 (depends on how engine handles null bytes)
    assert response.status_code in [
        200,
        400,
    ], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_json_escape_sequences(api_client, request, stream):
    """Content contains JSON escape characters, boundary test"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": 'Line1\nLine2\tTabbed"Quoted"\\Backslash'}],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint 1: status code 200
    assertion.assert_status_code_200(response)

    # Checkpoint 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_unicode_edge_cases(api_client, request, stream):
    """Content contains Unicode boundary characters (e.g., zero-width characters, combining characters)"""
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

    # Checkpoint 1: status code 200
    assertion.assert_status_code_200(response)

    # Checkpoint 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_rare_unicode_blocks(api_client, request, stream):
    """Content contains rare Unicode block characters (emoji variants, math symbols, etc.)"""
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

    # Checkpoint 1: status code 200
    assertion.assert_status_code_200(response)

    # Checkpoint 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_rtl_languages(api_client, request, stream):
    """Content contains right-to-left languages (Arabic, Hebrew, etc.)"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "مرحبا بالعالم (Arabic) שלום עולם (Hebrew)"}],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint 1: status code 200
    assertion.assert_status_code_200(response)

    # Checkpoint 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_mixed_encoding_simulation(api_client, request, stream):
    """Content simulates mixed encoding scenario (correctly encoded UTF-8)"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Mixed: English中文العربية日本語🌍"}],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint 1: status code 200
    assertion.assert_status_code_200(response)

    # Checkpoint 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


# ==================== Content Array Format Boundary Tests ====================


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_array_type_field_case_sensitive(api_client, request, stream):
    """Content array format type field case sensitivity boundary test"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "TEXT", "text": "你好"}]}],  # uppercase TEXT
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint: status code should be 200 (insensitive) or 400 (sensitive)
    assert response.status_code in [
        200,
        400,
    ], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_array_extra_fields(api_client, request, stream):
    """Content array format contains extra fields, boundary test"""
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

    # Checkpoint: status code should be 200 (if engine ignores extra fields) or 400 (if strict validation)
    assert response.status_code in [
        200,
        400,
    ], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_array_text_whitespace_only(api_client, request, stream):
    """Content array format text field contains only whitespace characters, boundary test"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "   \t\n  "}]}],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint: status code should be 200 or 400 (depends on engine implementation)
    assert response.status_code in [
        200,
        400,
    ], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_array_very_long_text(api_client, request, stream):
    """Content array format text field is extremely long text, boundary test"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "A" * 5000}]}],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint: status code should be 200 or 400
    assert response.status_code in [
        200,
        400,
    ], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_array_many_objects(api_client, request, stream):
    """Content array format contains many text objects, boundary test"""
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

    # Checkpoint: status code should be 200 or 400
    assert response.status_code in [
        200,
        400,
    ], f"Status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_array_unicode_text(api_client, request, stream):
    """Content array format text field contains Unicode characters, boundary test"""
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

    # Checkpoint 1: status code 200
    assertion.assert_status_code_200(response)

    # Checkpoint 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)
