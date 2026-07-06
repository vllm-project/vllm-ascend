import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_simple_string(api_client, request, stream):
    """Content is a plain string, request should succeed normally"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好，请简单介绍一下自己"}],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint 1: status code 200
    assertion.assert_status_code_200(response)

    # Checkpoint 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Checkpoint 3: finish_reason is valid
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_empty_string(api_client, request, stream):
    """Content is an empty string, request should succeed normally"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": ""}],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint 1: status code 200
    assertion.assert_status_code_200(response)

    # Checkpoint 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Checkpoint 3: finish_reason is stop or length
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_null(api_client, request, stream):
    """Content is null, request should succeed normally"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": None}],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: error code 400, or finish_reason stop/length both pass
    if assertion.has_error_code(response):
        # Error code exists, validate it is 400
        assertion.assert_error_code_400(response)
    else:
        # No error code, check finish_reason is stop or length
        if stream:
            assertion.assert_stream_has_done(response.text)

        if stream:
            finish_reason = assertion.assert_stream_single_finish_reason(response.text)
        else:
            finish_reason = response.json()["choices"][0]["finish_reason"]
        assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_array_empty(api_client, request, stream):
    """Content is an empty array [], request should succeed normally"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": []}],
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Checkpoint 1: status code 200
    assertion.assert_status_code_200(response)

    # Checkpoint 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Checkpoint 3: finish_reason is stop or length
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_missing(api_client, request, stream):
    """Message object missing content field, request should succeed normally"""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "user"
                # missing content field
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

    # Checkpoint 3: finish_reason is stop or length
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_with_special_chars(api_client, request, stream):
    """Content contains special characters (punctuation, symbols, etc.), request should succeed normally"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Hello! 你好~ @#$%^&*()_+-=[]{}|;':\",./<>?"}],
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
def test_content_multiline_text(api_client, request, stream):
    """Content contains multiline text (newline characters), request should succeed normally"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "第一行\n第二行\n\n空行后的第三行"}],
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
def test_content_with_emoji(api_client, request, stream):
    """Content contains emoji, request should succeed normally"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好👋 很高兴见到你😊 这是一颗星星⭐"}],
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
def test_content_unicode_chinese(api_client, request, stream):
    """Content contains Chinese characters and Unicode characters, request should succeed normally"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": " apples 中文测试 日本語テスト 한국어"}],
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
def test_content_long_text(api_client, request, stream):
    """Content is a long text (approx. 1000 characters), request should succeed normally"""
    long_content = "这是测试文本。" * 100
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": long_content}],
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
def test_content_code_snippet(api_client, request, stream):
    """Content is a code snippet, request should succeed normally"""
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

    # Checkpoint 1: status code 200
    assertion.assert_status_code_200(response)

    # Checkpoint 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


# ==================== Content Array Format Tests ====================


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_array_text_objects(api_client, request, stream):
    """Content is an array of multiple text objects (OpenAI multimodal standard format)"""
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

    # Checkpoint 1: status code 200
    assertion.assert_status_code_200(response)

    # Checkpoint 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Checkpoint 3: finish_reason is valid
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_array_single_text_object(api_client, request, stream):
    """Content is an array with a single text object"""
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

    # Checkpoint 1: status code 200
    assertion.assert_status_code_200(response)

    # Checkpoint 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_content_array_empty_text(api_client, request, stream):
    """Content is an array format but text is an empty string"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "text", "text": ""}]}],
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
def test_content_array_many_text_objects(api_client, request, stream):
    """Content is an array containing many text objects (boundary test)"""
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

    # Checkpoint 1: status code 200
    assertion.assert_status_code_200(response)

    # Checkpoint 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)
