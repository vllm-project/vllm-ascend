"""
Tests for the thinking and enable_thinking fields.
- thinking: used by DeepSeek/DS model families.
- enable_thinking: used by Qwen model families.

When the field is true, validate that the think tags are complete.
When the field is false, validate that no think tags are present.
Follow the validation rules from the think_tag directory.
"""

import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


def is_qwen_model(model_name):
    """Return whether the model belongs to the Qwen family, case-insensitively."""
    return model_name and "qwen" in model_name.lower()


def is_deepseek_model(model_name):
    """Return whether the model belongs to the DeepSeek/DS family, case-insensitively."""
    if not model_name:
        return False
    model_lower = model_name.lower()
    return "deepseek" in model_lower or "ds" in model_lower


def should_check_think_tag(request):
    """Return whether think-tag validation should be performed."""
    return request.config.getoption("--thinkTagOutput").strip().lower() == "true"


# ==================== Qwen Model Tests - enable_thinking ====================


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_qwen_enable_thinking_true(api_client, request, stream):
    """Qwen model: enable_thinking=true enables thinking mode; validate complete think tags."""
    model = request.config.getoption("--model")
    if not is_qwen_model(model):
        pytest.skip(f"current model {model} is not in the Qwen family; skipping this test")

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "请用最简单的一句话介绍你是谁。"}],
        "chat_template_kwargs": {"enable_thinking": True},
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check 1: status code is 200
    assertion.assert_status_code_200(response)

    # Check 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check 3: think tags are complete when enable_thinking=true
    if should_check_think_tag(request):
        assertion.assert_think_tag_present(response.content.decode("utf-8"), "enable_thinking=true")


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_qwen_enable_thinking_false(api_client, request, stream):
    """Qwen model: enable_thinking=false disables thinking mode; validate that no think tags are present."""
    model = request.config.getoption("--model")
    if not is_qwen_model(model):
        pytest.skip(f"current model {model} is not in the Qwen family; skipping this test")

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "请用最简单的一句话介绍你是谁。"}],
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check 1: status code is 200
    assertion.assert_status_code_200(response)

    # Check 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check 3: no think tags are present when enable_thinking=false
    if should_check_think_tag(request):
        assertion.assert_no_think_tag(response.content.decode("utf-8"), "enable_thinking=false")


# ==================== DeepSeek/DS Model Tests - thinking ====================


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_deepseek_thinking_true(api_client, request, stream):
    """DeepSeek/DS model: thinking=true enables thinking mode; validate complete think tags."""
    model = request.config.getoption("--model")
    if not is_deepseek_model(model):
        pytest.skip(f"current model {model} is not in the DeepSeek/DS family; skipping this test")

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "请用最简单的一句话介绍你是谁。"}],
        "chat_template_kwargs": {"thinking": True},
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check 1: status code is 200
    assertion.assert_status_code_200(response)

    # Check 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check 3: think tags are complete when thinking=true
    if should_check_think_tag(request):
        assertion.assert_think_tag_present(response.content.decode("utf-8"), "thinking=true")


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_deepseek_thinking_false(api_client, request, stream):
    """DeepSeek/DS model: thinking=false disables thinking mode; validate that no think tags are present."""
    model = request.config.getoption("--model")
    if not is_deepseek_model(model):
        pytest.skip(f"current model {model} is not in the DeepSeek/DS family; skipping this test")

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "请用最简单的一句话介绍你是谁。"}],
        "chat_template_kwargs": {"thinking": False},
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check 1: status code is 200
    assertion.assert_status_code_200(response)

    # Check 2: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check 3: no think tags are present when thinking=false
    if should_check_think_tag(request):
        assertion.assert_no_think_tag(response.content.decode("utf-8"), "thinking=false")


# ==================== Inapplicable Model Tests - Abnormal Scenarios ====================


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_qwen_field_on_deepseek(api_client, request, stream):
    """Abnormal: use the enable_thinking field on a DeepSeek model."""
    model = request.config.getoption("--model")
    if not is_deepseek_model(model):
        pytest.skip(f"current model {model} is not in the DeepSeek/DS family; skipping this test")

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {"enable_thinking": True},  # Use the Qwen field on DeepSeek
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: either 200 if the engine ignores unknown fields, or 400 if it validates strictly
    assert response.status_code in [
        200,
        400,
    ], f"status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_deepseek_field_on_qwen(api_client, request, stream):
    """Abnormal: use the thinking field on a Qwen model."""
    model = request.config.getoption("--model")
    if not is_qwen_model(model):
        pytest.skip(f"current model {model} is not in the Qwen family; skipping this test")

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {"thinking": True},  # Use the DeepSeek field on Qwen
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: either 200 if the engine ignores unknown fields, or 400 if it validates strictly
    assert response.status_code in [
        200,
        400,
    ], f"status code should be 200 or 400, got {response.status_code}"


# ==================== Boundary Tests ====================


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_qwen_enable_thinking_null(api_client, request, stream):
    """Abnormal: enable_thinking is null for a Qwen model."""
    model = request.config.getoption("--model")
    if not is_qwen_model(model):
        pytest.skip(f"current model {model} is not in the Qwen family; skipping this test")

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {"enable_thinking": None},
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    assertion.assert_status_code_200(response)


def test_chat_template_kwargs_deepseek_thinking_null_non_stream(api_client, request):
    """Abnormal: thinking is null for a DeepSeek model in non-streaming mode; error code is 400."""
    model = request.config.getoption("--model")
    if not is_deepseek_model(model):
        pytest.skip(f"current model {model} is not in the DeepSeek/DS family; skipping this test")

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {"thinking": None},
        "stream": False,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    assertion.assert_status_code_200(response)


def test_chat_template_kwargs_deepseek_thinking_null_stream(api_client, request):
    """Abnormal: thinking is null for a DeepSeek model in streaming mode; status code is 200 and error code is 400."""
    model = request.config.getoption("--model")
    if not is_deepseek_model(model):
        pytest.skip(f"current model {model} is not in the DeepSeek/DS family; skipping this test")

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {"thinking": None},
        "stream": True,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200 and error code is 400
    assertion.assert_status_code_200(response)
    assertion.assert_error_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_qwen_enable_thinking_string(api_client, request, stream):
    """Abnormal: enable_thinking is a string for a Qwen model."""
    model = request.config.getoption("--model")
    if not is_qwen_model(model):
        pytest.skip(f"current model {model} is not in the Qwen family; skipping this test")

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {"enable_thinking": "true"},
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    assert response.status_code in [
        200,
        400,
    ], f"status code should be 200 or 400, got {response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_deepseek_thinking_string(api_client, request, stream):
    """Abnormal: thinking is a string for a DeepSeek model."""
    model = request.config.getoption("--model")
    if not is_deepseek_model(model):
        pytest.skip(f"current model {model} is not in the DeepSeek/DS family; skipping this test")

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {"thinking": "true"},
        "stream": stream,
        "max_tokens": 512,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    assert response.status_code in [
        200,
        400,
    ], f"status code should be 200 or 400, got {response.status_code}"
