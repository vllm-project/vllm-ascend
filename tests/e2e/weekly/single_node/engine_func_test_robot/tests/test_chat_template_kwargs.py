import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import completion_request


def _is_qwen_model(model_name):
    return model_name and "qwen" in model_name.lower()


def _is_deepseek_model(model_name):
    if not model_name:
        return False
    model_lower = model_name.lower()
    return "deepseek" in model_lower or "ds" in model_lower


def _should_check_think_tag(request):
    return request.config.getoption("--thinkTagOutput").strip().lower() == "true"


@pytest.mark.parametrize(
    "chat_template_kwargs",
    [
        None,
        {},
        {"add_generation_prompt": True},
        {"enable_system_prompt": True},
        {"date": "2025-04-01", "time": "10:00:00"},
        {"tools_prompt": "default", "custom_var": "custom_value"},
        {"add_special_tokens": True},
        {"skip_special_tokens": False},
    ],
    ids=["null", "empty", "generation_prompt", "system_prompt", "date", "multiple", "special_tokens", "skip_tokens"],
)
def test_chat_template_kwargs_accepts_supported_values(api_client, chat_template_kwargs):
    response = completion_request.send_chat_request(api_client, chat_template_kwargs=chat_template_kwargs)
    assertion.assert_chat_completion_success(response)


def test_chat_template_kwargs_accepts_streaming_request(api_client):
    response = completion_request.send_chat_request(
        api_client, chat_template_kwargs={"add_generation_prompt": True}, stream=True
    )
    assertion.assert_chat_completion_success(response, stream=True)


@pytest.mark.parametrize(
    "chat_template_kwargs",
    ["invalid_string", ["item1", "item2"], 123, True],
    ids=["string", "array", "integer", "boolean"],
)
def test_chat_template_kwargs_rejects_invalid_top_level_types(api_client, chat_template_kwargs):
    response = completion_request.send_chat_request(api_client, chat_template_kwargs=chat_template_kwargs)
    assertion.assert_validation_error_response(response)


@pytest.mark.parametrize(
    "chat_template_kwargs",
    [
        {"custom_param": "a" * 10000},
        {f"param_{i}": f"value_{i}" for i in range(100)},
        {"param-with-dash": "value", "param.with.dot": "value", "param:with:colon": "value"},
        {"number_as_string": "12345", "float_as_string": "3.14159", "bool_as_string": "true"},
        {"int_value": 42, "float_value": 3.14, "bool_value": True, "null_value": None},
        {"model": "overridden_model", "messages": "overridden", "stream": True},
        {"level1": {"level2": {"level3": {"deep_value": "found"}}}},
    ],
    ids=["long_value", "many_keys", "special_keys", "numeric_strings", "mixed_types", "reserved_keys", "nested"],
)
def test_chat_template_kwargs_accepts_or_rejects_boundary_objects(api_client, chat_template_kwargs):
    response = completion_request.send_chat_request(api_client, chat_template_kwargs=chat_template_kwargs)
    assertion.assert_status_code_200_or_400(response)


@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
def test_chat_template_kwargs_qwen_enable_thinking(api_client, request, enabled):
    model = request.config.getoption("--model")
    if not _is_qwen_model(model):
        pytest.skip(f"current model {model} is not in the Qwen family; skipping this test")

    response = completion_request.send_chat_request(
        api_client, chat_template_kwargs={"enable_thinking": enabled}, max_tokens=512
    )
    assertion.assert_status_code_200(response)
    if _should_check_think_tag(request):
        response_text = response.content.decode("utf-8")
        if enabled:
            assertion.assert_think_tag_present(response_text, "enable_thinking=true")
        else:
            assertion.assert_no_think_tag(response_text, "enable_thinking=false")


@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
def test_chat_template_kwargs_deepseek_thinking(api_client, request, enabled):
    model = request.config.getoption("--model")
    if not _is_deepseek_model(model):
        pytest.skip(f"current model {model} is not in the DeepSeek/DS family; skipping this test")

    response = completion_request.send_chat_request(
        api_client, chat_template_kwargs={"thinking": enabled}, max_tokens=512
    )
    assertion.assert_status_code_200(response)
    if _should_check_think_tag(request):
        response_text = response.content.decode("utf-8")
        if enabled:
            assertion.assert_think_tag_present(response_text, "thinking=true")
        else:
            assertion.assert_no_think_tag(response_text, "thinking=false")


@pytest.mark.parametrize(
    ("model_family", "chat_template_kwargs"),
    [
        ("qwen", {"thinking": True}),
        ("qwen", {"enable_thinking": None}),
        ("qwen", {"enable_thinking": "true"}),
        ("deepseek", {"enable_thinking": True}),
        ("deepseek", {"thinking": None}),
        ("deepseek", {"thinking": "true"}),
    ],
    ids=[
        "qwen_deepseek_field",
        "qwen_null",
        "qwen_string",
        "deepseek_qwen_field",
        "deepseek_null",
        "deepseek_string",
    ],
)
def test_chat_template_kwargs_thinking_accepts_or_rejects_model_specific_boundaries(
    api_client, request, model_family, chat_template_kwargs
):
    model = request.config.getoption("--model")
    if model_family == "qwen" and not _is_qwen_model(model):
        pytest.skip(f"current model {model} is not in the Qwen family; skipping this test")
    if model_family == "deepseek" and not _is_deepseek_model(model):
        pytest.skip(f"current model {model} is not in the DeepSeek/DS family; skipping this test")

    response = completion_request.send_chat_request(
        api_client, chat_template_kwargs=chat_template_kwargs, max_tokens=512
    )
    assertion.assert_status_code_200_or_400(response)
