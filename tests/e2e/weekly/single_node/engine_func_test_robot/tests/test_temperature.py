import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion, completion_request


@pytest.mark.parametrize("temperature", [0.0, 0.7, 2.0], ids=["greedy", "typical", "max"])
def test_temperature_accepts_supported_range(api_client, temperature):
    response = completion_request.send_chat_request(api_client, temperature=temperature)
    assertion.assert_chat_completion_success(response)


def test_temperature_accepts_streaming_request(api_client):
    response = completion_request.send_chat_request(api_client, temperature=0.7, stream=True)
    assertion.assert_chat_completion_success(response, stream=True)


@pytest.mark.parametrize("temperature", [-0.01, [], {}])
def test_temperature_rejects_invalid_values(api_client, temperature):
    response = completion_request.send_chat_request(api_client, temperature=temperature)
    assertion.assert_validation_error_response(response)


@pytest.mark.parametrize(
    "temperature",
    [2.01, "0.7", None],
    ids=["above_openai_range", "coerced_string", "default_null"],
)
def test_temperature_accepts_vllm_compatible_values(api_client, temperature):
    response = completion_request.send_chat_request(api_client, temperature=temperature)
    assertion.assert_success_or_validation_error(response)


def test_temperature_combines_with_top_p(api_client):
    response = completion_request.send_chat_request(api_client, temperature=0.8, top_p=0.9)
    assertion.assert_chat_completion_success(response)
