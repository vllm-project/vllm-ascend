import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion, completion_request


@pytest.mark.parametrize("top_k", [0, 1, 50, -1], ids=["zero_disabled", "min", "typical", "minus_one_disabled"])
def test_top_k_accepts_representative_values(api_client, top_k):
    response = completion_request.send_chat_request(api_client, top_k=top_k)
    assertion.assert_chat_completion_success(response)


def test_top_k_accepts_streaming_request(api_client):
    response = completion_request.send_chat_request(api_client, top_k=50, stream=True)
    assertion.assert_chat_completion_success(response, stream=True)


@pytest.mark.parametrize("top_k", [-2, 1.5, []])
def test_top_k_rejects_invalid_values(api_client, top_k):
    response = completion_request.send_chat_request(api_client, top_k=top_k)
    assertion.assert_validation_error_response(response)


@pytest.mark.parametrize("top_k", [1000000, "10", None], ids=["large_value", "coerced_string", "default_null"])
def test_top_k_accepts_vllm_compatible_values(api_client, top_k):
    response = completion_request.send_chat_request(api_client, top_k=top_k)
    assertion.assert_success_or_validation_error(response)


def test_top_k_combines_with_other_sampling_options(api_client):
    response = completion_request.send_chat_request(api_client, top_k=50, top_p=0.9, temperature=0.8)
    assertion.assert_chat_completion_success(response)
