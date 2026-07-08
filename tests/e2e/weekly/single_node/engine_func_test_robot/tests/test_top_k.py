import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import completion_request


@pytest.mark.parametrize("top_k", [1, 50, -1], ids=["min", "typical", "disabled"])
def test_top_k_accepts_representative_values(api_client, top_k):
    response = completion_request.send_chat_request(api_client, top_k=top_k)
    assertion.assert_chat_completion_success(response)


def test_top_k_accepts_streaming_request(api_client):
    response = completion_request.send_chat_request(api_client, top_k=50, stream=True)
    assertion.assert_chat_completion_success(response, stream=True)


@pytest.mark.parametrize("top_k", [0, -2, 1000000, 1.5, "10", None])
def test_top_k_rejects_invalid_values(api_client, top_k):
    response = completion_request.send_chat_request(api_client, top_k=top_k)
    assertion.assert_validation_error_response(response)


def test_top_k_combines_with_other_sampling_options(api_client):
    response = completion_request.send_chat_request(api_client, top_k=50, top_p=0.9, temperature=0.8)
    assertion.assert_chat_completion_success(response)
