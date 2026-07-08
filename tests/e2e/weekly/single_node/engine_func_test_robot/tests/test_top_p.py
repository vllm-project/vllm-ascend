import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import completion_request


@pytest.mark.parametrize("top_p", [0.1, 0.9, 1.0], ids=["low", "typical", "disabled"])
def test_top_p_accepts_representative_values(api_client, top_p):
    response = completion_request.send_chat_request(api_client, top_p=top_p)
    assertion.assert_chat_completion_success(response)


def test_top_p_accepts_streaming_request(api_client):
    response = completion_request.send_chat_request(api_client, top_p=0.9, stream=True)
    assertion.assert_chat_completion_success(response, stream=True)


@pytest.mark.parametrize("top_p", [0.0, -0.01, 1.01, "0.9", None, []])
def test_top_p_rejects_invalid_values(api_client, top_p):
    response = completion_request.send_chat_request(api_client, top_p=top_p)
    assertion.assert_validation_error_response(response)


def test_top_p_combines_with_other_sampling_options(api_client):
    response = completion_request.send_chat_request(api_client, top_p=0.9, top_k=50, temperature=0.8)
    assertion.assert_chat_completion_success(response)
