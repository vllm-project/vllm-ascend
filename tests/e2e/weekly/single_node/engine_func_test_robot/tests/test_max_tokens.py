import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import completion_request


@pytest.mark.parametrize("max_tokens", [1, 32, 512], ids=["min", "typical", "large"])
def test_max_tokens_accepts_positive_integers(api_client, max_tokens):
    response = completion_request.send_chat_request(api_client, max_tokens=max_tokens)
    assertion.assert_chat_completion_success(response)


def test_max_tokens_accepts_streaming_request(api_client):
    response = completion_request.send_chat_request(api_client, max_tokens=32, stream=True)
    assertion.assert_chat_completion_success(response, stream=True)


@pytest.mark.parametrize("max_tokens", [0, -1, 1.5, "32", True])
def test_max_tokens_rejects_invalid_values(api_client, max_tokens):
    response = completion_request.send_chat_request(api_client, max_tokens=max_tokens)
    assertion.assert_validation_error_response(response)


def test_max_tokens_combines_with_stop_and_sampling(api_client):
    response = completion_request.send_chat_request(api_client, max_tokens=32, stop=["."], temperature=0.7, top_p=0.9)
    assertion.assert_chat_completion_success(response)
