import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import completion_request

PARAM_NAME = "frequency_penalty"


@pytest.mark.parametrize("value", [-2.0, 0.0, 2.0], ids=["min", "default", "max"])
def test_frequency_penalty_accepts_supported_range(api_client, value):
    response = completion_request.send_chat_request(api_client, **{PARAM_NAME: value})
    assertion.assert_chat_completion_success(response)


def test_frequency_penalty_accepts_streaming_request(api_client):
    response = completion_request.send_chat_request(api_client, **{PARAM_NAME: 0.5, "stream": True})
    assertion.assert_chat_completion_success(response, stream=True)


@pytest.mark.parametrize("value", [-2.1, 2.1, "1", None, [], {}])
def test_frequency_penalty_rejects_invalid_values(api_client, value):
    response = completion_request.send_chat_request(api_client, **{PARAM_NAME: value})
    assertion.assert_validation_error_response(response)


def test_frequency_penalty_combines_with_other_penalties(api_client):
    response = completion_request.send_chat_request(api_client, frequency_penalty=0.5, presence_penalty=0.5)
    assertion.assert_chat_completion_success(response)
