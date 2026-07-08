import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import completion_request


@pytest.mark.parametrize("repetition_penalty", [1.0, 1.2, 2.0], ids=["neutral", "typical", "high"])
def test_repetition_penalty_accepts_supported_values(api_client, repetition_penalty):
    response = completion_request.send_chat_request(api_client, repetition_penalty=repetition_penalty)
    assertion.assert_chat_completion_success(response)


def test_repetition_penalty_accepts_streaming_request(api_client):
    response = completion_request.send_chat_request(api_client, repetition_penalty=1.2, stream=True)
    assertion.assert_chat_completion_success(response, stream=True)


@pytest.mark.parametrize("repetition_penalty", [0.99, 0, -1, "1.2", None, [], {}])
def test_repetition_penalty_rejects_invalid_values(api_client, repetition_penalty):
    response = completion_request.send_chat_request(api_client, repetition_penalty=repetition_penalty)
    assertion.assert_validation_error_response(response)


def test_repetition_penalty_combines_with_sampling_options(api_client):
    response = completion_request.send_chat_request(
        api_client, repetition_penalty=1.2, frequency_penalty=0.5, presence_penalty=0.5
    )
    assertion.assert_chat_completion_success(response)
