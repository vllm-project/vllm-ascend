import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import completion_request


@pytest.mark.parametrize("top_logprobs", [0, 5, 20], ids=["none", "typical", "max"])
def test_logprobs_accepts_supported_top_logprobs(api_client, top_logprobs):
    response = completion_request.send_chat_request(api_client, logprobs=True, top_logprobs=top_logprobs)
    assertion.assert_status_code_200(response)
    assertion.assert_top_logprobs_count(response, top_logprobs)


def test_logprobs_accepts_streaming_request(api_client):
    response = completion_request.send_chat_request(api_client, logprobs=True, top_logprobs=5, stream=True)
    assertion.assert_status_code_200(response)
    assertion.assert_stream_has_done(response.text)
    assertion.assert_top_logprobs_count(response, 5)


def test_logprobs_rejects_top_logprobs_above_limit(api_client):
    response = completion_request.send_chat_request(api_client, logprobs=True, top_logprobs=21)
    assertion.assert_validation_error_response(response)


def test_logprobs_combines_with_logit_bias(api_client):
    response = completion_request.send_chat_request(
        api_client, logprobs=True, logit_bias={"6002": -100}, max_tokens=128
    )
    assertion.assert_chat_completion_success(response)
