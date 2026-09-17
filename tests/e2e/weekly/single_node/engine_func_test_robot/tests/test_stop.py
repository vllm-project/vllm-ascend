import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion, completion_request


@pytest.mark.parametrize("stop", [".", [".", "!"], [], ["??"]], ids=["string", "list", "empty", "unicode"])
def test_stop_accepts_supported_shapes(api_client, stop):
    response = completion_request.send_chat_request(api_client, stop=stop, max_tokens=64)
    assertion.assert_chat_completion_success(response)


def test_stop_accepts_streaming_request(api_client):
    response = completion_request.send_chat_request(api_client, stop=["."], max_tokens=64, stream=True)
    assertion.assert_chat_completion_success(response, stream=True)


@pytest.mark.parametrize("stop", [1, [["."]], {"value": "."}])
def test_stop_rejects_invalid_values(api_client, stop):
    response = completion_request.send_chat_request(api_client, stop=stop)
    assertion.assert_validation_error_response(response)


@pytest.mark.parametrize("stop", [None, [str(i) for i in range(100)]], ids=["default_null", "large_list"])
def test_stop_accepts_default_or_large_values(api_client, stop):
    response = completion_request.send_chat_request(api_client, stop=stop)
    assertion.assert_success_or_validation_error(response)


def test_stop_accepts_duplicate_sequences(api_client):
    response = completion_request.send_chat_request(api_client, stop=["END", "END"], max_tokens=64)
    assertion.assert_chat_completion_success(response)
