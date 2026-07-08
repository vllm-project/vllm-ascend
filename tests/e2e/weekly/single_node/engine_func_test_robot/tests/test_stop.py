import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import completion_request


@pytest.mark.parametrize("stop", [".", [".", "!"], [], ["??"]], ids=["string", "list", "empty", "unicode"])
def test_stop_accepts_supported_shapes(api_client, stop):
    response = completion_request.send_chat_request(api_client, stop=stop, max_tokens=64)
    assertion.assert_chat_completion_success(response)


def test_stop_accepts_streaming_request(api_client):
    response = completion_request.send_chat_request(api_client, stop=["."], max_tokens=64, stream=True)
    assertion.assert_chat_completion_success(response, stream=True)


@pytest.mark.parametrize("stop", [1, None, [["."]], {"value": "."}, [str(i) for i in range(100)]])
def test_stop_rejects_invalid_values(api_client, stop):
    response = completion_request.send_chat_request(api_client, stop=stop)
    assertion.assert_validation_error_response(response)


def test_stop_accepts_duplicate_sequences(api_client):
    response = completion_request.send_chat_request(api_client, stop=["END", "END"], max_tokens=64)
    assertion.assert_chat_completion_success(response)
