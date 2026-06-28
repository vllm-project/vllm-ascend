import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


def test_chat_non_stream_think_tag_complete(api_client, request):
    """Test chat non stream think tag complete."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Introduce yourself.'
        }],
        "stream": False,
        "max_tokens": 5120000
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: response behavior is valid
    if request.config.getoption("--thinkTagOutput").strip().lower() == "true":
        assertion.assert_think_tag_present(response.content.decode("utf-8"), 'response should be valid')
