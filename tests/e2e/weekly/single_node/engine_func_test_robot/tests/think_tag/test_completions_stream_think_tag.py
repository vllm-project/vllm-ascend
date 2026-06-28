import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


@pytest.mark.skip(reason='Deprecated start marker.')
def test_completions_stream_no_think_tag(api_client, request):
    """Test completions stream no think tag."""
    request_body = {
        "model": "auto",
        "prompt": 'Introduce yourself.',
        "stream": True,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/completions", request_body)

    # Check: response behavior is valid
    if request.config.getoption("--thinkTagOutput").strip().lower() == "true":
        assertion.assert_no_think_tag(response.text, 'response')
