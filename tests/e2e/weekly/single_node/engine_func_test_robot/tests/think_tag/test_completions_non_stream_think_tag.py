import pytest
from ...utility import request_helper as helper
from ...utility import assertion


@pytest.mark.skip(reason='Deprecated start marker.')
def test_completions_non_stream_no_think_tag(api_client, request):
    """Test completions non stream no think tag."""
    request_body = {
        "model": "auto",
        "prompt": 'Introduce yourself.',
        "stream": False,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/completions", request_body)

    # Check: response behavior is valid
    if request.config.getoption("--thinkTagOutput").strip().lower() == "true":
        assertion.assert_no_think_tag(response.content.decode("utf-8"), 'response should be valid')