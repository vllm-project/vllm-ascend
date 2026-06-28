import random
import pytest
from ...utility import request_helper as helper
from ...utility import assertion


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("top_logprobs", [random.randint(1, 19), 0, 20], ids=["in_1_19", "0", "20"])
def test_logprobs_normal_boundary(api_client, request, stream, top_logprobs):
    """Test logprobs normal boundary."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Say hello.'
        }],
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check: response behavior is valid
    assertion.assert_top_logprobs_count(response, top_logprobs)


def test_logprobs_exceed_20_non_stream(api_client, request):
    """Test logprobs exceed 20 non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Say hello.'
        }],
        "logprobs": True,
        "top_logprobs": 21,
        "stream": False,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_logprobs_exceed_20_stream(api_client, request):
    """Test logprobs exceed 20 stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Say hello.'
        }],
        "logprobs": True,
        "top_logprobs": 21,
        "stream": True,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_200(response)
    assertion.assert_error_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_logprobs_with_logit_bias(api_client, request, stream):
    """Test logprobs with logit bias."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Say hello.'
        }],
        "logprobs": True,
        "logit_bias": {"6002": -100},
        "stream": stream,
        "max_tokens": 5120
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check: finish_reason is valid
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)