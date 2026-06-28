import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("frequency_penalty", [-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0],
                         ids=["fp-2.0", "fp-1.0", "fp0", "fp0.5", "fp1.0", "fp1.5", "fp2.0"])
def test_frequency_penalty_normal_values(api_client, request, stream, frequency_penalty):
    """Test frequency penalty normal values."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "frequency_penalty": frequency_penalty,
        "stream": stream,
        "max_tokens": 100
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


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_frequency_penalty_zero_no_effect(api_client, request, stream):
    """Test frequency penalty zero no effect."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "frequency_penalty": 0.0,
        "stream": stream,
        "max_tokens": 100
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


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_frequency_penalty_positive_reduce_repetition(api_client, request, stream):
    """Test frequency penalty positive reduce repetition."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "frequency_penalty": 1.5,
        "stream": stream,
        "max_tokens": 100
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


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_frequency_penalty_negative_encourage_repetition(api_client, request, stream):
    """Test frequency penalty negative encourage repetition."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "frequency_penalty": -1.5,
        "stream": stream,
        "max_tokens": 50
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


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_frequency_penalty_without_setting(api_client, request, stream):
    """Test frequency penalty without setting."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        # Check: response behavior is valid
        "stream": stream,
        "max_tokens": 100
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


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_frequency_penalty_with_presence_penalty(api_client, request, stream):
    """Test frequency penalty with presence penalty."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "frequency_penalty": 1.0,
        "presence_penalty": 0.5,
        "stream": stream,
        "max_tokens": 100
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


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_frequency_penalty_with_repetition_penalty(api_client, request, stream):
    """Test frequency penalty with repetition penalty."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "frequency_penalty": 0.8,
        "repetition_penalty": 1.2,
        "stream": stream,
        "max_tokens": 100
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


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_frequency_penalty_all_penalties(api_client, request, stream):
    """Test frequency penalty all penalties."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5,
        "repetition_penalty": 1.1,
        "stream": stream,
        "max_tokens": 100
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
