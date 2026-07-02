import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize(
    "presence_penalty",
    [-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0],
    ids=["pp-2.0", "pp-1.0", "pp0", "pp0.5", "pp1.0", "pp1.5", "pp2.0"],
)
def test_presence_penalty_normal_values(api_client, request, stream, presence_penalty):
    """Test presence penalty normal values."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "List fruit names."}],
        "presence_penalty": presence_penalty,
        "stream": stream,
        "max_tokens": 100,
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
def test_presence_penalty_zero_no_effect(api_client, request, stream):
    """Test presence penalty zero no effect."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "List fruit names."}],
        "presence_penalty": 0.0,
        "stream": stream,
        "max_tokens": 100,
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
def test_presence_penalty_positive_diversify_topics(api_client, request, stream):
    """Test presence penalty positive diversify topics."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "List fruit names."}],
        "presence_penalty": 1.5,
        "stream": stream,
        "max_tokens": 150,
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
def test_presence_penalty_negative_stay_on_topic(api_client, request, stream):
    """Test presence penalty negative stay on topic."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "List fruit names."}],
        "presence_penalty": -1.5,
        "stream": stream,
        "max_tokens": 100,
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
def test_presence_penalty_without_setting(api_client, request, stream):
    """Test presence penalty without setting."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "List fruit names."}],
        # Check: response behavior is valid
        "stream": stream,
        "max_tokens": 100,
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
def test_presence_penalty_with_frequency_penalty(api_client, request, stream):
    """Test presence penalty with frequency penalty."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "List fruit names."}],
        "presence_penalty": 1.0,
        "frequency_penalty": 1.0,
        "stream": stream,
        "max_tokens": 100,
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
def test_presence_penalty_highest_value(api_client, request, stream):
    """Test presence penalty highest value."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "List fruit names."}],
        "presence_penalty": 2.0,
        "stream": stream,
        "max_tokens": 100,
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
def test_presence_penalty_lowest_value(api_client, request, stream):
    """Test presence penalty lowest value."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "List fruit names."}],
        "presence_penalty": -2.0,
        "stream": stream,
        "max_tokens": 100,
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
