import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize(
    "temperature",
    [0.0, 0.1, 0.5, 0.8, 1.0, 1.5, 2.0],
    ids=["t0", "t0.1", "t0.5", "t0.8", "t1.0", "t1.5", "t2.0"],
)
def test_temperature_normal_values(api_client, request, stream, temperature):
    """Test temperature normal values."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        "temperature": temperature,
        "stream": stream,
        "max_tokens": 50,
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
def test_temperature_zero_deterministic(api_client, request, stream):
    """Test temperature zero deterministic."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        "temperature": 0.0,
        "stream": stream,
        "max_tokens": 10,
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
def test_temperature_high_creative(api_client, request, stream):
    """Test temperature high creative."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        "temperature": 2.0,
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
def test_temperature_with_top_p(api_client, request, stream):
    """Test temperature with top p."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": stream,
        "max_tokens": 50,
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
def test_temperature_without_setting(api_client, request, stream):
    """Test temperature without setting."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        # Check: response behavior is valid
        "stream": stream,
        "max_tokens": 50,
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
def test_temperature_balanced_default(api_client, request, stream):
    """Test temperature balanced default."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        "temperature": 1.0,
        "stream": stream,
        "max_tokens": 50,
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
