import pytest
from ...utility import request_helper as helper
from ...utility import assertion


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("repetition_penalty", [1.0, 1.05, 1.1, 1.2, 1.5, 2.0], 
                         ids=["rp1.0", "rp1.05", "rp1.1", "rp1.2", "rp1.5", "rp2.0"])
def test_repetition_penalty_normal_values(api_client, request, stream, repetition_penalty):
    """Test repetition penalty normal values."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "repetition_penalty": repetition_penalty,
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
def test_repetition_penalty_one_no_effect(api_client, request, stream):
    """Test repetition penalty one no effect."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "repetition_penalty": 1.0,
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
def test_repetition_penalty_moderate_reduce_repetition(api_client, request, stream):
    """Test repetition penalty moderate reduce repetition."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
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
def test_repetition_penalty_strong_no_repeat(api_client, request, stream):
    """Test repetition penalty strong no repeat."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "repetition_penalty": 2.0,
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
def test_repetition_penalty_without_setting(api_client, request, stream):
    """Test repetition penalty without setting."""
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
def test_repetition_penalty_with_frequency_penalty(api_client, request, stream):
    """Test repetition penalty with frequency penalty."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "repetition_penalty": 1.2,
        "frequency_penalty": 0.5,
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
def test_repetition_penalty_with_presence_penalty(api_client, request, stream):
    """Test repetition penalty with presence penalty."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "repetition_penalty": 1.3,
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
def test_repetition_penalty_all_penalties(api_client, request, stream):
    """Test repetition penalty all penalties."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "repetition_penalty": 1.1,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.3,
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
def test_repetition_penalty_with_temperature(api_client, request, stream):
    """Test repetition penalty with temperature."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'List fruit names.'
        }],
        "repetition_penalty": 1.15,
        "temperature": 0.8,
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
