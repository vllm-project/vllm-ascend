import re
import json
import pytest
from ...utility import request_helper as helper
from ...utility import assertion


def parse_stream_choices_count(response_text):
    """Test parse stream choices count."""
    # Check: response behavior is valid
    # Check: response behavior is valid
    indexes = re.findall(r'"index"\s*:\s*(\d+)', response_text)
    if not indexes:
        return 0
    # Check: response behavior is valid
    return max(int(idx) for idx in indexes) + 1


def parse_stream_finish_reasons(response_text):
    """Test parse stream finish reasons."""
    # Check: finish_reason is valid
    finish_reasons = re.findall(r'"finish_reason"\s*:\s*"([^"]+)"', response_text)
    # Check: response behavior is valid
    return [fr for fr in finish_reasons if fr]


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_default_value(api_client, request, stream):
    """Test n default value."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Say hello.'
        }],
        "stream": stream,
        "max_tokens": 10
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check: response behavior is valid
    if stream:
        choices_count = parse_stream_choices_count(response.text)
    else:
        choices_count = len(response.json()["choices"])
    assert choices_count == 1, f"Expected 1 choice, got {choices_count}"

    # Check: finish_reason is valid
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_explicit_one(api_client, request, stream):
    """Test n explicit one."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Say hello.'
        }],
        "n": 1,
        "stream": stream,
        "max_tokens": 10
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check: response behavior is valid
    if stream:
        choices_count = parse_stream_choices_count(response.text)
    else:
        choices_count = len(response.json()["choices"])
    assert choices_count == 1, f"Expected 1 choice, got {choices_count}"

    # Check: finish_reason is valid
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_two_with_temperature(api_client, request, stream):
    """Test n two with temperature."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Say hello.'
        }],
        "n": 2,
        "temperature": 0.8,
        "stream": stream,
        "max_tokens": 10
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check: response behavior is valid
    if stream:
        choices_count = parse_stream_choices_count(response.text)
    else:
        choices_count = len(response.json()["choices"])
    assert choices_count == 2, f"Expected 2 choices, got {choices_count}"

    # Check: finish_reason is valid
    if stream:
        finish_reasons = parse_stream_finish_reasons(response.text)
    else:
        finish_reasons = [c["finish_reason"] for c in response.json()["choices"]]
    for i, finish_reason in enumerate(finish_reasons):
        assertion.assert_finish_reason_valid(finish_reason, f"choice[{i}] ")


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("n_value", [3, 5], ids=["n3", "n5"])
def test_n_multiple_values(api_client, request, stream, n_value):
    """Test n multiple values."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Say hello.'
        }],
        "n": n_value,
        "temperature": 0.9,
        "stream": stream,
        "max_tokens": 10
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check: response behavior is valid
    if stream:
        choices_count = parse_stream_choices_count(response.text)
    else:
        choices_count = len(response.json()["choices"])
    assert choices_count == n_value, f"Expected {n_value} choices, got {choices_count}"

    # Check: finish_reason is valid
    if stream:
        finish_reasons = parse_stream_finish_reasons(response.text)
    else:
        finish_reasons = [c["finish_reason"] for c in response.json()["choices"]]
    for i, finish_reason in enumerate(finish_reasons):
        assertion.assert_finish_reason_valid(finish_reason, f"choice[{i}] ")


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_null_uses_default(api_client, request, stream):
    """Test n null uses default."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Say hello.'
        }],
        "n": None,
        "stream": stream,
        "max_tokens": 10
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check: response behavior is valid
    if stream:
        choices_count = parse_stream_choices_count(response.text)
    else:
        choices_count = len(response.json()["choices"])
    assert choices_count == 1, f"Expected 1 choice (default), got {choices_count}"

    # Check: finish_reason is valid
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_one_with_greedy_sampling(api_client, request, stream):
    """Test n one with greedy sampling."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Say hello.'
        }],
        "n": 1,
        "temperature": 0,
        "stream": stream,
        "max_tokens": 10
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: streaming response contains [DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # Check: response behavior is valid
    if stream:
        choices_count = parse_stream_choices_count(response.text)
    else:
        choices_count = len(response.json()["choices"])
    assert choices_count == 1, f"Expected 1 choice, got {choices_count}"

    # Check: finish_reason is valid
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)
