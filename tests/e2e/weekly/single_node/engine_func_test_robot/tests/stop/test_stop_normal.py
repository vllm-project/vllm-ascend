import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_stop_single_string(api_client, request, stream):
    """Test stop single string."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Tell a short story.'
        }],
        "stop": "5",
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
    # Check: response behavior is valid
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_stop_single_in_array(api_client, request, stream):
    """Test stop single in array."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Tell a short story.'
        }],
        "stop": ["E"],
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
def test_stop_multiple_strings(api_client, request, stream):
    """Test stop multiple strings."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Tell a short story.'
        }],
        "stop": ['response', 'response', 'response'],
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
def test_stop_long_sequence(api_client, request, stream):
    """Test stop long sequence."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Tell a short story.'
        }],
        "stop": ['response', 'response', 'response'],
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
def test_stop_with_special_chars(api_client, request, stream):
    """Test stop with special chars."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Tell a short story.'
        }],
        "stop": ["，", "。", "！"],
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
def test_stop_empty_array(api_client, request, stream):
    """Test stop empty array."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Tell a short story.'
        }],
        "stop": [],
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
def test_stop_without_setting(api_client, request, stream):
    """Test stop without setting."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Tell a short story.'
        }],
        # Check: response behavior is valid
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
def test_stop_unicode_chars(api_client, request, stream):
    """Test stop unicode chars."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Tell a short story.'
        }],
        "stop": ["😀", "🎉"],
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
def test_stop_large_number_of_sequences(api_client, request, stream):
    """Test stop large number of sequences."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Tell a short story.'
        }],
        "stop": ['response', 'response', 'response', 'response'],
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
def test_stop_partial_duplicate_elements(api_client, request, stream):
    """Test stop partial duplicate elements."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Tell a short story.'
        }],
        "stop": ["5", "5", "6"],  # Check: response behavior is valid
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
def test_stop_all_duplicate_elements(api_client, request, stream):
    """Test stop all duplicate elements."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Tell a short story.'
        }],
        "stop": ["E", "E", "E"],  # Check: response behavior is valid
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
def test_stop_duplicate_with_various_count(api_client, request, stream):
    """Test stop duplicate with various count."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Tell a short story.'
        }],
        # Check: duplicate stop words are accepted.
        "stop": ['response', 'response', 'response', 'response', 'response', 'response'],
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
def test_stop_two_identical_elements(api_client, request, stream):
    """Test stop two identical elements."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Tell a short story.'
        }],
        "stop": ['response', 'response'],  # Check: response behavior is valid
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
