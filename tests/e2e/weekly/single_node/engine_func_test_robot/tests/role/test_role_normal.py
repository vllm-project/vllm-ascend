import pytest
from ...utility import request_helper as helper
from ...utility import assertion


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_role_single_user(api_client, request, stream):
    """Test role single user."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Hello.'
        }],
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
def test_role_system_user(api_client, request, stream):
    """Test role system user."""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "system",
                "content": 'Hello.'
            },
            {
                "role": "user",
                "content": 'Hello.'
            }
        ],
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
def test_role_multi_turn(api_client, request, stream):
    """Test role multi turn."""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": 'Hello.'
            },
            {
                "role": "assistant",
                "content": 'Hello.'
            },
            {
                "role": "user",
                "content": 'Hello.'
            }
        ],
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
def test_role_system_user_assistant_user(api_client, request, stream):
    """Test role system user assistant user."""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "system",
                "content": 'Hello.'
            },
            {
                "role": "user",
                "content": 'Hello.'
            },
            {
                "role": "assistant",
                "content": 'Hello.'
            },
            {
                "role": "user",
                "content": 'Hello.'
            }
        ],
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
def test_role_only_system(api_client, request, stream):
    """Test role only system."""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "system",
                "content": 'Hello.'
            }
        ],
        "stream": stream,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: error code is not 500
    assertion.assert_error_code_not_500(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_role_multiple_system(api_client, request, stream):
    """Test role multiple system."""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "system",
                "content": 'Hello.'
            },
            {
                "role": "system",
                "content": 'Hello.'
            },
            {
                "role": "user",
                "content": 'Hello.'
            }
        ],
        "stream": stream,
        "max_tokens": 100
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: error code is not 500
    assertion.assert_error_code_not_500(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_role_long_history(api_client, request, stream):
    """Test role long history."""
    # Check: response behavior is valid
    messages = []
    for i in range(10):
        messages.append({
            "role": "user",
            "content": f"Conversation turn {i + 1}"
        })
        messages.append({
            "role": "assistant",
            "content": f"Conversation turn {i + 1}"
        })
    messages.append({
        "role": "user",
        "content": 'Hello.'
    })

    request_body = {
        "model": "auto",
        "messages": messages,
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
def test_role_empty_content(api_client, request, stream):
    """Test role empty content."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": ""
        }],
        "stream": stream,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    # Check: response behavior is valid
    assertion.assert_status_code_200(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_role_only_assistant(api_client, request, stream):
    """Test role only assistant."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "assistant",
            "content": 'Hello.'
        }],
        "stream": stream,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: error code is not 500
    assertion.assert_error_code_not_500(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_role_user_assistant_ending(api_client, request, stream):
    """Test role user assistant ending."""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": 'Hello.'
            },
            {
                "role": "assistant",
                "content": 'Hello.'
            }
        ],
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
def test_role_system_user_assistant_ending(api_client, request, stream):
    """Test role system user assistant ending."""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "system",
                "content": 'Hello.'
            },
            {
                "role": "user",
                "content": 'Hello.'
            },
            {
                "role": "assistant",
                "content": 'Hello.'
            }
        ],
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
def test_role_multi_turn_assistant_ending(api_client, request, stream):
    """Test role multi turn assistant ending."""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": 'Hello.'
            },
            {
                "role": "assistant",
                "content": 'Hello.'
            },
            {
                "role": "user",
                "content": 'Hello.'
            },
            {
                "role": "assistant",
                "content": 'Hello.'
            }
        ],
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
def test_role_long_history_assistant_ending(api_client, request, stream):
    """Test role long history assistant ending."""
    # Check: response behavior is valid
    messages = []
    for i in range(5):
        messages.append({
            "role": "user",
            "content": f"Conversation turn {i + 1}"
        })
        messages.append({
            "role": "assistant",
            "content": f"Conversation turn {i + 1}"
        })

    request_body = {
        "model": "auto",
        "messages": messages,
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
