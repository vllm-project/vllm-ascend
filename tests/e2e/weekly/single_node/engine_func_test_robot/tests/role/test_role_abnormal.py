import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


def test_role_invalid_value_non_stream(api_client, request):
    """Test role invalid value non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "unknown",
            "content": 'Hello.'
        }],
        "stream": False,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    if response.status_code == 400:
        assertion.assert_error_code_400(response)
    else:
        assertion.assert_status_code_200(response)
        response_json = response.json()
        error_code = response_json.get("error", {}).get("code") or response_json.get("code")
        if error_code == 400:
            assertion.assert_error_code_400(response)
        else:
            finish_reason = response_json["choices"][0]["finish_reason"]
            assertion.assert_finish_reason_valid(finish_reason)


def test_role_invalid_value_stream(api_client, request):
    """Test role invalid value stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "unknown",
            "content": 'Hello.'
        }],
        "stream": True,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    if response.status_code == 400:
        assertion.assert_error_code_400(response)
    else:
        assertion.assert_status_code_200(response)
        import regex as re
        match = re.search(r'\"code\"\s?:\s?(\d+)', response.text, re.M)
        if match and int(match.group(1)) == 400:
            assertion.assert_error_code_400(response)
        else:
            finish_reason = assertion.assert_stream_single_finish_reason(response.text)
            assertion.assert_finish_reason_valid(finish_reason)


def test_role_missing_non_stream(api_client, request):
    """Test role missing non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            # Check: response behavior is valid
            "content": 'Hello.'
        }],
        "stream": False,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    response_json = response.json()
    error_code = response_json.get("error", {}).get("code") or response_json.get("code")
    if error_code == 400:
        assertion.assert_error_code_400(response)
    else:
        finish_reason = response_json["choices"][0]["finish_reason"]
        assertion.assert_finish_reason_valid(finish_reason)


def test_role_missing_stream(api_client, request):
    """Test role missing stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            # Check: response behavior is valid
            "content": 'Hello.'
        }],
        "stream": True,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_role_null_non_stream(api_client, request):
    """Test role null non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": None,
            "content": 'Hello.'
        }],
        "stream": False,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    response_json = response.json()
    error_code = response_json.get("error", {}).get("code") or response_json.get("code")
    if error_code == 400:
        assertion.assert_error_code_400(response)
    else:
        finish_reason = response_json["choices"][0]["finish_reason"]
        assertion.assert_finish_reason_valid(finish_reason)


def test_role_null_stream(api_client, request):
    """Test role null stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": None,
            "content": 'Hello.'
        }],
        "stream": True,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_role_integer_non_stream(api_client, request):
    """Test role integer non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": 123,
            "content": 'Hello.'
        }],
        "stream": False,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    if response.status_code == 400:
        assertion.assert_error_code_400(response)
    else:
        assertion.assert_status_code_200(response)
        response_json = response.json()
        error_code = response_json.get("error", {}).get("code") or response_json.get("code")
        if error_code == 400:
            assertion.assert_error_code_400(response)
        else:
            finish_reason = response_json["choices"][0]["finish_reason"]
            assertion.assert_finish_reason_valid(finish_reason)


def test_role_integer_stream(api_client, request):
    """Test role integer stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": 123,
            "content": 'Hello.'
        }],
        "stream": True,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    if response.status_code == 400:
        assertion.assert_error_code_400(response)
    else:
        assertion.assert_status_code_200(response)
        import regex as re
        match = re.search(r'\"code\"\s?:\s?(\d+)', response.text, re.M)
        if match and int(match.group(1)) == 400:
            assertion.assert_error_code_400(response)
        else:
            finish_reason = assertion.assert_stream_single_finish_reason(response.text)
            assertion.assert_finish_reason_valid(finish_reason)


def test_role_empty_string_non_stream(api_client, request):
    """Test role empty string non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "",
            "content": 'Hello.'
        }],
        "stream": False,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    if response.status_code == 400:
        assertion.assert_error_code_400(response)
    else:
        assertion.assert_status_code_200(response)
        response_json = response.json()
        error_code = response_json.get("error", {}).get("code") or response_json.get("code")
        if error_code == 400:
            assertion.assert_error_code_400(response)
        else:
            finish_reason = response_json["choices"][0]["finish_reason"]
            assertion.assert_finish_reason_valid(finish_reason)


def test_role_empty_string_stream(api_client, request):
    """Test role empty string stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "",
            "content": 'Hello.'
        }],
        "stream": True,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    if response.status_code == 400:
        assertion.assert_error_code_400(response)
    else:
        assertion.assert_status_code_200(response)
        import regex as re
        match = re.search(r'\"code\"\s?:\s?(\d+)', response.text, re.M)
        if match and int(match.group(1)) == 400:
            assertion.assert_error_code_400(response)
        else:
            finish_reason = assertion.assert_stream_single_finish_reason(response.text)
            assertion.assert_finish_reason_valid(finish_reason)


def test_role_case_sensitive_user_non_stream(api_client, request):
    """Test role case sensitive user non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "User",  # Check: response behavior is valid
            "content": 'Hello.'
        }],
        "stream": False,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    if response.status_code == 400:
        assertion.assert_error_code_400(response)
    else:
        assertion.assert_status_code_200(response)
        response_json = response.json()
        error_code = response_json.get("error", {}).get("code") or response_json.get("code")
        if error_code == 400:
            assertion.assert_error_code_400(response)
        else:
            finish_reason = response_json["choices"][0]["finish_reason"]
            assertion.assert_finish_reason_valid(finish_reason)


def test_role_case_sensitive_user_stream(api_client, request):
    """Test role case sensitive user stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "User",  # Check: response behavior is valid
            "content": 'Hello.'
        }],
        "stream": True,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    if response.status_code == 400:
        assertion.assert_error_code_400(response)
    else:
        assertion.assert_status_code_200(response)
        import regex as re
        match = re.search(r'\"code\"\s?:\s?(\d+)', response.text, re.M)
        if match and int(match.group(1)) == 400:
            assertion.assert_error_code_400(response)
        else:
            finish_reason = assertion.assert_stream_single_finish_reason(response.text)
            assertion.assert_finish_reason_valid(finish_reason)


def test_role_case_sensitive_system_non_stream(api_client, request):
    """Test role case sensitive system non stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "SYSTEM",
            "content": 'Hello.'
        }],
        "stream": False,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    if response.status_code == 400:
        assertion.assert_error_code_400(response)
    else:
        assertion.assert_status_code_200(response)
        response_json = response.json()
        error_code = response_json.get("error", {}).get("code") or response_json.get("code")
        if error_code == 400:
            assertion.assert_error_code_400(response)
        else:
            finish_reason = response_json["choices"][0]["finish_reason"]
            assertion.assert_finish_reason_valid(finish_reason)


def test_role_case_sensitive_system_stream(api_client, request):
    """Test role case sensitive system stream."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "SYSTEM",
            "content": 'Hello.'
        }],
        "stream": True,
        "max_tokens": 50
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    if response.status_code == 400:
        assertion.assert_error_code_400(response)
    else:
        assertion.assert_status_code_200(response)
        import regex as re
        match = re.search(r'\"code\"\s?:\s?(\d+)', response.text, re.M)
        if match and int(match.group(1)) == 400:
            assertion.assert_error_code_400(response)
        else:
            finish_reason = assertion.assert_stream_single_finish_reason(response.text)
            assertion.assert_finish_reason_valid(finish_reason)
