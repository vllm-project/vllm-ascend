from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


def test_temperature_negative_non_stream(api_client, request):
    """Test temperature negative non stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        "temperature": -0.5,
        "stream": False,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_temperature_negative_stream(api_client, request):
    """Test temperature negative stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        "temperature": -0.5,
        "stream": True,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_error_code_400(response)


def test_temperature_exceed_upper_limit_non_stream(api_client, request):
    """Test temperature exceed upper limit non stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        "temperature": 100.0,  # Check: response behavior is valid
        "stream": False,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: finish_reason is valid
    assertion.assert_status_code_200(response)
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_temperature_exceed_upper_limit_stream(api_client, request):
    """Test temperature exceed upper limit stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        "temperature": 100.0,  # Check: response behavior is valid
        "stream": True,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: finish_reason is valid
    assertion.assert_status_code_200(response)
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    assertion.assert_finish_reason_valid(finish_reason)


def test_temperature_string_non_stream(api_client, request):
    """Test temperature string non stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        "temperature": "1",
        "stream": False,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: finish_reason is valid
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_temperature_string_stream(api_client, request):
    """Test temperature string stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        "temperature": "0.7",
        "stream": True,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: finish_reason is valid
    assertion.assert_status_code_200(response)
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    assertion.assert_finish_reason_valid(finish_reason)


def test_temperature_null_non_stream(api_client, request):
    """Test temperature null non stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        "temperature": None,
        "stream": False,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: finish_reason is valid
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_temperature_null_stream(api_client, request):
    """Test temperature null stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        "temperature": None,
        "stream": True,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: finish_reason is valid
    assertion.assert_status_code_200(response)
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    assertion.assert_finish_reason_valid(finish_reason)


def test_temperature_array_non_stream(api_client, request):
    """Test temperature array non stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        "temperature": [0.5, 0.7],
        "stream": False,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_temperature_array_stream(api_client, request):
    """Test temperature array stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        "temperature": [0.5, 0.7],
        "stream": True,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_temperature_object_non_stream(api_client, request):
    """Test temperature object non stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        "temperature": {"value": 0.7},
        "stream": False,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_temperature_object_stream(api_client, request):
    """Test temperature object stream."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        "temperature": {"value": 0.7},
        "stream": True,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)
