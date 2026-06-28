import pytest
from ...utility import request_helper as helper
from ...utility import assertion


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_zero(api_client, request, stream):
    """Test n zero."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Say hello.'
        }],
        "n": 0,
        "stream": stream,
        "max_tokens": 10
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("n_value", [-1, -10, -100], ids=["n-1", "n-10", "n-100"])
def test_n_negative(api_client, request, stream, n_value):
    """Test n negative."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Say hello.'
        }],
        "n": n_value,
        "stream": stream,
        "max_tokens": 10
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_string_invalid(api_client, request, stream):
    """Test n string invalid."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Say hello.'
        }],
        "n": "abc",
        "stream": stream,
        "max_tokens": 10
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_array(api_client, request, stream):
    """Test n array."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Say hello.'
        }],
        "n": [1],
        "stream": stream,
        "max_tokens": 10
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_object(api_client, request, stream):
    """Test n object."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Say hello.'
        }],
        "n": {},
        "stream": stream,
        "max_tokens": 10
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_boolean_false(api_client, request, stream):
    """Test n boolean false."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Say hello.'
        }],
        "n": False,
        "stream": stream,
        "max_tokens": 10
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_two_with_greedy_sampling(api_client, request, stream):
    """Test n two with greedy sampling."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Say hello.'
        }],
        "n": 2,
        "temperature": 0,
        "stream": stream,
        "max_tokens": 10
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("n_value", [3, 5], ids=["n3", "n5"])
def test_n_greater_than_one_with_greedy_sampling(api_client, request, stream, n_value):
    """Test n greater than one with greedy sampling."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Say hello.'
        }],
        "n": n_value,
        "temperature": 0,
        "stream": stream,
        "max_tokens": 10
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_float_less_than_one(api_client, request, stream):
    """Test n float less than one."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Say hello.'
        }],
        "n": 0.5,
        "stream": stream,
        "max_tokens": 10
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_float_zero_point_nine(api_client, request, stream):
    """Test n float zero point nine."""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": 'Say hello.'
        }],
        "n": 0.9,
        "stream": stream,
        "max_tokens": 10
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code and error code are 400
    assertion.assert_status_code_400(response)
