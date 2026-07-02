from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


def test_temperature_low_conservative(api_client, request):
    """Test temperature low conservative."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        "temperature": 0.1,
        "stream": False,
        "max_tokens": 30,
    }

    results = []
    for i in range(3):
        response = helper.send_request(api_client, "/v1/chat/completions", request_body)

        assertion.assert_status_code_200(response, f"request #{i + 1}")
        content = response.json()["choices"][0]["message"]["content"]
        results.append(content)

    # Check: response behavior is valid
    for i, result in enumerate(results):
        assert result is not None, f"result #{i + 1} is None"
        assert len(result) > 0, f"result #{i + 1} is empty"


def test_temperature_medium_balanced(api_client, request):
    """Test temperature medium balanced."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        "temperature": 1.0,
        "stream": False,
        "max_tokens": 30,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: response behavior is valid
    content = response.json()["choices"][0]["message"]["content"]
    assert content is not None, "response should be valid"
    assert len(content) > 0, "response should be valid"

    # Check: finish_reason is valid
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_temperature_high_diverse(api_client, request):
    """Test temperature high diverse."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        "temperature": 2.0,
        "stream": False,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: response behavior is valid
    content = response.json()["choices"][0]["message"]["content"]
    assert content is not None, "response should be valid"
    assert len(content) > 0, "response should be valid"

    # Check: finish_reason is valid
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_temperature_with_top_p_combination(api_client, request):
    """Test temperature with top p combination."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the weather."}],
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 50,
        "stream": False,
        "max_tokens": 30,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: status code is 200
    assertion.assert_status_code_200(response)

    # Check: response behavior is valid
    content = response.json()["choices"][0]["message"]["content"]
    assert content is not None, "response should be valid"
    assert len(content) > 0, "response should be valid"

    # Check: finish_reason is valid
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)
