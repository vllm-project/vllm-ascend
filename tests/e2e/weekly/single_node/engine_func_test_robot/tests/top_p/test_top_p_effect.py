from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


def test_top_p_zero_deterministic(api_client):
    """top_p=0.0 is outside the supported range and should return error code 400."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Name the capital of China."}],
        "top_p": 0.0,
        "temperature": 0.0,
        "stream": False,
        "max_tokens": 10,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: error code is 400
    assertion.assert_error_code_400(response)


def test_top_p_small_conservative(api_client):
    """top_p=0.1 should produce valid non-empty responses."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Recommend one Chinese poem."}],
        "top_p": 0.1,
        "temperature": 0.7,
        "stream": False,
        "max_tokens": 30,
    }

    results = []
    for _ in range(3):
        response = helper.send_request(api_client, "/v1/chat/completions", request_body)
        results.append(assertion.assert_non_empty_message_content(response))

    # Check: all responses contain generated text
    assert len(results) == 3


def test_top_p_medium_balanced(api_client):
    """top_p=0.7 should produce valid non-empty responses."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe the sea in one sentence."}],
        "top_p": 0.7,
        "temperature": 0.8,
        "stream": False,
        "max_tokens": 30,
    }

    results = []
    for _ in range(5):
        response = helper.send_request(api_client, "/v1/chat/completions", request_body)
        results.append(assertion.assert_non_empty_message_content(response))

    # Check: all responses contain generated text
    assert len(results) == 5


def test_top_p_large_full_vocab(api_client):
    """top_p=1.0 should use the full probability distribution and produce a valid response."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe winter in three words."}],
        "top_p": 1.0,
        "temperature": 0.9,
        "stream": False,
        "max_tokens": 30,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: response contains generated text and finish_reason is valid
    assertion.assert_non_empty_message_content(response)
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_top_p_combined_with_top_k(api_client):
    """top_p and top_k should work together and produce a valid response."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe sunset in one sentence."}],
        "top_p": 0.85,
        "top_k": 50,
        "temperature": 0.75,
        "stream": False,
        "max_tokens": 30,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: response contains generated text and finish_reason is valid
    assertion.assert_non_empty_message_content(response)
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_top_p_temperature_interaction(api_client):
    """A low temperature with low top_p should still produce valid non-empty responses."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Name the largest province in China."}],
        "top_p": 0.3,
        "temperature": 0.2,
        "stream": False,
        "max_tokens": 10,
    }

    results = []
    for _ in range(3):
        response = helper.send_request(api_client, "/v1/chat/completions", request_body)
        results.append(assertion.assert_non_empty_message_content(response))

    # Check: all responses contain generated text
    assert len(results) == 3
