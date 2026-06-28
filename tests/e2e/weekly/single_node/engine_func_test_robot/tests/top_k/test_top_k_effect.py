import pytest_check as check

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import (
    request_helper as helper,
)


def _assert_non_empty_content(response):
    assertion.assert_status_code_200(response)
    content = response.json()["choices"][0]["message"]["content"]
    check.is_not_none(content, "response content is None")
    check.is_true(len(content) > 0, "response content is empty")
    return content


def test_top_k_small_conservative(api_client):
    """top_k=5 should produce valid non-empty responses."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe spring in three words."}],
        "top_k": 5,
        "temperature": 1.0,
        "stream": False,
        "max_tokens": 20,
    }

    results = []
    for _ in range(3):
        response = helper.send_request(api_client, "/v1/chat/completions", request_body)
        results.append(_assert_non_empty_content(response))

    # Check: all responses contain generated text
    check.equal(len(results), 3)


def test_top_k_large_diverse(api_client):
    """top_k=100 should produce valid non-empty responses."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Describe sunset in one sentence."}],
        "top_k": 100,
        "temperature": 0.9,
        "stream": False,
        "max_tokens": 30,
    }

    results = []
    for _ in range(5):
        response = helper.send_request(api_client, "/v1/chat/completions", request_body)
        results.append(_assert_non_empty_content(response))

    # Check: all responses contain generated text
    check.equal(len(results), 5)


def test_top_k_disabled_full_vocab(api_client):
    """top_k=-1 should disable top-k filtering and produce a valid response."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Write one short greeting."}],
        "top_k": -1,
        "temperature": 0.8,
        "stream": False,
        "max_tokens": 30,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: response contains generated text
    _assert_non_empty_content(response)


def test_top_k_combined_with_top_p_priority(api_client):
    """top_k and top_p should work together and produce a valid response."""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Summarize AI changing the world in one sentence."}],
        "top_k": 50,
        "top_p": 0.85,
        "temperature": 0.7,
        "stream": False,
        "max_tokens": 50,
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # Check: response contains generated text and finish_reason is valid
    _assert_non_empty_content(response)
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)
