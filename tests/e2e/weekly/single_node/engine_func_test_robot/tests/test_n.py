import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import completion_request


def _assert_choice_count(response, expected):
    assert len(response.json()["choices"]) == expected


@pytest.mark.parametrize("n", [1, 2], ids=["single", "multiple"])
def test_n_accepts_supported_values(api_client, n):
    response = completion_request.send_chat_request(api_client, n=n, temperature=0.7)
    assertion.assert_chat_completion_success(response)
    _assert_choice_count(response, n)


def test_n_omitted_uses_default_single_choice(api_client):
    response = completion_request.send_chat_request(api_client)
    assertion.assert_chat_completion_success(response)
    _assert_choice_count(response, 1)


@pytest.mark.parametrize("n", [0, -1, 1.5, "2", [], {}, False])
def test_n_rejects_invalid_values(api_client, n):
    response = completion_request.send_chat_request(api_client, n=n)
    assertion.assert_validation_error_response(response)


def test_n_rejects_multiple_choices_with_greedy_sampling(api_client):
    response = completion_request.send_chat_request(api_client, n=2, temperature=0.0)
    assertion.assert_validation_error_response(response)
