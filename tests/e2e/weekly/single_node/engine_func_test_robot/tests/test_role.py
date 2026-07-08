import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion, completion_request


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "Hello."}],
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Introduce yourself."}],
        [
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Hi."},
            {"role": "user", "content": "Continue."},
        ],
        [{"role": "system", "content": "Be brief."}],
    ],
    ids=["user", "system_user", "multi_turn", "system_only"],
)
def test_role_accepts_supported_message_sequences(api_client, messages):
    response = completion_request.send_chat_request(api_client, messages=messages)
    assertion.assert_chat_completion_success(response)


def test_role_accepts_streaming_multi_turn_request(api_client):
    messages = [
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": "Hi."},
        {"role": "user", "content": "Again."},
    ]
    response = completion_request.send_chat_request(api_client, messages=messages, stream=True)
    assertion.assert_chat_completion_success(response, stream=True)


@pytest.mark.parametrize(
    "message",
    [
        {"role": "invalid", "content": "Hello."},
        {"content": "Hello."},
        {"role": None, "content": "Hello."},
        {"role": 1, "content": "Hello."},
        {"role": "", "content": "Hello."},
        {"role": "User", "content": "Hello."},
    ],
)
def test_role_rejects_invalid_role_values(api_client, message):
    response = completion_request.send_chat_request(api_client, messages=[message])
    assertion.assert_validation_error_response(response)


def test_system_instruction_can_guide_response(api_client):
    messages = [{"role": "system", "content": "Always answer with OK."}, {"role": "user", "content": "Reply now."}]
    response = completion_request.send_chat_request(api_client, messages=messages, max_tokens=16)
    assertion.assert_chat_completion_success(response)
