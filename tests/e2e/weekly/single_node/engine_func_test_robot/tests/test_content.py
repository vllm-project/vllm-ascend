import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion, completion_request


@pytest.mark.parametrize(
    "content",
    [
        "Hello.",
        "",
        [],
        [{"type": "text", "text": "Hello."}],
        [{"type": "text", "text": "Hello."}, {"type": "text", "text": "Continue."}],
        'Line1\nLine2\tTabbed"Quoted"\\Backslash',
        "A" * 5000,
    ],
    ids=["string", "empty_string", "empty_array", "text_object", "many_text_objects", "escaped", "long_text"],
)
def test_content_accepts_supported_shapes(api_client, content):
    response = completion_request.send_chat_request(api_client, messages=[{"role": "user", "content": content}])
    assertion.assert_chat_completion_success(response)


def test_content_accepts_streaming_request(api_client):
    response = completion_request.send_chat_request(
        api_client, messages=[{"role": "user", "content": [{"type": "text", "text": "Hello."}]}], stream=True
    )
    assertion.assert_chat_completion_success(response, stream=True)


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user"},
        {"role": "user", "content": None},
        {"role": "user", "content": "   \t\n\n   "},
        {"role": "user", "content": "Hello\x00World"},
        {"role": "user", "content": [{"type": "TEXT", "text": "Hello."}]},
        {"role": "user", "content": [{"type": "text", "text": "Hello.", "extra_field": "extra_value"}]},
    ],
    ids=["missing", "null", "whitespace", "null_byte", "type_case", "extra_field"],
)
def test_content_accepts_or_rejects_implementation_dependent_boundaries(api_client, message):
    response = completion_request.send_chat_request(api_client, messages=[message])
    assertion.assert_status_code_200_or_400(response)


@pytest.mark.parametrize(
    "content",
    [
        12345,
        {"text": "hello", "extra": "data"},
        True,
        ["invalid", "array", "format"],
        [{"text": "Hello."}],
        [{"type": "text"}],
        [{"type": "invalid_type", "text": "Hello."}],
        [{"type": "text", "text": None}],
        [{"type": "text", "text": 12345}],
    ],
    ids=[
        "integer",
        "object",
        "boolean",
        "string_array",
        "missing_type",
        "missing_text",
        "invalid_type",
        "text_null",
        "text_integer",
    ],
)
def test_content_rejects_invalid_shapes(api_client, content):
    response = completion_request.send_chat_request(api_client, messages=[{"role": "user", "content": content}])
    assertion.assert_validation_error_response(response)
