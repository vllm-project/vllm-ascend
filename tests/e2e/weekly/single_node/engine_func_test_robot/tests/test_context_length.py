import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion, completion_request

CONTEXT_OVERFLOW_MARGIN = 4096


def _context_prompt(request, scale):
    max_length = int(request.config.getoption("--maxModelLength"))
    base_text = "Input: hello world. "
    repeat_count = max(1, max_length // scale)
    return base_text * repeat_count


def _exceed_context_prompt(request):
    configured_length = int(request.config.getoption("--maxModelLength"))
    repeat_count = configured_length + CONTEXT_OVERFLOW_MARGIN
    return "token " * repeat_count


@pytest.mark.parametrize(
    ("endpoint", "stream"),
    [
        ("chat", False),
        ("chat", True),
        ("completions", False),
        ("completions", True),
    ],
    ids=["chat", "chat_stream", "completions", "completions_stream"],
)
def test_context_length_within_limit(api_client, request, endpoint, stream):
    prompt = _context_prompt(request, scale=20)

    if endpoint == "chat":
        response = completion_request.send_chat_request(
            api_client,
            messages=[{"role": "user", "content": prompt}],
            stream=stream,
            max_tokens=512,
        )
    else:
        response = completion_request.send_completion_request(
            api_client,
            prompt=prompt,
            stream=stream,
            max_tokens=512,
        )

    assertion.assert_chat_completion_success(response, stream=stream)


@pytest.mark.parametrize(
    ("endpoint", "stream"),
    [
        ("chat", False),
        ("chat", True),
        ("completions", False),
        ("completions", True),
    ],
    ids=["chat", "chat_stream", "completions", "completions_stream"],
)
def test_context_length_exceed_limit_returns_validation_error(api_client, request, endpoint, stream):
    # Keep --maxModelLength aligned with the server's --max-model-len and build
    # a prompt that exceeds the configured service limit.
    prompt = _exceed_context_prompt(request)

    if endpoint == "chat":
        response = completion_request.send_chat_request(
            api_client,
            messages=[{"role": "user", "content": prompt}],
            stream=stream,
            max_tokens=512,
            stop=["Input:"],
        )
    else:
        response = completion_request.send_completion_request(
            api_client,
            prompt=prompt,
            stream=stream,
            max_tokens=512,
            stop=["Input:"],
        )

    assertion.assert_validation_error_response(response)
