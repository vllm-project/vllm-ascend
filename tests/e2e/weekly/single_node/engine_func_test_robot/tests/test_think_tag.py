import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion, completion_request


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_think_tag_output_when_enabled(api_client, request, stream):
    response = completion_request.send_chat_request(
        api_client,
        messages=[{"role": "user", "content": "Introduce yourself."}],
        stream=stream,
        max_tokens=512,
    )
    if request.config.getoption("--thinkTagOutput").strip().lower() == "true":
        response_text = response.text if stream else response.content.decode("utf-8")
        assertion.assert_think_tag_present(response_text, "response should be valid")


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_completions_do_not_emit_think_tag(api_client, stream):
    response = completion_request.send_completion_request(
        api_client, prompt="Introduce yourself.", stream=stream, max_tokens=512
    )
    assertion.assert_status_code_200(response)
    assertion.assert_no_think_tag(response.text)
