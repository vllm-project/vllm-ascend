import pytest
from ...utility import request_helper as helper
from ...utility import assertion


@pytest.mark.skip(reason="20260421标记start摒弃")
def test_completions_stream_no_think_tag(api_client, request):
    """completions接口流式：无think标签"""
    request_body = {
        "model": "auto",
        "prompt": "请用最精简的一句话快速说出你的名字，不要解释，不要废话。",
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：无think标签
    if request.config.getoption("--thinkTagOutput").strip().lower() == "true":
        assertion.assert_no_think_tag(response.text, "流式响应")