import pytest
from ...utility import request_helper as helper
from ...utility import assertion


def test_chat_non_stream_think_tag_complete(api_client, request):
    """chat接口非流式：思考模型think标签完整"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请用最简单的一句话快速介绍你是谁，不要废话。"
        }],
        "stream": False,
        "max_tokens": 5120000
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：think标签完整性
    if request.config.getoption("--thinkTagOutput").strip().lower() == "true":
        assertion.assert_think_tag_present(response.content.decode("utf-8"), "非流式响应")