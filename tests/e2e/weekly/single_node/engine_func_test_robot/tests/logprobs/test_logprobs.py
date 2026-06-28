import random
import pytest
from ...utility import request_helper as helper
from ...utility import assertion


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("top_logprobs", [random.randint(1, 19), 0, 20], ids=["in_1_19", "0", "20"])
def test_logprobs_normal_boundary(api_client, request, stream, top_logprobs):
    """logprobs为true，top_logprobs在[0,20]范围内，返回正确数量的top_logprobs"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天星期几，立马回复，精简语言，不要废话。"
        }],
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # 校验点3：top_logprobs数量正确
    assertion.assert_top_logprobs_count(response, top_logprobs)


def test_logprobs_exceed_20_non_stream(api_client, request):
    """非流式：logprobs为true，top_logprobs大于20，http status code:400, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天星期几，不要废话"
        }],
        "logprobs": True,
        "top_logprobs": 21,
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码400，错误码400
    assertion.assert_status_code_400(response)
    assertion.assert_error_code_400(response)


def test_logprobs_exceed_20_stream(api_client, request):
    """流式：logprobs为true，top_logprobs大于20，http status code:200, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天星期几，不要废话"
        }],
        "logprobs": True,
        "top_logprobs": 21,
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：流式场景状态码200，错误码400
    assertion.assert_status_code_200(response)
    assertion.assert_error_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_logprobs_with_logit_bias(api_client, request, stream):
    """logprobs为true，logit_bias存在，请求正常"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天星期几，不要废话"
        }],
        "logprobs": True,
        "logit_bias": {"6002": -100},
        "stream": stream,
        "max_tokens": 5120
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：请求成功（状态码200）
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # 校验点3：finish_reason有效
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)