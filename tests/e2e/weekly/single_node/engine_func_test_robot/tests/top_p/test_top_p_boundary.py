import pytest
from ...utility import request_helper as helper
from ...utility import assertion


def test_top_p_boundary_zero_non_stream(api_client, request):
    """非流式：top_p边界值=0.0，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "中国首都是？只回答城市名。"
        }],
        "top_p": 0.0,
        "stream": False,
        "max_tokens": 10
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_top_p_boundary_zero_stream(api_client, request):
    """流式：top_p边界值=0.0，状态码200但错误码400"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "中国首都是？只回答城市名。"
        }],
        "top_p": 0.0,
        "stream": True,
        "max_tokens": 10
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_boundary_one(api_client, request, stream):
    """top_p边界值=1.0，应该正常工作（禁用nucleus）"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "用一句话形容晚霞。"
        }],
        "top_p": 1.0,
        "stream": stream,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
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


def test_top_p_boundary_negative_small_non_stream(api_client, request):
    """非流式：top_p边界值为很小的负数，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天天气怎么样"
        }],
        "top_p": -0.0001,
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_top_p_boundary_negative_small_stream(api_client, request):
    """流式：top_p边界值为很小的负数，状态码200但错误码400"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天天气怎么样"
        }],
        "top_p": -0.0001,
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_top_p_boundary_just_above_one_non_stream(api_client, request):
    """非流式：top_p边界值为略大于1.0，应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天天气怎么样"
        }],
        "top_p": 1.0001,
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_top_p_boundary_just_above_one_stream(api_client, request):
    """流式：top_p边界值为略大于1.0，状态码200但错误码400"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天天气怎么样"
        }],
        "top_p": 1.0001,
        "stream": True,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_boundary_precision_decimal(api_client, request, stream):
    """top_p边界值为高精度小数，应该正常工作"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天天气怎么样"
        }],
        "top_p": 0.999999,
        "stream": stream,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
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


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_without_explicit_setting(api_client, request, stream):
    """top_p不设置（使用默认值1.0），应该正常工作"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天天气怎么样"
        }],
        # 不设置top_p
        "stream": stream,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
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


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_top_p_boundary_very_small_positive(api_client, request, stream):
    """top_p边界值为很小的正数，应该正常工作"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天天气怎么样"
        }],
        "top_p": 0.0001,
        "stream": stream,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
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
