import pytest
from ...utility import request_helper as helper
from ...utility import assertion


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("presence_penalty", [-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0], 
                         ids=["pp-2.0", "pp-1.0", "pp0", "pp0.5", "pp1.0", "pp1.5", "pp2.0"])
def test_presence_penalty_normal_values(api_client, request, stream, presence_penalty):
    """presence_penalty正常取值范围[-2.0, 2.0]，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请列举一些城市名称"
        }],
        "presence_penalty": presence_penalty,
        "stream": stream,
        "max_tokens": 100
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
def test_presence_penalty_zero_no_effect(api_client, request, stream):
    """presence_penalty=0.0，无存在惩罚效果"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请列举一些城市名称"
        }],
        "presence_penalty": 0.0,
        "stream": stream,
        "max_tokens": 100
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
def test_presence_penalty_positive_diversify_topics(api_client, request, stream):
    """presence_penalty正值，鼓励讨论新话题"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请分别讨论科技、文化、艺术三个不同的话题"
        }],
        "presence_penalty": 1.5,
        "stream": stream,
        "max_tokens": 150
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
def test_presence_penalty_negative_stay_on_topic(api_client, request, stream):
    """presence_penalty负值，鼓励停留在当前话题"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请深入讨论人工智能这个话题"
        }],
        "presence_penalty": -1.5,
        "stream": stream,
        "max_tokens": 100
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
def test_presence_penalty_without_setting(api_client, request, stream):
    """请求体中不设置presence_penalty参数，使用默认值0.0，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请列举一些城市名称"
        }],
        # 不设置presence_penalty
        "stream": stream,
        "max_tokens": 100
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
def test_presence_penalty_with_frequency_penalty(api_client, request, stream):
    """presence_penalty与frequency_penalty组合使用"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请列举不同领域的知识，避免重复"
        }],
        "presence_penalty": 1.0,
        "frequency_penalty": 1.0,
        "stream": stream,
        "max_tokens": 100
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
def test_presence_penalty_highest_value(api_client, request, stream):
    """presence_penalty=2.0最大值，最大程度鼓励新话题"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请描述多个不同的场景"
        }],
        "presence_penalty": 2.0,
        "stream": stream,
        "max_tokens": 100
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
def test_presence_penalty_lowest_value(api_client, request, stream):
    """presence_penalty=-2.0最小值，最大程度鼓励重复话题"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请深入讨论编程这个话题"
        }],
        "presence_penalty": -2.0,
        "stream": stream,
        "max_tokens": 100
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
