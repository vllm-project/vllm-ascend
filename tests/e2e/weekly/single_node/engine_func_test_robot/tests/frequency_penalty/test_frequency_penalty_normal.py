import pytest
from ...utility import request_helper as helper
from ...utility import assertion


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("frequency_penalty", [-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0], 
                         ids=["fp-2.0", "fp-1.0", "fp0", "fp0.5", "fp1.0", "fp1.5", "fp2.0"])
def test_frequency_penalty_normal_values(api_client, request, stream, frequency_penalty):
    """frequency_penalty正常取值范围[-2.0, 2.0]，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请列举一些水果名称"
        }],
        "frequency_penalty": frequency_penalty,
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
def test_frequency_penalty_zero_no_effect(api_client, request, stream):
    """frequency_penalty=0.0，无频率惩罚效果"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请列举一些水果名称"
        }],
        "frequency_penalty": 0.0,
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
def test_frequency_penalty_positive_reduce_repetition(api_client, request, stream):
    """frequency_penalty正值，降低重复token概率"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "用不同的词描述美丽，避免重复"
        }],
        "frequency_penalty": 1.5,
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
def test_frequency_penalty_negative_encourage_repetition(api_client, request, stream):
    """frequency_penalty负值，增加重复token概率"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请重复'重要'这个词多次"
        }],
        "frequency_penalty": -1.5,
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
def test_frequency_penalty_without_setting(api_client, request, stream):
    """请求体中不设置frequency_penalty参数，使用默认值0.0，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请列举一些水果名称"
        }],
        # 不设置frequency_penalty
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
def test_frequency_penalty_with_presence_penalty(api_client, request, stream):
    """frequency_penalty与presence_penalty组合使用"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请列举不同类别的物品，避免重复"
        }],
        "frequency_penalty": 1.0,
        "presence_penalty": 0.5,
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
def test_frequency_penalty_with_repetition_penalty(api_client, request, stream):
    """frequency_penalty与repetition_penalty组合使用"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请用丰富的词汇描述风景"
        }],
        "frequency_penalty": 0.8,
        "repetition_penalty": 1.2,
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
def test_frequency_penalty_all_penalties(api_client, request, stream):
    """所有惩罚参数组合使用"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请描述一个场景，用词要丰富多样"
        }],
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5,
        "repetition_penalty": 1.1,
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
