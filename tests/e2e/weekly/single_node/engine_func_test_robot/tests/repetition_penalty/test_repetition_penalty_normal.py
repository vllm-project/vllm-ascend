import pytest
from ...utility import request_helper as helper
from ...utility import assertion


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("repetition_penalty", [1.0, 1.05, 1.1, 1.2, 1.5, 2.0], 
                         ids=["rp1.0", "rp1.05", "rp1.1", "rp1.2", "rp1.5", "rp2.0"])
def test_repetition_penalty_normal_values(api_client, request, stream, repetition_penalty):
    """repetition_penalty正常取值范围[1.0, 2.0]，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请列举一些动物名称"
        }],
        "repetition_penalty": repetition_penalty,
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
def test_repetition_penalty_one_no_effect(api_client, request, stream):
    """repetition_penalty=1.0，无重复惩罚效果"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请列举一些动物名称"
        }],
        "repetition_penalty": 1.0,
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
def test_repetition_penalty_moderate_reduce_repetition(api_client, request, stream):
    """repetition_penalty=1.2，适度降低重复"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请用不同的词汇描述美丽"
        }],
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
def test_repetition_penalty_strong_no_repeat(api_client, request, stream):
    """repetition_penalty=2.0，强烈抑制重复"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请列举不同种类的食物，避免重复用词"
        }],
        "repetition_penalty": 2.0,
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
def test_repetition_penalty_without_setting(api_client, request, stream):
    """请求体中不设置repetition_penalty参数，使用默认值1.0，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请列举一些动物名称"
        }],
        # 不设置repetition_penalty
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
def test_repetition_penalty_with_frequency_penalty(api_client, request, stream):
    """repetition_penalty与frequency_penalty组合使用"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请列举不同的物品，用词丰富"
        }],
        "repetition_penalty": 1.2,
        "frequency_penalty": 0.5,
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
def test_repetition_penalty_with_presence_penalty(api_client, request, stream):
    """repetition_penalty与presence_penalty组合使用"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请描述多个不同的场景"
        }],
        "repetition_penalty": 1.3,
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
def test_repetition_penalty_all_penalties(api_client, request, stream):
    """所有惩罚参数组合使用"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请描述一个四季变化的场景，用词要丰富"
        }],
        "repetition_penalty": 1.1,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.3,
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
def test_repetition_penalty_with_temperature(api_client, request, stream):
    """repetition_penalty与temperature组合使用"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "写一个简短的故事"
        }],
        "repetition_penalty": 1.15,
        "temperature": 0.8,
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
