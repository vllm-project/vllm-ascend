import pytest
from ...utility import request_helper as helper
from ...utility import assertion


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("temperature", [0.0, 0.1, 0.5, 0.8, 1.0, 1.5, 2.0], ids=["t0", "t0.1", "t0.5", "t0.8", "t1.0", "t1.5", "t2.0"])
def test_temperature_normal_values(api_client, request, stream, temperature):
    """temperature正常取值范围[0.0, 2.0]，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "用一句话形容夏天的夜晚。"
        }],
        "temperature": temperature,
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
def test_temperature_zero_deterministic(api_client, request, stream):
    """temperature=0.0，贪婪解码，确定性输出"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "中国首都是哪里？只回答城市名。"
        }],
        "temperature": 0.0,
        "stream": stream,
        "max_tokens": 10
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # 校验点3：finish_reason为stop或length
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_temperature_high_creative(api_client, request, stream):
    """temperature=2.0，高创造性，多样化输出"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "写一首简短的诗"
        }],
        "temperature": 2.0,
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
def test_temperature_with_top_p(api_client, request, stream):
    """temperature与top_p组合使用，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "用一句话介绍自己。"
        }],
        "temperature": 0.7,
        "top_p": 0.9,
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
def test_temperature_without_setting(api_client, request, stream):
    """请求体中不设置temperature参数，使用默认值1.0，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天天气怎么样"
        }],
        # 不设置temperature
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
def test_temperature_balanced_default(api_client, request, stream):
    """temperature=1.0（默认值），平衡的采样温度"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "用一句话形容秋天。"
        }],
        "temperature": 1.0,
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
