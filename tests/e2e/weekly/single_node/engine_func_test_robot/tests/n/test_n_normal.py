import re
import json
import pytest
from ...utility import request_helper as helper
from ...utility import assertion


def parse_stream_choices_count(response_text):
    """解析流式响应中choices的数量（通过统计不同的index值）"""
    # 流式响应中每次SSE只吐一个单元素的choices列表
    # n>1时通过不同的index来区分，index范围是0到n-1
    indexes = re.findall(r'"index"\s*:\s*(\d+)', response_text)
    if not indexes:
        return 0
    # 返回最大index+1即为n的值
    return max(int(idx) for idx in indexes) + 1


def parse_stream_finish_reasons(response_text):
    """解析流式响应中所有choices的finish_reason（非空值）"""
    # 流式响应中finish_reason可能为空字符串，只在最后才有值
    finish_reasons = re.findall(r'"finish_reason"\s*:\s*"([^"]+)"', response_text)
    # 过滤掉空字符串
    return [fr for fr in finish_reasons if fr]


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_default_value(api_client, request, stream):
    """n参数默认值为1，返回1个结果"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
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

    # 校验点3：返回1个choice
    if stream:
        choices_count = parse_stream_choices_count(response.text)
    else:
        choices_count = len(response.json()["choices"])
    assert choices_count == 1, f"Expected 1 choice, got {choices_count}"

    # 校验点4：finish_reason有效
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_explicit_one(api_client, request, stream):
    """n=1显式指定，返回1个结果"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "n": 1,
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

    # 校验点3：返回1个choice
    if stream:
        choices_count = parse_stream_choices_count(response.text)
    else:
        choices_count = len(response.json()["choices"])
    assert choices_count == 1, f"Expected 1 choice, got {choices_count}"

    # 校验点4：finish_reason有效
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_two_with_temperature(api_client, request, stream):
    """n=2配合temperature>0，返回2个结果"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "n": 2,
        "temperature": 0.8,
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

    # 校验点3：返回2个choices
    if stream:
        choices_count = parse_stream_choices_count(response.text)
    else:
        choices_count = len(response.json()["choices"])
    assert choices_count == 2, f"Expected 2 choices, got {choices_count}"

    # 校验点4：每个choice的finish_reason有效
    if stream:
        finish_reasons = parse_stream_finish_reasons(response.text)
    else:
        finish_reasons = [c["finish_reason"] for c in response.json()["choices"]]
    for i, finish_reason in enumerate(finish_reasons):
        assertion.assert_finish_reason_valid(finish_reason, f"choice[{i}] ")


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("n_value", [3, 5], ids=["n3", "n5"])
def test_n_multiple_values(api_client, request, stream, n_value):
    """n=3/5，返回对应数量结果"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "n": n_value,
        "temperature": 0.9,
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

    # 校验点3：返回n个choices
    if stream:
        choices_count = parse_stream_choices_count(response.text)
    else:
        choices_count = len(response.json()["choices"])
    assert choices_count == n_value, f"Expected {n_value} choices, got {choices_count}"

    # 校验点4：每个choice的finish_reason有效
    if stream:
        finish_reasons = parse_stream_finish_reasons(response.text)
    else:
        finish_reasons = [c["finish_reason"] for c in response.json()["choices"]]
    for i, finish_reason in enumerate(finish_reasons):
        assertion.assert_finish_reason_valid(finish_reason, f"choice[{i}] ")


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_null_uses_default(api_client, request, stream):
    """n=null使用默认值1"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "n": None,
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

    # 校验点3：返回1个choice（默认值）
    if stream:
        choices_count = parse_stream_choices_count(response.text)
    else:
        choices_count = len(response.json()["choices"])
    assert choices_count == 1, f"Expected 1 choice (default), got {choices_count}"

    # 校验点4：finish_reason有效
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_one_with_greedy_sampling(api_client, request, stream):
    """n=1配合temperature=0贪婪采样，正常返回"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "中国首都是哪里？只回答城市名。"
        }],
        "n": 1,
        "temperature": 0,
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

    # 校验点3：返回1个choice
    if stream:
        choices_count = parse_stream_choices_count(response.text)
    else:
        choices_count = len(response.json()["choices"])
    assert choices_count == 1, f"Expected 1 choice, got {choices_count}"

    # 校验点4：finish_reason有效
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)
