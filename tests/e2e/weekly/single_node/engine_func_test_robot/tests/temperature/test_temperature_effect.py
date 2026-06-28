import pytest_check as check
from ...utility import request_helper as helper
from ...utility import assertion


def test_temperature_low_conservative(api_client, request):
    """temperature=0.1低温，采样保守，结果变化较小"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "推荐一首中文诗。"
        }],
        "temperature": 0.1,
        "stream": False,
        "max_tokens": 30
    }

    results = []
    for i in range(3):
        helper.attach_request_body(request_body, f"第{i+1}次请求")
        response = helper.send_request(api_client, "/v1/chat/completions", request_body)
        helper.attach_response_body(response, f"第{i+1}次响应")
        
        assertion.assert_status_code_200(response, f"第{i+1}次请求")
        content = response.json()["choices"][0]["message"]["content"]
        results.append(content)
    
    # 校验点：结果应该有合理长度
    for i, result in enumerate(results):
        check.is_not_none(result, f"第{i+1}次结果为None")
        check.is_true(len(result) > 0, f"第{i+1}次结果为空")


def test_temperature_medium_balanced(api_client, request):
    """temperature=1.0（默认），平衡的创造性"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "用一句话形容大海。"
        }],
        "temperature": 1.0,
        "stream": False,
        "max_tokens": 30
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：返回内容有效
    content = response.json()["choices"][0]["message"]["content"]
    check.is_not_none(content, "返回内容为None")
    check.is_true(len(content) > 0, "返回内容为空")
    
    # 校验点3：finish_reason有效
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_temperature_high_diverse(api_client, request):
    """temperature=2.0高温，采样更具多样性"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "写一首简短的诗"
        }],
        "temperature": 2.0,
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：返回内容有效
    content = response.json()["choices"][0]["message"]["content"]
    check.is_not_none(content, "返回内容为None")
    check.is_true(len(content) > 0, "返回内容为空")
    
    # 校验点3：finish_reason有效
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


def test_temperature_with_top_p_combination(api_client, request):
    """temperature与top_p的协同效果"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "用一句话形容日落。"
        }],
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 50,
        "stream": False,
        "max_tokens": 30
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：返回内容有效
    content = response.json()["choices"][0]["message"]["content"]
    check.is_not_none(content, "返回内容为None")
    check.is_true(len(content) > 0, "返回内容为空")
    
    # 校验点3：finish_reason有效
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)
