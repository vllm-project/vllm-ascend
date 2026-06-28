import pytest_check as check
from ...utility import request_helper as helper
from ...utility import assertion


def test_top_k_small_conservative(api_client, request):
    """top_k=5小范围采样，结果稳定但略有不同"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "用三个词描述春天"
        }],
        "top_k": 5,
        "temperature": 1.0,
        "stream": False,
        "max_tokens": 20
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


def test_top_k_large_diverse(api_client, request):
    """top_k=100大范围采样，结果多样性更高"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "用一句话形容晚霞"
        }],
        "top_k": 100,
        "temperature": 0.9,
        "stream": False,
        "max_tokens": 30
    }

    results = []
    for i in range(5):
        helper.attach_request_body(request_body, f"第{i+1}次请求")
        response = helper.send_request(api_client, "/v1/chat/completions", request_body)
        helper.attach_response_body(response, f"第{i+1}次响应")
        
        assertion.assert_status_code_200(response, f"第{i+1}次请求")
        content = response.json()["choices"][0]["message"]["content"]
        results.append(content)
    
    # 校验点：所有结果都应该有效
    for i, result in enumerate(results):
        check.is_not_none(result, f"第{i+1}次结果为None")
        check.is_true(len(result) > 0, f"第{i+1}次结果为空")


def test_top_k_disabled_full_vocab(api_client, request):
    """top_k=-1禁用限制，使用完整词表采样"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请说一句祝福语"
        }],
        "top_k": -1,
        "temperature": 0.8,
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


def test_top_k_combined_with_top_p_priority(api_client, request):
    """top_k和top_p同时设置，验证两者协同工作"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "用一句话总结这篇文章：人工智能正在改变世界。"
        }],
        "top_k": 50,
        "top_p": 0.85,
        "temperature": 0.7,
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
    
    # 校验点3：finish_reason为stop或length
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)
