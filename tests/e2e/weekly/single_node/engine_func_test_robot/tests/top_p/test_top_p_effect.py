import pytest_check as check
from ...utility import request_helper as helper
from ...utility import assertion


def test_top_p_zero_deterministic(api_client, request):
    """top_p=0.0应该返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "中国首都是哪里？（只回答城市名）"
        }],
        "top_p": 0.0,
        "temperature": 0.0,
        "stream": False,
        "max_tokens": 10
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：错误码400
    assertion.assert_error_code_400(response)


def test_top_p_small_conservative(api_client, request):
    """top_p=0.1小概率质量，采样保守"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "推荐一首中文诗。"
        }],
        "top_p": 0.1,
        "temperature": 0.7,
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


def test_top_p_medium_balanced(api_client, request):
    """top_p=0.7中等概率质量，平衡保守和多样"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "用一句话形容大海。"
        }],
        "top_p": 0.7,
        "temperature": 0.8,
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


def test_top_p_large_full_vocab(api_client, request):
    """top_p=1.0使用全部词表概率分布"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请用三个词描述冬天。"
        }],
        "top_p": 1.0,
        "temperature": 0.9,
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


def test_top_p_combined_with_top_k(api_client, request):
    """top_p和top_k同时设置，验证两者协同效果（先k后p）"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "用一句话形容日落。"
        }],
        "top_p": 0.85,
        "top_k": 50,
        "temperature": 0.75,
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


def test_top_p_temperature_interaction(api_client, request):
    """top_p和temperature的交互：低temperature+低top_p应该很保守"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "中国最大的省份是？只回答名称。"
        }],
        "top_p": 0.3,
        "temperature": 0.2,
        "stream": False,
        "max_tokens": 10
    }

    results = []
    for i in range(3):
        helper.attach_request_body(request_body, f"第{i+1}次请求")
        response = helper.send_request(api_client, "/v1/chat/completions", request_body)
        helper.attach_response_body(response, f"第{i+1}次响应")
        
        assertion.assert_status_code_200(response, f"第{i+1}次请求")
        content = response.json()["choices"][0]["message"]["content"]
        results.append(content)
    
    # 在非常保守的设置下，3次结果很可能相似（不是强制一样，只是概率高）
    for i, result in enumerate(results):
        check.is_not_none(result, f"第{i+1}次结果为None")
        check.is_true(len(result) > 0, f"第{i+1}次结果为空")
