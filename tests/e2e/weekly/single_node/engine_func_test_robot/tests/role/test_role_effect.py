import re
import json
import pytest_check as check
from ...utility import request_helper as helper
from ...utility import assertion


def test_role_system_instruction_follow(api_client, request):
    """system角色指令影响输出行为"""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "system",
                "content": "请用JSON格式回答所有问题。"
            },
            {
                "role": "user",
                "content": "请告诉我你的名字。"
            }
        ],
        "stream": False,
        "max_tokens": 10240
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

    # 校验点3：内容应该为有效的JSON格式
    json_str = re.sub(r"\s*<think>[\s\S]*?</think>\s*", "", content)
    json_str = re.search(r"(\{.*\})\s*(?:$|`|```)$", json_str, re.S)
    check.is_true(json_str, "response content未匹配到JSON格式")
    if json_str:
        check.is_true(json.loads(json_str.group(1)), "system指令未被遵循，返回内容非有效的JSON格式")


def test_role_long_conversation_pruning(api_client, request):
    """超长对话历史，模型应该能处理或截断"""
    # 构建20轮对话历史
    messages = [{"role": "system", "content": "你是一个耐心的助手。"}]
    for i in range(20):
        messages.append({
            "role": "user",
            "content": f"这是第{i+1}个问题。"
        })
        messages.append({
            "role": "assistant",
            "content": f"这是第{i+1}个回答。"
        })
    messages.append({
        "role": "user",
        "content": "请问这是第几轮对话？"
    })

    request_body = {
        "model": "auto",
        "messages": messages,
        "stream": False,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点1：状态码200（即使被截断也应该返回）
    assertion.assert_status_code_200(response)

    # 校验点2：返回内容有效
    content = response.json()["choices"][0]["message"]["content"]
    check.is_not_none(content, "返回内容为None")
    check.is_true(len(content) > 0, "返回内容为空")
