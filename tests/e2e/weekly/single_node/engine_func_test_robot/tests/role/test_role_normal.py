import pytest
from ...utility import request_helper as helper
from ...utility import assertion


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_role_single_user(api_client, request, stream):
    """单条user角色的消息，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
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
def test_role_system_user(api_client, request, stream):
    """system和user角色组合，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "system",
                "content": "你是一个乐于助人的助手。"
            },
            {
                "role": "user",
                "content": "你好"
            }
        ],
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
def test_role_multi_turn(api_client, request, stream):
    """多轮对话：user-assistant-user角色交替，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": "你好"
            },
            {
                "role": "assistant",
                "content": "你好！很高兴见到你，有什么我可以帮助你的吗？"
            },
            {
                "role": "user",
                "content": "今天天气怎么样"
            }
        ],
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
def test_role_system_user_assistant_user(api_client, request, stream):
    """完整对话：system-user-assistant-user，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "system",
                "content": "你是一个专业的技术文档撰写助手。"
            },
            {
                "role": "user",
                "content": "请帮我写一个函数说明文档的开头"
            },
            {
                "role": "assistant",
                "content": "好的，请问这个函数的主要功能是什么？"
            },
            {
                "role": "user",
                "content": "这是一个计算两个数之和的函数"
            }
        ],
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
def test_role_only_system(api_client, request, stream):
    """只有system角色，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "system",
                "content": "这是一个测试。"
            }
        ],
        "stream": stream,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：如果响应体存在error code，则error code不是500
    assertion.assert_error_code_not_500(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_role_multiple_system(api_client, request, stream):
    """多条system消息，后面的覆盖前面的或合并处理"""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "system",
                "content": "你是一个很严肃的助手。"
            },
            {
                "role": "system",
                "content": "你是一个很幽默的助手。"
            },
            {
                "role": "user",
                "content": "讲个笑话"
            }
        ],
        "stream": stream,
        "max_tokens": 100
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：如果响应体存在error code，则error code不是500
    assertion.assert_error_code_not_500(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_role_long_history(api_client, request, stream):
    """长历史对话（多轮user-assistant交替），请求正常处理"""
    # 构建10轮对话历史
    messages = []
    for i in range(10):
        messages.append({
            "role": "user",
            "content": f"这是第{i+1}轮的用户问题"
        })
        messages.append({
            "role": "assistant",
            "content": f"这是第{i+1}轮的助手回答"
        })
    messages.append({
        "role": "user",
        "content": "请总结上面的对话"
    })

    request_body = {
        "model": "auto",
        "messages": messages,
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
def test_role_empty_content(api_client, request, stream):
    """role为user但content为空字符串，请求正常处理或返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": ""
        }],
        "stream": stream,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码200或400（取决于实现是否允许空消息）
    # 此处假设大部分实现允许空消息
    assertion.assert_status_code_200(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_role_only_assistant(api_client, request, stream):
    """单条assistant角色结尾，请求处理（部分实现支持prefill场景）"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "assistant",
            "content": "我将为你解答这个问题。"
        }],
        "stream": stream,
        "max_tokens": 50
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：如果响应体存在error code，则error code不是500
    assertion.assert_error_code_not_500(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_role_user_assistant_ending(api_client, request, stream):
    """user-assistant角色对话，以assistant结尾"""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": "你好"
            },
            {
                "role": "assistant",
                "content": "你好！很高兴见到你。"
            }
        ],
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
def test_role_system_user_assistant_ending(api_client, request, stream):
    """system-user-assistant角色组合，以assistant结尾"""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "system",
                "content": "你是一个专业的助手。"
            },
            {
                "role": "user",
                "content": "请自我介绍一下。"
            },
            {
                "role": "assistant",
                "content": "你好，我是一个AI助手，可以帮助你解答问题。"
            }
        ],
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
def test_role_multi_turn_assistant_ending(api_client, request, stream):
    """多轮user-assistant交替，以assistant角色结尾"""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": "北京有什么好玩的？"
            },
            {
                "role": "assistant",
                "content": "北京有很多著名景点，比如故宫、长城、颐和园等。"
            },
            {
                "role": "user",
                "content": "还有呢？"
            },
            {
                "role": "assistant",
                "content": "还有天坛、圆明园、什刹海等景点也值得游览。"
            }
        ],
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
def test_role_long_history_assistant_ending(api_client, request, stream):
    """长历史对话，以assistant角色结尾"""
    # 构建5轮对话历史，以assistant结尾
    messages = []
    for i in range(5):
        messages.append({
            "role": "user",
            "content": f"第{i+1}个问题"
        })
        messages.append({
            "role": "assistant",
            "content": f"第{i+1}个回答"
        })

    request_body = {
        "model": "auto",
        "messages": messages,
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
