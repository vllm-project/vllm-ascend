import pytest
from ...utility import request_helper as helper
from ...utility import assertion


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_zero(api_client, request, stream):
    """n=0，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "n": 0,
        "stream": stream,
        "max_tokens": 10
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码400
    assertion.assert_status_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("n_value", [-1, -10, -100], ids=["n-1", "n-10", "n-100"])
def test_n_negative(api_client, request, stream, n_value):
    """n为负数，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "n": n_value,
        "stream": stream,
        "max_tokens": 10
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码400
    assertion.assert_status_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_string_invalid(api_client, request, stream):
    """n="abc"无效字符串，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "n": "abc",
        "stream": stream,
        "max_tokens": 10
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码400
    assertion.assert_status_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_array(api_client, request, stream):
    """n=[1]数组类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "n": [1],
        "stream": stream,
        "max_tokens": 10
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码400
    assertion.assert_status_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_object(api_client, request, stream):
    """n={}对象类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "n": {},
        "stream": stream,
        "max_tokens": 10
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码400
    assertion.assert_status_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_boolean_false(api_client, request, stream):
    """n=false，应返回400错误（false转为0）"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "n": False,
        "stream": stream,
        "max_tokens": 10
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码400
    assertion.assert_status_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_two_with_greedy_sampling(api_client, request, stream):
    """n=2配合temperature=0贪婪采样，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "n": 2,
        "temperature": 0,
        "stream": stream,
        "max_tokens": 10
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码400（贪婪采样时n必须为1）
    assertion.assert_status_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("n_value", [3, 5], ids=["n3", "n5"])
def test_n_greater_than_one_with_greedy_sampling(api_client, request, stream, n_value):
    """n>1配合temperature=0贪婪采样，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "n": n_value,
        "temperature": 0,
        "stream": stream,
        "max_tokens": 10
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码400
    assertion.assert_status_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_float_less_than_one(api_client, request, stream):
    """n=0.5浮点数小于1，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "n": 0.5,
        "stream": stream,
        "max_tokens": 10
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码400（0.5截断为0）
    assertion.assert_status_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_n_float_zero_point_nine(api_client, request, stream):
    """n=0.9浮点数接近1，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "n": 0.9,
        "stream": stream,
        "max_tokens": 10
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：状态码400（0.9截断为0）
    assertion.assert_status_code_400(response)
