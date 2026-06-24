"""
thinking 和 enable_thinking 字段测试
- thinking: 用于 deepseek/ds 系列模型
- enable_thinking: 用于 qwen 系列模型

当字段为 true 时，需要校验 `` 标签完整性
当字段为 false 时，需要校验不存在 `` 标签
参照 think_tag 目录的校验规则
"""
import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion
from tests.e2e.weekly.single_node.engine_func_test_robot.utility import request_helper as helper


def is_qwen_model(model_name):
    """判断是否qwen系列模型（大小写不敏感）"""
    return model_name and "qwen" in model_name.lower()


def is_deepseek_model(model_name):
    """判断是否deepseek/ds系列模型（大小写不敏感）"""
    if not model_name:
        return False
    model_lower = model_name.lower()
    return "deepseek" in model_lower or "ds" in model_lower


def should_check_think_tag(request):
    """判断是否需要进行think标签校验"""
    return request.config.getoption("--thinkTagOutput").strip().lower() == "true"


# ==================== Qwen系列模型测试 - enable_thinking ====================

@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_qwen_enable_thinking_true(api_client, request, stream):
    """qwen模型：enable_thinking=true，正常开启思考模式，需要校验think标签完整性"""
    model = request.config.getoption("--model")
    if not is_qwen_model(model):
        pytest.skip(f"当前模型 {model} 不是qwen系列，跳过此测试")

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请用最简单的一句话介绍你是谁。"
        }],
        "chat_template_kwargs": {
            "enable_thinking": True
        },
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # 校验点3：think标签完整性（enable_thinking=true时）
    if should_check_think_tag(request):
        assertion.assert_think_tag_present(response.content.decode("utf-8"), "enable_thinking=true")


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_qwen_enable_thinking_false(api_client, request, stream):
    """qwen模型：enable_thinking=false，关闭思考模式，需要校验不存在think标签"""
    model = request.config.getoption("--model")
    if not is_qwen_model(model):
        pytest.skip(f"当前模型 {model} 不是qwen系列，跳过此测试")

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请用最简单的一句话介绍你是谁。"
        }],
        "chat_template_kwargs": {
            "enable_thinking": False
        },
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # 校验点3：不存在think标签（enable_thinking=false时）
    if should_check_think_tag(request):
        assertion.assert_no_think_tag(response.content.decode("utf-8"), "enable_thinking=false")


# ==================== DeepSeek/DS系列模型测试 - thinking ====================

@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_deepseek_thinking_true(api_client, request, stream):
    """deepseek/ds模型：thinking=true，正常开启思考模式，需要校验think标签完整性"""
    model = request.config.getoption("--model")
    if not is_deepseek_model(model):
        pytest.skip(f"当前模型 {model} 不是deepseek/ds系列，跳过此测试")

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请用最简单的一句话介绍你是谁。"
        }],
        "chat_template_kwargs": {
            "thinking": True
        },
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # 校验点3：think标签完整性（thinking=true时）
    if should_check_think_tag(request):
        assertion.assert_think_tag_present(response.content.decode("utf-8"), "thinking=true")


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_deepseek_thinking_false(api_client, request, stream):
    """deepseek/ds模型：thinking=false，关闭思考模式，需要校验不存在think标签"""
    model = request.config.getoption("--model")
    if not is_deepseek_model(model):
        pytest.skip(f"当前模型 {model} 不是deepseek/ds系列，跳过此测试")

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请用最简单的一句话介绍你是谁。"
        }],
        "chat_template_kwargs": {
            "thinking": False
        },
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点1：状态码200
    assertion.assert_status_code_200(response)

    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # 校验点3：不存在think标签（thinking=false时）
    if should_check_think_tag(request):
        assertion.assert_no_think_tag(response.content.decode("utf-8"), "thinking=false")


# ==================== 非适用模型测试 - 异常场景 ====================

@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_qwen_field_on_deepseek(api_client, request, stream):
    """异常：在deepseek模型上使用enable_thinking字段"""
    model = request.config.getoption("--model")
    if not is_deepseek_model(model):
        pytest.skip(f"当前模型 {model} 不是deepseek/ds系列，跳过此测试")

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "chat_template_kwargs": {
            "enable_thinking": True  # 在deepseek上使用qwen的字段
        },
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：可能200（引擎忽略未知字段）或400（引擎严格校验）
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_deepseek_field_on_qwen(api_client, request, stream):
    """异常：在qwen模型上使用thinking字段"""
    model = request.config.getoption("--model")
    if not is_qwen_model(model):
        pytest.skip(f"当前模型 {model} 不是qwen系列，跳过此测试")

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "chat_template_kwargs": {
            "thinking": True  # 在qwen上使用deepseek的字段
        },
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：可能200（引擎忽略未知字段）或400（引擎严格校验）
    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


# ==================== 边界测试 ====================

@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_qwen_enable_thinking_null(api_client, request, stream):
    """异常：qwen模型enable_thinking为null"""
    model = request.config.getoption("--model")
    if not is_qwen_model(model):
        pytest.skip(f"当前模型 {model} 不是qwen系列，跳过此测试")

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "chat_template_kwargs": {
            "enable_thinking": None
        },
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    assertion.assert_status_code_200(response)


def test_chat_template_kwargs_deepseek_thinking_null_non_stream(api_client, request):
    """异常：deepseek模型thinking为null（非流式），错误码400"""
    model = request.config.getoption("--model")
    if not is_deepseek_model(model):
        pytest.skip(f"当前模型 {model} 不是deepseek/ds系列，跳过此测试")

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "chat_template_kwargs": {
            "thinking": None
        },
        "stream": False,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    assertion.assert_status_code_200(response)


def test_chat_template_kwargs_deepseek_thinking_null_stream(api_client, request):
    """异常：deepseek模型thinking为null（流式），状态码200，错误码400"""
    model = request.config.getoption("--model")
    if not is_deepseek_model(model):
        pytest.skip(f"当前模型 {model} 不是deepseek/ds系列，跳过此测试")

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "chat_template_kwargs": {
            "thinking": None
        },
        "stream": True,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    # 校验点：状态码200，错误码400
    assertion.assert_status_code_200(response)
    assertion.assert_error_code_400(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_qwen_enable_thinking_string(api_client, request, stream):
    """异常：qwen模型enable_thinking为字符串"""
    model = request.config.getoption("--model")
    if not is_qwen_model(model):
        pytest.skip(f"当前模型 {model} 不是qwen系列，跳过此测试")

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "chat_template_kwargs": {
            "enable_thinking": "true"
        },
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_deepseek_thinking_string(api_client, request, stream):
    """异常：deepseek模型thinking为字符串"""
    model = request.config.getoption("--model")
    if not is_deepseek_model(model):
        pytest.skip(f"当前模型 {model} 不是deepseek/ds系列，跳过此测试")

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "chat_template_kwargs": {
            "thinking": "true"
        },
        "stream": stream,
        "max_tokens": 512
    }

    response = helper.send_request(api_client, "/v1/chat/completions", request_body)

    assert response.status_code in [200, 400], f"状态码应为200或400，实际为{response.status_code}"
