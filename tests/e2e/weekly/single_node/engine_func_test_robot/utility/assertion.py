import re
import json
import pytest_check as check

# think标签定义
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def assert_status_code_200(response, msg=""):
    """校验HTTP状态码为200"""
    check.equal(response.status_code, 200, f"{msg}响应状态码不为200")


def assert_status_code_400(response, msg=""):
    """校验HTTP状态码为400"""
    check.equal(response.status_code, 400, f"{msg}响应状态码不为400")


def assert_finish_reason_stop(finish_reason, msg=""):
    """校验finish_reason为stop"""
    check.equal(finish_reason, "stop", f"{msg}finish_reason不为stop")


def assert_finish_reason_valid(finish_reason, msg=""):
    """校验finish_reason为stop或length"""
    check.is_in(finish_reason, ["stop", "length"], f"{msg}finish_reason不为stop或length")


def assert_stream_has_done(response_text, msg=""):
    """校验流式响应包含[DONE]"""
    check.is_true(
        re.search(r'^data:\s*\[DONE\](?:\n|$)', response_text, re.M),
        f"{msg}流式响应没有[DONE]"
    )


def assert_stream_single_finish_reason(response_text, msg=""):
    """校验流式响应只有一个finish_reason，返回该值"""
    finish_reasons = re.findall(r'finish_reason":\s*"([^"]+)"', response_text, re.M)
    check.equal(len(finish_reasons), 1, f"{msg}流式响应finish_reason存在多个值")
    return finish_reasons[0] if finish_reasons else None


def assert_think_tag_present(response_text, msg=""):
    """校验存在完整的think标签对"""
    think_open_count = response_text.count(THINK_OPEN)
    think_close_count = response_text.count(THINK_CLOSE)
    check.equal(think_open_count, think_close_count, f"{msg}think标签不完整, OPEN: {think_open_count}, CLOSE: {think_close_count}")
    check.equal(think_open_count, 1, f"{msg}没有think标签")


def assert_no_think_tag(response_text, msg=""):
    """校验不存在think标签"""
    check.equal(response_text.count(THINK_OPEN), 0, f"{msg}存在think标签")


def assert_json_response_content(response_text, msg=""):
    """校验响应内容为有效JSON（过滤think标签后）"""
    pattern = rf"\s*{re.escape(THINK_OPEN)}[\s\S]*?{re.escape(THINK_CLOSE)}"
    json_str = re.sub(pattern, "", response_text)
    match = re.search(r"(\{.*\})\s*(?:$|`|```)$", json_str, re.S)
    check.is_true(match, f"{msg}内容非JSON格式")
    if match:
        json.loads(match.group(1))


def has_error_code(response):
    """判断响应中是否存在error code"""
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        response_json = response.json()
        error_code = response_json.get("error", {}).get("code") or response_json.get("code")
        return error_code is not None
    elif "text/event-stream" in content_type or "text/plain" in content_type:
        match = re.search(r'\"code\"\s?:\s?(\d+)', response.text, re.M)
        return match is not None
    return False


def assert_error_code_400(response, msg=""):
    """校验错误码为400"""
    if "application/json" in response.headers.get("Content-Type", ""):
        error_code = response.json().get("error", {}).get("code") or response.json().get("code")
        check.equal(error_code, 400, f"{msg}错误码不为400")
    elif "text/event-stream" in response.headers.get("Content-Type", ""):
        match = re.search(r'\"code\"\s?:\s?(\d+)', response.text, re.M)
        if match:
            check.equal(int(match.group(1)), 400, f"{msg}流式响应错误码不为400")


def assert_error_code_422(response, msg=""):
    """校验错误码为422 Unprocessable Entity（数据验证失败）"""
    if "application/json" in response.headers.get("Content-Type", ""):
        error_code = response.json().get("error", {}).get("code") or response.json().get("code")
        check.equal(error_code, 422, f"{msg}错误码不为422（Unprocessable Entity-数据验证失败）")
    elif "text/event-stream" in response.headers.get("Content-Type", ""):
        match = re.search(r'\"code\"\s?:\s?(\d+)', response.text, re.M)
        if match:
            check.equal(int(match.group(1)), 422, f"{msg}流式响应错误码不为422")


def assert_error_code_not_500(response, msg=""):
    """校验如果响应体存在error code，则error code不是500"""
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        error_code = response.json().get("error", {}).get("code") or response.json().get("code")
        if error_code is not None:
            check.not_equal(error_code, 500, f"{msg}error code不应为500，实际: {error_code}")
    elif "text/event-stream" in content_type:
        match = re.search(r'\"code\"\s?:\s?(\d+)', response.text, re.M)
        if match:
            error_code = int(match.group(1))
            check.not_equal(error_code, 500, f"{msg}error code不应为500，实际: {error_code}")


def assert_image_edit_response_fields(response, msg=""):
    """校验图像编辑接口响应字段完整性

    Args:
        response: HTTP响应对象
        msg: 错误消息前缀
    """
    resp_json = response.json()

    # 校验顶层字段
    check.is_true("created" in resp_json, f"{msg}响应应包含created字段")
    check.is_true("data" in resp_json, f"{msg}响应应包含data字段")
    check.is_true("output_format" in resp_json, f"{msg}响应应包含output_format字段")
    check.is_true("size" in resp_json, f"{msg}响应应包含size字段")

    # 校验data数组
    data = resp_json.get("data", [])
    check.is_true(len(data) > 0, f"{msg}data应包含至少一个结果")

    # 校验data数组元素字段
    for idx, item in enumerate(data):
        has_b64 = "b64_json" in item and item["b64_json"]
        has_url = "url" in item and item["url"]
        check.is_true(has_b64 or has_url, f"{msg}data[{idx}]应包含b64_json或url字段")
        check.is_true("revised_prompt" in item, f"{msg}data[{idx}]应包含revised_prompt字段")

    return resp_json


def assert_top_logprobs_count(response, top_logprobs_value, msg=""):
    """校验logprobs的top_logprobs数量"""
    content_type = response.headers.get("Content-Type", "")

    if "application/json" in content_type:
        logprobs_content_list = response.json()["choices"][0]["logprobs"]["content"]
        for item_dict in logprobs_content_list:
            check.equal(
                len(item_dict.get("top_logprobs")),
                top_logprobs_value,
                f"{msg}logprobs top_logprobs长度不为{top_logprobs_value}"
            )
    elif "text/event-stream" in content_type:
        chunk_list = re.findall(r'^data:\s*(.*)(?:\n|$)', response.text, re.M)[1:-1]
        for chunk_item in chunk_list:
            chunk_json = json.loads(chunk_item)
            content = chunk_json["choices"][0]["delta"].get("content", "")    # todo⚠️：content为空时，top_logprobs为None,没有值。需要确认 or <|im_end|>...。即如下 if xxx else yyy 的 else 分支
            if content:
                logprobs_content_list = chunk_json["choices"][0]["logprobs"]["content"]
                for item_dict in logprobs_content_list:
                    check.equal(
                        len(item_dict.get("top_logprobs")),
                        top_logprobs_value,
                        f"{msg}流式logprobs top_logprobs长度不为{top_logprobs_value}"
                    )
