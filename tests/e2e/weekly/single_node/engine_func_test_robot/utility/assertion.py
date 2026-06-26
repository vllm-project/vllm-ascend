import json

import regex as re

# think tag definitions
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


class Check:
    @staticmethod
    def equal(a, b, msg=""):
        assert a == b, msg

    @staticmethod
    def not_equal(a, b, msg=""):
        assert a != b, msg

    @staticmethod
    def is_true(v, msg=""):
        assert v, msg

    @staticmethod
    def is_in(v, seq, msg=""):
        assert v in seq, msg


check = Check()


def assert_status_code_200(response, msg=""):
    """Verify HTTP status code is 200"""
    check.equal(response.status_code, 200, f"{msg}Response status code is not 200")


def assert_status_code_400(response, msg=""):
    """Verify HTTP status code is 400"""
    check.equal(response.status_code, 400, f"{msg}Response status code is not 400")


def assert_finish_reason_stop(finish_reason, msg=""):
    """Verify finish_reason is stop"""
    check.equal(finish_reason, "stop", f"{msg}finish_reason is not stop")


def assert_finish_reason_valid(finish_reason, msg=""):
    """Verify finish_reason is stop or length"""
    check.is_in(finish_reason, ["stop", "length"], f"{msg}finish_reason is not stop or length")


def assert_stream_has_done(response_text, msg=""):
    """Verify streaming response contains [DONE]"""
    check.is_true(
        re.search(r"^data:\s*\[DONE\](?:\n|$)", response_text, re.M),
        f"{msg}Streaming response does not contain [DONE]",
    )


def assert_stream_single_finish_reason(response_text, msg=""):
    """Verify streaming response has exactly one finish_reason, return its value"""
    finish_reasons = re.findall(r'finish_reason":\s*"([^"]+)"', response_text, re.M)
    check.equal(len(finish_reasons), 1, f"{msg}Streaming response has multiple finish_reason values")
    return finish_reasons[0] if finish_reasons else None


def assert_think_tag_present(response_text, msg=""):
    """Verify complete think tag pairs exist"""
    think_open_count = response_text.count(THINK_OPEN)
    think_close_count = response_text.count(THINK_CLOSE)
    check.equal(
        think_open_count,
        think_close_count,
        f"{msg}think tags are not balanced, OPEN: {think_open_count}, CLOSE: {think_close_count}",
    )
    check.equal(think_open_count, 1, f"{msg}No think tag present")


def assert_no_think_tag(response_text, msg=""):
    """Verify think tags do not exist"""
    check.equal(response_text.count(THINK_OPEN), 0, f"{msg}think tag exists")


def assert_json_response_content(response_text, msg=""):
    """Verify response content is valid JSON (after filtering think tags)"""
    pattern = rf"\s*{re.escape(THINK_OPEN)}[\s\S]*?{re.escape(THINK_CLOSE)}"
    json_str = re.sub(pattern, "", response_text)
    match = re.search(r"(\{.*\})\s*(?:$|`|```)$", json_str, re.S)
    check.is_true(match, f"{msg}Content is not in JSON format")
    if match:
        json.loads(match.group(1))


def has_error_code(response):
    """Determine if the response contains an error code"""
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        response_json = response.json()
        error_code = response_json.get("error", {}).get("code") or response_json.get("code")
        return error_code is not None
    elif "text/event-stream" in content_type or "text/plain" in content_type:
        match = re.search(r"\"code\"\s?:\s?(\d+)", response.text, re.M)
        return match is not None
    return False


def assert_error_code_400(response, msg=""):
    """Verify error code is 400"""
    if "application/json" in response.headers.get("Content-Type", ""):
        error_code = response.json().get("error", {}).get("code") or response.json().get("code")
        check.equal(error_code, 400, f"{msg}Error code is not 400")
    elif "text/event-stream" in response.headers.get("Content-Type", ""):
        match = re.search(r"\"code\"\s?:\s?(\d+)", response.text, re.M)
        if match:
            check.equal(int(match.group(1)), 400, f"{msg}Streaming response error code is not 400")


def assert_error_code_422(response, msg=""):
    """Verify error code is 422 Unprocessable Entity (data validation failure)"""
    if "application/json" in response.headers.get("Content-Type", ""):
        error_code = response.json().get("error", {}).get("code") or response.json().get("code")
        check.equal(error_code, 422, f"{msg}Error code is not 422 (Unprocessable Entity - data validation failure)")
    elif "text/event-stream" in response.headers.get("Content-Type", ""):
        match = re.search(r"\"code\"\s?:\s?(\d+)", response.text, re.M)
        if match:
            check.equal(int(match.group(1)), 422, f"{msg}Streaming response error code is not 422")


def assert_error_code_not_500(response, msg=""):
    """If response body contains an error code, verify it is not 500"""
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        error_code = response.json().get("error", {}).get("code") or response.json().get("code")
        if error_code is not None:
            check.not_equal(error_code, 500, f"{msg}Error code should not be 500, actual: {error_code}")
    elif "text/event-stream" in content_type:
        match = re.search(r"\"code\"\s?:\s?(\d+)", response.text, re.M)
        if match:
            error_code = int(match.group(1))
            check.not_equal(error_code, 500, f"{msg}Error code should not be 500, actual: {error_code}")


def assert_image_edit_response_fields(response, msg=""):
    """Verify completeness of response fields for image edit API

    Args:
        response: HTTP response object
        msg: Prefix for error messages
    """
    resp_json = response.json()

    # Verify top-level fields
    check.is_true("created" in resp_json, f"{msg}Response should contain created field")
    check.is_true("data" in resp_json, f"{msg}Response should contain data field")
    check.is_true("output_format" in resp_json, f"{msg}Response should contain output_format field")
    check.is_true("size" in resp_json, f"{msg}Response should contain size field")

    # Verify data array
    data = resp_json.get("data", [])
    check.is_true(len(data) > 0, f"{msg}data should contain at least one result")

    # Verify fields of each data array element
    for idx, item in enumerate(data):
        has_b64 = "b64_json" in item and item["b64_json"]
        has_url = "url" in item and item["url"]
        check.is_true(has_b64 or has_url, f"{msg}data[{idx}] should contain b64_json or url field")
        check.is_true("revised_prompt" in item, f"{msg}data[{idx}] should contain revised_prompt field")

    return resp_json


def assert_top_logprobs_count(response, top_logprobs_value, msg=""):
    """Verify the number of top_logprobs in logprobs"""
    content_type = response.headers.get("Content-Type", "")

    if "application/json" in content_type:
        logprobs_content_list = response.json()["choices"][0]["logprobs"]["content"]
        for item_dict in logprobs_content_list:
            check.equal(
                len(item_dict.get("top_logprobs")),
                top_logprobs_value,
                f"{msg}logprobs top_logprobs length is not {top_logprobs_value}",
            )
    elif "text/event-stream" in content_type:
        chunk_list = re.findall(r"^data:\s*(.*)(?:\n|$)", response.text, re.M)[1:-1]
        for chunk_item in chunk_list:
            chunk_json = json.loads(chunk_item)
            content = chunk_json["choices"][0]["delta"].get("content", "")
            if content:
                logprobs_content_list = chunk_json["choices"][0]["logprobs"]["content"]
                for item_dict in logprobs_content_list:
                    check.equal(
                        len(item_dict.get("top_logprobs")),
                        top_logprobs_value,
                        f"{msg}Streaming logprobs top_logprobs length is not {top_logprobs_value}",
                    )
