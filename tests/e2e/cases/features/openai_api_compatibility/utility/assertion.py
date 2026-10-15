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


def assert_finish_reason_valid(finish_reason, msg=""):
    """Verify finish_reason is stop or length"""
    check.is_in(finish_reason, ["stop", "length"], f"{msg}finish_reason is not stop or length")


def assert_chat_completion_success(response, stream=False, msg=""):
    """Verify chat completion success response and return finish_reason."""
    assert_status_code_200(response, msg)

    if stream:
        assert_stream_has_done(response.text, msg)
        finish_reason = assert_stream_single_finish_reason(response.text, msg)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]

    assert_finish_reason_valid(finish_reason, msg)
    return finish_reason


def assert_completion_response_shape(response, stream=False, msg=""):
    """Verify a successful OpenAI completion response has a usable choices payload."""
    assert_status_code_200(response, msg)
    if stream:
        assert_stream_has_done(response.text, msg)
        return None

    choices = response.json().get("choices")
    check.is_true(isinstance(choices, list) and choices, f"{msg}Response choices should be a non-empty list")
    return choices


def assert_success_or_validation_error(response, stream=False, msg=""):
    """Verify an implementation-specific request either succeeds or is rejected cleanly."""
    if response.status_code == 200 and _extract_error_code(response) is None:
        return assert_chat_completion_success(response, stream=stream, msg=msg)

    assert_validation_error_response(response, msg)
    return None


def assert_validation_error_response(response, msg=""):
    """Verify request validation failed.

    vLLM may reject bad OpenAI requests either with an HTTP 400/422 response or
    with a streamed error event. Do not require an `error.code` field when the
    HTTP status itself already proves validation failed.
    """
    error_code = _extract_error_code(response)
    if response.status_code in (400, 422):
        if error_code is not None:
            check.is_in(error_code, [400, 422], f"{msg}Error code is not 400 or 422")
        return

    check.equal(response.status_code, 200, f"{msg}Response status code is not 200, 400 or 422")
    check.is_in(error_code, [400, 422], f"{msg}Successful HTTP response does not contain a validation error")


def assert_status_code_200_or_400(response, msg=""):
    """Verify status is accepted or rejected by implementation-specific validation."""
    check.is_in(response.status_code, [200, 400, 422], f"{msg}Response status code is not 200, 400 or 422")


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


def assert_error_code_400(response, msg=""):
    """Verify error code is 400"""
    error_code = _extract_error_code(response)
    check.equal(error_code, 400, f"{msg}Error code is not 400")


def _extract_error_code(response):
    """Return OpenAI-compatible error code from JSON or SSE responses."""
    if "application/json" in response.headers.get("Content-Type", ""):
        response_json = response.json()
        return response_json.get("error", {}).get("code") or response_json.get("code")
    elif "text/event-stream" in response.headers.get("Content-Type", ""):
        match = re.search(r"\"code\"\s?:\s?(\d+)", response.text, re.M)
        if match:
            return int(match.group(1))
    return None


def assert_image_edit_response_fields(response, msg=""):
    """Verify image edit response contains the required output fields."""
    assert_status_code_200(response, msg)
    response_json = response.json()
    check.is_true("data" in response_json, f"{msg}Response should contain data field")
    check.is_true(response_json.get("data"), f"{msg}data should contain at least one result")

    for idx, item in enumerate(response_json["data"]):
        has_b64 = bool(item.get("b64_json"))
        has_url = bool(item.get("url"))
        check.is_true(has_b64 or has_url, f"{msg}data[{idx}] should contain b64_json or url")

    return response_json


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
