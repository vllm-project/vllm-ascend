# from typing import Any, Optional
#
# import requests
#
#
# def validate_response(response_json: dict, expected: Optional[dict], max_model_len: Optional[int] = None) -> None:
#     """Validate token usage from API response."""
#     usage = response_json.get("usage", {})
#     prompt_tokens = usage.get("prompt_tokens", 0)
#     completion_tokens = usage.get("completion_tokens", 0)
#     total_tokens = prompt_tokens + completion_tokens
#
#     print(f"Token usage - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}")
#
#     if not expected:
#         return
#
#     if "prompt_tokens" in expected:
#         expected_prompt_tokens = expected["prompt_tokens"]
#         assert prompt_tokens == expected_prompt_tokens, (
#             f"prompt_tokens mismatch: got {prompt_tokens}, expected {expected_prompt_tokens}"
#         )
#
#     if "completion_tokens" in expected:
#         expected_completion_tokens = expected["completion_tokens"]
#         assert completion_tokens == expected_completion_tokens, (
#             f"completion_tokens mismatch: got {completion_tokens}, expected {expected_completion_tokens}"
#         )
#
#     limit = expected.get("max_model_len") or max_model_len
#     if limit is not None:
#         assert total_tokens <= int(limit), (
#             f"total_tokens ({total_tokens}) exceeds max_model_len ({limit})"
#         )
#
#
# def send_v1_completions(prompt, model, server, request_args=None, expected: Optional[dict] = None,
#                         max_model_len: Optional[int] = None):
#     data: dict[str, Any] = {"model": model, "prompt": prompt}
#     if request_args:
#         data.update(request_args)
#     url = server.url_for("v1", "completions")
#     response = requests.post(url, json=data)
#     print(f"Status Code: {response.status_code}")
#     response_json = response.json()
#     print(f"Response json: {response_json}")
#     response_text = response_json["choices"][0]["text"]
#     print(f"Response: {response_text}")
#     assert response_text, "empty response"
#     validate_response(response_json, expected, max_model_len)
#
#
# def send_v1_chat_completions(prompt, model, server, request_args=None, expected: Optional[dict] = None,
#                              max_model_len: Optional[int] = None):
#     data: dict[str, Any] = {
#         "model": model,
#         "messages": [
#             {
#                 "role": "user",
#                 "content": prompt,
#             }
#         ],
#     }
#     if request_args:
#         data.update(request_args)
#     url = server.url_for("v1", "chat", "completions")
#     response = requests.post(url, json=data)
#     print(f"Status Code: {response.status_code}")
#     response_json = response.json()
#     print(f"Response json: {response_json}")
#     response_text = response_json["choices"][0]["message"]["content"]
#     print(f"Response: {response_text}")
#     assert response_text, "empty response"
#     validate_response(response_json, expected, max_model_len)

from typing import Any, Optional

import requests


def validate_response(response_json: dict, expected: Optional[dict], max_model_len: Optional[int] = None) -> None:
    """Validate token usage from API response."""
    usage = response_json.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = prompt_tokens + completion_tokens

    print(f"Token usage - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}")

    if not expected:
        return

    if "prompt_tokens" in expected:
        expected_prompt_tokens = expected["prompt_tokens"]
        assert prompt_tokens == expected_prompt_tokens, (
            f"prompt_tokens mismatch: got {prompt_tokens}, expected {expected_prompt_tokens}"
        )

    if "completion_tokens" in expected:
        expected_completion_tokens = expected["completion_tokens"]
        assert completion_tokens == expected_completion_tokens, (
            f"completion_tokens mismatch: got {completion_tokens}, expected {expected_completion_tokens}"
        )

    limit = expected.get("max_model_len") or max_model_len
    if limit is not None:
        assert total_tokens <= int(limit), (
            f"total_tokens ({total_tokens}) exceeds max_model_len ({limit})"
        )


def send_v1_completions(prompt, model, server, request_args=None, expected: Optional[dict] = None,
                        max_model_len: Optional[int] = None):
    data: dict[str, Any] = {"model": model, "prompt": prompt}
    if request_args:
        data.update(request_args)
    url = server.url_for("v1", "completions")
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    response_json = response.json()
    print(f"Response json: {response_json}")
    response_text = response_json["choices"][0]["text"]
    print(f"Response: {response_text}")
    assert response_text, "empty response"
    validate_response(response_json, expected, max_model_len)


def send_v1_chat_completions(prompt, model, server, request_args=None, expected: Optional[dict] = None,
                             max_model_len: Optional[int] = None):
    data: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }
    if request_args:
        data.update(request_args)
    url = server.url_for("v1", "chat", "completions")
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    response_json = response.json()
    print(f"Response json: {response_json}")
    response_text = response_json["choices"][0]["message"]["content"]
    print(f"Response: {response_text}")
    assert response_text, "empty response"
    validate_response(response_json, expected, max_model_len)