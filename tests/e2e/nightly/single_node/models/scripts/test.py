from typing import Any, Optional

import requests


def _tokenize_count(server, text: str, use_chat: bool = False) -> int:
    """Return the number of tokens the server would tokenize for `text`.

    When `use_chat` is True, the text is wrapped as a single user message so
    that the chat template is applied, matching what /v1/chat/completions
    would count as prompt_tokens.
    """
    url = server.url_for("tokenize")
    if use_chat:
        payload: dict[str, Any] = {"messages": [{"role": "user", "content": text}]}
    else:
        payload = {"prompt": text}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    body = r.json()
    return len(body.get("tokens") or body.get("token_ids", []))


def _generate_prompt_for_length(server, seed: str, target_tokens: int,
                                use_chat: bool = False) -> tuple[str, int]:
    single_count = _tokenize_count(server, seed, use_chat=use_chat)
    if single_count == 0:
        raise ValueError(f"seed {seed!r} tokenizes to 0 tokens")

    if target_tokens <= single_count:
        return seed, single_count

    est = max(1, target_tokens // single_count)
    body = "\n".join([seed] * est)
    actual = _tokenize_count(server, body, use_chat=use_chat)

    while actual > target_tokens and est > 1:
        est -= 1
        body = "\n".join([seed] * est)
        actual = _tokenize_count(server, body, use_chat=use_chat)

    if actual == target_tokens:
        return body, actual

    gap = target_tokens - actual
    if gap <= 0:
        return body, actual

    lo, hi = 0, max(gap * 4, 1)
    best_pad, best_count = 0, actual
    while lo < hi:
        mid = (lo + hi + 1) // 2
        cand = body + "a" * mid
        cand_count = _tokenize_count(server, cand, use_chat=use_chat)
        if cand_count <= target_tokens:
            lo = mid
            best_pad, best_count = mid, cand_count
        else:
            hi = mid - 1

    return body + "a" * best_pad, best_count


def resolve_prompt(server, raw, use_chat: bool = False) -> tuple[str, Optional[int]]:
    """Resolve a raw prompt spec. If dict {'seed':..., 'target_tokens':...},
    generate a prompt of that length and return (prompt, actual_count).
    Otherwise return (raw_string, None).

    Pass use_chat=True when the caller uses /v1/chat/completions so that the
    chat template overhead is accounted for during tokenization.
    """
    if isinstance(raw, dict):
        seed = str(raw.get("seed", ""))
        target = int(raw.get("target_tokens", 0))
        if not seed or not target:
            raise ValueError(f"prompt dict needs both 'seed' and 'target_tokens', got {raw}")
        prompt, actual = _generate_prompt_for_length(server, seed, target, use_chat=use_chat)
        print(f"[generate_prompt] seed={seed!r} target_tokens={target} actual={actual}")
        return prompt, actual
    return raw, None


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
    if expected and "completion_tokens" in expected:
        ct = expected["completion_tokens"]
        data["max_tokens"] = ct
        data["min_tokens"] = ct
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
    if expected and "completion_tokens" in expected:
        ct = expected["completion_tokens"]
        data["max_tokens"] = ct
        data["min_tokens"] = ct
    url = server.url_for("v1", "chat", "completions")
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    response_json = response.json()
    print(f"Response json: {response_json}")
    response_text = response_json["choices"][0]["message"]["content"]
    print(f"Response: {response_text}")
    assert response_text, "empty response"
    validate_response(response_json, expected, max_model_len)

# test_cases:
#   - name: "MyModel-Test"
#     model: "/path/to/model"
#     envs:
#       SERVER_PORT: "DEFAULT_PORT"
#     server_cmd:
#       - "--tensor-parallel-size"
#       - "8"
#       - "--port"
#       - "$SERVER_PORT"
#       - "--max-model-len"
#       - "8192"
#       - "--trust-remote-code"
#     test_content:
#       - chat_completion
#     prompts:
#       - seed: "Hello, how are you?"
#         target_tokens: 512
#       - seed: "请解释什么是大语言模型"
#         target_tokens: 1024
#       - "普通字符串也还可以用"
#     api_keyword_args:
#       max_tokens: 128
#     expected_response:
#       per_prompt:
#         - completion_tokens: 80
#         - completion_tokens: 120
#         - prompt_tokens: 8
#           completion_tokens: 30
