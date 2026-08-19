from typing import Any, Optional

import requests


def _tokenize_count(server, prompt: str) -> int:
    """Return the number of tokens the server would tokenize for `prompt`."""
    url = server.url_for("tokenize")
    r = requests.post(url, json={"prompt": prompt}, timeout=60)
    r.raise_for_status()
    body = r.json()
    return len(body.get("tokens") or body.get("token_ids", []))


def _generate_prompt_for_length(server, seed: str, target_tokens: int) -> tuple[str, int]:
    """Repeat `seed` until its tokenized length equals `target_tokens` (best-effort <=).

    Uses binary search on the repetition count via the server's /tokenize endpoint.
    Returns (generated_prompt, actual_token_count).
    """
    if target_tokens <= 0:
        base_count = _tokenize_count(server, seed)
        return seed, base_count

    single_count = _tokenize_count(server, seed)
    if single_count == 0:
        raise ValueError(f"seed {seed!r} tokenizes to 0 tokens")

    lo, hi = 1, max(1, target_tokens)
    while _tokenize_count(server, "\n".join([seed] * hi)) < target_tokens:
        hi *= 2

    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _tokenize_count(server, "\n".join([seed] * mid)) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1

    prompt = "\n".join([seed] * lo)
    return prompt, _tokenize_count(server, prompt)


def resolve_prompt(server, raw) -> tuple[str, Optional[int]]:
    """Resolve a raw prompt spec. If dict {'seed':..., 'target_tokens':...},
    generate a prompt of that length and return (prompt, actual_count).
    Otherwise return (raw_string, None).
    """
    if isinstance(raw, dict):
        seed = str(raw.get("seed", ""))
        target = int(raw.get("target_tokens", 0))
        if not seed or not target:
            raise ValueError(f"prompt dict needs both 'seed' and 'target_tokens', got {raw}")
        prompt, actual = _generate_prompt_for_length(server, seed, target)
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