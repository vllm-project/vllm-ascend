import json
import re
from typing import Any, Optional

import requests


def validate_response(text: str, expected: dict, prompt: str = "") -> None:
    """Validate response text against expected rules."""
    if not expected:
        return

    # Keyword check
    contains = expected.get("contains", [])
    if isinstance(contains, str):
        contains = [contains]
    for keyword in contains:
        assert keyword in text, (
            f"Keyword '{keyword}' not found in response for prompt '{prompt}': {text[:200]}"
        )

    # Regex check
    regex = expected.get("regex")
    if regex:
        assert re.search(regex, text), (
            f"Regex '{regex}' not matched in response for prompt '{prompt}': {text[:200]}"
        )

    # Exact match
    equals = expected.get("equals")
    if equals is not None:
        assert text.strip() == str(equals).strip(), (
            f"Expected '{equals}', got '{text[:200]}' for prompt '{prompt}'"
        )

    # Fuzzy match (normalize whitespace)
    if expected.get("fuzzy_match") and equals:
        normalized_expected = re.sub(r'\s+', ' ', str(equals)).strip()
        normalized_actual = re.sub(r'\s+', ' ', text).strip()
        assert normalized_actual == normalized_expected, (
            f"Fuzzy match failed for prompt '{prompt}': "
            f"expected '{normalized_expected[:100]}', got '{normalized_actual[:100]}'"
        )

    # Length validation
    min_len = expected.get("min_length")
    max_len = expected.get("max_length")
    actual_len = len(text)
    if min_len is not None:
        assert actual_len >= min_len, (
            f"Response too short ({actual_len} < {min_len}) for prompt '{prompt}'"
        )
    if max_len is not None:
        assert actual_len <= max_len, (
            f"Response too long ({actual_len} > {max_len}) for prompt '{prompt}'"
        )

    # JSON validity check
    if expected.get("assert_json"):
        try:
            json.loads(text)
        except json.JSONDecodeError:
            raise AssertionError(
                f"Response is not valid JSON for prompt '{prompt}': {text[:200]}"
            )


def send_v1_completions(prompt, model, server, request_args=None, expected: Optional[dict] = None):
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
    validate_response(response_text, expected, prompt)


def send_v1_chat_completions(prompt, model, server, request_args=None, expected: Optional[dict] = None):
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
    validate_response(response_text, expected, prompt)