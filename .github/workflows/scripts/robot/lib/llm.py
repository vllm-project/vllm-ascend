"""Shared LLM client for robot workflows.

Provides a single ``call_llm()`` entry point used by both the description
check and commit message check bots.
"""

import os
import time

import requests

DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 120
DEFAULT_TEMPERATURE = 0.3
DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = 2


def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> str:
    """Call the OpenAI-compatible chat completions endpoint and return the response.

    Reads API credentials from the environment:
    ``LLM_API_KEY``, ``LLM_BASE_URL``, ``LLM_MODEL``.

    Retries up to *retries* times on empty response or transient errors.

    Args:
        system_prompt: System message content.
        user_prompt: User message content.
        max_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.
        retries: Number of retry attempts on empty or failed response.
        retry_delay: Seconds to wait between retries.

    Returns:
        The model response text.

    Raises:
        KeyError: If ``LLM_API_KEY`` or ``LLM_BASE_URL`` is not set.
        RuntimeError: If all retries are exhausted and response is still empty.
        requests.HTTPError: If the API returns a non-2xx status.
        requests.Timeout: If the request exceeds *timeout*.
    """
    api_key = os.environ["LLM_API_KEY"]
    base_url = os.environ["LLM_BASE_URL"]
    model = os.environ.get("LLM_MODEL", "default")

    last_content = ""
    for attempt in range(1, retries + 1):
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": DEFAULT_TEMPERATURE,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        if content:
            return content
        last_content = content
        if attempt < retries:
            print(f"  WARN: empty LLM response (attempt {attempt}/{retries}), retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    return last_content
