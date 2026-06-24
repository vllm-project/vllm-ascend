"""Shared LLM client for robot workflows.

Provides a single ``call_llm()`` entry point used by both the description
check and commit message check bots.
"""

import os

import requests

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT = 120
DEFAULT_TEMPERATURE = 0.3


def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Call the OpenAI-compatible chat completions endpoint and return the response.

    Reads API credentials from the environment:
    ``VLLM_API_KEY`` / ``LLM_API_KEY``, ``VLLM_BASE_URL`` / ``LLM_BASE_URL``,
    ``LLM_MODEL``.

    Args:
        system_prompt: System message content.
        user_prompt: User message content.
        max_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.

    Returns:
        The model response text.

    Raises:
        KeyError: If neither ``VLLM_API_KEY`` nor ``LLM_API_KEY`` is set.
        requests.HTTPError: If the API returns a non-2xx status.
        requests.Timeout: If the request exceeds *timeout*.
    """
    api_key = os.environ.get("LLM_API_KEY") or os.environ["VLLM_API_KEY"]
    base_url = os.environ.get("LLM_BASE_URL") or os.environ["VLLM_BASE_URL"]
    model = os.environ.get("LLM_MODEL", "default")

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
    return resp.json()["choices"][0]["message"]["content"]
