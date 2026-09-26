# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Require a correlated pool load after the prefiller's local cache is reset."""

import regex as re


def pool_load_observed(lines: list[str], request_id: str) -> bool:
    request = re.escape(request_id)
    allocated = re.compile(
        rf"KV pool load spec enabled req={request} num_external_tokens=[1-9]\d* "
        r"vllm_cached=0 kvpool_cached=[1-9]\d* "
    )
    completed = re.compile(
        rf"KV pool worker backend get returned request={request} token_len=[1-9]\d* "
        r"groups=\[[^\]]+\] keys=[1-9]\d*"
    )
    prefill_lines = [line for line in lines if line.startswith("[PD_0]")]
    if any("Failed to get " in line or "Failed to put " in line for line in prefill_lines):
        return False
    return any(allocated.search(line) for line in prefill_lines) and any(
        completed.search(line) for line in prefill_lines
    )
