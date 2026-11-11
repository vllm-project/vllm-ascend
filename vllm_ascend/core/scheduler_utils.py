# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.request import Request


def should_skip_external_kv(request: Request) -> bool:
    """Use the native cache-read policy across the v0.33.0 field rename."""
    try:
        return request.skip_reading_external_kv
    except AttributeError:
        return request.skip_reading_prefix_cache
