#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading

from vllm.logger import init_logger
from vllm.v1.metrics.stats import PromptTokenStats

logger = init_logger(__name__)

_original_update_from_output = PromptTokenStats.update_from_output
_warned_invalid_prompt_token_stats = False
_warn_lock = threading.Lock()


def _patched_update_from_output(
    self,
    num_cached_tokens: int,
    num_external_computed_tokens: int,
    prompt_len: int,
) -> None:
    safe_prompt_len = max(prompt_len, 0)
    safe_cached_tokens = max(0, min(num_cached_tokens, safe_prompt_len))
    # ``num_external_computed_tokens`` is a subset of cached tokens and must
    # never exceed the normalized cached-token count.
    max_external_tokens = safe_cached_tokens
    safe_external_tokens = max(0, min(num_external_computed_tokens, max_external_tokens))

    global _warned_invalid_prompt_token_stats
    has_invalid_token_stats = (
        safe_prompt_len != prompt_len
        or safe_cached_tokens != num_cached_tokens
        or safe_external_tokens != num_external_computed_tokens
    )
    if has_invalid_token_stats and not _warned_invalid_prompt_token_stats:
        with _warn_lock:
            if not _warned_invalid_prompt_token_stats:
                logger.warning(
                    "Detected invalid prompt token stats: prompt_len=%d, num_cached_tokens=%d, "
                    "num_external_computed_tokens=%d; normalized to prompt_len=%d, num_cached_tokens=%d, "
                    "num_external_computed_tokens=%d",
                    prompt_len,
                    num_cached_tokens,
                    num_external_computed_tokens,
                    safe_prompt_len,
                    safe_cached_tokens,
                    safe_external_tokens,
                )
                _warned_invalid_prompt_token_stats = True

    _original_update_from_output(
        self,
        safe_cached_tokens,
        safe_external_tokens,
        safe_prompt_len,
    )


PromptTokenStats.update_from_output = _patched_update_from_output
