# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Install DeepSeek V4.1 frontend fallbacks for vLLM v0.29."""

from vllm_ascend.compat.deepseek_v41.registration import (
    register_deepseek_v41_compat,
)

register_deepseek_v41_compat()
