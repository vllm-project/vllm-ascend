# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""DeepSeek V4.1 frontend compatibility for vLLM releases before native support."""

try:
    from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config
except ImportError:
    from .config import DeepseekV41Config

__all__ = ["DeepseekV41Config"]
