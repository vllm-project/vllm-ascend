# SPDX-License-Identifier: Apache-2.0
"""Register Kimi K3 config before vLLM asks Transformers to load it."""

from vllm_ascend.transformers_utils.configs.kimi_k3 import register_kimi_k3_config

register_kimi_k3_config()
