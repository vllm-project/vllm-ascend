# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend Qwen3.5 MoE adapters with opt-in expert reduction."""

from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForCausalLM
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MoeMTP

from vllm_ascend.models.qwen3_5_reduction import Qwen3_5ExpertReductionMixin


class AscendQwen3_5MoeForCausalLM(Qwen3_5ExpertReductionMixin, Qwen3_5MoeForCausalLM):
    pass


class AscendQwen3_5MoeMTP(Qwen3_5ExpertReductionMixin, Qwen3_5MoeMTP):
    pass
