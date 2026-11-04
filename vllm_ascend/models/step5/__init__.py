# SPDX-License-Identifier: Apache-2.0
"""Ascend Step5 model: DSA sparse attention on the MRV2 runner.

The model skeleton and the CSA state layout (per-request tail ring + FP8
summary pages) come from the vLLM 0.33 upstream implementation; the
compute runs on the Ascend kernels proven in the MRV1 port.
"""

from vllm_ascend.models.step5.dsa_attention import (
    AscendStep5DSABackend,
    AscendStep5DSAImpl,
)
from vllm_ascend.models.step5.model import (
    AscendStep5DSAAttention,
    AscendStep5Attention,
    AscendStep5DecoderLayer,
    AscendStep5ForCausalLM,
    AscendStep5Model,
)
from vllm_ascend.models.step5.mtp import AscendStep5MTP
from vllm_ascend.models.step5.vl import AscendStep5ForConditionalGeneration

__all__ = [
    "AscendStep5DSABackend",
    "AscendStep5DSAImpl",
    "AscendStep5DSAAttention",
    "AscendStep5Attention",
    "AscendStep5DecoderLayer",
    "AscendStep5ForCausalLM",
    "AscendStep5Model",
    "AscendStep5MTP",
    "AscendStep5ForConditionalGeneration",
]
