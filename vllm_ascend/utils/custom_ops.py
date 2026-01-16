#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""
Custom operation registration utilities for vLLM Ascend.

This module provides functionality for:
- Enabling lazy initialization of custom ops
- Registering Ascend-specific custom operations
"""

from typing import TYPE_CHECKING, Optional

from vllm.logger import logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

_CUSTOM_OP_ENABLED = None
_ASCEND_CUSTOMOP_IS_REIGISTERED = False
REGISTERED_ASCEND_OPS = {}


def enable_custom_op():
    """
    Enable lazy init for vllm_ascend_C to avoid early initialization of CANN's RTS component.
    Ensure that ASCEND_RT_VISIBLE_DEVICES can be dynamically modified before torch.npu.set_device().
    """
    global _CUSTOM_OP_ENABLED

    if _CUSTOM_OP_ENABLED is not None:
        return _CUSTOM_OP_ENABLED
    try:
        # isort: off
        # register custom ops into torch_library here
        import vllm_ascend.vllm_ascend_C  # type: ignore  # noqa: F401
        # register the meta implementation for custom kernel if necessary
        import vllm_ascend.meta_registration  # type: ignore  # noqa: F401
        # isort: on
        _CUSTOM_OP_ENABLED = True
    except ImportError:
        _CUSTOM_OP_ENABLED = False
        logger.warning(
            "Warning: Failed to register custom ops, all custom ops will be disabled"
        )
    return _CUSTOM_OP_ENABLED


def register_ascend_customop(vllm_config: Optional[VllmConfig] = None):
    """Register Ascend CustomOP

    NOTE: if the register branch requires model type, please use `vllm.config.get_current_vllm_config`,
    and ensure this will execute after model config is initilazed.
    """
    global _ASCEND_CUSTOMOP_IS_REIGISTERED
    if _ASCEND_CUSTOMOP_IS_REIGISTERED:
        return
    from vllm.model_executor.custom_op import CustomOp

    from vllm_ascend.ops.activation import AscendQuickGELU, AscendSiluAndMul
    from vllm_ascend.ops.fused_moe.fused_moe import (AscendFusedMoE,
                                                     AscendSharedFusedMoE)
    from vllm_ascend.ops.layernorm import AscendGemmaRMSNorm, AscendRMSNorm
    from vllm_ascend.ops.linear import (AscendColumnParallelLinear,
                                        AscendMergedColumnParallelLinear,
                                        AscendQKVParallelLinear,
                                        AscendReplicatedLinear,
                                        AscendRowParallelLinear)
    from vllm_ascend.ops.mla import AscendMultiHeadLatentAttention
    from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention
    from vllm_ascend.ops.rotary_embedding import (
        AscendApplyRotaryEmb, AscendDeepseekScalingRotaryEmbedding,
        AscendMRotaryEmbedding, AscendRotaryEmbedding,
        AscendYaRNRotaryEmbedding)
    from vllm_ascend.ops.vocab_parallel_embedding import (
        AscendLogitsProcessor, AscendParallelLMHead,
        AscendVocabParallelEmbedding)

    global REGISTERED_ASCEND_OPS
    REGISTERED_ASCEND_OPS = {
        "QuickGELU": AscendQuickGELU,
        "SiluAndMul": AscendSiluAndMul,
        "RotaryEmbedding": AscendRotaryEmbedding,
        "MRotaryEmbedding": AscendMRotaryEmbedding,
        "ColumnParallelLinear": AscendColumnParallelLinear,
        "RowParallelLinear": AscendRowParallelLinear,
        "YaRNScalingRotaryEmbedding": AscendYaRNRotaryEmbedding,
        "MergedColumnParallelLinear": AscendMergedColumnParallelLinear,
        "QKVParallelLinear": AscendQKVParallelLinear,
        "ReplicatedLinear": AscendReplicatedLinear,
        "DeepseekScalingRotaryEmbedding": AscendDeepseekScalingRotaryEmbedding,
        "VocabParallelEmbedding": AscendVocabParallelEmbedding,
        "ParallelLMHead": AscendParallelLMHead,
        "LogitsProcessor": AscendLogitsProcessor,
        "RMSNorm": AscendRMSNorm,
        "GemmaRMSNorm": AscendGemmaRMSNorm,
        "FusedMoE": AscendFusedMoE,
        "SharedFusedMoE": AscendSharedFusedMoE,
        "MultiHeadLatentAttentionWrapper": AscendMultiHeadLatentAttention,
        "MMEncoderAttention": AscendMMEncoderAttention,
        "ApplyRotaryEmb": AscendApplyRotaryEmb,
    }

    for name, op_cls in REGISTERED_ASCEND_OPS.items():
        CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)

    # NOTE: Keep this at last to ensure all custom actions are registered
    _ASCEND_CUSTOMOP_IS_REIGISTERED = True
