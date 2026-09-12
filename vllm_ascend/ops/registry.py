#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

"""Ascend OOT CustomOp registry.

This module is the single source of truth for the Ascend out-of-tree
CustomOp catalog and its registration entry points, including
``register_all_custom_ops`` which moved here from ``vllm_ascend.utils``.

Custom ops are imported lazily inside cached helpers so that merely
importing this module does not trigger CANN/torch-npu initialization.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING

from vllm.model_executor.custom_op import CustomOp

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@cache
def _get_ops_base() -> dict[str, type]:
    from vllm_ascend.ops.activation import (
        AscendQuickGELU,
        AscendSiluAndMul,
        AscendSiluAndMulWithClamp,
    )
    from vllm_ascend.ops.bailing_moe_linear_attn import AscendBailingMoELinearAttention
    from vllm_ascend.ops.conv import AscendConv3dLayer
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    from vllm_ascend.ops.fused_moe.gate_linear import AscendGateLinear
    from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
    from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
    from vllm_ascend.ops.layernorm import AscendGemmaRMSNorm, AscendRMSNorm, AscendRMSNormGated
    from vllm_ascend.ops.linear import (
        AscendColumnParallelLinear,
        AscendMergedColumnParallelLinear,
        AscendQKVParallelLinear,
        AscendReplicatedLinear,
        AscendRowParallelLinear,
    )
    from vllm_ascend.ops.mla import AscendMultiHeadLatentAttention
    from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention
    from vllm_ascend.ops.qwen2_decoder import AscendCustomQwen2Decoder
    from vllm_ascend.ops.rel_pos_attention import AscendRelPosAttention
    from vllm_ascend.ops.rotary_embedding import (
        AscendApplyRotaryEmb,
        AscendDeepseekScalingRotaryEmbedding,
        AscendMRotaryEmbedding,
        AscendRotaryEmbedding,
        AscendYaRNRotaryEmbedding,
    )
    from vllm_ascend.ops.vocab_parallel_embedding import (
        AscendLogitsProcessor,
        AscendParallelLMHead,
        AscendVocabParallelEmbedding,
    )

    return {
        "QuickGELU": AscendQuickGELU,
        "SiluAndMul": AscendSiluAndMul,
        "SiluAndMulClamp": AscendSiluAndMulWithClamp,
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
        "MultiHeadLatentAttentionWrapper": AscendMultiHeadLatentAttention,
        "MMEncoderAttention": AscendMMEncoderAttention,
        "ApplyRotaryEmb": AscendApplyRotaryEmb,
        "RMSNormGated": AscendRMSNormGated,
        "Conv3dLayer": AscendConv3dLayer,
        "RelPosAttention": AscendRelPosAttention,
        "CustomQwen2Decoder": AscendCustomQwen2Decoder,
        "GatedDeltaNetAttention": AscendGatedDeltaNetAttention,
        "BailingMoELinearAttention": AscendBailingMoELinearAttention,
        "MoERunner": AscendMoERunner,
        "RoutedExperts": AscendRoutedExperts,
        "GateLinear": AscendGateLinear,
    }


@cache
def _get_ops_310p() -> dict[str, type]:
    from vllm_ascend._310p.fused_moe.fused_moe import AscendMoERunner310, AscendRoutedExperts310
    from vllm_ascend._310p.ops.activation import AscendSiluAndMul310
    from vllm_ascend._310p.ops.conv import AscendConv3dLayer310
    from vllm_ascend._310p.ops.fla.gdn_310 import AscendGatedDeltaNetAttention310
    from vllm_ascend._310p.ops.layernorm import (
        AscendGemmaRMSNorm310,
        AscendRMSNorm310,
        AscendRMSNormGated310,
    )
    from vllm_ascend._310p.ops.mm_encoder_attention import AscendMMEncoderAttention310
    from vllm_ascend._310p.ops.rotary_embedding import AscendMRotaryEmbedding310, AscendRotaryEmbedding310
    from vllm_ascend._310p.ops.vocab_parallel_embedding import (
        AscendParallelLMHead310,
        AscendVocabParallelEmbedding310,
    )

    return {
        "SiluAndMul": AscendSiluAndMul310,
        "RotaryEmbedding": AscendRotaryEmbedding310,
        "RMSNorm": AscendRMSNorm310,
        "GemmaRMSNorm": AscendGemmaRMSNorm310,
        "RMSNormGated": AscendRMSNormGated310,
        "ParallelLMHead": AscendParallelLMHead310,
        "VocabParallelEmbedding": AscendVocabParallelEmbedding310,
        "MMEncoderAttention": AscendMMEncoderAttention310,
        "Conv3dLayer": AscendConv3dLayer310,
        "GatedDeltaNetAttention": AscendGatedDeltaNetAttention310,
        "MRotaryEmbedding": AscendMRotaryEmbedding310,
        "MoERunner": AscendMoERunner310,
        "RoutedExperts": AscendRoutedExperts310,
    }


def ascend_custom_ops() -> Mapping[str, type]:
    if get_current_hardware_profile().supports(HardwareCapability.COMPATIBILITY_OP_IMPLEMENTATIONS):
        return MappingProxyType({**_get_ops_base(), **_get_ops_310p()})
    return MappingProxyType(_get_ops_base())


def register_custom_op(name: str, op_cls: type | None = None) -> None:
    """Registers a single Ascend OOT CustomOp."""
    if op_cls is None:
        op_cls = ascend_custom_ops()[name]
    CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)


def register_custom_ops(
    include: Collection[str] | None = None,
    exclude: Collection[str] = (),
) -> None:
    """Registers a subset of the Ascend OOT CustomOps."""
    custom_ops = ascend_custom_ops()
    names = include if include is not None else custom_ops
    for name in names:
        if name not in exclude:
            register_custom_op(name)


# register_all_custom_ops runs at most once per process.
_registered_all_custom_ops = False


def register_all_custom_ops(vllm_config: VllmConfig | None = None):
    """Register Ascend CustomOP

    NOTE: if the register branch requires model type, please use `vllm.config.get_current_vllm_config`,
    and ensure this will execute after model config is initilazed.
    """
    global _registered_all_custom_ops
    if _registered_all_custom_ops:
        return
    _registered_all_custom_ops = True

    if vllm_config is None:
        try:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
        except AssertionError:
            vllm_config = None

    exclude = set()
    if vllm_config is None or vllm_config.model_config is None or not vllm_config.model_config.is_deepseek_mla:
        # GateLinear is needed by DeepSeek MLA models.
        exclude.add("GateLinear")
    register_custom_ops(exclude=exclude)
