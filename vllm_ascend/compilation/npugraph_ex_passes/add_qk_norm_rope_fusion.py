#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
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
import functools

import torch
from torch._inductor.pattern_matcher import Match
from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import logger


def _extra_stream_scope_check(match: Match) -> bool:
    """
    Checks if all nodes in the same stream.
    """
    non_default_streams = set()
    has_default = False

    for node in match.nodes:
        if node.op == "call_function":
            current_stream = node.meta.get("stream_label")
            if current_stream is None:
                has_default = True
            else:
                non_default_streams.add(current_stream)
                if len(non_default_streams) > 1:
                    logger.debug(
                        f"Cross-stream operation detected in pattern match for QKNormRope. "
                        f"Multiple streams found: {non_default_streams}. "
                        f"Fusion is not supported for cross-stream operations."
                    )
                    return False

    if has_default and len(non_default_streams) > 0:
        logger.debug(
            f"Cross-stream operation detected in pattern match for QKNormRope. "
            f"Multiple streams found: {non_default_streams}. "
            f"Fusion is not supported for cross-stream operations.")
        return False

    return True


@functools.lru_cache(None)
# The replacement registered here will be actually executed after AOT.
def register_qknorm_rope_fusion(head_dim, num_heads, num_kv_heads,
                                eps):

    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    def pattern(qkv: torch.Tensor, q_weight: torch.Tensor,
                k_weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):

        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_dim, head_dim)
        q_norm_out, _ = torch.ops.npu.npu_rms_norm(q_by_head, q_weight, eps)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // head_dim, head_dim)
        k_norm_out, _ = torch.ops.npu.npu_rms_norm(k_by_head, k_weight, eps)

        q_flat = q_norm_out.view(q.shape)
        q_reshape = q_flat.contiguous().view(1, q_flat.shape[0], -1, head_dim)

        k_flat = k_norm_out.view(k.shape)
        k_reshape = k_flat.contiguous().view(1, k_flat.shape[0], -1, head_dim)

        q_rope, k_rope = torch.ops.npu.npu_apply_rotary_pos_emb(
            q_reshape, k_reshape, cos, sin)

        return q_rope, k_rope, v

    def replacement(qkv: torch.Tensor, q_weight: torch.Tensor,
                    k_weight: torch.Tensor, cos: torch.Tensor,
                    sin: torch.Tensor):
        results = torch.ops.vllm.qkv_rmsnorm_rope(input=qkv,
                                                  q_weight=q_weight,
                                                  k_weight=k_weight,
                                                  q_hidden_size=q_size,
                                                  kv_hidden_size=kv_size,
                                                  head_dim=head_dim,
                                                  eps=eps,
                                                  q_bias=None,
                                                  k_bias=None,
                                                  sin=sin,
                                                  cos=cos)

        return results

    def get_inputs():
        T = 5
        qkv = torch.empty(T,
                          q_size + 2 * kv_size,
                          dtype=torch.bfloat16,
                          device="npu")
        q_weight = torch.empty(head_dim, dtype=torch.bfloat16, device="npu")
        k_weight = torch.empty(head_dim, dtype=torch.bfloat16, device="npu")
        cos = torch.empty(1,
                          T,
                          1,
                          head_dim,
                          dtype=torch.bfloat16,
                          device="npu")
        sin = torch.empty(1,
                          T,
                          1,
                          head_dim,
                          dtype=torch.bfloat16,
                          device="npu")
        return [qkv, q_weight, k_weight, cos, sin]

    import torchair

    torchair.register_replacement(search_fn=pattern,
                                  replace_fn=replacement,
                                  example_inputs=get_inputs(),
                                  extra_check=_extra_stream_scope_check)


# The replacement registered here will be actually executed after AOT.
def register_qknorm_rope_fusion_with_bias(head_dim, num_heads,
                                          num_kv_heads, eps):

    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    def pattern(qkv: torch.Tensor, q_weight: torch.Tensor,
                k_weight: torch.Tensor, q_bias: torch.Tensor,
                k_bias: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_dim, head_dim)
        q_norm_out, _ = torch.ops.npu.npu_rms_norm(q_by_head, q_weight, eps)
        q_normed = q_norm_out + q_bias

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // head_dim, head_dim)
        k_norm_out, _ = torch.ops.npu.npu_rms_norm(k_by_head, k_weight, eps)
        k_normed = k_norm_out + k_bias

        q_flat = q_normed.view(q.shape)
        q_reshape = q_flat.contiguous().view(1, q_flat.shape[0], -1, head_dim)

        k_flat = k_normed.view(k.shape)
        k_reshape = k_flat.contiguous().view(1, k_flat.shape[0], -1, head_dim)

        q_rope, k_rope = torch.ops.npu.npu_apply_rotary_pos_emb(
            q_reshape, k_reshape, cos, sin)

        return q_rope, k_rope, v

    def replacement(qkv: torch.Tensor, q_weight: torch.Tensor,
                    k_weight: torch.Tensor, q_bias: torch.Tensor,
                    k_bias: torch.Tensor, cos: torch.Tensor,
                    sin: torch.Tensor):
        results = torch.ops.vllm.qkv_rmsnorm_rope(input=qkv,
                                                  q_weight=q_weight,
                                                  k_weight=k_weight,
                                                  q_hidden_size=q_size,
                                                  kv_hidden_size=kv_size,
                                                  head_dim=head_dim,
                                                  eps=eps,
                                                  q_bias=q_bias,
                                                  k_bias=k_bias,
                                                  sin=sin,
                                                  cos=cos)

        return results

    def get_inputs():
        T = 5
        qkv = torch.empty(T,
                          q_size + 2 * kv_size,
                          dtype=torch.bfloat16,
                          device="npu")
        q_weight = torch.empty(head_dim, dtype=torch.bfloat16, device="npu")
        k_weight = torch.empty(head_dim, dtype=torch.bfloat16, device="npu")
        q_bias = torch.empty(head_dim, dtype=torch.bfloat16, device="npu")
        k_bias = torch.empty(head_dim, dtype=torch.bfloat16, device="npu")
        cos = torch.empty(1,
                          T,
                          1,
                          head_dim,
                          dtype=torch.bfloat16,
                          device="npu")
        sin = torch.empty(1,
                          T,
                          1,
                          head_dim,
                          dtype=torch.bfloat16,
                          device="npu")
        return [qkv, q_weight, k_weight, q_bias, k_bias, cos, sin]

    import torchair

    torchair.register_replacement(search_fn=pattern,
                                  replace_fn=replacement,
                                  example_inputs=get_inputs(),
                                  extra_check=_extra_stream_scope_check)


def register_qknorm_rope_fusions(vllm_config: VllmConfig) -> None:
    """
    Register QKNorm and Rope fusion patterns with the given vllm_config.
    
    Args:
        vllm_config: The vLLM configuration object containing model settings.
    """
    attn_layers: dict[str, Attention] = get_layers_from_vllm_config(
        vllm_config, Attention)
    
    if len(attn_layers) == 0:
        logger.debug(
            "QKNorm and Rope fusion enabled, but no Attention layers were discovered."
        )
        return
    
    layer = next(iter(attn_layers.values()))
    if layer.head_size != 128:
        logger.debug(
            "QKNorm and Rope fusion not enabled: head_dim %d is not equal to 128",
            layer.head_size)
        return
    
    # register converter for pass
    common_epsilons = [1e-5, 1e-6]
    for eps in common_epsilons:
        logger.info(
            f"Start register fusion pattern for QKNormRope with epsilons={eps}"
        )
        register_qknorm_rope_fusion(vllm_config, layer.head_size,
                                    layer.num_heads, layer.num_kv_heads,
                                    eps)
        register_qknorm_rope_fusion_with_bias(vllm_config, layer.head_size,
                                              layer.num_heads,
                                              layer.num_kv_heads, eps)


def get_qknorm_rope_vllm_config() -> VllmConfig:
    """
    Get the vllm_config from the current compiler instance.
    This function is called at module load time to lazily register fusions.
    
    Returns:
        VllmConfig object from the compiler, or a default one if not available.
    """
    try:
        from vllm_ascend.compilation.compiler_interface import get_current_compiler
        
        compiler = get_current_compiler()
        if compiler and hasattr(compiler, 'vllm_config'):
            return compiler.vllm_config
    except (ImportError, AttributeError):
        pass
    
    # Fallback to default config
    return VllmConfig()


# Lazy initialization: try to register with compiler's vllm_config if available
try:
    vllm_config = get_qknorm_rope_vllm_config()
    attn_layers: dict[str, Attention] = get_layers_from_vllm_config(
        vllm_config, Attention)
    
    if len(attn_layers) == 0:
        logger.debug(
            "QKNorm and Rope fusion enabled, but no Attention layers were discovered."
        )

    layer = next(iter(attn_layers.values()))
    if layer.head_size != 128:
        logger.debug(
            "QKNorm and Rope fusion not enabled: head_dim %d is not equal to 128",
            layer.head_size)

    # register converter for pass
    common_epsilons = [1e-5, 1e-6]
    for eps in common_epsilons:
        logger.info(
            f"Start register fusion pattern for QKNormRope with epsilons={eps}"
        )
        register_qknorm_rope_fusion(layer.head_size, layer.num_heads,
                                    layer.num_kv_heads, eps)
        register_qknorm_rope_fusion_with_bias(layer.head_size, layer.num_heads,
                                              layer.num_kv_heads, eps)
except Exception as e:
    logger.debug(
        f"Failed to register QKNorm and Rope fusions at module load time: {e}. "
        f"This is expected if the compiler hasn't been initialized yet."
    )
