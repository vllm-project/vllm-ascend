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
#
"""
Qwen2-specific C8 KV Cache quantization method.

This module provides a specialized C8 KV cache quantization method for Qwen2/Qwen3 models,
which use fused qkv_proj and have different parameter naming conventions.
"""

import torch
import torch_npu
from typing import Optional
from vllm.attention.backends.abstract import AttentionType
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

from vllm_ascend.attention.attention_v1 import AscendAttentionState


def quant_per_tensor(in_tensor: torch.Tensor,
                     input_scale: torch.Tensor,
                     input_offset: Optional[torch.Tensor],
                     function=False):
    """Quantize tensor to int8."""
    return torch_npu.npu_quantize(in_tensor, input_scale, input_offset,
                                  torch.qint8, -1, function)


class Qwen2C8KVCacheMethod(BaseKVCacheMethod):
    """
    C8 KV Cache quantization method specifically for Qwen2/Qwen3 models.
    
    This method is designed to work with Qwen2's fused qkv_proj architecture,
    where k_proj.kv_cache_scale and v_proj.kv_cache_scale from the quantized
    model are remapped to attn.key_antiquant_scale and attn.value_antiquant_scale.
    
    Key differences from AscendC8KVCacheMethod:
    - Dynamically adapts to model's dtype (bfloat16/float16) instead of hardcoded float16
    - Registers parameters with "key_antiquant_scale" and "value_antiquant_scale" names
    - Compatible with the parameter remapping in patch_qwen2_kv_cache.py
    """

    def __init__(self, quant_config=None, prefix: str = "") -> None:
        """
        Initialize Qwen2 C8 KV Cache method.
        
        Args:
            quant_config: Quantization config to get model dtype
            prefix: Layer prefix (not used, for compatibility)
        """
        self.antiquant_scale_comb = None
        # Get dtype from quant_config or default to bfloat16
        if quant_config and hasattr(quant_config, 'quant_description'):
            from vllm.config import get_current_vllm_config
            vllm_config = get_current_vllm_config()
            self.params_dtype = vllm_config.model_config.dtype
        else:
            # Default to bfloat16 for Qwen3 models
            self.params_dtype = torch.bfloat16

    def create_weights(self, layer) -> None:
        """
        Create KV cache quantization parameters for Qwen2 Attention layer.
        
        Parameters are registered on the Attention layer as:
        - key_antiquant_scale: scale for key cache dequantization
        - value_antiquant_scale: scale for value cache dequantization
        
        Note: These will be accessed as layer.key_antiquant_scale and layer.value_antiquant_scale.
        When loading weights, the patch remaps:
        - k_proj.kv_cache_scale -> attn.key_antiquant_scale
        - v_proj.kv_cache_scale -> attn.value_antiquant_scale
        
        The dtype is set to match model's dtype (typically bfloat16 or float16) to ensure
        compatibility with NPU operators like aclnnIncreFlashAttentionV4.
        """
        param_dict = {}
        
        # Use the params_dtype determined during initialization
        # For Qwen3 models, this is typically bfloat16
        scale_dtype = self.params_dtype
        
        # Create scale parameters for key and value cache dequantization
        # Shape: (num_kv_heads * head_size,)
        param_dict["key_antiquant_scale"] = torch.empty(
            layer.num_kv_heads * layer.head_size,
            dtype=scale_dtype,
            requires_grad=False
        )
        param_dict["value_antiquant_scale"] = torch.empty(
            layer.num_kv_heads * layer.head_size,
            dtype=scale_dtype,
            requires_grad=False
        )
        
        # Register parameters on the layer
        for weight_name, weight_param in param_dict.items():
            param = torch.nn.Parameter(weight_param, requires_grad=False)
            layer.register_parameter(weight_name, param)

    def process_weights_after_loading(self, layer):
        """
        Process weights after loading from checkpoint.
        
        Combines key and value antiquant scales into a single tensor for efficient
        kernel operations. The dtype is preserved from the loaded scales to ensure
        compatibility with NPU operators.
        """
        # Combine key and value scales, preserving their dtype
        self.antiquant_scale_comb = torch.cat(
            (layer.key_antiquant_scale.data.unsqueeze(0),
             layer.value_antiquant_scale.data.unsqueeze(0)),
            dim=0
        ).contiguous()

    def apply(self, layer, query, key, value, kv_cache, attn_metadata,
              attn_type, scale, output) -> torch.Tensor:
        """
        Apply C8 KV cache quantization during forward pass.
        
        This method:
        1. Quantizes key and value to int8 using the antiquant scales
        2. Stores quantized KV in the cache
        3. Performs attention with int8 KV cache
        4. Dequantizes during attention computation
        
        Args:
            layer: Attention layer
            query: Query tensor [num_tokens, num_heads, head_size]
            key: Key tensor [num_tokens, num_kv_heads, head_size]
            value: Value tensor [num_tokens, num_kv_heads, head_size]
            kv_cache: Tuple of (key_cache, value_cache)
            attn_metadata: Attention metadata
            attn_type: Type of attention (DECODER/ENCODER/etc)
            scale: Attention scale factor
            output: Output tensor
            
        Returns:
            Attention output tensor
        """
        num_tokens = query.shape[0]
        
        # Skip if no metadata (e.g., during model initialization)
        if attn_metadata is None:
            return output.view(num_tokens, layer.num_heads * layer.head_size)
        
        # Ensure k_scale and v_scale are 1.0 (no additional scaling)
        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        
        # Only support decoder attention for now
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and encoder/decoder cross-attention "
                "are not implemented for Qwen2C8KVCacheMethod"
            )

        # Quantize key and value to int8
        quant_key = quant_per_tensor(
            key.view(-1, layer.num_kv_heads * layer.head_size),
            layer.key_antiquant_scale.data.view(-1),
            None,
            True
        )
        quant_value = quant_per_tensor(
            value.view(-1, layer.num_kv_heads * layer.head_size),
            layer.value_antiquant_scale.data.view(-1),
            None,
            True
        )

        # Reshape tensors for attention computation
        query = query.view(-1, layer.num_heads, layer.head_size)
        key = key.view(-1, layer.num_kv_heads, layer.head_size)
        value = value.view(-1, layer.num_kv_heads, layer.head_size)
        value = value.contiguous()

        # Store quantized KV in cache
        if kv_cache[0].numel() > 0:
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            slots = attn_metadata.slot_mapping

            # Calculate block indices and slot indices
            block_size = key_cache.shape[1]
            slots_indices = slots.reshape(-1, 1)
            block_indices = slots_indices // block_size
            slots_indices = slots_indices % block_size
            indices = torch.cat((block_indices, slots_indices), dim=1)

            # Update KV cache with quantized tensors
            torch_npu.npu_scatter_nd_update_(key_cache, indices, quant_key)
            torch_npu.npu_scatter_nd_update_(value_cache, indices, quant_value)

        # Perform attention based on state
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            # Prefill attention without KV cache hit
            assert attn_metadata.attn_mask is not None
            mask = attn_metadata.attn_mask
            torch_npu._npu_flash_attention(
                query=query,
                key=key,
                value=value,
                mask=mask,
                seq_len=attn_metadata.seq_lens,
                scale_value=scale,
                num_heads=layer.num_heads,
                num_kv_heads=layer.num_kv_heads,
                out=output.reshape(query.shape)
            )

        elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            raise NotImplementedError(
                "KV cache int8 quantization is not implemented for PrefillCacheHit"
            )

        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            # Decode-only attention (incremental decoding)
            if hasattr(attn_metadata, "decode"):
                # torch_air compatibility
                decode_meta = attn_metadata.decode
                seq_lens = decode_meta.seq_lens_list
            else:
                seq_lens = attn_metadata.seq_lens

            block_size = key_cache.shape[1]
            query = query.view(num_tokens, 1, layer.num_heads * layer.head_size).contiguous()

            # Use cached KV (already quantized)
            key = key_cache
            value = value_cache

            # Incremental flash attention with int8 KV cache
            output = torch_npu.npu_incre_flash_attention(
                query,
                key,
                value,
                num_key_value_heads=layer.num_kv_heads,
                num_heads=layer.num_heads,
                actual_seq_lengths=seq_lens,
                scale_value=scale,
                input_layout='BSH',
                block_size=block_size,
                block_table=attn_metadata.block_tables,
                antiquant_scale=self.antiquant_scale_comb,  # Dequantization scales
            )

        else:
            raise NotImplementedError(
                "KV cache int8 quantization is not implemented for other attention states"
            )

        return output
