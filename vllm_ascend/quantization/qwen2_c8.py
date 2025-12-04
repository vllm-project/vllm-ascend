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
from typing import Optional, List
from vllm.attention.backends.abstract import AttentionType
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ, is_310p)


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
    - Registers parameters with "key_antiquant_scale" and "value_antiquant_scale" names
      (without "attn." prefix, as they are registered on the Attention layer directly)
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
        
        # Counter for chunked prefill dequantization
        self.count_chunk_prefill = 0

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

    def anti_quant_int8(self, key_cache, value_cache, layer) -> List[torch.Tensor]:
        dst_type = self.params_dtype
        assert key_cache.dtype == torch.int8
        assert value_cache.dtype == torch.int8
        assert dst_type != torch.int8

        key_cache_anti_quant = torch_npu.npu_anti_quant(
            x = key_cache,
            scale = layer.key_antiquant_scale.data.view(-1),
            dst_dtype = dst_type
        )
        value_cache_anti_quant = torch_npu.npu_anti_quant(
            x = value_cache,
            scale = layer.value_antiquant_scale.data.view(-1),
            dst_dtype = dst_type
        )

        return [key_cache_anti_quant, value_cache_anti_quant]

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
        self.current_layer_antiquant = layer
        
        # Debug: Count state calls and print when state changes
        if attn_metadata is not None:
            state_name = str(attn_metadata.attn_state).split('.')[-1]
            
            # Initialize tracking variables
            if not hasattr(self, '_current_state'):
                self._current_state = None
                self._state_count = 0
            
            if state_name != self._current_state:
                # State changed: print previous state with count
                if self._current_state is not None:
                    print(f"[ATTN_STATE] {self._current_state} x{self._state_count}", flush=True)
                
                # Switch to new state
                self._current_state = state_name
                self._state_count = 1
            else:
                # Same state: increment counter
                self._state_count += 1
        
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

            self.key_cache = key_cache
            self.value_cache = value_cache  

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
            assert attn_metadata is not None
            assert attn_metadata.attn_mask is not None

            compress_mask = attn_metadata.attn_mask
            batch_size = attn_metadata.query_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]
            num_block, block_size, _ = self.key_cache.shape  # type: ignore

            # Read from cache and dequantize if needed
            key_from_cache = self.key_cache.view(num_block, block_size, -1)
            value_from_cache = self.value_cache.view(num_block, block_size, -1)
            
            if key_from_cache.dtype == torch.int8:
                key_cache_anti_quant, value_cache_anti_quant = self.anti_quant_int8(key_from_cache, value_from_cache, layer)
            else:
                key_cache_anti_quant = key_from_cache
                value_cache_anti_quant = value_from_cache

            # Check max sequence length to decide which attention operator to use
            # sparse_mode=3 requires mask dim <= 2048, so use alternative operator for longer sequences
            max_seq_len = max(attn_metadata.seq_lens_list) if hasattr(attn_metadata, 'seq_lens_list') and attn_metadata.seq_lens_list is not None else 0
            
            if block_size == 128 and max_seq_len <= 2048:
                key = key_cache_anti_quant.view(  # type: ignore
                    num_block, block_size, -1)
                value = value_cache_anti_quant.view(  # type: ignore
                    num_block, block_size, -1)

                output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    atten_mask=compress_mask,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=attn_metadata.seq_lens_list,
                    num_key_value_heads=layer.num_kv_heads,
                    num_heads=layer.num_heads,
                    scale=scale,
                    sparse_mode=3,
                )
            else:
                torch_npu._npu_flash_attention_qlens(
                    query=query,
                    key_cache=key_cache_anti_quant,
                    value_cache=value_cache_anti_quant,
                    block_table=block_table,
                    mask=compress_mask,
                    seq_len=attn_metadata.query_lens,
                    context_lens=attn_metadata.seq_lens,
                    num_kv_heads=layer.num_kv_heads,
                    num_heads=layer.num_heads,
                    scale_value=scale,
                    out=output)
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

        elif attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
            # Use chunked prefill for head size 192 scenario, like deepseek
            # paged_attention_splitfuse maybe crash at such scenario.
            # TODO: vanilla path will be removed after the kernel support
            # head_size 192 scenario.
            if layer.head_size == 192:
                raise NotImplementedError(
                    "KV cache int8 quantization is not implemented for head_size == 192"
                )

            # Use paged attention.
            assert attn_metadata is not None
            assert attn_metadata.attn_mask is not None

            if is_310p():
                # Do reformat in case of broadcasted tensors.
                attn_metadata.attn_mask = \
                    torch_npu.npu_format_cast(attn_metadata.attn_mask.contiguous(),
                                              ACL_FORMAT_FRACTAL_NZ)
                attn_metadata.seq_lens = \
                    attn_metadata.seq_lens.to(device=query.device)

            # TODO:The npu_fused_infer_attention_score op is planned to
            # be utilized in a wider range in upcoming versions.
            num_block, block_size, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1)
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1)

            if key.dtype == torch.int8:
                key, value = self.anti_quant_int8(key, value, layer)
                
                # Count and print dequantization for debugging
                self.count_chunk_prefill += 1
                print(f"[DEQUANT] 第{self.count_chunk_prefill}次触发chunked_prefill反量化")

            output, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=attn_metadata.block_tables,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=attn_metadata.seq_lens_list,
                num_key_value_heads=layer.num_kv_heads,
                num_heads=layer.num_heads,
                scale=scale,
                sparse_mode=3,
            )
        else:
            raise NotImplementedError(
                f"KV cache int8 quantization is not implemented for other attention states: {attn_metadata.attn_state}"
            )

        return output

