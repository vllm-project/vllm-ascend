#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# Adapted from vllm/model_executor/models/internvl.py and intern_vit.py
# This file is a part of the vllm-ascend project.

from typing import Optional

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.intern_vit import (InternParallelAttention,
                                                   InternVisionModel)
from vllm.model_executor.models.internvl import InternVLChatModel
from vllm.model_executor.models.utils import maybe_prefix


class AscendInternParallelAttention(InternParallelAttention):
    """
    NPU-optimized attention for InternVision encoder. 
    Uses torch.nn.functional.scaled_dot_product_attention instead of MultiHeadAttention.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store head dimensions for NPU attention
        self.head_size = self.head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        NPU-compatible forward pass using PyTorch's scaled_dot_product_attention.

        Uses F.scaled_dot_product_attention which automatically dispatches to NPU
        kernels when running on NPU device (via torch_npu).

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection using parent's layer
        qkv, _ = self.qkv(hidden_states)

        # Split into Q, K, V: [batch, seq_len, hidden_dim]
        q, k, v = qkv.chunk(3, dim=-1)

        # Apply QK normalization if configured (using parent's method)
        if self.qk_normalization:
            q, k = self._apply_qk_norm(q, k)

        # Reshape for multi-head attention: [batch, seq_len, num_heads, head_dim]
        q = q.reshape(batch_size, seq_len, self.num_heads_per_partition,
                      self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads_per_partition,
                      self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads_per_partition,
                      self.head_dim)

        # Transpose to [batch, num_heads, seq_len, head_dim] for attention
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # PyTorch's scaled_dot_product_attention (dispatches to NPU backend automatically)
        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )

        # Transpose back: [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Reshape: [batch, seq_len, num_heads * head_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        # Output projection using parent's layer
        output, _ = self.proj(attn_output)

        return output


class AscendInternVisionModel(InternVisionModel):
    """
    Ascend-optimized InternVision encoder.

    Replaces the standard MultiHeadAttention in each encoder layer with
    AscendInternParallelAttention for NPU hardware acceleration.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        # Initialize parent, but we'll replace attention layers
        super().__init__(
            config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=prefix,
            use_data_parallel=use_data_parallel,
        )

        # Replace each layer's attention with Ascend version

        for i, layer in enumerate(self.encoder.layers):
            # Replace the attention module
            layer.attn = AscendInternParallelAttention(
                config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, f"encoder.layers.{i}.attn"),
                use_data_parallel=use_data_parallel,
            )


class AscendInternVLChatModel(InternVLChatModel):
    """
    Ascend-optimized InternVL multimodal model.

    Key changes:
    1. Uses AscendInternVisionModel with NPU-optimized SDPA attention for vision encoding
    2. Inherits all other functionality from upstream InternVLChatModel
    """

    def _init_vision_model(
        self,
        config,
        quant_config: Optional[QuantizationConfig],
        *,
        is_mono: bool,
        prefix: str,
    ):
        """Override to use Ascend-optimized vision model"""
        if not is_mono:
            vision_feature_layer = config.select_layer
            if vision_feature_layer < 0:
                num_hidden_layers = (config.vision_config.num_hidden_layers +
                                     vision_feature_layer + 1)
            else:
                num_hidden_layers = vision_feature_layer + 1

            # Use Ascend-optimized vision model
            return AscendInternVisionModel(
                config.vision_config,
                quant_config=quant_config,
                num_hidden_layers_override=num_hidden_layers,
                prefix=prefix,
                use_data_parallel=self.use_data_parallel,
            )
        else:
            # For mono architecture, use parent implementation
            # (not commonly used, can be optimized later if needed)
            return super()._init_vision_model(config,
                                              quant_config,
                                              is_mono=is_mono,
                                              prefix=prefix)
