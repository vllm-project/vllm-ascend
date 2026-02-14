#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

from typing import Any

import torch_npu
from vllm.v1.attention.backends.registry import (  # type: ignore
    AttentionBackendEnum,
    register_backend,
)

from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310
from vllm_ascend._310p.attention.metadata_builder import AscendAttentionMetadataBuilder310
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)


@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend310(AscendAttentionBackend):
    def __init__(self, *args, **kwargs):
        """
        Initializes the 310P backend and sets up the device-specific mask builder.
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int):
        """
        Determines the shape of the Key-Value (KV) cache tensor.

        The 310P hardware requires specific memory alignment for optimal performance.
        This method defines a 5D tensor shape where the head size dimension is
        split to ensure alignment to multiples of 16.

        Args:
            num_blocks (int): Number of memory blocks.
            block_size (int): Size of each block.
            num_kv_heads (int): Number of KV heads.
            head_size (int): Dimension size of each head.

        Returns:
            tuple: The specific 5D shape required by the hardware
                   (2, num_blocks, hidden_dim_aligned, block_size, 16).
        """
        # Align to a multiple of 16, as required by the 310P device.
        return (2, num_blocks, (num_kv_heads * head_size) // 16, block_size, 16)

    @staticmethod
    def get_impl_cls():
        """
        Returns the implementation class for the attention operations.
        """
        return AscendAttentionBackendImpl310

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        """
        Returns the metadata builder class specifically for 310P.
        """
        return AscendAttentionMetadataBuilder310


class AscendAttentionBackendImpl310(AscendAttentionBackendImpl):
    """
    Implementation of attention operations (Prefill, Decode, Chunked Prefill)
    optimized for the Ascend 310P architecture.
    """

    def forward_paged_attention(
        self,
        query: Any,
        attn_metadata: AscendMetadata,
        output: Any | None = None,
    ) -> Any:
        """
        Executes Paged Attention (typically for the decode phase).

        Ensures that the sequence length metadata is on the correct device
        before invoking the base implementation.

        Args:
            query (Any): The query tensor.
            attn_metadata (AscendMetadata): Metadata associated with the attention request.
            output (Any | None): Optional output tensor.

        Returns:
            Any: The result of the attention operation.
        """
        if attn_metadata.seq_lens.device != query.device:
            attn_metadata.seq_lens = attn_metadata.seq_lens.to(
                device=query.device,
                non_blocking=True,
            )
        return super().forward_paged_attention(query, attn_metadata, output)

    def forward_prefill_310(self, query, key, value, attn_metadata, output):
        """
        Executes Flash Attention for the prefill phase on 310P.

        This method handles memory alignment padding. If the query shape implies
        padding (aligned_tokens > real_tokens), it adjusts the sequence length
        of the last request to account for the delta, ensuring the NPU operator
        processes the data correctly.

        Args:
            query, key, value: Input tensors.
            attn_metadata (AscendMetadata): Attention metadata containing masks and seq_lens.
            output: Output tensor.

        Returns:
            The output tensor after flash attention.
        """
        real_tokens = int(attn_metadata.seq_lens.sum().item())
        seq_len = attn_metadata.seq_lens
        aligned_tokens = int(query.shape[0])
        delta = aligned_tokens - real_tokens

        # Adjust sequence length if padding (alignment) was applied to the inputs
        if delta:
            seq_len = seq_len.clone()
            seq_len[-1] += delta

        mask = attn_metadata.attn_mask
        torch_npu._npu_flash_attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
            seq_len=seq_len,
            scale_value=self.scale,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            out=output,
        )
        return output

    def forward_chunked_prefill_310(self, query, attn_metadata, output):
        """
        Executes SplitFuse (Chunked Prefill) attention on 310P.

        This handles scenarios where the prefill is split into chunks. It prepares
        the necessary metadata (query lengths, block tables) and generates the
        specific splitfuse mask before calling the NPU operator.

        Args:
            query: The query tensor.
            attn_metadata (AscendMetadata): Metadata containing start locations and block tables.
            output: The output tensor.
        """
        num_actual_tokens = int(attn_metadata.num_actual_tokens)
        query = query[:num_actual_tokens]
        output = output[:num_actual_tokens]

        # Calculate query lengths from start locations
        qsl_cpu = attn_metadata.query_start_loc.cpu()
        qlens = qsl_cpu[1:] - qsl_cpu[:-1]

        context_lens = attn_metadata.seq_lens
        block_table = attn_metadata.block_tables

        # Generate the specific mask for splitfuse
        mask = AttentionMaskBuilder310.get_splitfuse_mask(attn_metadata, query.device)

        if context_lens.device != query.device:
            context_lens = context_lens.to(query.device, non_blocking=True)

        torch_npu._npu_paged_attention_splitfuse(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            mask=mask,
            block_table=block_table,
            seq_len=qlens,
            context_lens=context_lens,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            out=output,
        )

        return output

    def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
        """
        Main dispatch method for attention operations.

        Routes the execution to Decode, Prefill, or Chunked Prefill methods
        based on the current attention state found in metadata.

        Args:
            query, key, value: Input tensors (Key/Value usually empty for decode/chunked).
            kv_cache: The KV cache structure.
            attn_metadata: Metadata determining the state (Prefill vs Decode).
            output: Tensor to write results to.

        Returns:
            The output tensor.

        Raises:
            NotImplementedError: If the attention state is not supported on 310P.
        """
        state = attn_metadata.attn_state
        # Condition for PrefillNoCache: No previous tokens have been processed yet
        if state == AscendAttentionState.PrefillNoCache:
            output = self.forward_prefill_310(query, key, value, attn_metadata, output)
        # Condition for DecodeOnly: Pure decoding phase where each request generates one token
        elif state == AscendAttentionState.DecodeOnly:
            output = self.forward_paged_attention(query, attn_metadata, output)
        # Condition for ChunkedPrefill:
        # 1. During speculative decoding scenarios (except mtp)
        # 2. Processing large prefill requests in chunks
        # Condition for PrefillCacheHit: Indicates prefill with some cached tokens already processed
        elif state in [AscendAttentionState.ChunkedPrefill, AscendAttentionState.PrefillCacheHit]:
            output = self.forward_chunked_prefill_310(query, attn_metadata, output)
        # Condition for SpecDecoding: Specified for mtp, which is not supported yet.
        else:
            raise NotImplementedError(f"AscendAttentionState: {state} is not supported for 310P currently.")
        return output
