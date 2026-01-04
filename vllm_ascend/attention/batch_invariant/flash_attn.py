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

from typing import Optional, Tuple

import torch

from vllm_ascend.attention.attention_v1 import (AscendAttentionBackendImpl,
                                                AscendAttentionState,
                                                AscendMetadata)
from vllm_ascend.ops.triton.batch_invariant.attention.flash_attn import \
    flash_attn_with_kvcache


class BatchInvariantBackendImpl(AscendAttentionBackendImpl):
    """Batch-invariant attention backend implementation for Ascend NPUs."""

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            return self._forward_decode_only_batch_invariant(
                query,
                attn_metadata,
                output,
            )
        return self._forward_prefill_batch_invariant(
            query,
            key,
            value,
            attn_metadata,
            output,
        )

    def _forward_decode_only_batch_invariant(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Batch-invariant decode-only attention using Triton flash attention.

        This method converts from TND layout to BSHD layout, calls the
        batch-invariant flash attention kernel, and converts back.

        Args:
            query: [num_tokens, num_heads, head_size] in TND layout
            attn_metadata: Attention metadata containing sequence info
            output: Optional output tensor

        Returns:
            output: [num_tokens, num_heads, head_size]
        """
        batch_size = attn_metadata.seq_lens.shape[0]
        seq_lens = attn_metadata.seq_lens  # (batch_size,)

        # For decode, each request has exactly 1 query token
        # query shape: [batch_size, num_heads, head_size] (already in TND with T=batch)
        # Reshape to BSHD: [batch_size, 1, num_heads, head_size]
        q_bshd = query[:batch_size].view(batch_size, 1, self.num_heads,
                                         self.head_size)

        # Get KV cache: [num_blocks, block_size, num_kv_heads, head_size]
        # We need to gather the relevant blocks for each sequence
        block_tables = attn_metadata.block_tables  # [batch_size, max_blocks]

        # For simplicity in batch-invariant mode, we'll process each sequence
        # independently. This ensures batch invariance at the cost of parallelism.
        # TODO: Optimize this with a proper paged attention kernel
        max_seq_len = seq_lens.max().item()
        assert self.key_cache is not None, "KV cache must be provided for decode"
        block_size = self.key_cache.shape[1]

        # Allocate contiguous KV tensors for batch processing
        # Shape: [batch_size, max_seq_len, num_kv_heads, head_size]
        k_gathered = torch.zeros(batch_size,
                                 max_seq_len,
                                 self.num_kv_heads,
                                 self.head_size,
                                 dtype=query.dtype,
                                 device=query.device)
        v_gathered = torch.zeros(batch_size,
                                 max_seq_len,
                                 self.num_kv_heads,
                                 self.head_size,
                                 dtype=query.dtype,
                                 device=query.device)

        # Gather KV cache for each sequence
        for b in range(batch_size):
            seq_len = seq_lens[b].item()
            num_blocks_needed = (seq_len + block_size - 1) // block_size

            for block_idx in range(num_blocks_needed):
                block_num = block_tables[b, block_idx].item()
                start_pos = block_idx * block_size
                end_pos = min(start_pos + block_size, seq_len)
                actual_len = end_pos - start_pos

                k_gathered[b, start_pos:end_pos] = self.key_cache[
                    block_num, :actual_len]
                v_gathered[b, start_pos:end_pos] = self.value_cache[
                    block_num, :actual_len]

        # Call batch-invariant flash attention
        # Note: cache_seqlens tells the kernel the actual sequence lengths
        attn_output = flash_attn_with_kvcache(
            q_bshd,
            k_gathered,
            v_gathered,
            cache_seqlens=seq_lens.int(),
            causal=True,
            softmax_scale=self.scale,
        )

        # Convert back to TND layout: [batch_size, 1, num_heads, head_size] -> [batch_size, num_heads, head_size]
        attn_output = attn_output.view(batch_size, self.num_heads,
                                       self.head_size)

        if output is not None:
            output[:batch_size] = attn_output
            return output
        return attn_output

    def _forward_prefill_batch_invariant(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batch-invariant prefill attention using Triton flash attention.

        For prefill, we have variable-length sequences packed into a single
        tensor. We process each sequence independently to ensure batch invariance.

        Args:
            query: [num_tokens, num_heads, head_size] in TND layout
            key: [num_tokens, num_kv_heads, head_size] in TND layout
            value: [num_tokens, num_kv_heads, head_size] in TND layout
            attn_metadata: Attention metadata containing sequence info
            output: Output tensor [num_tokens, num_heads, head_size]

        Returns:
            output: [num_tokens, num_heads, head_size]
        """
        # Get sequence boundaries from actual_seq_lengths_q
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q  # List of cumulative positions
        num_seqs = len(actual_seq_lengths_q)

        # Process each sequence independently
        prev_end = 0
        for seq_idx in range(num_seqs):
            seq_end = actual_seq_lengths_q[seq_idx]
            seq_len = seq_end - prev_end

            if seq_len == 0:
                prev_end = seq_end
                continue

            # Extract Q, K, V for this sequence
            # Shape: [seq_len, num_heads, head_size]
            q_seq = query[prev_end:seq_end]
            k_seq = key[prev_end:seq_end]
            v_seq = value[prev_end:seq_end]

            # Convert to BSHD: [1, seq_len, num_heads, head_size]
            q_bshd = q_seq.unsqueeze(0)
            k_bshd = k_seq.unsqueeze(0)
            v_bshd = v_seq.unsqueeze(0)

            # Call batch-invariant flash attention
            attn_output = flash_attn_with_kvcache(
                q_bshd,
                k_bshd,
                v_bshd,
                causal=True,
                softmax_scale=self.scale,
            )

            # Convert back to TND and store
            # [1, seq_len, num_heads, head_size] -> [seq_len, num_heads, head_size]
            output[prev_end:seq_end] = attn_output.squeeze(0)

            prev_end = seq_end

        return output
