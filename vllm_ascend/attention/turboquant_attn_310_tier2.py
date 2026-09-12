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
"""TurboQuant Tier 2 for Ascend 310P: dequantize INSIDE the attention kernel.

Tier 0 (`turboquant_attn_310.py`) gathers the packed cache, dequantizes it into a
dense fp16 tensor and calls a stock attention op. That works but materialises
O(context) per decode step, which OOM-killed at 6k and hit an aicore timeout at
11k. Tier 2 calls the custom AscendC ops instead, so the dequantized block never
leaves UB:

    write : npu_turboquant_reshape_and_cache_v310
    read  : npu_turboquant_paged_attention_v310

Measured against Tier 0 on 310P (ctx=6144, b=4, d=256, 16 heads / 4 kv heads):
op-level 50055 -> 3622 us, end-to-end 8.3 -> 0.60 us/token.

NORM PLANES ARE CALLER-OWNED AND PERSISTENT
    [num_slots, 16] fp16 -- one whole 32B block per slot, only the first
    num_kv_heads lanes carrying data. This shape is load-bearing, not padding
    for its own sake:
      * each slot owns its block, so the write is a plain aligned DataCopy with
        no read-modify-write against a neighbouring slot on the same 64B line
        (the packed 8B-per-slot layout raced: 184/512 norms wrong under permuted
        slots, which is what real block-manager mappings look like);
      * the write is therefore idempotent, so SLOT REUSE overwrites rather than
        accumulating;
      * and the plane can be allocated ONCE and reused across decode steps.
    The op used to allocate the planes internally and return them, which threw
    away all history in serving: 64 tokens written one-per-call left 4 of 256
    entries set, output cosine 0.139670 against the single-call reference.
"""

import torch

from vllm_ascend.attention.turboquant_attn_310 import (
    AscendTurboQuantAttentionBackendImpl310,
)

# One 32B block per slot. Must match kNzC0 in the kernels and the [num_slots, 16]
# check in turboquant_reshape_and_cache_v310_torch_adpt.h.
NORM_LANES = 16


def norm_bytes_per_block(block_size: int) -> int:
    """Bytes of norm plane backing ONE cache block, both K and V.

    vLLM sizes num_blocks as `kv_cache_tensor.size // page_size_bytes`, and the
    turboquant backend's get_kv_cache_shape only describes the PACKED payload.
    Without adding this the norm planes are invisible to the block accounting and
    the allocation is over-committed. See `page_size_padded` in AttentionSpec --
    vLLM already budgets per-token-head scales this way, but with a
    num_kv_heads*fp32 constant that under-counts this layout by 2x.
    """
    return 2 * block_size * NORM_LANES * torch.finfo(torch.float16).bits // 8


class AscendTurboQuantTier2AttentionBackendImpl310(
    AscendTurboQuantAttentionBackendImpl310
):
    """Quantize on write; the read kernel dequantizes inside attention."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.tq_k_bits != self.tq_v_bits:
            raise ValueError(
                "TurboQuant Tier 2 takes a single `bits` per op, so K and V must "
                f"match; got k={self.tq_k_bits} v={self.tq_v_bits}. Asymmetric "
                "presets (e.g. k3v4) are Tier 0 only."
            )
        self._tq_signs = None
        self._tq_cents = None

    # ---------------------------------------------------------------- helpers

    def _tables(self, device):
        """Rotation signs + codebook, built once.

        `signs` comes from the codec so Tier 0 and Tier 2 cannot draw different
        rotations -- keys written under one and read under the other would decode
        to noise. `centroids` is unused in uniform mode (codebook_mode=0); the
        kernel derives the levels arithmetically. It is still passed because the
        op signature is shared with the Lloyd-Max path.
        """
        if self._tq_signs is None:
            kc, _ = self._codecs(device)
            self._tq_signs = kc.signs.to(device=device, dtype=torch.float32)
            self._tq_cents = torch.zeros(
                2 * (1 << self.tq_k_bits), dtype=torch.float32, device=device
            )
        return self._tq_signs, self._tq_cents

    def _ensure_norm_cache(self, device):
        """Override Tier 0's [num_slots, num_kv_heads] plane with [num_slots, 16]."""
        if self._k_norms is None:
            # cache is (num_blocks, C1, block_size, C0): block_size is dim 2
            nslots = self.key_cache.shape[0] * self.key_cache.shape[2]
            self._k_norms = torch.zeros(
                (nslots, NORM_LANES), dtype=torch.float16, device=device
            )
            self._v_norms = torch.zeros_like(self._k_norms)

    # ----------------------------------------------------------------- write

    def reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output):
        if len(kv_cache) <= 1:
            return query, key, value, output
        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        n = attn_metadata.num_actual_tokens
        self._ensure_norm_cache(key.device)
        signs, cents = self._tables(key.device)

        # The planes are passed IN and mutated in place; nothing is returned.
        torch.ops._C_ascend.npu_turboquant_reshape_and_cache_v310(
            key[:n].contiguous(),
            value[:n].contiguous(),
            self.key_cache,
            self.value_cache,
            self._k_norms,
            self._v_norms,
            # same rule as the read path: pin the device, never just the dtype
            attn_metadata.slot_mapping[:n].to(device=key.device, dtype=torch.int32),
            signs,
            cents,
            self.tq_k_bits,
            0,   # variant: production path
            0,   # codebook_mode: uniform
        )
        return query, key, value, output

    # ------------------------------------------------------------------ read

    def forward_paged_attention(self, query, attn_metadata, output=None):
        """Decode straight out of the packed cache -- no dense materialisation."""
        signs, cents = self._tables(query.device)
        # decode: one query position per sequence, so num_tokens == batch
        q = query.view(-1, self.num_heads, self.head_size).contiguous()

        attn = torch.ops._C_ascend.npu_turboquant_paged_attention_v310(
            q,
            self.key_cache,
            self.value_cache,
            self._k_norms,
            self._v_norms,
            # device= is load-bearing, not defensive. `seq_lens` arrives on the
            # HOST in this vLLM version -- the stock 310P impl carries an
            # explicit `if attn_metadata.seq_lens.device != query.device` fixup,
            # and Tier 0 spells it `.to(bt.device)`. A bare `.to(torch.int32)`
            # is a dtype cast that LEAVES IT ON CPU, so the kernel read a host
            # pointer as device memory: seqLen came back garbage, nBlocks
            # collapsed to 0, and the block loop never ran -- acc and runSum
            # stayed 0 and every decode returned EXACTLY zero. That is why
            # output was bit-identical at bits=2/3/4 and at tp=1/tp=2, and why
            # the first tokens (dense prefill path) were right and everything
            # after collapsed. Every device gate built its own tensors on the
            # NPU, so none of them could see it.
            attn_metadata.block_tables.to(device=query.device, dtype=torch.int32),
            attn_metadata.seq_lens.to(device=query.device, dtype=torch.int32),
            signs,
            cents,
            self.tq_k_bits,
            self.scale,
            0,   # variant
            0,   # codebook_mode
        )
        if output is not None:
            output[: attn.shape[0]] = attn.view_as(output[: attn.shape[0]])
            return output
        return attn
