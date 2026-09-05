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
"""TurboQuant KV cache for Ascend 310P -- Tier 0 (no custom AscendC kernel).

Stores the paged KV cache as packed b-bit codes plus one fp16 norm per
(token, kv_head), and dequantizes on read into a dense tensor that a stock
dense-attention operator consumes. This yields the real HBM saving (5.22x at
b=3, head_dim=256) at the cost of bandwidth: the point is to validate the
storage format and unlock long-context evaluation, not to be fast. A fused
dequant-inside-attention kernel (Tier 2) is what makes it fast.

Constraints discovered on 310P and encoded here:
  * `npu_fused_infer_attention_score` does not exist on this SoC -- the decode
    path uses `npu_incre_flash_attention` instead.
  * The KV cache is FRACTAL_NZ with the 5D shape
    `(2, num_blocks, (num_kv_heads*head_size)//16, block_size, 16)`.
  * The cache must stay **fp16-typed**. `_npu_reshape_and_cache` is a pure
    scatter, so packed bytes are reinterpreted as fp16 (nbytes/2 slots per head)
    and move verbatim -- verified byte-exact (0/192000 mismatches). An int8-typed
    cache is NOT usable: at 96B with C0=32 the op reports success while writing
    nothing, and at C0=16 it errors outright.
  * `_npu_reshape_and_cache` takes `slot_indices=` as a KEYWORD; passing it
    positionally silently writes nothing.

Scope: PrefillNoCache + DecodeOnly. Chunked prefill / prefix caching read the
paged cache through `_npu_paged_attention_splitfuse`, which cannot parse packed
codes; they are rejected explicitly rather than silently producing garbage.
"""

import torch
import torch_npu

from vllm_ascend._310p.attention.attention_v1 import (
    AscendAttentionBackend310,
    AscendAttentionBackendImpl310,
)
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.turboquant_codec import TurboQuantCodec, packed_bytes

NZ_C0 = 16  # fp16 NZ tile on 310P (the cache stays fp16-typed; packed bytes are viewed as fp16)

# Presets mirror the upstream vLLM TurboQuant naming so results are comparable.
# k3v3 chosen by the Phase-0 sweep: K and V bits are fungible above their floors
# (K>=2, V>=3), and a symmetric width means one unpack path.
PRESETS = {
    "turboquant_3bit": (3, 3),
    "turboquant_4bit": (4, 4),
    "turboquant_k3v4": (3, 4),
    "turboquant_k2v4": (2, 4),
}


def preset_bits(cache_dtype_str: str):
    return PRESETS.get(cache_dtype_str)


class AscendTurboQuantAttentionBackend310(AscendAttentionBackend310):
    """Backend whose KV cache holds packed codes instead of fp16."""

    # Set by the quantization method before the cache is allocated.
    tq_k_bits: int = 3
    tq_v_bits: int = 3

    @staticmethod
    def get_impl_cls():
        return AscendTurboQuantAttentionBackendImpl310

    @classmethod
    def get_kv_cache_shape(cls, num_blocks, block_size, num_kv_heads, head_size, cache_type=""):
        """5D NZ shape in the SAME form the fp16 base uses:
        `(2, num_blocks, (num_kv_heads*head_size)//16, block_size, 16)`.

        The cache stays **fp16-typed**; packed bytes are reinterpreted as fp16
        (`nbytes/2` slots per head). `_npu_reshape_and_cache` is a pure scatter and
        was verified byte-exact through this view on 310P -- an int8-typed cache is
        NOT usable here: at 96B/C0=32 the op reports success while writing nothing.

        K and V share one shape, so an asymmetric split (k2v4) would round the
        narrower plane up. k3v3 sidesteps that entirely.
        """
        bits = max(cls.tq_k_bits, cls.tq_v_bits)
        half = packed_bytes(head_size, bits) // 2          # fp16 slots per head
        feat = num_kv_heads * half
        if feat % NZ_C0:
            raise ValueError(
                f"num_kv_heads*{half} = {feat} must be a multiple of the NZ tile ({NZ_C0}); "
                f"head_size={head_size} bits={bits} num_kv_heads={num_kv_heads}"
            )
        return (2, num_blocks, feat // NZ_C0, block_size, NZ_C0)


class AscendTurboQuantAttentionBackendImpl310(AscendAttentionBackendImpl310):
    """Quantize on write; gather+dequantize on read, then dense attention.

    Attention runs in the ROTATED basis: because Pi is orthogonal,
    rotating the query and the output is equivalent to inverse-rotating every
    cached key/value, and costs O(1) per request instead of O(context).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tq_k_bits = getattr(self, "tq_k_bits", 3)
        self.tq_v_bits = getattr(self, "tq_v_bits", 3)
        self._k_codec = None
        self._v_codec = None
        self._k_norms = None  # (num_blocks*block_size, num_kv_heads) fp16
        self._v_norms = None

    # ---------------------------------------------------------------- helpers

    def _codecs(self, device):
        if self._k_codec is None:
            self._k_codec = TurboQuantCodec(self.head_size, self.tq_k_bits, device)
            self._v_codec = TurboQuantCodec(self.head_size, self.tq_v_bits, device)
        return self._k_codec, self._v_codec

    def _ensure_norm_cache(self, device):
        """Norms live outside the packed plane: 96+2=98 is not a multiple of the
        32B NZ tile, and padding to 128B would cost 5.22x -> 4.0x."""
        if self._k_norms is None:
            # cache is (num_blocks, C1, block_size, C0) -> block_size is dim 2, NOT dim 1
            nslots = self.key_cache.shape[0] * self.key_cache.shape[2]
            shape = (nslots, self.num_kv_heads)
            self._k_norms = torch.zeros(shape, dtype=torch.float16, device=device)
            self._v_norms = torch.zeros(shape, dtype=torch.float16, device=device)

    def _nz_gather(self, cache: torch.Tensor, block_table: torch.Tensor, nbytes: int):
        """Gather paged NZ blocks -> (batch, max_tokens, num_kv_heads, nbytes) int8.

        cache is (num_blocks, C1, block_size, C0) with C1*C0 = num_kv_heads*nbytes.
        Flattening (C1, C0) for a fixed block offset recovers the feature vector,
        since f = c1*C0 + c0.
        """
        _, c1, block_size, c0 = cache.shape
        b, nblk = block_table.shape
        g = cache[block_table.reshape(-1).long()]        # (b*nblk, C1, S, C0) fp16
        g = g.permute(0, 2, 1, 3).contiguous()           # (b*nblk, S, C1, C0)
        g = g.view(b, nblk * block_size, self.num_kv_heads, nbytes // 2)
        return g.view(torch.int8)                        # back to packed bytes

    # ----------------------------------------------------------------- write

    def reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output):
        if len(kv_cache) <= 1:
            return query, key, value, output
        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        n = attn_metadata.num_actual_tokens
        slots = attn_metadata.slot_mapping[:n]
        kc, vc = self._codecs(key.device)
        self._ensure_norm_cache(key.device)

        k_packed, k_norm = kc.quantize(key[:n])      # (n, H, nbytes) int8, (n, H) fp16
        v_packed, v_norm = vc.quantize(value[:n])

        # view packed bytes as fp16: the op is a pure scatter and moves them verbatim
        torch_npu._npu_reshape_and_cache(
            key=k_packed.contiguous().view(torch.float16),
            value=v_packed.contiguous().view(torch.float16),
            key_cache=self.key_cache, value_cache=self.value_cache,
            slot_indices=slots,                       # keyword is mandatory
        )
        idx = slots.long()
        self._k_norms[idx] = k_norm
        self._v_norms[idx] = v_norm
        return query, key, value, output

    # ------------------------------------------------------------------ read

    def forward_paged_attention(self, query, attn_metadata, output=None):
        """Decode: gather packed blocks, dequantize to dense, dense attention."""
        kc, vc = self._codecs(query.device)
        bt = attn_metadata.block_tables
        batch, nblk = bt.shape
        block_size = self.key_cache.shape[2]      # (num_blocks, C1, block_size, C0)

        k_packed = self._nz_gather(self.key_cache, bt, kc.nbytes)
        v_packed = self._nz_gather(self.value_cache, bt, vc.nbytes)

        # norms follow the same slot arithmetic as the packed planes
        slot = (bt.long().unsqueeze(-1) * block_size
                + torch.arange(block_size, device=bt.device).view(1, 1, -1)).reshape(batch, -1)
        k_norm = self._k_norms[slot]                 # (batch, tokens, H)
        v_norm = self._v_norms[slot]

        # dequantized, still in the rotated basis
        k = kc.dequantize(k_packed, k_norm)          # (batch, tokens, H, D)
        v = vc.dequantize(v_packed, v_norm)

        # mask out slots past the true sequence length
        toks = nblk * block_size
        valid = (torch.arange(toks, device=bt.device).view(1, -1)
                 < attn_metadata.seq_lens.to(bt.device).view(-1, 1))

        # Grouped attention computed directly, WITHOUT expanding K/V to num_heads.
        # npu_incre_flash_attention has no GQA on 310P, and repeat_interleave to 16
        # heads quadrupled an already-large transient (~90 MiB/layer/step at 11k ctx),
        # which triggered aicore timeouts (507014) partway through a long-context run.
        # Broadcasting over the group axis keeps K/V at num_kv_heads.
        rep = self.num_heads // self.num_kv_heads
        q = query[:batch].view(batch, self.num_kv_heads, rep, self.head_size)
        q = kc.rotate(q)                             # rotate the query, not the cache

        k = k.permute(0, 2, 1, 3).float()            # (b, n_kv, tokens, D)
        v = v.permute(0, 2, 1, 3).float()
        scores = torch.einsum("bkrd,bksd->bkrs", q.float(), k) * self.scale
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        attn = torch.einsum("bkrs,bksd->bkrd", torch.softmax(scores, -1), v)
        attn = attn.reshape(batch, self.num_heads, self.head_size).to(query.dtype)

        attn = vc.rotate(attn)                       # Pi is self-inverse; one rotation per request
        if output is not None:
            output[:batch] = attn
            return output
        return attn

    # -------------------------------------------------------------- dispatch

    def forward_chunked_prefill_310(self, query, attn_metadata, output):
        raise NotImplementedError(
            "TurboQuant Tier 0 does not support chunked prefill / prefix caching: "
            "_npu_paged_attention_splitfuse reads the paged cache directly and cannot "
            "parse packed codes. Run with --no-enable-chunked-prefill and prefix caching off."
        )

    def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
        state = attn_metadata.attn_state
        if state == AscendAttentionState.PrefillNoCache:
            # prefill consumes the freshly-computed fp16 K/V, never the cache
            return self.forward_prefill_310(query, key, value, attn_metadata, output)
        if state == AscendAttentionState.DecodeOnly:
            return self.forward_paged_attention(query, attn_metadata, output)
        raise NotImplementedError(f"TurboQuant Tier 0: unsupported attention state {state}")
