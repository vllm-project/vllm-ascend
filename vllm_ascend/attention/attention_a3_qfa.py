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
"""A3/A2 QFA INT8-GQA attention (quant_mode=12).

Two paths share this file:

1. Prefill (``VLLM_ASCEND_QFA_A3_PREFILL=1``, no KV cache): Q/K quantized
   per-token-head and V per-head to INT8, fed to the TND-packed operator.
   Only PrefillNoCache steps qualify; the KV cache stays BF16.

2. Decode (``VLLM_ASCEND_QFA_A3_DECODE=1``, paged KV cache): the decode
   operator contract (PA_BNBD paged K/V, TND Q with NT q_descale,
   per-token K descale stored per page, per-head static V descale). The
   INT8 KV cache infrastructure is inherited from
   AscendC8AttentionBackendImpl (C8/QuaRot channel); Step 1 covers
   DecodeOnly, other states keep the inherited FIA fallbacks.
"""

import torch
import torch_npu
from vllm.v1.attention.backend import AttentionType  # type: ignore[import-not-found]

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionState,
    AscendC8AttentionBackendImpl,
    AscendMetadata,
)
from vllm_ascend.utils import logger

QFA_INT8_GQA_QUANT_MODE = 12
QFA_INT8_SCALE_LEVEL = 127.0
QFA_MASK_NO_MASK = 0
QFA_MASK_CAUSAL = 3
QFA_ABSMAX_EPS = 1e-12


def _quantize_per_token_head(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[T, H, D] float -> INT8 payload + FP32 descale [T, H]."""
    descale = x.abs().amax(dim=-1).clamp_min(QFA_ABSMAX_EPS).float() / QFA_INT8_SCALE_LEVEL
    payload = (x.float() / descale.unsqueeze(-1)).round().clamp(-QFA_INT8_SCALE_LEVEL, QFA_INT8_SCALE_LEVEL)
    return payload.to(torch.int8), descale


def _quantize_per_head(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[T, H, D] float -> INT8 payload + FP32 descale [H] shared over tokens."""
    descale = x.abs().amax(dim=(0, 2)).clamp_min(QFA_ABSMAX_EPS).float() / QFA_INT8_SCALE_LEVEL
    payload = (x.float() / descale.view(1, -1, 1)).round().clamp(-QFA_INT8_SCALE_LEVEL, QFA_INT8_SCALE_LEVEL)
    return payload.to(torch.int8), descale


def _build_int8_cu_seqlens(attn_metadata) -> torch.Tensor:
    """Real per-request cumulative query lengths into the operator's INT32 [B+1].

    ``actual_seq_lengths_q`` carries the true batch (no padded tail) and is
    what the FIA prefill paths already consume; prefix the mandatory 0.
    """
    device = attn_metadata.query_start_loc_gpu.device
    return torch.tensor(
        [0, *attn_metadata.actual_seq_lengths_q],
        dtype=torch.int32,
        device=device,
    )


class AscendQFA3PrefillBackendImpl(AscendAttentionBackendImpl):
    """Stock backend impl with the PrefillNoCache branch routed to QFA INT8-GQA."""

    _enable_logged = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.head_size > 256:
            raise ValueError(
                f"A3 QFA INT8-GQA supports head_dim <= 256, got {self.head_size}."
            )
        if not AscendQFA3PrefillBackendImpl._enable_logged:
            logger.info(
                "A3 QFA INT8-GQA prefill attention enabled "
                "(VLLM_ASCEND_QFA_A3_PREFILL=1); decode and chunked/prefix-hit "
                "prefill keep the stock FIA path."
            )
            AscendQFA3PrefillBackendImpl._enable_logged = True

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata,
        output: torch.Tensor,
    ):
        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.sliding_window is None
            and self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            return self._forward_qfa_int8_prefill(query, key, value, attn_metadata, output)
        return super().forward_impl(query, key, value, kv_cache, attn_metadata, output)

    def _forward_qfa_int8_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        from cann_ops_transformer.ops import quant_flash_attn

        num_tokens = attn_metadata.num_actual_tokens
        # Q/K: fused per-token-head quantization (1 kernel each).
        q_int8, q_descale = torch_npu.npu_dynamic_quant(query[:num_tokens], dst_type=torch.int8)
        k_int8, k_descale = torch_npu.npu_dynamic_quant(key[:num_tokens], dst_type=torch.int8)
        # V: per-head over all tokens; no fused kernel available (npu_quantize
        # rejects BF16+FP32 scale), keep the manual chain.
        v_int8, v_descale = _quantize_per_head(value[:num_tokens])
        p_scale = torch.tensor([QFA_INT8_SCALE_LEVEL], dtype=torch.float32, device=query.device)

        # PrefillNoCache: this step's K/V is the whole history, so the query
        # prefix sums serve both sides.
        cu_seqlens = _build_int8_cu_seqlens(attn_metadata)
        max_seqlen = attn_metadata.max_query_len or num_tokens

        attn_out, _ = quant_flash_attn(
            q_int8,
            k_int8,
            v_int8,
            q_descale,
            k_descale,
            v_descale,
            QFA_INT8_GQA_QUANT_MODE,
            p_scale=p_scale,
            metadata=None,
            layout_q="TND",
            layout_q_descale="TND",
            layout_kv="TND",
            layout_out="TND",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            mask_mode=QFA_MASK_CAUSAL if attn_metadata.causal else QFA_MASK_NO_MASK,
            win_left=-1,
            win_right=-1,
            softmax_scale=self.scale,
            return_softmax_lse=False,
        )
        # attn_out is [T, Hq, D] and the runner's output buffer shares that layout.
        output[:num_tokens] = attn_out
        return output


class AscendQFA3C8Impl(AscendC8AttentionBackendImpl):
    """Decode path on the A3 QFA INT8-GQA paged operator (quant_mode=12).

    Inherits the C8 INT8 KV-cache infrastructure (static per-channel
    quantization, paged scatter writes, FIA fallbacks for prefill states)
    from AscendC8AttentionBackendImpl and replaces the DecodeOnly forward
    with the native INT8 QFA operator:

    - Q: per-token-head dynamic quantization (TND payload, NT descale)
    - K: INT8 paged cache + per-page FP32 descale cache (Step 2); until
      then the inherited per-channel K path feeds the operator as
      descale=scale broadcast (functional shape, unverified numerics)
    - V: static per-head descale (v_cache_scale from the fa_v.scale
      calibration channel)

    Step 1 scope: DecodeOnly dispatch + call-chain bring-up with in-test
    scales. Prefill states keep the inherited FIA dequant path.
    """

    _decode_logged = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not AscendQFA3C8Impl._decode_logged:
            logger.info(
                "A3 QFA INT8-GQA decode attention enabled "
                "(VLLM_ASCEND_QFA_A3_DECODE=1); DecodeOnly steps run the "
                "paged QFA operator, other states keep the C8 FIA path."
            )
            AscendQFA3C8Impl._decode_logged = True

    # ------------------------------------------------------------------
    # Step 1: DecodeOnly dispatch. The inherited forward() handles cache
    # writes + state routing; only the decode branch is replaced here.
    # ------------------------------------------------------------------
    def _forward_c8_decode(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer,
    ) -> torch.Tensor:
        return self._forward_qfa_decode(query, attn_metadata, output, layer)

    def _forward_qfa_decode(self, query, attn_metadata, output, layer):
        """Native INT8 decode via the paged QFA operator.

        The preconditions below are invariants of the supported deployment
        (causal decoder-only model, GPU-side length sources, bound caches);
        a violation means the assembly chain broke, so fail fast instead
        of silently falling back to FIA.
        """
        from cann_ops_transformer.ops import quant_flash_attn

        if self.sliding_window is not None:
            raise NotImplementedError(
                "A3 QFA decode does not support sliding-window attention "
                f"(layer={getattr(layer, 'layer_name', '?')})."
            )
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER_DECODER):
            raise NotImplementedError(
                f"A3 QFA decode does not support attn_type={self.attn_type} "
                f"(layer={getattr(layer, 'layer_name', '?')})."
            )
        qsl_gpu = attn_metadata.query_start_loc_gpu
        seq_lens_gpu = attn_metadata.seq_lens_gpu
        if qsl_gpu is None or seq_lens_gpu is None:
            raise RuntimeError(
                "A3 QFA decode requires the GPU-side length sources "
                "(query_start_loc_gpu / seq_lens_gpu) on AscendMetadata."
            )
        if self.key_cache is None or self.value_cache is None:
            raise RuntimeError(
                "A3 QFA decode requires bound K/V caches "
                f"(layer={getattr(layer, 'layer_name', '?')})."
            )

        num_tokens = query.shape[0]
        batch_size = len(attn_metadata.seq_lens_list)
        if num_tokens <= 0 or batch_size <= 0:
            return output

        # Device-side length derivation (graph-replay safe; sanitizes the
        # padded tail the same way the C8-MXFP backend does).
        cu_seqlens_q = qsl_gpu.clamp(min=0, max=num_tokens).cummax(dim=0).values[: batch_size + 1].to(torch.int32)
        seqused_kv = seq_lens_gpu.clamp(min=1)[:batch_size].to(torch.int32)
        max_seqlen_q = attn_metadata.max_query_len or num_tokens
        mask_mode = QFA_MASK_NO_MASK if max_seqlen_q == 1 else QFA_MASK_CAUSAL

        # Q per-token-head quantization (TND payload + NT descale).
        q_int8, q_descale = torch_npu.npu_dynamic_quant(
            query[:num_tokens], dst_type=torch.int8
        )
        q_descale_nt = q_descale.t().contiguous()  # (Q_N, Q_T)

        # Paged K/V: the LBNHC cache IS the operator's PA_BNBD layout.
        k_cache, v_cache = self.key_cache, self.value_cache

        # K descale per page: Step 1 uses the inherited per-channel scale
        # broadcast to the (Bn, KV_N, Bs) contract (functional placeholder;
        # Step 2 replaces it with the dynamic per-token descale cache).
        k_descale_pages = self._step1_k_descale_pages(k_cache, layer, batch_size)

        # V descale: static per-head, from the fa_v.scale channel. The
        # inherited _prepare_c8_scales shards per-channel (Hkv*D); the QFA
        # contract wants per-head, so reduce over D.
        self._prepare_c8_scales(layer, query.device)
        v_descale = layer._c8_v_scale.view(self.num_kv_heads, -1).amax(dim=1)

        p_scale = torch.tensor([QFA_INT8_SCALE_LEVEL], dtype=torch.float32, device=query.device)
        block_table = self._qfa_block_table(attn_metadata, batch_size)

        attn_out, _ = quant_flash_attn(
            q_int8,
            k_cache,
            v_cache,
            q_descale_nt,
            k_descale_pages,
            v_descale,
            QFA_INT8_GQA_QUANT_MODE,
            block_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            seqused_kv=seqused_kv,
            p_scale=p_scale,
            metadata=None,
            softmax_scale=self.scale,
            mask_mode=mask_mode,
            win_left=-1,
            win_right=-1,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=int(attn_metadata.seq_lens_list[batch_size - 1]) if batch_size else 1,
            layout_q="TND",
            layout_q_descale="NT",
            layout_kv="PA_BNBD",
            layout_out="TND",
            return_softmax_lse=False,
        )
        output[:num_tokens] = attn_out
        return output

    def _step1_k_descale_pages(self, k_cache: torch.Tensor, layer, batch_size: int) -> torch.Tensor:
        """Placeholder K descale (Bn, KV_N, Bs): the inherited per-channel
        K scale reduced per head, broadcast over pages/tokens. Numerically
        wrong for real quantization (per-channel cache holds round(x *
        inv_perchannel)) -- Step 1 only proves the call chain; Step 2's
        dynamic descale cache replaces this entirely."""
        self._prepare_c8_scales(layer, k_cache.device)
        k_scale = layer._c8_k_scale.view(self.num_kv_heads, -1).amax(dim=1)  # (Hkv,)
        num_blocks = k_cache.shape[0]
        block_size = k_cache.shape[2] if k_cache.dim() == 4 else 128
        return k_scale.view(1, self.num_kv_heads, 1).expand(num_blocks, self.num_kv_heads, block_size).contiguous()

    def _qfa_block_table(self, attn_metadata, batch_size: int) -> torch.Tensor:
        """block_table for the operator, at kernel-block granularity.

        The cache tensors are laid out in kernel blocks (Bs=128, see
        model_runner get_kv_cache_shape(num_blocks * chunk, ...)). The
        metadata table follows the scheduler page; for hybrid models that
        page is 896 = 7 kernel blocks, so each scheduler entry expands to
        `ratio` consecutive kernel blocks (device-side, replay-safe).
        """
        bt = attn_metadata.block_tables[:batch_size]
        cache_bs = self.key_cache.shape[2] if self.key_cache.dim() == 4 else 128
        sched_bs = self.vllm_config.cache_config.block_size
        bt = bt.to(torch.int32)
        if sched_bs == cache_bs:
            return bt
        ratio = sched_bs // cache_bs
        # entry [i, j] covers kernel pages j*ratio .. j*ratio+ratio-1
        expanded = bt.view(batch_size, -1, 1) * ratio + torch.arange(
            ratio, device=bt.device, dtype=torch.int32
        ).view(1, 1, ratio)
        return expanded.view(batch_size, -1)
