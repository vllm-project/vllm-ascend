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
"""A3/A2 QFA INT8-GQA prefill attention (quant_mode=12, no KV cache).

Prefill-only path enabled by ``VLLM_ASCEND_QFA_A3_PREFILL=1``. Q/K are
quantized per-token-head and V per-head to INT8 (FP32 descales) and fed
to the A2/A3 FULL_INT8_GQA operator on TND-packed inputs. Only
PrefillNoCache steps qualify: chunked prefill and prefix-cache hits must
fall back to the FIA path because the operator sees no paged cache, so
the current step's K/V must already be the full history. The KV cache
itself stays BF16 -- reshape_and_cache writes the unquantized K/V before
forward_impl runs, so decode keeps the stock path unchanged.
"""

import torch

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionState,
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


def _build_int8_cu_seqlens(attn_metadata, num_tokens: int) -> torch.Tensor:
    """Sanitized INT32 [B+1] prefix sums from the persistent query buffer.

    The padded tail of ``query_start_loc_gpu`` may hold -1/stale entries;
    clamp + cummax turn it into zero-length requests, matching the
    operator's "non-decreasing, ends at T" contract.
    """
    num_reqs = attn_metadata.num_reqs
    cu = (
        attn_metadata.query_start_loc_gpu[: num_reqs + 1]
        .clamp(min=0, max=num_tokens)
        .cummax(dim=0)
        .values
    )
    return cu.to(torch.int32).contiguous()


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
        q_int8, q_descale = _quantize_per_token_head(query[:num_tokens])
        k_int8, k_descale = _quantize_per_token_head(key[:num_tokens])
        v_int8, v_descale = _quantize_per_head(value[:num_tokens])
        p_scale = torch.tensor([QFA_INT8_SCALE_LEVEL], dtype=torch.float32, device=query.device)

        # PrefillNoCache: this step's K/V is the whole history, so the query
        # prefix sums serve both sides.
        cu_seqlens = _build_int8_cu_seqlens(attn_metadata, num_tokens)
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
        # [T, Hq, D] BF16 -> the padded output buffer's layout.
        output[:num_tokens] = attn_out.reshape(num_tokens, self.hidden_size)
        return output
