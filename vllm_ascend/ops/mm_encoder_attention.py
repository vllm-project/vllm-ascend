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

"""Ascend implementation of upstream :class:`MMEncoderAttention`.

Non-graph execution keeps using ``torch_npu._npu_flash_attention_unpad`` (CPU ``seq_lens``).

Vision encoder ACL graph capture swaps to Fused Infer Attention (``npu_fused_infer_attention_score``)
with ``graph_task_group_begin/end`` so replay-time host metadata can be rebound from the update stream,
matching the LLM full-graph pattern in :mod:`vllm_ascend.attention.attention_v1`.
"""

from __future__ import annotations

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention  # type: ignore
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.multimodal.encoder_forward_context import get_encoder_graph_runtime_state
from vllm_ascend.multimodal.encoder_graph_params import get_encoder_graph_params, update_encoder_graph_workspace
from vllm_ascend.utils import weak_ref_tensors

MIN_PAD_SIZE: int = 64  # min_size to pad weight
MAX_PAD_SIZE: int = 128  # max_size to pad weight

# ViT self-attention runs without a paged KV block table; FIA still expects a block_size compatible with CANN.
_VIT_FIA_BLOCK_SIZE: int = 128


def _per_seq_lengths_to_fia_endpoints(per_seq: list[int]) -> list[int]:
    acc = 0
    ends: list[int] = []
    for raw in per_seq:
        L = int(raw)
        if L == 0:
            continue
        acc += L
        ends.append(acc)
    return ends


class AscendMMEncoderAttention(MMEncoderAttention):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
    ) -> None:
        """
        Args:
            num_heads: number of attention heads per partition.
            head_size: hidden_size per attention head.
            scale: scale factor.
            num_kv_heads: number of kv heads.
            prefix: This has no effect, it is only here to make it easier to
                    swap between Attention and MMEncoderAttention.
            multimodal_config: configs for multi-modal.
        """
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
        )

        self.enable_pad = self.head_size > MIN_PAD_SIZE and self.head_size < MAX_PAD_SIZE
        self.scale_value = self.head_size**-0.5

    def _ascend_attn_scale(self) -> float:
        return float(self.scale) if self.scale is not None else self.scale_value

    @classmethod
    def maybe_compute_seq_lens(
        cls,
        attn_backend: AttentionBackendEnum,
        cu_seqlens: np.ndarray,
        device: torch.device,
    ) -> np.ndarray | None:
        if cu_seqlens is None:
            return None

        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        seq_lens = torch.from_numpy(seq_lens).to("cpu", non_blocking=True)

        return seq_lens

    def _reshape_qkv_to_3d(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bsz: int,
        q_len: int,
        kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape query, key, value to 3D tensors:
        (batch_size * seq_len, num_heads, head_size)
        """
        query = query.view(bsz * q_len, self.num_heads, self.head_size)
        key = key.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if (num_repeat := self.num_queries_per_kv) > 1:
            key = torch.repeat_interleave(key, num_repeat, dim=1)
            value = torch.repeat_interleave(value, num_repeat, dim=1)

        return query, key, value

    def _maybe_compute_cu_seqlens(
        self,
        bsz: int,
        q_len: int,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cu_seqlens is not None:
            return cu_seqlens

        cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device="cpu")
        return cu_seqlens

    def _prepare_seq_metadata_cpu(
        self,
        bsz: int,
        q_len: int,
        cu_seqlens: torch.Tensor | None,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """CPU tensor ``seq_lens`` consumed by ``_npu_flash_attention_unpad``."""

        if sequence_lengths is not None:
            if sequence_lengths.device.type != "cpu":
                sequence_lengths = sequence_lengths.to("cpu")
            return sequence_lengths

        cu_seqlens = self._maybe_compute_cu_seqlens(bsz, q_len, cu_seqlens)
        return torch.diff(cu_seqlens).to("cpu")

    def _forward_eager_flash(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        seq_lens_cpu: torch.Tensor,
        is_reshaped: bool,
        bsz: int,
    ) -> torch.Tensor:
        q, k, v = query, key, value

        origin_shape: int | None = None
        if self.enable_pad:
            origin_shape = q.shape[-1]
            pad_len = MAX_PAD_SIZE - origin_shape
            q = F.pad(q, (0, pad_len), mode="constant", value=0)
            k = F.pad(k, (0, pad_len), mode="constant", value=0)
            v = F.pad(v, (0, pad_len), mode="constant", value=0)

        context_layer = DeviceOperator.npu_flash_attention(
            query=q,
            key=k,
            value=v,
            seq_lens_cpu=seq_lens_cpu,
            head_num=self.num_heads,
            scale_value=self.scale_value,
            num_kv_heads=self.num_kv_heads,
        )

        if self.enable_pad and origin_shape is not None:
            context_layer = context_layer[..., :origin_shape]

        if is_reshaped:
            context_layer = einops.rearrange(context_layer, "(b s) h d -> b s h d", b=bsz).contiguous()
        else:
            context_layer = einops.rearrange(context_layer, "(b s) h d -> b s (h d)", b=bsz).contiguous()
        return context_layer

    def _forward_capture_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None,
        sequence_lengths: torch.Tensor | None,
        is_reshaped: bool,
        bsz: int,
        q_len: int,
        kv_len: int,
    ) -> torch.Tensor:
        runtime = get_encoder_graph_runtime_state()
        token_budget = runtime.token_budget
        params = get_encoder_graph_params()
        if token_budget is None or params is None:
            raise RuntimeError("Encoder graph capture state was not initialized (missing token_budget).")

        q, k, v = self._reshape_qkv_to_3d(query, key, value, bsz, q_len, kv_len)

        origin_shape: int | None = None
        if self.enable_pad:
            origin_shape = q.shape[-1]
            pad_len = MAX_PAD_SIZE - origin_shape
            q = F.pad(q, (0, pad_len), mode="constant", value=0)
            k = F.pad(k, (0, pad_len), mode="constant", value=0)
            v = F.pad(v, (0, pad_len), mode="constant", value=0)

        out = torch.empty_like(q)
        softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)

        vit_layer_idx = runtime.capture_layer_cursor
        runtime.capture_layer_cursor = vit_layer_idx + 1

        if sequence_lengths is not None:
            actual_seq_lengths_q = _per_seq_lengths_to_fia_endpoints(
                sequence_lengths.detach().cpu().view(-1).tolist()
            )
            uses_sequence_lengths_host = True
        else:
            cu_for_lengths = self._maybe_compute_cu_seqlens(bsz, q_len, cu_seqlens)
            actual_seq_lengths_q = cu_for_lengths[1:].detach().cpu().tolist()
            uses_sequence_lengths_host = False

        workspace = params.workspaces.get(token_budget)
        attn_mask = None
        block_table = None
        scale = self._ascend_attn_scale()

        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=q,
                key=k,
                value=v,
                atten_mask=attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=_VIT_FIA_BLOCK_SIZE,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_q,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=0,
                scale=scale,
            )
            update_encoder_graph_workspace(token_budget, workspace)

        stream = torch_npu.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        params.events[token_budget].append(event)

        packed = (
            weak_ref_tensors(q),
            weak_ref_tensors(k),
            weak_ref_tensors(v),
            block_table,
            attn_mask,
            _VIT_FIA_BLOCK_SIZE,
            uses_sequence_lengths_host,
            vit_layer_idx,
            self.num_kv_heads,
            self.num_heads,
            scale,
            weak_ref_tensors(out),
            weak_ref_tensors(softmax_lse),
        )
        params.attn_params[token_budget].append(packed)

        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            query=q,
            key=k,
            value=v,
            atten_mask=attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=_VIT_FIA_BLOCK_SIZE,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_q,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=scale,
            sparse_mode=0,
            workspace=workspace,
            out=[out, softmax_lse],
        )
        handle = torch.npu.graph_task_group_end(stream)
        params.handles[token_budget].append(handle)

        context_layer = out
        if self.enable_pad and origin_shape is not None:
            context_layer = context_layer[..., :origin_shape]

        if is_reshaped:
            context_layer = einops.rearrange(context_layer, "(b s) h d -> b s h d", b=bsz).contiguous()
        else:
            context_layer = einops.rearrange(context_layer, "(b s) h d -> b s (h d)", b=bsz).contiguous()
        return context_layer

    def forward_oot(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
        sequence_lengths: torch.Tensor | None = None,
    ):
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() == 4

        if get_encoder_graph_runtime_state().capturing:
            return self._forward_capture_fia(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                sequence_lengths=sequence_lengths,
                is_reshaped=is_reshaped,
                bsz=bsz,
                q_len=q_len,
                kv_len=kv_len,
            )

        seq_lens_cpu = self._prepare_seq_metadata_cpu(bsz, q_len, cu_seqlens, sequence_lengths)
        q, k, v = self._reshape_qkv_to_3d(query, key, value, bsz, q_len, kv_len)
        return self._forward_eager_flash(q, k, v, seq_lens_cpu=seq_lens_cpu, is_reshaped=is_reshaped, bsz=bsz)
