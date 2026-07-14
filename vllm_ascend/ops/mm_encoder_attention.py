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

Eager and ACL-graph capture both use Fused Infer Attention (``npu_fused_infer_attention_score``)
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

from vllm_ascend.utils import weak_ref_tensors
from vllm_ascend.worker.encoder_acl_graph import (
    get_encoder_forward_context,
    get_encoder_graph_params,
    update_encoder_graph_workspace,
)

MIN_PAD_SIZE: int = 64
MAX_PAD_SIZE: int = 128
SWA_INT_MAX: int = 2147483647
FIA_BLOCK_SIZE: int = 128


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

    def _maybe_compute_actual_seq_lengths(
        self,
        bsz: int,
        q_len: int,
        cu_seqlens: torch.Tensor | None,
        sequence_lengths: torch.Tensor | None,
    ) -> tuple[list[int], list[int]]:
        """Build FIA ``actual_seq_lengths`` as cumulative host-side ``list[int]``."""
        if sequence_lengths is not None:
            seq_lens_cpu = sequence_lengths
            if seq_lens_cpu.device.type != "cpu":
                seq_lens_cpu = seq_lens_cpu.to("cpu")
            actual = seq_lens_cpu.cumsum(0).to(torch.int64).tolist()
        else:
            cu = self._maybe_compute_cu_seqlens(bsz, q_len, cu_seqlens)
            if cu.device.type != "cpu":
                cu = cu.to("cpu")
            actual = cu[1:].to(torch.int64).tolist()

        return actual, actual

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
            # Handle MQA and GQA
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

        # If cu_seqlens is not provided, we create a default one assuming all sequences have the same length.
        # This is used by models such as Hunyuan-OCR, which always pass None as cu_seqlens and rely on the operator to
        # compute it internally.
        cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device="cpu")
        return cu_seqlens

    def _maybe_pad_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int | None]:
        if not self.enable_pad:
            return q, k, v, None
        origin_head_dim = q.shape[-1]
        pad_len = MAX_PAD_SIZE - origin_head_dim
        q = F.pad(q, (0, pad_len), mode="constant", value=0)
        k = F.pad(k, (0, pad_len), mode="constant", value=0)
        v = F.pad(v, (0, pad_len), mode="constant", value=0)
        return q, k, v, origin_head_dim

    @staticmethod
    def _maybe_unpad_output(
        context_layer: torch.Tensor,
        origin_head_dim: int | None,
    ) -> torch.Tensor:
        if origin_head_dim is not None:
            return context_layer[..., :origin_head_dim]
        return context_layer

    @staticmethod
    def _restore_batch_layout(
        context_layer: torch.Tensor,
        *,
        bsz: int,
        q_len: int,
        is_reshaped: bool,
    ) -> torch.Tensor:
        if is_reshaped:
            return einops.rearrange(context_layer, "(b s) h d -> b s h d", b=bsz, s=q_len).contiguous()
        return einops.rearrange(context_layer, "(b s) h d -> b s (h d)", b=bsz, s=q_len).contiguous()

    def _run_vit_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        actual_seq_lengths_q: list[int],
        actual_seq_lengths_kv: list[int],
        *,
        out: torch.Tensor | None = None,
        softmax_lse: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fia_kwargs = dict(
            query=query,
            key=key,
            value=value,
            atten_mask=None,
            block_table=None,
            input_layout="TND",
            block_size=FIA_BLOCK_SIZE,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale_value,
            sparse_mode=0,
            pre_tokens=SWA_INT_MAX,
            next_tokens=SWA_INT_MAX,
        )
        if out is None:
            context_layer, _ = torch_npu.npu_fused_infer_attention_score(**fia_kwargs)
            return context_layer
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**fia_kwargs)
        if softmax_lse is None:
            softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        torch_npu.npu_fused_infer_attention_score.out(
            workspace=workspace,
            out=[out, softmax_lse],
            **fia_kwargs,
        )
        return out

    def _forward_eager_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        seq_lens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        is_reshaped: bool,
        bsz: int,
        q_len: int,
    ) -> torch.Tensor:
        actual_seq_lengths_q, actual_seq_lengths_kv = self._maybe_compute_actual_seq_lengths(  # TODO
            bsz,
            q_len,
            cu_seqlens,
            seq_lens,
        )
        q, k, v, origin_head_dim = self._maybe_pad_qkv(query, key, value)
        context_layer = self._run_vit_fia(q, k, v, actual_seq_lengths_q, actual_seq_lengths_kv)
        context_layer = self._maybe_unpad_output(context_layer, origin_head_dim)
        return self._restore_batch_layout(
            context_layer,
            bsz=bsz,
            q_len=q_len,
            is_reshaped=is_reshaped,
        )

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
        context = get_encoder_forward_context()
        token_budget = context.token_budget
        params = get_encoder_graph_params()
        if token_budget is None or params is None:
            raise RuntimeError("Encoder graph capture state was not initialized (missing token_budget).")

        actual_seq_lengths_q, actual_seq_lengths_kv = self._maybe_compute_actual_seq_lengths(
            bsz,
            q_len,
            cu_seqlens,
            sequence_lengths,
        )
        q, k, v, origin_head_dim = self._maybe_pad_qkv(query, key, value)

        out = torch.empty_like(q)
        softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)

        workspace = params.workspaces.get(token_budget)
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=q,
                key=k,
                value=v,
                atten_mask=None,
                block_table=None,
                input_layout="TND",
                block_size=FIA_BLOCK_SIZE,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=0,
                scale=self.scale_value,
                pre_tokens=SWA_INT_MAX,
                next_tokens=SWA_INT_MAX,
            )
            update_encoder_graph_workspace(token_budget, workspace)

        stream = torch_npu.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)

        torch.npu.graph_task_group_begin(stream)
        self._run_vit_fia(
            q,
            k,
            v,
            actual_seq_lengths_q,
            actual_seq_lengths_kv,
            out=out,
            softmax_lse=softmax_lse,
            workspace=workspace,
        )
        handle = torch.npu.graph_task_group_end(stream)

        vit_layer_idx = context.capture_layer_cursor
        context.capture_layer_cursor = vit_layer_idx + 1
        uses_sequence_lengths_host = sequence_lengths is not None
        packed = (
            weak_ref_tensors(q),
            weak_ref_tensors(k),
            weak_ref_tensors(v),
            None,
            None,
            FIA_BLOCK_SIZE,
            uses_sequence_lengths_host,
            vit_layer_idx,
            self.num_kv_heads,
            self.num_heads,
            self.scale_value,
            weak_ref_tensors(out),
            weak_ref_tensors(softmax_lse),
        )
        params.attn_params[token_budget].append(packed)
        params.events[token_budget].append(event)
        params.handles[token_budget].append(handle)

        context_layer = self._maybe_unpad_output(out, origin_head_dim)
        return self._restore_batch_layout(
            context_layer,
            bsz=bsz,
            q_len=q_len,
            is_reshaped=is_reshaped,
        )

    def forward_oot(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Unused on Ascend (upstream API compat)
        sequence_lengths: torch.Tensor | None = None,
    ):
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() == 4

        q, k, v = self._reshape_qkv_to_3d(query, key, value, bsz, q_len, kv_len)

        if get_encoder_forward_context().capturing:
            return self._forward_capture_fia(
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                sequence_lengths=sequence_lengths,
                is_reshaped=is_reshaped,
                bsz=bsz,
                q_len=q_len,
                kv_len=kv_len,
            )

        return self._forward_eager_fia(
            q, k, v, seq_lens=sequence_lengths, cu_seqlens=cu_seqlens, is_reshaped=is_reshaped, bsz=bsz, q_len=q_len
        )
