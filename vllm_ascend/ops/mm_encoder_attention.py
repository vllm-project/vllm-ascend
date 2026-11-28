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

By default, eager and ACL-graph capture use Fused Infer Attention
(``npu_fused_infer_attention_score``). An opt-in A5 path uses eager MXFP8
QuantFlashAttn for multimodal encoder self-attention.
"""

from __future__ import annotations

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention  # type: ignore
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import weak_ref_tensors
from vllm_ascend.worker.encoder_acl_graph import (
    EncoderGraphKey,
    get_encoder_forward_context,
    get_encoder_graph_params,
    maybe_compute_actual_seq_lengths,
    update_encoder_graph_workspace,
)

MIN_PAD_SIZE: int = 64
MAX_PAD_SIZE: int = 128
SWA_INT_MAX: int = 2147483647
FIA_BLOCK_SIZE: int = 128
QFA_QUANT_MODE_MXFP8: int = 1
QFA_MASK_MODE_NO_MASK: int = 0
QFA_V_GROUP_SIZE: int = 64
QFA_SUPPORTED_HEAD_SIZES: frozenset[int] = frozenset({64, 128, 192, 256})


class AscendMMEncoderAttention(MMEncoderAttention):
    @classmethod
    def maybe_recompute_cu_seqlens(
        cls,
        attn_backend: AttentionBackendEnum,
        cu_seqlens: np.ndarray | torch.Tensor | None,
        hidden_size: int,
        tp_size: int,
        device: torch.device,
        fp8_padded_hidden_size: int | None = None,
    ) -> torch.Tensor | None:
        if cu_seqlens is None:
            return None

        # Ascend FIA consumes host token endpoints, not device offsets. Keep
        # CPU-generated metadata on CPU instead of uploading it here and
        # synchronously copying it back in every vision attention block.
        # Own the storage so later mutation of the NumPy input cannot change
        # metadata retained by the encoder graph manager.
        return torch.as_tensor(cu_seqlens, dtype=torch.int32, device="cpu").clone()

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
        self.use_quant_flash_attn = get_ascend_config().mm_encoder_quant_flash_attn
        if self.use_quant_flash_attn and not get_current_hardware_profile().supports(
            HardwareCapability.MM_ENCODER_MXFP8_QFA
        ):
            raise ValueError("mm_encoder_quant_flash_attn requires an A5 QuantFlashAttn runtime.")

    def _reshape_qkv_to_3d(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bsz: int,
        q_len: int,
        kv_len: int,
        *,
        repeat_kv: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape query, key, value to 3D tensors:
        (batch_size * seq_len, num_heads, head_size)
        """
        query = query.view(bsz * q_len, self.num_heads, self.head_size)
        key = key.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if repeat_kv and (num_repeat := self.num_queries_per_kv) > 1:
            # Handle MQA and GQA
            key = torch.repeat_interleave(key, num_repeat, dim=1)
            value = torch.repeat_interleave(value, num_repeat, dim=1)

        return query, key, value

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

    def _maybe_compute_cu_seqlens(
        self,
        bsz: int,
        q_len: int,
        cu_seqlens: torch.Tensor | None,
        *,
        is_capturing: bool = False,
    ) -> torch.Tensor:
        # In the eager path, if cu_seqlens is provided by the model we use it; if it is not provided, we create a
        # default one assuming all sequences have the same length. This is used by models such as Hunyuan-OCR, which
        # always pass None as cu_seqlens and rely on the operator to compute it internally.
        # In the capture path, we always create the default cu_seqlens on CPU instead of using the model tensor, to
        # avoid a device-to-host sync (.cpu()).
        if is_capturing or cu_seqlens is None:
            cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device="cpu")
            return cu_seqlens

        return cu_seqlens.cpu()

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
            scale=self.scale,
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
        cu_seqlens: torch.Tensor | None = None,
        is_reshaped: bool,
        bsz: int,
        q_len: int,
    ) -> torch.Tensor:
        actual_seq_lengths_q, actual_seq_lengths_kv = maybe_compute_actual_seq_lengths(
            self._maybe_compute_cu_seqlens(bsz, q_len, cu_seqlens),
            query.shape[0],
            key.shape[0],
            cudagraph_mm_encoder=False,
        )
        q, k, v, origin_head_dim = self._maybe_pad_qkv(query, key, value)
        k = k.contiguous()
        v = v.contiguous()
        context_layer = self._run_vit_fia(q, k, v, actual_seq_lengths_q, actual_seq_lengths_kv)
        context_layer = self._maybe_unpad_output(context_layer, origin_head_dim)
        return self._restore_batch_layout(
            context_layer,
            bsz=bsz,
            q_len=q_len,
            is_reshaped=is_reshaped,
        )

    def _forward_eager_qfa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None,
        is_reshaped: bool,
        bsz: int,
        q_len: int,
    ) -> torch.Tensor:
        """Run non-paged, non-causal MXFP8 attention with per-sequence V scales.

        Q/K are quantized over the head dimension. V is quantized over tokens,
        resetting its 64-token groups at each sequence boundary as required by
        the TND QuantFlashAttn contract. The eager path uses the current host
        boundaries; a captured copy would reuse stale boundaries on replay.
        """
        from cann_ops_transformer.ops import quant_flash_attn, quant_flash_attn_metadata

        q, k, v, origin_head_dim = self._maybe_pad_qkv(query, key, value)
        if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("mm_encoder_quant_flash_attn requires BF16 Q/K/V inputs.")
        if q.shape[-1] not in QFA_SUPPORTED_HEAD_SIZES:
            raise ValueError(f"Unsupported QuantFlashAttn head size: {q.shape[-1]}.")
        if q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0]:
            raise ValueError("mm_encoder_quant_flash_attn currently requires self-attention with equal Q/K/V lengths.")
        if q.shape[0] == 0:
            return self._restore_batch_layout(q, bsz=bsz, q_len=q_len, is_reshaped=is_reshaped)

        cu_cpu = self._maybe_compute_cu_seqlens(bsz, q_len, cu_seqlens).to(torch.int32)
        boundaries = cu_cpu.tolist()
        if not boundaries or boundaries[0] != 0 or boundaries[-1] != q.shape[0]:
            raise ValueError("QuantFlashAttn cu_seqlens must start at zero and end at the Q/K/V token count.")
        if any(end < start for start, end in zip(boundaries, boundaries[1:])):
            raise ValueError("QuantFlashAttn cu_seqlens must be nondecreasing.")

        q_quant, q_scale = torch_npu.npu_dynamic_mx_quant(q.contiguous(), dst_type=torch.float8_e4m3fn)
        k_quant, k_scale = torch_npu.npu_dynamic_mx_quant(k.contiguous(), dst_type=torch.float8_e4m3fn)
        v_quant_parts = []
        v_scale_parts = []
        for start, end in zip(boundaries, boundaries[1:]):
            if start == end:
                continue
            # Make the token dimension the last axis for DynamicMxQuant.
            part = v[start:end].permute(1, 2, 0).contiguous()
            pad = (-part.shape[-1]) % QFA_V_GROUP_SIZE
            if pad:
                part = F.pad(part, (0, pad))
            quant, scale = torch_npu.npu_dynamic_mx_quant(part, dst_type=torch.float8_e4m3fn)
            v_quant_parts.append(quant.permute(2, 0, 1)[: end - start].contiguous())
            v_scale_parts.append(scale.permute(2, 0, 1, 3).contiguous())
        v_quant = torch.cat(v_quant_parts, dim=0)
        v_scale = torch.cat(v_scale_parts, dim=0)
        cu_device = cu_cpu.to(device=q.device, non_blocking=True)
        max_seqlen = max(end - start for start, end in zip(boundaries, boundaries[1:]))
        common = dict(
            cu_seqlens_q=cu_device,
            cu_seqlens_kv=cu_device,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            mask_mode=QFA_MASK_MODE_NO_MASK,
            layout_q="TND",
            layout_q_descale="TND",
            layout_kv="TND",
            layout_out="TND",
        )
        metadata = quant_flash_attn_metadata(
            self.num_heads,
            self.num_kv_heads,
            q.shape[-1],
            QFA_QUANT_MODE_MXFP8,
            **common,
        )
        result = quant_flash_attn(
            q_quant,
            k_quant,
            v_quant,
            q_scale.view(torch.float8_e8m0fnu),
            k_scale.view(torch.float8_e8m0fnu),
            v_scale.view(torch.float8_e8m0fnu),
            QFA_QUANT_MODE_MXFP8,
            metadata=metadata,
            softmax_scale=self.scale,
            return_softmax_lse=False,
            **common,
        )
        context_layer = result[0] if isinstance(result, tuple) else result
        context_layer = self._maybe_unpad_output(context_layer, origin_head_dim)
        return self._restore_batch_layout(context_layer, bsz=bsz, q_len=q_len, is_reshaped=is_reshaped)

    def _forward_capture_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        is_reshaped: bool,
        bsz: int,
        q_len: int,
    ) -> torch.Tensor:
        context = get_encoder_forward_context()
        token_budget = context.token_budget
        path = context.path
        axis_keys = context.axis_keys
        is_capturing = context.capturing
        params = get_encoder_graph_params()
        if token_budget is None or params is None:
            raise RuntimeError("Encoder graph capture state was not initialized (missing token_budget).")

        actual_seq_lengths_q, actual_seq_lengths_kv = maybe_compute_actual_seq_lengths(
            self._maybe_compute_cu_seqlens(bsz, q_len, cu_seqlens, is_capturing=is_capturing),
            query.shape[0],
            key.shape[0],
            cudagraph_mm_encoder=True,
        )
        q, k, v, origin_head_dim = self._maybe_pad_qkv(query, key, value)
        # FIA requires contiguous K/V tensors. Materialize them before graph-task
        # capture so the tensors retained in ``attn_params`` are exactly the
        # tensors whose addresses were captured by the operator.
        k = k.contiguous()
        v = v.contiguous()

        out = torch.empty_like(q)
        softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)

        graph_key: EncoderGraphKey = (path, token_budget, axis_keys) if axis_keys else (path, token_budget)
        workspace = params.workspaces.get(graph_key)
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
                scale=self.scale,
                pre_tokens=SWA_INT_MAX,
                next_tokens=SWA_INT_MAX,
            )
            update_encoder_graph_workspace(token_budget, workspace, path=path, axis_keys=axis_keys)

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

        packed = (
            weak_ref_tensors(q),
            weak_ref_tensors(k),
            weak_ref_tensors(v),
            None,
            None,
            FIA_BLOCK_SIZE,
            self.num_kv_heads,
            self.num_heads,
            self.scale,
            weak_ref_tensors(out),
            weak_ref_tensors(softmax_lse),
        )
        params.attn_params[graph_key].append(packed)
        params.events[graph_key].append(event)
        params.handles[graph_key].append(handle)

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
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ):
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() == 4

        q, k, v = self._reshape_qkv_to_3d(
            query, key, value, bsz, q_len, kv_len, repeat_kv=not self.use_quant_flash_attn
        )

        if self.use_quant_flash_attn:
            return self._forward_eager_qfa(
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                is_reshaped=is_reshaped,
                bsz=bsz,
                q_len=q_len,
            )

        if get_encoder_forward_context().capturing:
            return self._forward_capture_fia(
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                is_reshaped=is_reshaped,
                bsz=bsz,
                q_len=q_len,
            )

        return self._forward_eager_fia(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            is_reshaped=is_reshaped,
            bsz=bsz,
            q_len=q_len,
        )
