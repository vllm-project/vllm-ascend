#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Apply replay-time FIA task updates for captured ViT encoder graphs."""

from __future__ import annotations

from collections.abc import AbstractSet

import torch
import torch_npu

from vllm_ascend.multimodal.encoder_forward_context import get_encoder_graph_runtime_state
from vllm_ascend.multimodal.encoder_graph_params import get_encoder_graph_params


def _resolve_vit_actual_lengths(
    *,
    uses_sequence_lengths_host: bool,
    vit_layer_idx: int,
    fullatt_block_indexes: AbstractSet[int] | frozenset[int] | None,
) -> tuple[list[int], list[int]]:
    """Map replay buffers + ``fullatt_block_indexes`` to FIA ``actual_seq_lengths``."""

    runtime = get_encoder_graph_runtime_state()
    if uses_sequence_lengths_host:
        seq = runtime.host_sequence_lengths
        label = "host_sequence_lengths"
    elif fullatt_block_indexes is not None:
        if vit_layer_idx in fullatt_block_indexes:
            seq = runtime.host_cu_seqlens_ends
            label = "host_cu_seqlens_ends (full-attn)"
        else:
            seq = runtime.host_cu_window_seqlens_ends
            label = "host_cu_window_seqlens_ends (window-attn)"
    else:
        seq = runtime.host_cu_seqlens_ends
        label = "host_cu_seqlens_ends"
    if seq is None:
        raise RuntimeError(
            f"Encoder replay missing {label} for vit_layer_idx={vit_layer_idx}; "
            "EncoderAclGraphManager must populate encoder_graph_replay_scope()."
        )
    return seq, seq


def update_encoder_full_graph_params(
    update_stream: torch.npu.Stream,
    token_budget: int,
    *,
    fullatt_block_indexes: AbstractSet[int] | frozenset[int] | None = None,
) -> None:
    """Re-bind fused infer attention host tensors inside the encoder NPUGraph (parallel to LLM path).

    Qwen2.5-VL: layers listed in ``fullatt_block_indexes`` use ``cu_seqlens`` host endpoints;
    others use ``cu_window_seqlens``. Those layouts are **not** baked at capture — only here.

    This deliberately bypasses :class:`AttentionBackend` — ViT attention is not registered there — but reuses
    the same ``graph_task_update_{begin,end}`` + ``ExternalEvent`` ordering pattern as
    :meth:`AscendAttentionBackendImpl.update_graph_params`.
    """

    params = get_encoder_graph_params()
    if params is None or token_budget not in params.handles:
        return

    handles = params.handles[token_budget]
    events = params.events[token_budget]
    attn_blocks = params.attn_params[token_budget]
    workspace = params.workspaces.get(token_budget)

    if len(handles) != len(events) or len(handles) != len(attn_blocks):
        raise RuntimeError(
            "Encoder graph bookkeeping is inconsistent: "
            f"budget={token_budget} handles={len(handles)} "
            f"events={len(events)} attn_blocks={len(attn_blocks)}"
        )

    with torch.npu.stream(update_stream):
        for handle, event, packed in zip(handles, events, attn_blocks):
            (
                query,
                key,
                value,
                block_table,
                attn_mask,
                block_size,
                uses_sequence_lengths_host,
                vit_layer_idx,
                num_kv_heads,
                num_heads,
                scale,
                output,
                softmax_lse,
            ) = packed

            actual_seq_lengths_q, actual_seq_lengths_kv = _resolve_vit_actual_lengths(
                uses_sequence_lengths_host=uses_sequence_lengths_host,
                vit_layer_idx=vit_layer_idx,
                fullatt_block_indexes=fullatt_block_indexes,
            )

            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_fused_infer_attention_score.out(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=num_kv_heads,
                num_heads=num_heads,
                scale=scale,
                sparse_mode=0,
                workspace=workspace,
                out=[output, softmax_lse],
            )
            torch.npu.graph_task_update_end(update_stream)
            event.record(update_stream)
