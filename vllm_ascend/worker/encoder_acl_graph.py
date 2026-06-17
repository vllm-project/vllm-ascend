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
"""NPU-specific encoder ACL graph: params, runtime context, FIA replay updates, and manager."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch_npu
from vllm.logger import logger
from vllm.platforms import current_platform
from vllm.v1.worker.encoder_cudagraph import BudgetGraphMetadata, EncoderCudaGraphManager

from vllm_ascend.utils import weak_ref_tensors

_ENCODER_CAPTURE_SUPPORTS_GRAPH_POOL = "graph_pool" in inspect.signature(EncoderCudaGraphManager.capture).parameters


# ---------------------------------------------------------------------------
# Per–encoder-budget ACL graph bookkeeping (ViT FIA tasks)
# ---------------------------------------------------------------------------


@dataclass
class EncoderGraphParams:
    """Mirrors :class:`vllm_ascend.compilation.acl_graph.GraphParams` but keyed by encoder token budget."""

    events: dict[int, list[torch.npu.ExternalEvent]] = field(default_factory=dict)
    workspaces: dict[int, torch.Tensor | None] = field(default_factory=dict)
    handles: dict[int, list[Any]] = field(default_factory=dict)
    # Flattened per-forward insertion order (one entry per ViT block invocation).
    attn_params: dict[int, list[tuple]] = field(default_factory=dict)


_encoder_graph_params: EncoderGraphParams | None = None


def set_encoder_graph_params(token_budgets: list[int]) -> None:
    global _encoder_graph_params
    budgets_sorted_unique = sorted(token_budgets)
    _encoder_graph_params = EncoderGraphParams(
        events={b: [] for b in budgets_sorted_unique},
        workspaces={b: None for b in budgets_sorted_unique},
        handles={b: [] for b in budgets_sorted_unique},
        attn_params={b: [] for b in budgets_sorted_unique},
    )


def get_encoder_graph_params() -> EncoderGraphParams | None:
    return _encoder_graph_params


def update_encoder_graph_workspace(token_budget: int, workspace: torch.Tensor) -> None:
    if _encoder_graph_params is None:
        return
    _encoder_graph_params.workspaces[token_budget] = workspace


# ---------------------------------------------------------------------------
# Capture / replay runtime state (thread-local module singleton)
# ---------------------------------------------------------------------------


@dataclass
class EncoderForwardContext:
    """Vision encoder NPUGraph runtime flags and host-side FIA arguments.

    Captured tensors stay on device; FIA ``graph_task_update`` needs Python ``list[int]``
    lengths that are refreshed each replay from encoder metadata buffers on device (see RFC).
    """

    token_budget: int | None = None
    capturing: bool = False
    capture_layer_cursor: int = 0
    cu_seqlens_cpu: torch.Tensor | None = None
    cu_window_seqlens_cpu: torch.Tensor | None = None
    sequence_lengths_cpu: torch.Tensor | None = None


_context = EncoderForwardContext()


def get_encoder_forward_context() -> EncoderForwardContext:
    return _context


def _reset_encoder_forward_context() -> None:
    """Clear replay-time host length fields."""

    _context.token_budget = None
    _context.capturing = False
    _context.cu_seqlens_cpu = None
    _context.cu_window_seqlens_cpu = None
    _context.sequence_lengths_cpu = None


@contextmanager
def set_encoder_forward_context(
    token_budget: int,
    capturing: bool,
    *,
    cu_seqlens_cpu: list[int] | None = None,
    cu_window_seqlens_cpu: list[int] | None = None,
    sequence_lengths_cpu: list[int] | None = None,
):
    """Enter encoder graph replay (FIA host args): callers must pass lengths each time.

    On exit, replay host fields are **cleared** (not restored). Lists must not be reused
    across replays without repopulating from the current batch buffers.
    """

    _context.token_budget = token_budget
    _context.capturing = capturing
    _context.cu_seqlens_cpu = cu_seqlens_cpu
    _context.cu_window_seqlens_cpu = cu_window_seqlens_cpu
    _context.sequence_lengths_cpu = sequence_lengths_cpu
    _context.capture_layer_cursor = 0
    try:
        yield _context
    finally:
        _reset_encoder_forward_context()


# ---------------------------------------------------------------------------
# Replay-time FIA task updates
# ---------------------------------------------------------------------------


def _pad_actual_seq_lengths_for_fia(actual_seq_lengths: list[int], num_tokens: int) -> list[int]:
    """TND FIA requires ``query.shape[0] == actual_seq_lengths[-1]``."""
    if not actual_seq_lengths or actual_seq_lengths[-1] != num_tokens:
        actual_seq_lengths.append(num_tokens)
    return actual_seq_lengths


def _maybe_compute_actual_seq_lengths(
    *,
    num_query_tokens: int,
    uses_seq_len_host: bool,
    vit_layer_idx: int,
    fullatt_block_indexes: set[int] | frozenset[int] | None,
) -> tuple[list[int], list[int]]:
    context = get_encoder_forward_context()
    if uses_seq_len_host:
        if context.sequence_lengths_cpu is None:
            raise RuntimeError("context.sequence_lengths_cpu is None during encoder replay.")
        actual = context.sequence_lengths_cpu.cumsum(0).to(torch.int64).tolist()
    elif fullatt_block_indexes is not None:
        if vit_layer_idx in fullatt_block_indexes:
            if context.cu_seqlens_cpu is None:
                raise RuntimeError("context.cu_seqlens_cpu is None during encoder replay.")
            actual = context.cu_seqlens_cpu[1:].to(torch.int64).tolist()
        else:
            if context.cu_window_seqlens_cpu is None:
                raise RuntimeError("context.cu_window_seqlens_cpu is None during encoder replay.")
            actual = context.cu_window_seqlens_cpu[1:].to(torch.int64).tolist()
    else:
        if context.cu_seqlens_cpu is None:
            raise RuntimeError("context.cu_seqlens_cpu is None during encoder replay.")
        actual = context.cu_seqlens_cpu[1:].to(torch.int64).tolist()

    aligned = _pad_actual_seq_lengths_for_fia(actual, num_query_tokens)
    return aligned, aligned


def update_encoder_graph_params(
    update_stream: torch.npu.Stream,
    token_budget: int,
    *,
    fullatt_block_indexes: set[int] | frozenset[int] | None = None,
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

            num_query_tokens = query.shape[0]
            actual_seq_lengths_q, actual_seq_lengths_kv = _maybe_compute_actual_seq_lengths(
                num_query_tokens=num_query_tokens,
                uses_seq_len_host=uses_sequence_lengths_host,
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


# ---------------------------------------------------------------------------
# Encoder NPUGraph manager
# ---------------------------------------------------------------------------
class EncoderAclGraphManager(EncoderCudaGraphManager):
    """Hooks encoder capture/replay into Ascend FIA graph-task infrastructure."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.graph_pool = current_platform.get_global_graph_pool()
        self.update_stream: torch.npu.Stream | None = None
        visual = getattr(self.model, "visual", None)
        fa_raw = getattr(visual, "fullatt_block_indexes", None) if visual is not None else None
        self.fullatt = frozenset(fa_raw) if fa_raw is not None else None

    def capture(self, graph_pool: Any | None = None):
        encoder_graph_pool = graph_pool if graph_pool is not None else self.graph_pool
        self.graph_pool = encoder_graph_pool

        set_encoder_graph_params(self.token_budgets)

        if _ENCODER_CAPTURE_SUPPORTS_GRAPH_POOL:
            super().capture(graph_pool=encoder_graph_pool)
        else:
            super().capture()

        weak_ref_encoder_graph_workspaces()

    def _capture_budget_graph(self, token_budget: int):
        logger.debug(
            "Capturing encoder aclgraph for budget=%d, max_batch_size=%d, max_frames_per_batch=%d",
            token_budget,
            self.max_batch_size,
            self.max_frames_per_batch,
        )

        capture_inputs = self.model.prepare_encoder_cudagraph_capture_inputs(
            token_budget,
            self.max_batch_size,
            self.max_frames_per_batch,
            self.device,
            self.dtype,
        )

        mm_kwargs = capture_inputs.mm_kwargs
        buffers = capture_inputs.buffers

        with torch.inference_mode():
            output = self.model.encoder_cudagraph_forward(mm_kwargs, buffers)
            output_buffer = torch.empty_like(output)

        graph = torch.npu.NPUGraph()
        with (
            set_encoder_forward_context(token_budget, True),
            torch.inference_mode(),
            torch.npu.graph(graph, self.graph_pool),
        ):
            output = self.model.encoder_cudagraph_forward(mm_kwargs, buffers)
            output_buffer.copy_(output)

        input_key = self.config.input_key_by_modality["image"]
        self.budget_graphs[token_budget] = BudgetGraphMetadata(
            token_budget=token_budget,
            max_batch_size=self.max_batch_size,
            max_frames_per_batch=self.max_frames_per_batch,
            graph=graph,
            input_buffer=mm_kwargs[input_key],
            metadata_buffers=buffers,
            output_buffer=weak_ref_tensors(output_buffer),
        )

    def _run_budget_graph(
        self,
        mm_kwargs: dict[str, Any],
        token_budget: int,
        replay_buffers: dict[str, torch.Tensor | None],
    ) -> torch.Tensor | None:
        num_items = len(self._get_item_specs(mm_kwargs))
        if token_budget not in self.budget_graphs:
            self.graph_misses += num_items
            return None

        graph_meta = self.budget_graphs[token_budget]

        input_key = self.config.input_key_by_modality[self.model.get_input_modality(mm_kwargs)]
        src = mm_kwargs[input_key]
        n = src.shape[0]
        graph_meta.input_buffer[:n].copy_(src)

        for key in self.config.buffer_keys:
            src_buf = replay_buffers.get(key)
            if src_buf is None:
                continue
            buf = graph_meta.metadata_buffers[key]
            if src_buf.ndim == 0:
                buf.copy_(src_buf)
            else:
                slice_n = src_buf.shape[0]
                buf.zero_()
                buf[:slice_n].copy_(src_buf)

        meta = graph_meta.metadata_buffers
        cu_seqlens_cpu = None if meta.get("cu_seqlens") is None else meta.get("cu_seqlens").cpu()
        cu_window_seqlens_cpu = None if meta.get("cu_window_seqlens") is None else meta.get("cu_window_seqlens").cpu()
        seq_lens_cpu = None if meta.get("sequence_lengths") is None else meta.get("sequence_lengths").cpu()

        update_stream = self.update_stream
        if update_stream is None:
            update_stream = torch.npu.Stream()

        graph_meta.graph.replay()

        with set_encoder_forward_context(
            token_budget,
            False,
            cu_seqlens_cpu=cu_seqlens_cpu,
            cu_window_seqlens_cpu=cu_window_seqlens_cpu,
            sequence_lengths_cpu=seq_lens_cpu,
        ):
            update_encoder_graph_params(update_stream, token_budget, fullatt_block_indexes=self.fullatt)

        self.graph_hits += num_items
        return graph_meta.output_buffer


def weak_ref_encoder_graph_workspaces() -> None:
    params = get_encoder_graph_params()
    if params is None:
        return
    for budget, ws in list(params.workspaces.items()):
        if ws is None:
            continue
        params.workspaces[budget] = weak_ref_tensors(ws)
