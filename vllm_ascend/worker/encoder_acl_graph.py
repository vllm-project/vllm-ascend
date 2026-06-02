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

from collections.abc import AbstractSet
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch_npu
from vllm.logger import init_logger
from vllm.v1.worker.encoder_cudagraph import BudgetGraphMetadata, EncoderCudaGraphManager

from vllm_ascend.utils import weak_ref_tensors

logger = init_logger(__name__)


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
    if _encoder_graph_params is not None:
        existing = sorted(_encoder_graph_params.events.keys())
        if existing == budgets_sorted_unique:
            return
        raise RuntimeError(
            "Encoder graph params already initialized with different budgets "
            f"(existing={existing}, new={budgets_sorted_unique})"
        )
    _encoder_graph_params = EncoderGraphParams(
        events={b: [] for b in budgets_sorted_unique},
        workspaces={b: None for b in budgets_sorted_unique},
        handles={b: [] for b in budgets_sorted_unique},
        attn_params={b: [] for b in budgets_sorted_unique},
    )


def get_encoder_graph_params() -> EncoderGraphParams | None:
    return _encoder_graph_params


def reset_encoder_graph_params_for_testing() -> None:
    global _encoder_graph_params
    _encoder_graph_params = None


def update_encoder_graph_workspace(token_budget: int, workspace: torch.Tensor) -> None:
    if _encoder_graph_params is None:
        return
    _encoder_graph_params.workspaces[token_budget] = workspace


# ---------------------------------------------------------------------------
# Capture / replay runtime state (thread-local module singleton)
# ---------------------------------------------------------------------------


@dataclass
class EncoderGraphRuntimeState:
    """Vision encoder NPUGraph runtime flags and host-side FIA arguments.

    Captured tensors stay on device; FIA ``graph_task_update`` needs Python ``list[int]``
    lengths that are refreshed each replay from encoder metadata buffers on device (see RFC).
    """

    # --- Capture / replay scheduling ---
    # Current encoder CUDA-graph token budget key (matches EncoderCudaGraphManager budget).
    token_budget: int | None = None
    # True while recording the encoder NPUGraph forward (ViT uses FIA graph-task path).
    capturing: bool = False

    # --- Capture-only: order of ViT attention registrations ---
    # Increments once per ViT attention during capture (order matches ``enumerate(blocks)``).
    capture_layer_cursor: int = 0

    # --- Host layouts consumed by update_encoder_full_graph_params / FIA ---
    # Cumulative token ends per variable-length sequence from ``cu_seqlens`` (``cu[i:]`` prefix sums).
    host_cu_seqlens_ends: list[int] | None = None
    # Same semantics for window layout (``cu_window_seqlens``).
    host_cu_window_seqlens_ends: list[int] | None = None
    # Cumulative ends derived from per-seq ``sequence_lengths`` (e.g. Qwen3-VL FlashInfer-style metadata).
    host_sequence_lengths: list[int] | None = None


_state = EncoderGraphRuntimeState()


def get_encoder_graph_runtime_state() -> EncoderGraphRuntimeState:
    return _state


def reset_encoder_graph_runtime_state() -> None:
    global _state
    _state = EncoderGraphRuntimeState()


def _reset_capture_scope_fields() -> None:
    """Clear capture-only fields. Host replay lists are untouched."""

    _state.token_budget = None
    _state.capturing = False
    _state.capture_layer_cursor = 0


def _reset_replay_scope_fields() -> None:
    """Clear replay-time host length fields."""

    _state.token_budget = None
    _state.capturing = False
    _state.host_cu_seqlens_ends = None
    _state.host_cu_window_seqlens_ends = None
    _state.host_sequence_lengths = None


@contextmanager
def encoder_graph_capture_scope(token_budget: int):
    """Enter encoder NPUGraph capture: callers must pass ``token_budget`` each time.

    ``fullatt_block_indexes`` is **not** stored here; replay resolves cu vs cu_window host lengths
    in ``update_encoder_full_graph_params`` using the model config and ``vit_layer_idx`` captured
    per attention layer.

    On exit, capture-related fields are **cleared** (not restored). Nested scopes are not
    supported; each capture episode must open this context with fresh arguments.
    """

    _state.token_budget = token_budget
    _state.capturing = True
    _state.capture_layer_cursor = 0
    try:
        yield _state
    finally:
        _reset_capture_scope_fields()


@contextmanager
def encoder_graph_replay_scope(
    token_budget: int,
    *,
    host_cu_seqlens_ends: list[int] | None = None,
    host_cu_window_seqlens_ends: list[int] | None = None,
    host_sequence_lengths: list[int] | None = None,
):
    """Enter encoder graph replay (FIA host args): callers must pass lengths each time.

    On exit, replay host fields are **cleared** (not restored). Lists must not be reused
    across replays without repopulating from the current batch buffers.
    """

    _state.token_budget = token_budget
    _state.host_cu_seqlens_ends = host_cu_seqlens_ends
    _state.host_cu_window_seqlens_ends = host_cu_window_seqlens_ends
    _state.host_sequence_lengths = host_sequence_lengths
    _state.capturing = False
    try:
        yield _state
    finally:
        _reset_replay_scope_fields()


# ---------------------------------------------------------------------------
# Replay-time FIA task updates
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Encoder NPUGraph manager
# ---------------------------------------------------------------------------


def _cu_prefix_to_host_endpoints(cu: torch.Tensor | None) -> list[int] | None:
    if cu is None:
        return None
    flat = cu.detach().cpu().view(-1).tolist()
    return flat[1:] if flat else flat


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


class EncoderAclGraphManager(EncoderCudaGraphManager):
    """Hooks encoder capture/replay into Ascend FIA graph-task infrastructure."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.update_stream: torch.npu.Stream | None = None

    def capture(self):
        set_encoder_graph_params(self.token_budgets)
        super().capture()
        weak_ref_encoder_graph_workspaces()

    def _capture_budget_graph(self, token_budget: int):
        logger.debug(
            "Capturing encoder aclgraph for budget=%d, max_batch_size=%d, "
            "max_frames_per_batch=%d",
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
        with encoder_graph_capture_scope(token_budget):
            with torch.inference_mode(), torch.npu.graph(graph):
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
            output_buffer=output_buffer,
        )

    def _run_budget_graph(
        self,
        mm_kwargs: dict[str, Any],
        token_budget: int,
        replay_buffers: dict[str, torch.Tensor | None],
    ) -> torch.Tensor | None:
        num_items = self.model.get_encoder_cudagraph_num_items(mm_kwargs)
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
        host_full = _cu_prefix_to_host_endpoints(meta.get("cu_seqlens"))
        host_win = _cu_prefix_to_host_endpoints(meta.get("cu_window_seqlens"))
        seq_lens_tensor = meta.get("sequence_lengths")
        host_seq_lens = None
        if isinstance(seq_lens_tensor, torch.Tensor):
            host_seq_lens = _per_seq_lengths_to_fia_endpoints(seq_lens_tensor.detach().cpu().view(-1).tolist())

        update_stream = self.update_stream
        if update_stream is None:
            update_stream = torch.npu.Stream()

        visual = getattr(self.model, "visual", None)
        fa_raw = getattr(visual, "fullatt_block_indexes", None) if visual is not None else None
        fullatt = frozenset(fa_raw) if fa_raw is not None else None

        with encoder_graph_replay_scope(
            token_budget,
            host_cu_seqlens_ends=host_full,
            host_cu_window_seqlens_ends=host_win,
            host_sequence_lengths=host_seq_lens,
        ):
            update_encoder_full_graph_params(
                update_stream, token_budget, fullatt_block_indexes=fullatt
            )

        torch.npu.current_stream().wait_stream(update_stream)
        graph_meta.graph.replay()

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
