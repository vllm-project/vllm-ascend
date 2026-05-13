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
"""Thread-local / context state for vision encoder ACL graph capture and replay."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
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


