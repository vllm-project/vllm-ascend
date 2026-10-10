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
# This file is a part of the vllm-ascend project.
#
"""Central registry for named NPU streams.

Historically the plugin created a handful of module-level singleton streams
(``vllm_ascend.utils`` accessors, ``moe_utils.COMM_STREAM``, and the DSA
overlap stream). This module consolidates their ownership behind one
registry keyed by purpose so later overlap work can reason about "which
stream does what" in a single place.

Design decisions:

- **Lazy creation.** A stream is constructed on first access and never
  earlier. This preserves the timing of the legacy module-level singletons:
  devices are initialized and a forward pass has already run by the time
  most auxiliary streams are needed.
- **Module-level singleton registry.** vLLM workers run one process per NPU
  and the registry is only mutated on first access per name. The legacy
  globals were likewise plain module attributes, so the concurrency exposure
  is unchanged. A lock would only serialize the first lookup; later lookups
  are lock-free reads either way.
- **No new names.** The registry hosts exactly the streams that already
  existed plus nothing else. ``current_stream`` keeps its special semantics:
  it caches the ambient compute stream (following ``torch.npu.set_stream``)
  rather than creating a new one.
"""

from __future__ import annotations

import threading
from typing import Literal

import torch
import torch_npu

StreamName = Literal[
    # Ambient compute stream the worker currently runs on. Populated from
    # torch.npu.current_stream() on first access, NOT a newly created stream.
    "current_computation",
    # Standalone compute stream used by the sampler (see
    # vllm_ascend/sample/sampler.py).
    "global_computation",
    # Auxiliary compute stream for shared-expert MLP overlap
    # (see vllm_ascend/ops/fused_moe/shared_experts.py).
    "shared_experts",
    # Communication stream for context-parallel chunked prefill
    # (see vllm_ascend/attention/context_parallel/attention_cp.py).
    "cp_chunked_prefill",
    # Communication stream for MoE all-to-all dispatch/combine
    # (see vllm_ascend/ops/fused_moe/moe_utils.py).
    "moe_comm",
    # Auxiliary compute stream for DSA quant/matmul part overlap
    # (see vllm_ascend/attention/dsa_v1.py).
    "dsa_overlap",
]

# Registry entries that construct a brand-new torch_npu.npu.Stream.
# "current_computation" is excluded: it snapshots the ambient stream instead
# of allocating one.
_NEW_STREAM_NAMES: frozenset[str] = frozenset(
    {
        "global_computation",
        "shared_experts",
        "cp_chunked_prefill",
        "moe_comm",
        "dsa_overlap",
    }
)

# Note: "attention_calculation" is intentionally NOT registered here. The
# legacy accessor ``vllm_ascend.utils.attention_calculation_stream`` currently
# has zero in-tree consumers; its module-level global stays in ``utils.py``
# untouched so any external references keep working. If a consumer appears,
# register the name here and delegate the accessor.


class StreamRegistry:
    """Owns named singleton NPU streams and creates them lazily.

    Access the process-wide instance through :func:`get_stream_registry`.
    """

    def __init__(self) -> None:
        self._streams: dict[str, torch.npu.Stream] = {}
        # Guards the first construction of each stream. Contention is
        # negligible (streams are created once per process), but the lock
        # keeps concurrent first accesses from allocating duplicates.
        self._lock = threading.Lock()

    def get_stream(self, name: StreamName) -> torch.npu.Stream:
        """Return the singleton stream for ``name``, creating it if needed."""
        stream = self._streams.get(name)
        if stream is not None:
            return stream
        with self._lock:
            stream = self._streams.get(name)
            if stream is not None:
                return stream
            stream = self._create_stream(name)
            self._streams[name] = stream
            return stream

    def peek_stream(self, name: StreamName) -> torch.npu.Stream | None:
        """Return the stream for ``name`` if it exists, else ``None``.

        Unlike :meth:`get_stream` this never allocates. Useful for tests
        asserting lazy-creation behavior.
        """
        return self._streams.get(name)

    def _create_stream(self, name: StreamName) -> torch.npu.Stream:
        if name == "current_computation":
            # Snapshot the ambient stream instead of allocating a new one.
            # torch.npu.current_stream() constructs a fresh wrapper object on
            # every call, so caching the first result avoids that overhead in
            # hot paths (mirrors the legacy vllm_ascend.utils behavior).
            return torch.npu.current_stream()
        if name in _NEW_STREAM_NAMES:
            # moe_comm pins the current device explicitly, matching the
            # legacy moe_utils.COMM_STREAM construction; a plain Stream()
            # resolves the current device the same way for the other names.
            if name == "moe_comm":
                return torch_npu.npu.Stream(device=torch.npu.current_device())
            return torch_npu.npu.Stream()
        raise KeyError(f"Unknown stream name: {name!r}")

    def clear(self) -> None:
        """Drop all cached streams while preserving the registry singleton."""
        with self._lock:
            self._streams.clear()

    def registered_names(self) -> tuple[str, ...]:
        """Return the sorted names the registry knows how to build."""
        return ("current_computation",) + tuple(sorted(_NEW_STREAM_NAMES))


_REGISTRY = StreamRegistry()


def get_stream_registry() -> StreamRegistry:
    """Return the process-wide :class:`StreamRegistry`."""
    return _REGISTRY


def reset_stream_registry() -> None:
    """Drop all cached streams. Intended for tests only."""
    _REGISTRY.clear()
