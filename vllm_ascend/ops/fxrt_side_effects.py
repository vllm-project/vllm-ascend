# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Small FXRT custom-call boundaries used by DeepSeek V4 prefill.

FX graphs cannot carry ``torch.npu.Event`` or ``torch.npu.Stream`` Python
objects.  Keep those objects in process-local registries and expose only stable
integer handles to Dynamo.  FXRT lowers the registered torch operators below
as ``custom_call`` nodes and invokes their Python implementations at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import torch

from vllm_ascend.attention.utils import (
    maybe_save_kv_layer_to_connector,
    notify_kv_cache_written,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.memcache_comm_fence import publish_attention_compute_start_event


@dataclass(frozen=True)
class NPUEventHandle:
    index: int


@dataclass(frozen=True)
class NPUStreamHandle:
    index: int


_registry_lock = RLock()
_npu_events: list[torch.npu.Event] = []
_npu_streams: list[torch.npu.Stream] = []
_named_event_indices: dict[str, int] = {}
_stream_indices: dict[int, int] = {}


def create_npu_event_handle() -> NPUEventHandle:
    """Create an event during worker initialization, outside Dynamo tracing."""
    with _registry_lock:
        index = len(_npu_events)
        _npu_events.append(torch.npu.Event())
    return NPUEventHandle(index)


def register_npu_stream(stream: torch.npu.Stream) -> NPUStreamHandle:
    """Register a long-lived stream and return its graph-safe integer handle."""
    existing = _stream_indices.get(id(stream))
    if existing is not None:
        return NPUStreamHandle(existing)
    with _registry_lock:
        existing = _stream_indices.get(id(stream))
        if existing is not None:
            return NPUStreamHandle(existing)
        index = len(_npu_streams)
        _npu_streams.append(stream)
        _stream_indices[id(stream)] = index
    return NPUStreamHandle(index)


def get_npu_stream_index(stream: torch.npu.Stream) -> int:
    """Return a stable graph-safe handle for a process-local NPU stream."""
    return register_npu_stream(stream).index


def get_fxrt_event_index(name: str) -> int:
    """Return a reusable event slot allocated outside the FX graph.

    The slot is intentionally keyed by semantic stage rather than by MoE/DSA
    layer.  Stages are ordered by stream dependencies, so reusing a slot for
    the next layer is safe and avoids creating Python ``Event`` objects while
    Dynamo is tracing.
    """
    index = _named_event_indices.get(name)
    if index is not None:
        return index
    with _registry_lock:
        index = _named_event_indices.get(name)
        if index is None:
            index = create_npu_event_handle().index
            _named_event_indices[name] = index
        return index


def _stream_from_index(stream_index: int) -> torch.npu.Stream:
    if stream_index < 0:
        return torch.npu.current_stream()
    return _npu_streams[stream_index]


@torch.library.custom_op("vllm_ascend::fxrt_record_event", mutates_args=())
def fxrt_record_event(event_index: int, stream_index: int) -> None:
    _npu_events[event_index].record(_stream_from_index(stream_index))


@fxrt_record_event.register_fake
def _fxrt_record_event_fake(event_index: int, stream_index: int) -> None:
    return None


@torch.library.custom_op("vllm_ascend::fxrt_wait_event", mutates_args=())
def fxrt_wait_event(event_index: int, stream_index: int) -> None:
    _stream_from_index(stream_index).wait_event(_npu_events[event_index])


@fxrt_wait_event.register_fake
def _fxrt_wait_event_fake(event_index: int, stream_index: int) -> None:
    return None


@torch.library.custom_op("vllm_ascend::fxrt_wait_stream", mutates_args=())
def fxrt_wait_stream(stream_index: int, wait_for_stream_index: int) -> None:
    _stream_from_index(stream_index).wait_stream(
        _stream_from_index(wait_for_stream_index)
    )


@fxrt_wait_stream.register_fake
def _fxrt_wait_stream_fake(
    stream_index: int, wait_for_stream_index: int
) -> None:
    return None


@torch.library.custom_op(
    "vllm_ascend::fxrt_wait_for_kv_layer", mutates_args=()
)
def fxrt_wait_for_kv_layer(layer_name: str) -> None:
    wait_for_kv_layer_from_connector(layer_name)


@fxrt_wait_for_kv_layer.register_fake
def _fxrt_wait_for_kv_layer_fake(layer_name: str) -> None:
    return None


@torch.library.custom_op(
    "vllm_ascend::fxrt_save_kv_layer", mutates_args=()
)
def fxrt_save_kv_layer(
    layer_name: str, kv_cache_layer: list[torch.Tensor | None]
) -> None:
    maybe_save_kv_layer_to_connector(layer_name, kv_cache_layer)


@fxrt_save_kv_layer.register_fake
def _fxrt_save_kv_layer_fake(
    layer_name: str, kv_cache_layer: list[torch.Tensor | None]
) -> None:
    return None


@torch.library.custom_op(
    "vllm_ascend::fxrt_notify_kv_cache_written", mutates_args=()
)
def fxrt_notify_kv_cache_written(layer_name: str) -> None:
    notify_kv_cache_written(layer_name)


@fxrt_notify_kv_cache_written.register_fake
def _fxrt_notify_kv_cache_written_fake(layer_name: str) -> None:
    return None


@torch.library.custom_op(
    "vllm_ascend::fxrt_record_attention_compute_start", mutates_args=()
)
def fxrt_record_attention_compute_start(event_index: int) -> None:
    # A gate retained by the asynchronous KV-transfer worker may still refer
    # to the preceding invocation's event.  Publish a fresh event on every
    # execution instead of re-recording that object in place.  Only the stable
    # integer slot crosses the FX graph boundary.
    event = torch.npu.Event()
    with _registry_lock:
        _npu_events[event_index] = event
    event.record(torch.npu.current_stream())
    publish_attention_compute_start_event(event)


@fxrt_record_attention_compute_start.register_fake
def _fxrt_record_attention_compute_start_fake(event_index: int) -> None:
    return None


def get_attention_event_index() -> int:
    """Allocate one stable event index for a DSA layer during its init."""
    return create_npu_event_handle().index


@torch.library.custom_op(
    "vllm_ascend::fxrt_dsa_scatter_if_nonempty",
    mutates_args=("cache",),
)
def fxrt_dsa_scatter_if_nonempty(
    cache: torch.Tensor,
    compressed_kv: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Hide only the data-dependent empty check, not the DSA tensor body."""
    if compressed_kv.shape[0] > 0:
        DeviceOperator.dsa_kv_compress_scatter(
            cache, compressed_kv, slot_mapping
        )


@fxrt_dsa_scatter_if_nonempty.register_fake
def _fxrt_dsa_scatter_if_nonempty_fake(
    cache: torch.Tensor,
    compressed_kv: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    return None


@torch.library.custom_op(
    "vllm_ascend::fxrt_dsa_indexer_scatter_if_nonempty",
    mutates_args=(
        "indexer_k_cache",
        "indexer_scale_cache",
        "indexer_full_cache",
    ),
)
def fxrt_dsa_indexer_scatter_if_nonempty(
    kv: torch.Tensor,
    indexer_k_cache: torch.Tensor,
    indexer_scale_cache: torch.Tensor,
    indexer_full_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor,
) -> None:
    """Keep the empty-KV guard local to one custom call."""
    if kv.numel() > 0:
        DeviceOperator.indexer_scatter_kv(
            kv,
            indexer_k_cache,
            indexer_scale_cache,
            indexer_full_cache,
            slot_mapping,
        )


@fxrt_dsa_indexer_scatter_if_nonempty.register_fake
def _fxrt_dsa_indexer_scatter_if_nonempty_fake(
    kv: torch.Tensor,
    indexer_k_cache: torch.Tensor,
    indexer_scale_cache: torch.Tensor,
    indexer_full_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor,
) -> None:
    return None
