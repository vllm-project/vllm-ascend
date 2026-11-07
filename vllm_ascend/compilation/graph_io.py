# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Address binding for inputs of captured ACL graphs under VPP.

An ACL graph replays the kernels recorded at capture time, and those kernels
read their inputs from the *addresses* the tensors had at capture. A captured
graph therefore only produces correct results if every runtime input sits at
the captured address when the graph is replayed.

Virtual pipeline parallelism (VPP) breaks that assumption twice:

* ``_prepare_vpp_context`` hands every in-flight batch its own clones of
  ``input_ids`` / ``positions`` / ``inputs_embeds`` (the runner's persistent
  buffers are overwritten by the next batch while this one is suspended across
  a VPP yield), and the fold-back carry is the previous stage's graph output.
* ``_clone_vpp_attn_metadata`` likewise clones the attention metadata, while
  the KV write inside a FULL graph reads ``slot_mapping`` by address (unlike
  the FIA/PA parameters, it is not re-supplied by an ``UpdatableGraph`` update
  task at replay time).

``GraphIOBinder`` is the single place that knows about those addresses: the
runner records the anchors while ``_dummy_run`` captures, and copies each
batch's tensors back into the anchors immediately before the replay.

The binder only exists when ``virtual_pipeline_parallel_size > 1``; the runner
never creates one otherwise, so non-VPP graph execution is untouched.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from vllm.config import CUDAGraphMode
from vllm.logger import logger
from vllm.sequence import IntermediateTensors

# The binder whose capture forward is currently running, or None. Attention
# backends consult it to verify that tensors they read by address are covered
# by an anchor. A plain module global (not a contextvar) keeps the check on the
# non-VPP hot path down to a single global lookup; graph capture runs on the
# worker thread only.
_ACTIVE_CAPTURE_BINDER: GraphIOBinder | None = None


def graph_io_capture_active() -> bool:
    """True while a VPP graph-capture forward is running."""
    return _ACTIVE_CAPTURE_BINDER is not None


def note_graph_direct_read(field: str, tensor: torch.Tensor) -> None:
    """Check that an attention op reads an anchored metadata buffer."""
    binder = _ACTIVE_CAPTURE_BINDER
    if binder is not None:
        binder.note_direct_read(field, tensor)


# Metadata fields that the KV write (``npu_scatter_pa_kv_cache``) reads by
# address and that ``UpdatableGraph`` does *not* refresh at replay time.
# Everything else (FIA/PA parameters, block tables, seq_lens) is either
# re-supplied through an update task or consumed by an eager splitting op.
DIRECT_READ_METADATA_FIELDS = ("slot_mapping",)

# Model inputs that a captured graph reads by address. Failing to bind one of
# these silently replays stale activations, so it is an error, not a warning.
_CRITICAL_INPUT_KEYS = ("input_ids", "positions", "inputs_embeds")


def _aligned_views(anchor: torch.Tensor, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return ``(anchor_view, source_view)`` with identical shapes.

    The token dimension is dim 0 for ``[num_tokens, ...]`` tensors and the
    last dim for the mrope ``[3, num_tokens]`` layout, so both are tried.
    Returns ``None`` when the tensors cannot be aligned (different rank or
    mismatching non-token dimensions).
    """
    if anchor.dim() != source.dim():
        return None
    if anchor.shape == source.shape:
        return anchor, source
    for dim in (0, anchor.dim() - 1):
        if dim < 0:
            continue
        if anchor.shape[:dim] == source.shape[:dim] and anchor.shape[dim + 1 :] == source.shape[dim + 1 :]:
            n = min(anchor.shape[dim], source.shape[dim])
            return anchor.narrow(dim, 0, n), source.narrow(dim, 0, n)
    return None


def _iter_metadata(attn_metadata: Any) -> Iterator[tuple[str, Any]]:
    """Yield ``(layer_name, metadata)`` from a per-layer metadata container."""
    if attn_metadata is None:
        return
    # ubatching hands the runner a list of per-ubatch dicts.
    containers = attn_metadata if isinstance(attn_metadata, list) else [attn_metadata]
    for container in containers:
        if isinstance(container, Mapping):
            yield from container.items()


class GraphIOBinder:
    """Binds per-batch VPP tensors onto the addresses captured ACL graphs read.

    Anchors are recorded lazily during the capture forward (``_dummy_run``),
    so they are exactly the tensors the graph kernels were recorded with.
    """

    def __init__(
        self,
        intermediate_sync: Callable[[int, IntermediateTensors], IntermediateTensors] | None = None,
    ) -> None:
        # Persistent buffer a captured graph reads for each model input path.
        self._anchors: dict[str, torch.Tensor] = {}
        # layer name -> {metadata field -> persistent buffer}
        self._metadata_anchors: dict[str, dict[str, torch.Tensor]] = {}
        # Pins the carry into the runner's persistent intermediate buffer.
        self._intermediate_sync = intermediate_sync
        self._capturing = False
        self._warned: set[str] = set()

        from vllm_ascend import envs as ascend_envs

        self._strict = ascend_envs.VLLM_ASCEND_GRAPH_IO_STRICT
        self._debug = ascend_envs.VLLM_ASCEND_GRAPH_IO_DEBUG

    # ------------------------------------------------------------------
    # capture side
    # ------------------------------------------------------------------
    @contextmanager
    def capture_context(self):
        """Mark the enclosed forwards as graph-capture (or profile) forwards.

        Anchor recording has to happen here rather than by inspecting
        ``is_current_stream_capturing()`` inside ``_model_forward``: ACL graph
        capture happens *inside* the wrapper, after the inputs are bound.
        """
        global _ACTIVE_CAPTURE_BINDER
        prev_binder = _ACTIVE_CAPTURE_BINDER
        prev_capturing = self._capturing
        self._capturing = True
        _ACTIVE_CAPTURE_BINDER = self
        try:
            yield self
        finally:
            _ACTIVE_CAPTURE_BINDER = prev_binder
            self._capturing = prev_capturing

    def record_metadata_anchors(self, attn_metadata: Any) -> None:
        """Remember the metadata buffers a capturing graph reads by address."""
        for layer_name, metadata in _iter_metadata(attn_metadata):
            for field in DIRECT_READ_METADATA_FIELDS:
                tensor = getattr(metadata, field, None)
                if not isinstance(tensor, torch.Tensor):
                    continue
                fields = self._metadata_anchors.setdefault(layer_name, {})
                anchor = fields.get(field)
                if anchor is None:
                    fields[field] = tensor
                elif anchor.data_ptr() != tensor.data_ptr():
                    self._fail(
                        f"metadata anchor for {layer_name}.{field} moved between "
                        f"captures ({anchor.data_ptr():#x} -> {tensor.data_ptr():#x})"
                    )

    def note_direct_read(self, field: str, tensor: torch.Tensor) -> None:
        """Verify a capturing attention op reads an anchored metadata buffer.

        Called by attention backends right before they launch an op that reads
        ``tensor`` by address. A tensor derived from an anchored buffer (a
        slice/view) is fine; anything else means the replay would read stale
        data, and adding such a field requires registering it in
        ``DIRECT_READ_METADATA_FIELDS`` and pinning it in ``pin_metadata``.
        """
        start = tensor.data_ptr()
        end = start + tensor.numel() * tensor.element_size()
        for fields in self._metadata_anchors.values():
            anchor = fields.get(field)
            if anchor is None:
                continue
            base = anchor.data_ptr()
            if base <= start and end <= base + anchor.numel() * anchor.element_size():
                return
        self._fail(
            f"attention op reads {field} at {start:#x}, which is not backed by "
            "any captured metadata anchor; the ACL graph would replay stale data"
        )

    # ------------------------------------------------------------------
    # replay side
    # ------------------------------------------------------------------
    def bind_forward_inputs(
        self,
        model_inputs: dict[str, Any],
        mode: CUDAGraphMode,
    ) -> dict[str, Any]:
        """Return ``model_inputs`` whose tensors sit on the captured addresses."""
        if self._capturing:
            self._record_inputs(model_inputs)
            return model_inputs
        if mode == CUDAGraphMode.NONE:
            # Eager fallback for this batch: the caller's tensors are used
            # directly, no captured graph reads them.
            return model_inputs
        return self._pin_inputs(model_inputs)

    def pin_metadata(self, attn_metadata: Any, mode: CUDAGraphMode) -> None:
        """Copy this batch's metadata onto the captured metadata buffers.

        Only FULL graphs need this: their FIA/PA parameters are refreshed by
        ``UpdatableGraph`` update tasks, but the KV write reads ``slot_mapping``
        by address, and the eager splitting ops of PIECEWISE read the runtime
        metadata object directly.
        """
        if mode != CUDAGraphMode.FULL:
            return
        copied: set[int] = set()
        for layer_name, metadata in _iter_metadata(attn_metadata):
            for field in DIRECT_READ_METADATA_FIELDS:
                source = getattr(metadata, field, None)
                if not isinstance(source, torch.Tensor):
                    continue
                anchor = self._metadata_anchors.get(layer_name, {}).get(field)
                if anchor is None:
                    # The layer did not run during capture (e.g. a stage that
                    # was never replayed), so there is no graph to feed.
                    continue
                if id(source) in copied:
                    continue
                copied.add(id(source))
                views = _aligned_views(anchor, source)
                if views is None:
                    self._fail(
                        f"cannot bind {layer_name}.{field}: anchor shape "
                        f"{tuple(anchor.shape)} vs batch shape "
                        f"{tuple(source.shape)}"
                    )
                    continue
                anchor_view, source_view = views
                if anchor_view.data_ptr() != source_view.data_ptr():
                    anchor_view.copy_(source_view, non_blocking=True)
                if self._debug:
                    logger.info(
                        "graph io: pinned %s.%s (%d elems) to %#x",
                        layer_name,
                        field,
                        anchor_view.numel(),
                        anchor_view.data_ptr(),
                    )

    def pin_intermediates(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors,
    ) -> IntermediateTensors:
        """Route a graph output carry through the persistent intermediate buffer."""
        if self._intermediate_sync is None:
            return intermediate_tensors
        return self._intermediate_sync(num_tokens, intermediate_tensors)

    @staticmethod
    def snapshot(intermediate_tensors: IntermediateTensors) -> IntermediateTensors:
        """Private copy of a graph output that stays alive across a yield.

        A graph always writes the same output buffer for a given
        ``(batch_descriptor, virtual stage)``: replaying the next in-flight
        batch overwrites it. The async send path protects eager tensors by
        retaining a reference, which cannot protect a graph buffer.
        """
        return IntermediateTensors({k: v.clone() for k, v in intermediate_tensors.tensors.items()})

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _record_inputs(self, model_inputs: dict[str, Any]) -> None:
        for key, value in model_inputs.items():
            if isinstance(value, IntermediateTensors):
                for name, tensor in value.tensors.items():
                    if isinstance(tensor, torch.Tensor):
                        self._record_anchor(f"intermediate.{name}", tensor)
            elif isinstance(value, torch.Tensor):
                self._record_anchor(f"input.{key}", value)

    def _record_anchor(self, path: str, tensor: torch.Tensor) -> None:
        anchor = self._anchors.get(path)
        if anchor is None:
            self._anchors[path] = tensor
            return
        if anchor.data_ptr() != tensor.data_ptr():
            # Keep the first anchor: every graph captured so far reads it, and
            # rebinding to a new buffer would leave those graphs unserviced.
            self._fail(
                f"anchor for {path} moved between captures "
                f"({anchor.data_ptr():#x} -> {tensor.data_ptr():#x}); a captured "
                "graph would read the wrong buffer"
            )
            return
        # Captures happen per shape; keep the widest view of the same buffer.
        if tensor.numel() > anchor.numel():
            self._anchors[path] = tensor

    def _pin_inputs(self, model_inputs: dict[str, Any]) -> dict[str, Any]:
        pinned: dict[str, Any] = {}
        changed = False
        for key, value in model_inputs.items():
            if isinstance(value, IntermediateTensors):
                new_tensors = {
                    name: self._pin_tensor(f"intermediate.{name}", tensor, critical=True)
                    for name, tensor in value.tensors.items()
                }
                changed |= any(new is not old for new, old in zip(new_tensors.values(), value.tensors.values()))
                pinned[key] = IntermediateTensors(new_tensors)
            elif isinstance(value, torch.Tensor):
                new_value = self._pin_tensor(f"input.{key}", value, critical=key in _CRITICAL_INPUT_KEYS)
                changed |= new_value is not value
                pinned[key] = new_value
            else:
                pinned[key] = value
        return pinned if changed else model_inputs

    def _pin_tensor(self, path: str, source: torch.Tensor, *, critical: bool) -> torch.Tensor:
        anchor = self._anchors.get(path)
        if anchor is None or anchor.data_ptr() == source.data_ptr():
            return source
        views = _aligned_views(anchor, source)
        if views is None:
            message = (
                f"cannot bind {path}: anchor shape {tuple(anchor.shape)} vs "
                f"batch shape {tuple(source.shape)}; the captured graph would "
                "read the anchor buffer, not this batch's data"
            )
            if critical or self._strict:
                raise RuntimeError(f"[vllm-ascend][VPP graph io] {message}")
            self._warn_once(path, message)
            return source
        anchor_view, source_view = views
        anchor_view.copy_(source_view, non_blocking=True)
        if self._debug:
            logger.info(
                "graph io: pinned %s (%d elems) to %#x",
                path,
                anchor_view.numel(),
                anchor_view.data_ptr(),
            )
        return anchor_view

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning("[vllm-ascend][VPP graph io] %s", message)

    def _fail(self, message: str) -> None:
        if self._strict:
            raise RuntimeError(f"[vllm-ascend][VPP graph io] {message}")
        self._warn_once(message, message)
