# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MRV2 draft KV sliding-window helpers.

Model Runner V2 bakes ``seq_lens`` / block tables into per-layer metadata inside
``_build_attn_metadata``, so the window must be applied to builder inputs
(windowed ``input_buffers.seq_lens``, cropped ``input_block_tables``, capped
``seq_lens_cpu_upper_bound``) around that call. Slot mappings stay absolute.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.v1.spec_decode.utils import SlidingWindowAdapter


def resolve_draft_window_size(method: str, vllm_config: VllmConfig) -> int | None:
    """Return ``draft_window_size`` when enabled for ``method``, else ``None``.

    Mirrors MRV1 validation in ``AscendSpecDecodeBaseProposer.load_model``:
    MTP is incompatible (reuses target layers); DeepSeek-V4 DSpark drafts warn
    that windowing can hurt already-stable long-context acceptance.
    """
    draft_window_size = get_ascend_config().draft_window_size
    if draft_window_size is None:
        return None

    if method == "mtp":
        logger.warning(
            "[sliding-window] draft method is MTP, which is not "
            "compatible with draft_window_size; the window is ignored."
        )
        return None

    target_hf_config = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
    target_model_type = getattr(target_hf_config, "model_type", "") or ""
    if method == "dspark" and target_model_type.startswith("deepseek_v4"):
        logger.warning(
            "[sliding-window] DSpark drafts trained natively with a "
            "DeepSeek-V4 target are long-context stable; enabling the "
            "sliding window degrades acceptance length."
        )
    return draft_window_size


def _checked_block_sizes(block_sizes: Sequence[int], num_groups: int) -> list[int]:
    """Require one positive block size per KV table."""
    resolved = [int(size) for size in block_sizes]
    if len(resolved) != num_groups:
        raise RuntimeError(
            "draft sliding window block sizes do not match KV groups: "
            f"{resolved} for {num_groups} tables"
        )
    if any(size <= 0 for size in resolved):
        raise RuntimeError(f"draft sliding window block sizes must be positive, got {resolved}")
    return resolved


def _kernel_block_sizes(block_tables: Any, num_groups: int) -> list[int]:
    """Prefer one Ascend kernel block size per KV table; fall back to the physical block size.

    ``len`` covers both a list and a tensor. A truthiness check on a multi-value
    tensor raises, which is the mixed-size case this crop has to accept.
    """
    kernel_block_sizes = getattr(block_tables, "kernel_block_sizes", None)
    if kernel_block_sizes is not None and len(kernel_block_sizes) > 0:
        return _checked_block_sizes(kernel_block_sizes, num_groups)
    block_sizes = getattr(block_tables, "block_sizes", None)
    if block_sizes is not None and len(block_sizes) > 0:
        return _checked_block_sizes(block_sizes, num_groups)
    raise RuntimeError("draft sliding window requires block_tables with a block size")


class MRV2DraftSlidingWindow:
    """Apply :class:`SlidingWindowAdapter` to MRV2 draft builder inputs."""

    def __init__(
        self,
        window_size: int,
        block_sizes: Sequence[int],
        max_num_reqs: int,
        future_offset: int,
        device: torch.device,
    ) -> None:
        if not block_sizes:
            raise ValueError("draft sliding window requires at least one KV cache group")
        self.block_sizes = tuple(int(size) for size in block_sizes)
        # One seq_lens tensor is shared by every group, so the token crop has to be a
        # multiple of every kernel block size. Each table then drops start_tokens / block_size
        # columns: a 128-token offset is 2 columns of 64 and 1 column of 128.
        self.align = math.lcm(*self.block_sizes)
        self.adapters = [
            SlidingWindowAdapter(
                window_size,
                block_size,
                max_num_reqs,
                future_offset,
                device,
                align=self.align,
            )
            for block_size in self.block_sizes
        ]
        self._clones = [
            torch.zeros(
                (max_num_reqs, adapter.max_window_blocks),
                dtype=torch.int32,
                device=device,
            )
            for adapter in self.adapters
        ]
        self._last_windowed_seq_lens: torch.Tensor | None = None

    @property
    def max_window_tokens(self) -> int:
        return min(adapter.max_window_blocks * adapter.block_size for adapter in self.adapters)

    @classmethod
    def maybe_create(
        cls,
        *,
        method: str,
        vllm_config: VllmConfig,
        block_tables: Any,
        max_num_reqs: int,
        device: torch.device,
        future_offset: int,
    ) -> MRV2DraftSlidingWindow | None:
        window_size = resolve_draft_window_size(method, vllm_config)
        if window_size is None:
            return None
        input_block_tables = getattr(block_tables, "input_block_tables", None)
        if input_block_tables is None or len(input_block_tables) == 0:
            return None
        return cls(
            window_size,
            _kernel_block_sizes(block_tables, len(input_block_tables)),
            max_num_reqs,
            future_offset,
            device,
        )

    def _subtract_crop(self, tensor: torch.Tensor, crop: torch.Tensor, num_reqs: int) -> None:
        """Subtract the block-table crop offset from a length tensor in-place."""
        aligned = crop.to(device=tensor.device, dtype=tensor.dtype)
        tensor[:num_reqs].sub_(aligned)

    @contextmanager
    def apply_builder_inputs(
        self,
        input_buffers: Any,
        block_tables: Any,
        seq_lens_cpu_upper_bound: torch.Tensor,
        num_reqs: int,
        extra_seq_lens: Sequence[torch.Tensor | None] = (),
    ) -> Iterator[torch.Tensor]:
        """Window seq_lens + block tables for one metadata build; restore after.

        Yields a windowed clone of ``seq_lens_cpu_upper_bound``. Slot mappings
        on ``block_tables`` are never modified.
        """
        if num_reqs <= 0:
            yield seq_lens_cpu_upper_bound
            return

        original_seq_lens = input_buffers.seq_lens
        original_tables = list(block_tables.input_block_tables)
        if len(original_tables) != len(self._clones):
            raise RuntimeError(
                "draft sliding window group count changed after initialization: "
                f"expected {len(self._clones)}, got {len(original_tables)}"
            )

        extras = [extra for extra in extra_seq_lens if extra is not None]

        # One token crop for every group. Each table divides it by its own block size.
        # Private clones are filled before any live table is replaced.
        seq_lens = original_seq_lens[:num_reqs]
        crop = self.adapters[0].token_crop(seq_lens)
        windowed_seq_lens = seq_lens - crop
        for i, full_table in enumerate(original_tables):
            meta = SimpleNamespace(seq_lens=seq_lens, block_table_tensor=full_table)
            adapter = self.adapters[i]
            adapter.full_block_table = full_table
            adapter.compute_sliding_window_block_table(meta, self._clones[i], start_tokens=crop)
        for clone in self._clones:
            # Keep max_num_reqs rows so a padded graph batch can slice past num_reqs.
            # Rows from an earlier, larger batch must not survive.
            clone[num_reqs:].zero_()
        for i in range(len(original_tables)):
            block_tables.input_block_tables[i] = self._clones[i]
        self._last_windowed_seq_lens = windowed_seq_lens

        # Separate storage: metadata may alias ``input_buffers.seq_lens``.
        windowed_buffer = original_seq_lens.clone()
        windowed_buffer[:num_reqs].copy_(windowed_seq_lens)
        input_buffers.seq_lens = windowed_buffer

        windowed_upper_bound = seq_lens_cpu_upper_bound.clone()
        self._subtract_crop(windowed_upper_bound, crop, num_reqs)
        for extra in extras:
            self._subtract_crop(extra, crop, num_reqs)

        try:
            yield windowed_upper_bound
        finally:
            input_buffers.seq_lens = original_seq_lens
            for i, table in enumerate(original_tables):
                block_tables.input_block_tables[i] = table

    def clamp_cpu_lengths(self, seq_lens: torch.Tensor, num_reqs: int) -> torch.Tensor:
        """Cap host lengths at the cropped table. ``seq_lens`` shape is unchanged."""
        if num_reqs <= 0:
            return seq_lens
        out = seq_lens.clone()
        limit = min(num_reqs, out.shape[0])
        active = out[:limit]
        active.clamp_(max=self.max_window_tokens)
        min_lens = self._last_windowed_seq_lens
        if min_lens is not None:
            n = min(limit, min_lens.shape[0])
            floor = min_lens[:n].to(device=active.device, dtype=active.dtype)
            active[:n].clamp_(min=floor)
        return out

    def clamp_metadata_after_step(self, attn_metadata: Any, num_reqs: int) -> None:
        """Keep length fields an upper bound of the cropped block table.

        Upstream adds ``step`` to ``seq_lens_cpu_upper_bound`` after our crop.
        Clamp so the bound cannot exceed the cropped table, and cannot fall
        below the windowed actual length.
        """
        if not attn_metadata or num_reqs <= 0:
            return
        max_tokens = self.max_window_tokens
        min_lens = self._last_windowed_seq_lens
        metadata_items: Iterable[Any] = attn_metadata.values() if isinstance(attn_metadata, dict) else (attn_metadata,)
        for metadata in metadata_items:
            if metadata is None:
                continue
            self._clamp_length_fields(metadata, num_reqs, max_tokens, min_lens)
            decode = getattr(metadata, "decode", None)
            if decode is not None:
                self._clamp_length_fields(decode, num_reqs, max_tokens, min_lens)

    def _clamp_length_fields(
        self,
        obj: Any,
        num_reqs: int,
        max_tokens: int,
        min_lens: torch.Tensor | None,
    ) -> None:
        for name in ("seq_lens_cpu", "_seq_lens_cpu", "seq_lens_cpu_upper_bound"):
            src = getattr(obj, name, None)
            if src is None or not isinstance(src, torch.Tensor) or src.device.type != "cpu":
                continue
            sl = src[: min(num_reqs, src.shape[0])]
            sl.clamp_(max=max_tokens)
            if min_lens is not None:
                sl.clamp_(min=min_lens[: sl.shape[0]].to(device=sl.device, dtype=sl.dtype))
        seq_lens_list = getattr(obj, "seq_lens_list", None)
        if seq_lens_list:
            floor = (
                min_lens[: min(num_reqs, len(seq_lens_list))].tolist()
                if min_lens is not None
                else [0] * len(seq_lens_list)
            )
            obj.seq_lens_list = [
                max(int(floor[i]) if i < len(floor) else 0, min(int(x), max_tokens))
                for i, x in enumerate(seq_lens_list)
            ]


class DraftSlidingWindowMixin:
    """Mixin that installs and applies MRV2 draft sliding-window builder hooks."""

    _draft_sliding_window: MRV2DraftSlidingWindow | None

    def _resolve_draft_sliding_window_method(self) -> str:
        method = getattr(self, "_draft_sliding_window_method", None)
        if method is not None:
            return method
        return getattr(getattr(self, "speculative_config", None), "method", None) or "eagle"

    def _draft_sliding_window_future_offset(self) -> int:
        method = self._resolve_draft_sliding_window_method()
        if method in ("dflash", "dspark"):
            return 0
        # EAGLE3: seq_lens is context-only; cover K draft positions beyond it.
        return int(getattr(self, "num_speculative_tokens", None) or getattr(self, "num_speculative_steps", 0))

    def _init_draft_sliding_window(self) -> None:
        self._draft_sliding_window = MRV2DraftSlidingWindow.maybe_create(
            method=self._resolve_draft_sliding_window_method(),
            vllm_config=self.vllm_config,
            block_tables=self.block_tables,
            max_num_reqs=self.max_num_reqs,
            device=self.device,
            future_offset=self._draft_sliding_window_future_offset(),
        )

    @contextmanager
    def _maybe_draft_sliding_window(
        self,
        seq_lens_cpu_upper_bound: torch.Tensor,
        num_reqs: int,
        extra_seq_lens: Sequence[torch.Tensor | None] = (),
    ) -> Iterator[torch.Tensor]:
        helper = getattr(self, "_draft_sliding_window", None)
        if helper is None:
            yield seq_lens_cpu_upper_bound
            return
        with helper.apply_builder_inputs(
            self.input_buffers,
            self.block_tables,
            seq_lens_cpu_upper_bound,
            num_reqs,
            extra_seq_lens=extra_seq_lens,
        ) as windowed_upper_bound:
            yield windowed_upper_bound

    def _clamp_windowed_metadata(self, attn_metadata: Any, num_reqs: int) -> None:
        helper = getattr(self, "_draft_sliding_window", None)
        if helper is None:
            return
        helper.clamp_metadata_after_step(attn_metadata, num_reqs)

    def _window_cpu_lengths(self, seq_lens: torch.Tensor, num_reqs: int) -> torch.Tensor:
        """Return ``seq_lens`` capped at the draft window. Unchanged when the window is off."""
        helper = getattr(self, "_draft_sliding_window", None)
        if helper is None:
            return seq_lens
        return helper.clamp_cpu_lengths(seq_lens, num_reqs)
