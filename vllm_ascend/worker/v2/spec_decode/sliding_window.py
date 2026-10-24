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

from collections.abc import Iterator
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


def _kernel_block_size(block_tables: Any) -> int:
    """Prefer Ascend kernel block size; fall back to physical block size."""
    kernel_block_sizes = getattr(block_tables, "kernel_block_sizes", None)
    if kernel_block_sizes:
        return int(kernel_block_sizes[0])
    block_sizes = getattr(block_tables, "block_sizes", None)
    if block_sizes:
        return int(block_sizes[0])
    raise RuntimeError("draft sliding window requires block_tables with a block size")


class MRV2DraftSlidingWindow:
    """Apply :class:`SlidingWindowAdapter` to MRV2 draft builder inputs."""

    def __init__(
        self,
        window_size: int,
        block_size: int,
        max_num_reqs: int,
        future_offset: int,
        device: torch.device,
        num_groups: int,
    ) -> None:
        if num_groups < 1:
            raise ValueError("draft sliding window requires at least one KV cache group")
        self.adapter = SlidingWindowAdapter(
            window_size,
            block_size,
            max_num_reqs,
            future_offset,
            device,
        )
        self._clones = [
            torch.zeros(
                (max_num_reqs, self.adapter.max_window_blocks),
                dtype=torch.int32,
                device=device,
            )
            for _ in range(num_groups)
        ]

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
        if not input_block_tables:
            return None
        return cls(
            window_size,
            _kernel_block_size(block_tables),
            max_num_reqs,
            future_offset,
            device,
            num_groups=len(input_block_tables),
        )

    @contextmanager
    def apply_builder_inputs(
        self,
        input_buffers: Any,
        block_tables: Any,
        seq_lens_cpu_upper_bound: torch.Tensor,
        num_reqs: int,
    ) -> Iterator[torch.Tensor]:
        """Window seq_lens + block tables for one metadata build; restore after.

        Yields a windowed clone of ``seq_lens_cpu_upper_bound``. Slot mappings
        on ``block_tables`` are never modified.
        """
        if num_reqs <= 0:
            yield seq_lens_cpu_upper_bound
            return

        seq_lens = input_buffers.seq_lens
        saved_seq_lens = seq_lens[:num_reqs].clone()
        original_tables = list(block_tables.input_block_tables)
        if len(original_tables) != len(self._clones):
            raise RuntimeError(
                "draft sliding window group count changed after initialization: "
                f"expected {len(self._clones)}, got {len(original_tables)}"
            )

        windowed_seq_lens: torch.Tensor | None = None
        for i, full_table in enumerate(original_tables):
            meta = SimpleNamespace(
                seq_lens=saved_seq_lens,
                block_table_tensor=full_table,
            )
            self.adapter.full_block_table = full_table
            self.adapter.compute_sliding_window_block_table(meta, self._clones[i])
            if windowed_seq_lens is None:
                windowed_seq_lens = self.adapter._windowed_seq_lens
            block_tables.input_block_tables[i] = self._clones[i][:num_reqs]

        assert windowed_seq_lens is not None
        seq_lens[:num_reqs].copy_(windowed_seq_lens)

        k_future = self.adapter._future_offset
        w = self.adapter.window_size
        b = self.adapter.block_size
        windowed_upper_bound = seq_lens_cpu_upper_bound.clone()
        src = windowed_upper_bound[:num_reqs]
        windowed_upper_bound[:num_reqs] = src - ((src + k_future - w).clamp(min=0) // b) * b

        try:
            yield windowed_upper_bound
        finally:
            seq_lens[:num_reqs].copy_(saved_seq_lens)
            for i, table in enumerate(original_tables):
                block_tables.input_block_tables[i] = table


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
        ) as windowed_upper_bound:
            yield windowed_upper_bound
