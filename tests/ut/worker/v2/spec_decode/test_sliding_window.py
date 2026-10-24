# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for MRV2 draft sliding-window builder wiring."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.worker.v2.spec_decode.sliding_window import (
    MRV2DraftSlidingWindow,
    resolve_draft_window_size,
)


def test_resolve_draft_window_size_ignores_mtp():
    with patch(
        "vllm_ascend.worker.v2.spec_decode.sliding_window.get_ascend_config",
        return_value=SimpleNamespace(draft_window_size=2048),
    ):
        assert resolve_draft_window_size("mtp", SimpleNamespace()) is None


def test_resolve_draft_window_size_returns_configured_value():
    with patch(
        "vllm_ascend.worker.v2.spec_decode.sliding_window.get_ascend_config",
        return_value=SimpleNamespace(draft_window_size=2048),
    ):
        assert resolve_draft_window_size("dspark", SimpleNamespace()) == 2048


def test_apply_builder_inputs_windows_seq_lens_and_block_tables_on_cpu():
    window_size = 128
    block_size = 64
    max_num_reqs = 2
    num_reqs = 1
    helper = MRV2DraftSlidingWindow(
        window_size=window_size,
        block_size=block_size,
        max_num_reqs=max_num_reqs,
        future_offset=0,
        device=torch.device("cpu"),
        num_groups=2,
    )

    full_table_a = torch.arange(10, dtype=torch.int32).view(1, 10).expand(max_num_reqs, 10).contiguous()
    full_table_b = (torch.arange(10, dtype=torch.int32) + 100).view(1, 10).expand(max_num_reqs, 10).contiguous()
    original_a = full_table_a.clone()
    original_b = full_table_b.clone()
    slot_mappings = torch.arange(8, dtype=torch.int32)

    input_buffers = SimpleNamespace(seq_lens=torch.tensor([300, 0], dtype=torch.int32))
    block_tables = SimpleNamespace(
        input_block_tables=[full_table_a, full_table_b],
        slot_mappings=slot_mappings,
    )
    upper_bound = torch.tensor([300, 0], dtype=torch.int32)
    saved_slots = slot_mappings.clone()

    with helper.apply_builder_inputs(input_buffers, block_tables, upper_bound, num_reqs) as windowed_ub:
        # 300 with W=128,b=64 -> start=128, windowed=172 (matches MRV1 adapter tests).
        assert input_buffers.seq_lens[0].item() == 172
        assert windowed_ub[0].item() == 172
        assert block_tables.input_block_tables[0].shape[1] == helper.adapter.max_window_blocks
        assert block_tables.input_block_tables[1].shape[1] == helper.adapter.max_window_blocks
        # Cropped columns are the recent blocks (start_block=128/64=2).
        assert block_tables.input_block_tables[0][0, 0].item() == 2
        assert block_tables.input_block_tables[1][0, 0].item() == 102
        # Slot mapping invariant: never windowed.
        torch.testing.assert_close(block_tables.slot_mappings, saved_slots)

    # Restored after the build scope.
    torch.testing.assert_close(input_buffers.seq_lens, torch.tensor([300, 0], dtype=torch.int32))
    torch.testing.assert_close(block_tables.input_block_tables[0], original_a)
    torch.testing.assert_close(block_tables.input_block_tables[1], original_b)
    torch.testing.assert_close(block_tables.slot_mappings, saved_slots)


def test_maybe_create_returns_none_when_disabled():
    block_tables = SimpleNamespace(
        input_block_tables=[torch.zeros(1, 4, dtype=torch.int32)],
        kernel_block_sizes=[64],
    )
    with patch(
        "vllm_ascend.worker.v2.spec_decode.sliding_window.get_ascend_config",
        return_value=SimpleNamespace(draft_window_size=None),
    ):
        assert (
            MRV2DraftSlidingWindow.maybe_create(
                method="dspark",
                vllm_config=SimpleNamespace(),
                block_tables=block_tables,
                max_num_reqs=1,
                device=torch.device("cpu"),
                future_offset=0,
            )
            is None
        )


def test_maybe_create_returns_none_for_mtp_even_when_configured():
    block_tables = SimpleNamespace(
        input_block_tables=[torch.zeros(1, 4, dtype=torch.int32)],
        kernel_block_sizes=[64],
    )
    with patch(
        "vllm_ascend.worker.v2.spec_decode.sliding_window.get_ascend_config",
        return_value=SimpleNamespace(draft_window_size=2048),
    ):
        assert (
            MRV2DraftSlidingWindow.maybe_create(
                method="mtp",
                vllm_config=SimpleNamespace(),
                block_tables=block_tables,
                max_num_reqs=1,
                device=torch.device("cpu"),
                future_offset=0,
            )
            is None
        )


def test_mixin_noop_when_helper_missing():
    from vllm_ascend.worker.v2.spec_decode.sliding_window import DraftSlidingWindowMixin

    class Dummy(DraftSlidingWindowMixin):
        pass

    dummy = Dummy()
    dummy._draft_sliding_window = None
    upper = torch.tensor([10], dtype=torch.int32)
    with dummy._maybe_draft_sliding_window(upper, 1) as out:
        assert out is upper
