# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for MRV2 draft sliding-window builder wiring."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
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
        block_sizes=(block_size, block_size),
        max_num_reqs=max_num_reqs,
        future_offset=0,
        device=torch.device("cpu"),
    )

    full_table_a = torch.arange(10, dtype=torch.int32).view(1, 10).expand(max_num_reqs, 10).contiguous()
    full_table_b = (torch.arange(10, dtype=torch.int32) + 100).view(1, 10).expand(max_num_reqs, 10).contiguous()
    original_a = full_table_a.clone()
    original_b = full_table_b.clone()
    slot_mappings = torch.arange(8, dtype=torch.int32)

    original_seq_lens = torch.tensor([300, 0], dtype=torch.int32)
    input_buffers = SimpleNamespace(seq_lens=original_seq_lens)
    block_tables = SimpleNamespace(
        input_block_tables=[full_table_a, full_table_b],
        slot_mappings=slot_mappings,
    )
    # Independent windowing of 320 would yield 128; the shared crop must keep
    # this an upper bound of the windowed actual length (172).
    upper_bound = torch.tensor([320, 0], dtype=torch.int32)
    seq_lens_cpu = torch.tensor([300, 0], dtype=torch.int32)
    saved_slots = slot_mappings.clone()

    with helper.apply_builder_inputs(
        input_buffers, block_tables, upper_bound, num_reqs, extra_seq_lens=(seq_lens_cpu,)
    ) as windowed_ub:
        # 300 with W=128,b=64 -> start=128, windowed=172 (matches MRV1 adapter tests).
        assert input_buffers.seq_lens is not original_seq_lens
        assert input_buffers.seq_lens[0].item() == 172
        metadata_seq_lens = input_buffers.seq_lens
        assert windowed_ub[0].item() == 192  # 320 - 128, still >= 172
        assert seq_lens_cpu[0].item() == 172
        assert block_tables.input_block_tables[0].shape[1] == helper.adapters[0].max_window_blocks
        assert block_tables.input_block_tables[1].shape[1] == helper.adapters[1].max_window_blocks
        # Cropped columns are the recent blocks (start_block=128/64=2).
        assert block_tables.input_block_tables[0][0, 0].item() == 2
        assert block_tables.input_block_tables[1][0, 0].item() == 102
        # Slot mapping invariant: never windowed.
        torch.testing.assert_close(block_tables.slot_mappings, saved_slots)

    # Buffer identity is restored; metadata that aliased the windowed clone
    # still sees 172.
    assert input_buffers.seq_lens is original_seq_lens
    torch.testing.assert_close(input_buffers.seq_lens, torch.tensor([300, 0], dtype=torch.int32))
    torch.testing.assert_close(metadata_seq_lens, torch.tensor([172, 0], dtype=torch.int32))
    torch.testing.assert_close(block_tables.input_block_tables[0], original_a)
    torch.testing.assert_close(block_tables.input_block_tables[1], original_b)
    torch.testing.assert_close(block_tables.slot_mappings, saved_slots)


def test_clamp_metadata_keeps_upper_bound_after_draft_step():
    helper = MRV2DraftSlidingWindow(
        window_size=128,
        block_sizes=(64,),
        max_num_reqs=1,
        future_offset=0,
        device=torch.device("cpu"),
    )
    helper._last_windowed_seq_lens = torch.tensor([172], dtype=torch.int32)
    # Upstream added step=5 to the crop-aligned bound 192.
    metadata = SimpleNamespace(
        seq_lens_cpu_upper_bound=torch.tensor([197], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([197], dtype=torch.int32),
        seq_lens_list=[197],
    )

    helper.clamp_metadata_after_step({"layer": metadata}, num_reqs=1)

    # Cropped table holds max_window_blocks * block_size = 192 tokens.
    assert metadata.seq_lens_cpu_upper_bound[0].item() == 192
    assert metadata.seq_lens_cpu[0].item() == 192
    assert metadata.seq_lens_list == [192]
    assert metadata.seq_lens_cpu_upper_bound[0].item() >= 172


def test_mixed_kernel_block_sizes_crop_each_group():
    """A 128-token offset is two 64-token blocks and one 128-token block.

    Sequence lengths are shared, so the token crop is the common multiple of
    the kernel block sizes. When the per-group block boundaries disagree
    (W=100, raw offset 200), the finer group keeps the shared 128-token start
    instead of its tighter 192-token start.
    """
    max_num_reqs = 2
    num_reqs = 1
    helper = MRV2DraftSlidingWindow(
        window_size=128,
        block_sizes=(64, 128),
        max_num_reqs=max_num_reqs,
        future_offset=0,
        device=torch.device("cpu"),
    )
    full_table_a = torch.arange(10, dtype=torch.int32).view(1, 10).expand(max_num_reqs, 10).contiguous()
    full_table_b = (torch.arange(10, dtype=torch.int32) + 100).view(1, 10).expand(max_num_reqs, 10).contiguous()
    original_a = full_table_a.clone()
    original_b = full_table_b.clone()
    slot_mappings = torch.arange(8, dtype=torch.int32)
    original_seq_lens = torch.tensor([300, 0], dtype=torch.int32)
    input_buffers = SimpleNamespace(seq_lens=original_seq_lens)
    block_tables = SimpleNamespace(
        input_block_tables=[full_table_a, full_table_b],
        slot_mappings=slot_mappings,
        kernel_block_sizes=[64, 128],
    )
    upper_bound = torch.tensor([320, 0], dtype=torch.int32)
    saved_slots = slot_mappings.clone()

    with helper.apply_builder_inputs(input_buffers, block_tables, upper_bound, num_reqs) as windowed_ub:
        # 300 with W=128 aligns to 128 tokens on both sizes -> windowed 172.
        assert input_buffers.seq_lens[0].item() == 172
        assert windowed_ub[0].item() == 192  # 320 - 128
        assert block_tables.input_block_tables[0][0, 0].item() == 2  # 128 / 64
        assert block_tables.input_block_tables[1][0, 0].item() == 101  # 128 / 128
        shared_crop = helper.adapters[0].start_tokens_in_window_rounding
        assert helper.adapters[1].start_tokens_in_window_rounding is shared_crop
        assert block_tables.input_block_tables[0].shape[1] == helper.adapters[0].max_window_blocks
        assert block_tables.input_block_tables[1].shape[1] == helper.adapters[1].max_window_blocks
        torch.testing.assert_close(block_tables.slot_mappings, saved_slots)

    torch.testing.assert_close(block_tables.input_block_tables[0], original_a)
    torch.testing.assert_close(block_tables.input_block_tables[1], original_b)
    torch.testing.assert_close(input_buffers.seq_lens, original_seq_lens)

    # raw = 300 - 100 = 200. Size 64 would start at 192; size 128 starts at 128.
    # The shared crop stays at 128 so both tables describe the same tokens.
    helper_coarse = MRV2DraftSlidingWindow(
        window_size=100,
        block_sizes=(64, 128),
        max_num_reqs=max_num_reqs,
        future_offset=0,
        device=torch.device("cpu"),
    )
    input_buffers.seq_lens = original_seq_lens.clone()
    upper_bound = torch.tensor([320, 0], dtype=torch.int32)
    with helper_coarse.apply_builder_inputs(input_buffers, block_tables, upper_bound, num_reqs) as windowed_ub:
        assert input_buffers.seq_lens[0].item() == 172  # 300 - 128, the shared token start
        assert windowed_ub[0].item() == 192
        assert block_tables.input_block_tables[0][0, 0].item() == 2
        assert block_tables.input_block_tables[1][0, 0].item() == 101
        shared_crop = helper_coarse.adapters[0].start_tokens_in_window_rounding
        assert helper_coarse.adapters[1].start_tokens_in_window_rounding is shared_crop

    torch.testing.assert_close(block_tables.input_block_tables[0], original_a)
    torch.testing.assert_close(block_tables.input_block_tables[1], original_b)


def test_cropped_block_table_keeps_padded_rows():
    helper = MRV2DraftSlidingWindow(
        window_size=128,
        block_sizes=(64,),
        max_num_reqs=2,
        future_offset=0,
        device=torch.device("cpu"),
    )
    helper._clones[0].fill_(7)
    table = torch.arange(8, dtype=torch.int32).view(1, 8).expand(2, 8).contiguous()
    original = table.clone()
    block_tables = SimpleNamespace(input_block_tables=[table])
    input_buffers = SimpleNamespace(seq_lens=torch.tensor([300, 0], dtype=torch.int32))
    upper_bound = torch.tensor([320, 0], dtype=torch.int32)

    with helper.apply_builder_inputs(input_buffers, block_tables, upper_bound, num_reqs=1):
        published = block_tables.input_block_tables[0]
        assert published.shape[0] == 2
        assert published[0, 0].item() == 2
        assert torch.count_nonzero(published[1]) == 0

    torch.testing.assert_close(block_tables.input_block_tables[0], original)


def test_clamp_cpu_lengths_caps_a_long_request():
    helper = MRV2DraftSlidingWindow(
        window_size=128,
        block_sizes=(64,),
        max_num_reqs=2,
        future_offset=0,
        device=torch.device("cpu"),
    )
    helper._last_windowed_seq_lens = torch.tensor([172], dtype=torch.int32)
    lengths = torch.tensor([302, 0], dtype=torch.int32)

    capped = helper.clamp_cpu_lengths(lengths, num_reqs=1)

    assert capped.shape == lengths.shape
    assert capped[0].item() == helper.max_window_tokens
    assert capped[1].item() == 0
    assert lengths[0].item() == 302


def test_maybe_create_keeps_each_kernel_block_size():
    block_tables = SimpleNamespace(
        input_block_tables=[
            torch.zeros(1, 4, dtype=torch.int32),
            torch.zeros(1, 4, dtype=torch.int32),
        ],
        kernel_block_sizes=[64, 128],
    )
    with patch(
        "vllm_ascend.worker.v2.spec_decode.sliding_window.get_ascend_config",
        return_value=SimpleNamespace(draft_window_size=128),
    ):
        helper = MRV2DraftSlidingWindow.maybe_create(
            method="dspark",
            vllm_config=SimpleNamespace(),
            block_tables=block_tables,
            max_num_reqs=1,
            device=torch.device("cpu"),
            future_offset=0,
        )
    assert helper is not None
    assert helper.block_sizes == (64, 128)
    assert helper.align == 128


def test_maybe_create_accepts_a_kernel_block_size_tensor():
    block_tables = SimpleNamespace(
        input_block_tables=[
            torch.zeros(1, 4, dtype=torch.int32),
            torch.zeros(1, 4, dtype=torch.int32),
        ],
        kernel_block_sizes=torch.tensor([64, 128], dtype=torch.int32),
    )
    with patch(
        "vllm_ascend.worker.v2.spec_decode.sliding_window.get_ascend_config",
        return_value=SimpleNamespace(draft_window_size=128),
    ):
        helper = MRV2DraftSlidingWindow.maybe_create(
            method="dspark",
            vllm_config=SimpleNamespace(),
            block_tables=block_tables,
            max_num_reqs=1,
            device=torch.device("cpu"),
            future_offset=0,
        )
    assert helper is not None
    assert helper.block_sizes == (64, 128)
    assert helper.align == 128


def test_maybe_create_falls_back_to_physical_block_sizes():
    block_tables = SimpleNamespace(
        input_block_tables=[torch.zeros(1, 4, dtype=torch.int32)],
        kernel_block_sizes=[],
        block_sizes=[128],
    )
    with patch(
        "vllm_ascend.worker.v2.spec_decode.sliding_window.get_ascend_config",
        return_value=SimpleNamespace(draft_window_size=128),
    ):
        helper = MRV2DraftSlidingWindow.maybe_create(
            method="dspark",
            vllm_config=SimpleNamespace(),
            block_tables=block_tables,
            max_num_reqs=1,
            device=torch.device("cpu"),
            future_offset=0,
        )
    assert helper is not None
    assert helper.block_sizes == (128,)


def test_maybe_create_rejects_a_block_size_list_that_does_not_match_groups():
    block_tables = SimpleNamespace(
        input_block_tables=[
            torch.zeros(1, 4, dtype=torch.int32),
            torch.zeros(1, 4, dtype=torch.int32),
        ],
        kernel_block_sizes=[64],
    )
    with patch(
        "vllm_ascend.worker.v2.spec_decode.sliding_window.get_ascend_config",
        return_value=SimpleNamespace(draft_window_size=128),
    ):
        with pytest.raises(RuntimeError, match="do not match KV groups"):
            MRV2DraftSlidingWindow.maybe_create(
                method="dspark",
                vllm_config=SimpleNamespace(),
                block_tables=block_tables,
                max_num_reqs=1,
                device=torch.device("cpu"),
                future_offset=0,
            )


def test_maybe_create_rejects_a_non_positive_block_size():
    block_tables = SimpleNamespace(
        input_block_tables=[torch.zeros(1, 4, dtype=torch.int32)],
        kernel_block_sizes=[0],
    )
    with patch(
        "vllm_ascend.worker.v2.spec_decode.sliding_window.get_ascend_config",
        return_value=SimpleNamespace(draft_window_size=128),
    ):
        with pytest.raises(RuntimeError, match="must be positive"):
            MRV2DraftSlidingWindow.maybe_create(
                method="dspark",
                vllm_config=SimpleNamespace(),
                block_tables=block_tables,
                max_num_reqs=1,
                device=torch.device("cpu"),
                future_offset=0,
            )


def test_constructor_requires_a_kv_group():
    with pytest.raises(ValueError, match="at least one KV cache group"):
        MRV2DraftSlidingWindow(
            window_size=128,
            block_sizes=(),
            max_num_reqs=1,
            future_offset=0,
            device=torch.device("cpu"),
        )


def test_apply_builder_inputs_rejects_a_changed_group_count():
    helper = MRV2DraftSlidingWindow(
        window_size=128,
        block_sizes=(64,),
        max_num_reqs=1,
        future_offset=0,
        device=torch.device("cpu"),
    )
    table = torch.arange(4, dtype=torch.int32).view(1, 4)
    original = table.clone()
    block_tables = SimpleNamespace(input_block_tables=[table, table.clone()])
    input_buffers = SimpleNamespace(seq_lens=torch.tensor([300], dtype=torch.int32))
    upper_bound = torch.tensor([320], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="group count changed"):
        with helper.apply_builder_inputs(input_buffers, block_tables, upper_bound, 1):
            pass

    torch.testing.assert_close(block_tables.input_block_tables[0], original)
    assert input_buffers.seq_lens[0].item() == 300


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
