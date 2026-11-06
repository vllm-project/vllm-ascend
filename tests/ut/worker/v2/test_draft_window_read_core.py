# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Read geometry and ownership boundaries on ordinary CPU tensors."""

from copy import copy
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from vllm_ascend.worker.v2.spec_decode.dflash2.resident_kv import apply_dflash2_reference_kv_specs
from vllm_ascend.worker.v2.spec_decode.draft_window_plan import compute_draft_window_start
from vllm_ascend.worker.v2.spec_decode.draft_window_read import materialize_draft_read


@pytest.mark.parametrize("end,window,future,expected", [(127, 128, 0, 0), (257, 128, 0, 128), (250, 128, 7, 128)])
def test_aligned_start(end, window, future, expected):
    assert (
        compute_draft_window_start(full_exact_end=end, window_size=window, block_span_tokens=128, future_offset=future)
        == expected
    )


def make_common():
    return SimpleNamespace(
        block_table_tensor=torch.tensor([[5, 6, 7, 99], [11, 12, 13, 77], [0, 0, 0, 0]], dtype=torch.int32),
        seq_lens=torch.tensor([128, 257, 0], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([144, 260, 0], dtype=torch.int32),
        num_reqs=3,
        slot_mapping=torch.arange(8, dtype=torch.int32),
        positions=torch.arange(8),
    )


def test_projection_keeps_full_target_and_write_operands():
    target = make_common()
    draft = copy(target)
    full_table, full_lengths = target.block_table_tensor.clone(), target.seq_lens.clone()
    owner = []
    materialize_draft_read(
        draft,
        [1, 3],
        128,
        exact_cpu=torch.tensor([128, 257], dtype=torch.int32),
        starts_cpu=torch.tensor([0, 128], dtype=torch.int32),
        retain=lambda *objects: owner.extend(objects),
    )
    assert torch.equal(draft.block_table_tensor, torch.tensor([[5, 0], [12, 13], [0, 0]], dtype=torch.int32))
    assert draft.seq_lens.tolist() == [128, 129, 0]
    assert draft._draft_window_exact_cpu.tolist() == [128, 129, 0]
    assert torch.equal(target.block_table_tensor, full_table)
    assert torch.equal(target.seq_lens, full_lengths)
    assert draft.slot_mapping is target.slot_mapping and draft.positions is target.positions
    assert any(item is target.block_table_tensor for item in owner)
    assert any(item is draft.block_table_tensor for item in owner)
    assert any(item is draft.seq_lens for item in owner)
    assert any(item is draft._draft_window_exact_cpu for item in owner)


@pytest.mark.parametrize(
    "columns,ends,starts,error",
    [
        ([-1, 3], [128, 257], [0, 128], "initialized compact"),
        ([1, 5], [128, 257], [0, 128], "initialized compact"),
        ([1, 2], [128, 257], [0, 128], "exact readable extent"),
        ([1, 3], [145, 257], [0, 128], "exact, upper"),
        ([1, 3], [128, 257], [0, 129], "intra-block"),
    ],
)
def test_invalid_read_does_not_publish(columns, ends, starts, error):
    common = make_common()
    table, lengths = common.block_table_tensor, common.seq_lens
    owner = []
    with pytest.raises(ValueError, match=error):
        materialize_draft_read(
            common,
            columns,
            128,
            exact_cpu=torch.tensor(ends, dtype=torch.int32),
            starts_cpu=torch.tensor(starts, dtype=torch.int32),
            retain=lambda *objects: owner.extend(objects),
        )
    assert common.block_table_tensor is table and common.seq_lens is lengths
    assert not owner


@pytest.mark.parametrize("enabled", [False, True])
def test_complete_draft_kv_retention_does_not_change_native_or_target(enabled):
    native = SlidingWindowSpec(block_size=128, num_kv_heads=4, head_size=128, dtype=torch.bfloat16, sliding_window=2048)
    target = FullAttentionSpec(block_size=128, num_kv_heads=4, head_size=128, dtype=torch.bfloat16)
    original = {"draft": native, "target": target}
    owner = SimpleNamespace(_dflash2_full_kv_reference=enabled, draft_attn_layer_names={"draft"})
    result = apply_dflash2_reference_kv_specs(original, owner)
    assert type(result["draft"]) is (FullAttentionSpec if enabled else SlidingWindowSpec)
    assert result["target"] is target and original["draft"] is native
    assert native.sliding_window == 2048
    assert result["draft"].block_size == native.block_size
    assert result["draft"].num_kv_heads == native.num_kv_heads


@pytest.mark.parametrize("combined", [False, True])
def test_dense_gather_uses_actual_slot_extent_and_preserves_page_offsets(combined):
    from vllm_ascend.worker.v2.spec_decode.draft_window_dense import gather_dense_draft_kv, prepare_dense_draft_read

    if combined:
        backing = torch.arange(6 * 2 * 128 * 2 * 4, dtype=torch.float32).reshape(6, 2, 128, 2, 4).bfloat16()
        key_cache, value_cache = backing[:, 0], backing[:, 1]
    else:
        key_cache = torch.arange(6 * 128 * 2 * 4, dtype=torch.float32).reshape(6, 128, 2, 4).bfloat16()
        value_cache = -key_cache
    owned = []
    read = prepare_dense_draft_read(
        torch.tensor([[3, 1], [4, 2]], dtype=torch.int32),
        [2, 2],
        128,
        exact_cpu=torch.tensor([129, 130], dtype=torch.int32),
        query_ends_cpu=torch.tensor([129, 130], dtype=torch.int32),
        upper_cpu=torch.tensor([129, 130], dtype=torch.int32),
        query_width=1,
        causal=True,
        retain=lambda *objects: owned.extend(objects),
        frozen_starts=torch.tensor([128, 128], dtype=torch.int32),
        full_rows_cpu=torch.tensor([False, False]),
    )
    actual = gather_dense_draft_kv(read, key_cache, value_cache)
    for cache, gathered in zip((key_cache, value_cache), actual):
        expected = torch.stack(
            (cache[1, 0].expand(32, -1, -1), torch.cat((cache[2, :2], cache[2, 1].expand(30, -1, -1))))
        ).float()
        torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
        assert any(value is gathered for value in owned)
    assert read.seq_lens.tolist() == [1, 2]
    assert read.mask[0, 0, 0].tolist() == [False] + [True] * 31
    assert read.mask[1, 0, 0].tolist() == [False, False] + [True] * 30
