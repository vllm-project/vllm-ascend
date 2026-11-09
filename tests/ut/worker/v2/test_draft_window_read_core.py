# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Read geometry and ownership boundaries on ordinary CPU tensors."""

from contextlib import nullcontext
from copy import copy
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from vllm_ascend.worker.v2.spec_decode.dflash2.resident_kv import apply_dflash2_resident_kv_specs
from vllm_ascend.worker.v2.spec_decode.draft_window.read import compute_draft_window_start, materialize_draft_read


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
    owner = SimpleNamespace(_dflash2_full_kv_retention=enabled, draft_attn_layer_names={"draft"})
    result = apply_dflash2_resident_kv_specs(original, owner)
    assert type(result["draft"]) is (FullAttentionSpec if enabled else SlidingWindowSpec)
    assert result["target"] is target and original["draft"] is native
    assert native.sliding_window == 2048
    assert result["draft"].block_size == native.block_size
    assert result["draft"].num_kv_heads == native.num_kv_heads


@pytest.mark.parametrize("combined", [False, True])
def test_dense_gather_uses_actual_slot_extent_and_preserves_page_offsets(combined):
    from vllm_ascend.worker.v2.spec_decode.draft_window.dense import gather_dense_draft_kv, prepare_dense_draft_read

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


@pytest.mark.parametrize("heads,dim,expected_tokens", [(4, 192, 8), (16, 128, 8), (16, 192, 4), (32, 256, 2)])
def test_dense_gather_launch_bounds_wide_kv_scratch(monkeypatch, heads, dim, expected_tokens):
    import torch_npu

    from vllm_ascend.ops.triton.spec_decode import draft_window as kernel
    from vllm_ascend.worker.v2.spec_decode.draft_window.dense import gather_dense_draft_kv

    device = SimpleNamespace(type="npu")
    token_size = heads * dim
    cache = SimpleNamespace(
        ndim=4,
        shape=(4, 128, heads, dim),
        dtype=torch.bfloat16,
        device=device,
        stride=lambda axis=None: (128 * token_size, token_size, dim, 1) if axis is None else 128 * token_size,
    )
    slots = SimpleNamespace(device=device, shape=(1, 33), numel=lambda: 33, is_contiguous=lambda: True)
    read = SimpleNamespace(slots=slots, _attention_workspaces={}, retain=lambda *objects: None)
    calls = []

    class Launch:
        def __getitem__(self, grid):
            return lambda *args: calls.append((grid, args))

    monkeypatch.setattr(torch_npu, "get_npu_format", lambda cache: 0)
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: object())
    monkeypatch.setattr(torch, "empty_like", lambda *args, **kwargs: object())
    monkeypatch.setattr(kernel, "_gather_draft_kv_fp32", Launch())
    gather_dense_draft_kv(read, cache, cache)
    grid, operands = calls.pop()
    assert grid == ((33 + expected_tokens - 1) // expected_tokens,)
    assert operands[-3] == -33 % expected_tokens
    assert operands[-2] == expected_tokens
    assert operands[-1] == 1 << (token_size - 1).bit_length()


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("fail_after_prepare", [False, True])
def test_ordinary_callback_preserves_mixed_full_rows_and_failed_reader_ownership(
    monkeypatch, prepared, fail_after_prepare
):
    import numpy as np
    from vllm.config.compilation import CUDAGraphMode

    from vllm_ascend.worker.v2.spec_decode.draft_window.runtime import DraftWindowRuntime

    batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping_np=np.array([1, 0]),
        prefill_len_np=np.array([249, 300]),
        num_computed_prefill_tokens_np=np.array([249, 120]),
        num_scheduled_tokens=np.array([8, 8]),
        num_draft_tokens_per_req=None,
    )
    speculator = SimpleNamespace(
        method="dflash",
        num_speculative_steps=7,
        num_query_per_req=8,
        input_batch=batch,
        block_tables=SimpleNamespace(kernel_block_sizes=[128], num_blocks=SimpleNamespace(np=np.array([[1, 3]]))),
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(
                enforce_eager=True, draft_model_config=SimpleNamespace(enforce_eager=False)
            ),
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL),
        ),
    )
    # Backend admission has separate coverage; exercise the real proposal and
    # metadata boundaries here with a direct-prefix consumer.
    monkeypatch.setattr(DraftWindowRuntime, "_validate_route", lambda *a: (0,))
    runtime = DraftWindowRuntime(speculator, 128)
    target = make_common()
    target.seq_lens[:2] = torch.tensor([257, 128], dtype=torch.int32)
    target.seq_lens_cpu_upper_bound[:2] = torch.tensor([260, 144], dtype=torch.int32)
    original_table, original_lengths = target.block_table_tensor.clone(), target.seq_lens.clone()
    common = copy(target)
    context = pytest.raises(RuntimeError, match="reader failed") if fail_after_prepare else nullcontext()
    with context, runtime.proposal_context(batch):
        callback = runtime.metadata_callback(8, 2, prepared_exact_cpu=target.seq_lens[:2].clone() if prepared else None)
        callback(0, common)
        assert runtime.reads_active
        assert common.seq_lens.tolist() == [129, 128, 0]
        assert common.block_table_tensor.tolist() == [[6, 7], [11, 0], [0, 0]]
        assert common.positions is target.positions and common.slot_mapping is target.slot_mapping
        if fail_after_prepare:
            raise RuntimeError("reader failed")
    assert not runtime.reads_active
    assert torch.equal(target.block_table_tensor, original_table) and torch.equal(target.seq_lens, original_lengths)
    assert any(value is common for value in runtime._pending)
    assert any(value is common.block_table_tensor for value in runtime._pending)
    with pytest.raises(RuntimeError, match="stale proposal"):
        callback(0, copy(target))
    if fail_after_prepare:
        with pytest.raises(RuntimeError, match="quarantined"), runtime.proposal_context(batch):
            pass
    else:
        pending = tuple(runtime._pending)
        with runtime.proposal_context(batch, context_kv_only=True):
            assert runtime._build_count == 0
        assert tuple(runtime._pending) == pending
