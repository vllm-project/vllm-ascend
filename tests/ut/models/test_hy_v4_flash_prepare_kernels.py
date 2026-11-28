# SPDX-License-Identifier: Apache-2.0
"""CPU interpreter checks; TRITON_INTERPRET=1, no NPU or model workload."""

import os
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.hy_v4_flash import attention_ops as attention_ops
from vllm_ascend.models.hy_v4_flash import cache as cache_ops
from vllm_ascend.models.hy_v4_flash import dfly_ops as dfly_ops
from vllm_ascend.models.hy_v4_flash import dfly_speculator

pytestmark = pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1", reason="requires CPU Triton interpreter")


@pytest.mark.parametrize("candidates", [1, 7, 128, 2048, 2054])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_compact_integer_indices_matches_stable_partition(candidates, dtype):
    rng = torch.Generator().manual_seed(1024)
    values = torch.randint(-2, 97, (7, candidates), generator=rng, dtype=dtype)
    values[0].fill_(-1)
    values[1].fill_(17)
    values[2, ::2] = -1
    values[3, candidates // 2 :] = -1
    values[4, : candidates // 2] = -1
    # Exercise a non-contiguous row stride, plus repeated values and holes.
    storage = torch.full((7, candidates + 3), -33, dtype=dtype)
    storage[:, :candidates].copy_(values)
    actual = cache_ops.compact_kv_positions(storage[:, :candidates])
    expected = torch.full_like(values, -1)
    for i, row in enumerate(values):
        valid = row[row >= 0]
        expected[i, : valid.numel()] = valid
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("batch,padded", [(1, 1), (1, 3), (3, 3), (9, 16), (17, 32), (33, 64)])
@pytest.mark.parametrize("mrope_dims", [0, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [19, 1025])
def test_fused_query_matches_reference(batch, padded, mrope_dims, dtype, hidden_size):
    width = 6
    count = padded * width
    ends = torch.arange(batch + 1, dtype=torch.int32) * width
    mapping = torch.arange(batch).flip(0)
    positions = torch.arange(batch * width, dtype=torch.int64) + 100
    # Include max-model-len rejection as well as num_sampled=0.
    positions[-width:] = 255
    sampled = torch.ones(batch, dtype=torch.int32)
    sampled[::3] = 0
    rejected = torch.arange(batch, dtype=torch.int32) % width
    previous = torch.arange(batch * 2).reshape(batch, 2)
    hidden = torch.arange(batch * width * (hidden_size + 3), dtype=torch.float32)
    hidden = hidden.to(dtype).reshape(batch * width, hidden_size + 3)[:, :hidden_size]
    delta = torch.arange(batch, dtype=torch.int32) * 13
    inputs = dfly_speculator.DFlyProposeInputs(ends, positions, mapping, sampled, rejected, previous, hidden)

    def buffers():
        spec = object.__new__(dfly_speculator.FlashDFlySpeculator)
        spec.num_query_per_req = width
        spec.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=256))
        spec.device = torch.device("cpu")
        spec.model = SimpleNamespace(config=SimpleNamespace(mask_token_id=99))
        spec.model_state = SimpleNamespace(rope_state=SimpleNamespace(prefill_delta=SimpleNamespace(gpu=delta)))
        spec.offsets = torch.arange(width)
        spec.input_buffers = SimpleNamespace(
            input_ids=torch.full((count,), -77, dtype=torch.int32),
            positions=torch.full((count,), -77, dtype=torch.int64),
            is_padding=torch.zeros(count, dtype=torch.bool),
            query_start_loc=torch.full((padded + 1,), -77, dtype=torch.int32),
            seq_lens=torch.full((padded,), -77, dtype=torch.int32),
        )
        spec.target_hidden_states = torch.full((padded, hidden_size), -77.0, dtype=dtype)
        spec.active_requests = torch.ones(padded, dtype=torch.bool)
        spec.source_rows = torch.full((padded,), -77, dtype=torch.int64)
        spec.anchors = torch.full((padded,), -77, dtype=torch.int64)
        spec.mrope_positions = torch.full((mrope_dims, count), -77, dtype=torch.int64) if mrope_dims else None
        return spec

    reference, actual = buffers(), buffers()
    reference._prepare_query_reference(inputs, batch, padded)
    b = actual.input_buffers
    dfly_ops._prepare_query_kernel[(padded,)](
        b.input_ids,
        b.positions,
        b.is_padding,
        b.query_start_loc,
        b.seq_lens,
        actual.target_hidden_states,
        actual.active_requests,
        actual.source_rows,
        actual.anchors,
        actual.mrope_positions if mrope_dims else b.positions,
        ends,
        positions,
        mapping,
        sampled,
        rejected,
        previous,
        hidden,
        delta,
        batch,
        padded,
        width,
        8,
        hidden_size,
        hidden.stride(0),
        hidden_size,
        previous.stride(0),
        mrope_dims,
        count if mrope_dims else 0,
        99,
        256,
        dfly_ops.HIDDEN_BLOCK,
    )
    for name in vars(b):
        torch.testing.assert_close(getattr(b, name), getattr(reference.input_buffers, name), rtol=0, atol=0)
    for name in ("target_hidden_states", "active_requests", "source_rows", "anchors"):
        torch.testing.assert_close(getattr(actual, name), getattr(reference, name), rtol=0, atol=0)
    if mrope_dims:
        torch.testing.assert_close(actual.mrope_positions, reference.mrope_positions, rtol=0, atol=0)


@pytest.mark.parametrize("batch,padded", [(1, 3), (3, 3), (9, 16), (33, 64)])
@pytest.mark.parametrize("window,topk", [(0, 4), (7, 19), (128, 1024)])
def test_fused_c8_metadata_overwrites_padding_and_matches_reference(batch, padded, window, topk):
    width, columns = 6, 17
    rng = torch.Generator().manual_seed(1024)
    snapshot = torch.randint(-2, 260, (batch * 3, topk * 2), generator=rng)[:, ::2]
    current = torch.arange(batch * columns * 2).reshape(batch, columns * 2)[:, ::2]
    active = torch.arange(padded) < batch
    active[::3] = False
    sources = torch.arange(padded).clamp_max(batch - 1) * 3
    anchors = torch.arange(padded) + 5
    shape = (padded, width, topk + window + width)
    actual = torch.full(shape, -77, dtype=torch.int64)
    table = torch.full((padded, columns), -77, dtype=torch.int32)
    dfly_ops._prepare_c8_metadata_kernel[(padded, width)](
        actual,
        table,
        snapshot,
        current,
        active,
        sources,
        anchors,
        batch,
        width,
        topk,
        window,
        snapshot.stride(0),
        snapshot.stride(1),
        current.stride(0),
        current.stride(1),
        table.stride(0),
        columns,
        shape[-1],
        dfly_ops.PREPARE_BLOCK,
    )
    expected = dfly_speculator.build_reuse_indices(snapshot[sources], anchors, width, window, active)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(table[:batch], current.to(torch.int32), rtol=0, atol=0)
    assert table[batch:].eq(0).all()


@pytest.mark.parametrize("prepacked", [False, True])
def test_native_context_respects_packed_reader_mask(prepacked):
    selected = torch.tensor([[1, -1, 2], [-1, -1, -1]])
    torch.manual_seed(1024)
    kv = torch.randn(2, 3, 576, dtype=torch.float32)
    query = torch.randn(2, 2, 576, dtype=torch.bfloat16)
    sink = torch.zeros(2)
    nope, rope = kv[..., :512].unsqueeze(2), kv[..., 512:].unsqueeze(2)
    if prepacked:
        nope, rope = cache_ops.compact_kv_rows(selected, nope, rope)
        mask = torch.tensor([[False, False, True], [True, True, True]])[:, None, None]
        rows = cache_ops.CompactKV(nope, rope, mask)
    else:
        rows = (nope, rope, selected.lt(0)[:, None, None])
    actual = attention_ops.native_mla_context(
        query,
        selected,
        rows,
        sink,
        0.04,
    )
    expected = attention_ops.native_mla_context(
        query,
        selected,
        kv,
        sink,
        0.04,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
