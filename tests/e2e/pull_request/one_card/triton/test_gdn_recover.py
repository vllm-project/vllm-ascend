# SPDX-License-Identifier: Apache-2.0
"""Independent recurrence oracle, storage guards and dynamic NPU graph replay."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.mamba.gdn_recover import (
    gdn_commit_conv_history,
    gdn_recover_commit,
    gdn_recover_verify,
    gdn_save_conv_history,
)


def _padded(shape, *, dtype=torch.float32, offset=7, padding=31, device="npu"):
    inner = 1
    for size in shape[1:]:
        inner *= size
    storage = torch.full((offset + shape[0] * (inner + padding) + 11,), 0.125, dtype=dtype, device=device)
    strides = [inner + padding]
    for i in range(1, len(shape)):
        stride = 1
        for size in shape[i + 1 :]:
            stride *= size
        strides.append(stride)
    return storage, storage.as_strided(shape, strides, offset)


def _inputs(lengths, qk_heads, heads, kd, vd, dtype):
    torch.manual_seed(20261014)
    tokens = sum(lengths)
    q = torch.nn.functional.normalize(torch.randn(tokens, qk_heads, kd), dim=-1).to(dtype)
    k = torch.nn.functional.normalize(torch.randn(tokens, qk_heads, kd), dim=-1).to(dtype)
    v = (torch.randn(tokens, heads, vd) * 0.2).to(dtype)
    g = -(torch.rand(tokens, heads) * 0.3 + 0.05)
    beta = torch.rand(tokens, heads) * 0.8
    return q, k, v, g, beta


def _oracle(initial, inputs, ids, lengths, scale):
    """CPU direct recurrence; never reads implementation replay records."""
    q, k, v, g, beta = [x.float() for x in inputs]
    heads, vd, _ = initial.shape[1:]
    headmap = torch.arange(heads) // (heads // k.shape[1])
    outputs = torch.zeros_like(v)
    snapshots = []
    start = 0
    for slot, length in zip(ids, lengths):
        state = initial[slot].clone() if slot > 0 else torch.zeros_like(initial[0])
        states = [state.clone()]
        for token in range(start, start + length):
            if slot > 0:
                key = k[token, headmap]
                state = state * g[token].exp()[:, None, None]
                correction = (v[token] - (state * key[:, None, :]).sum(-1)) * beta[token, :, None]
                state = state + correction[:, :, None] * key[:, None, :]
                outputs[token] = (state * (q[token, headmap] * scale)[:, None, :]).sum(-1)
            states.append(state.clone())
        snapshots.append(states)
        start += length
    return outputs, snapshots


def _write_mask(storage, view, selected):
    mask = torch.zeros(storage.numel(), dtype=torch.bool)
    inner = view[0].numel()
    for slot in selected:
        begin = view.storage_offset() + slot * view.stride(0)
        mask[begin : begin + inner] = True
    return mask


@pytest.mark.parametrize(
    "geometry",
    [(1, 3, 33, 17, 2), (2, 6, 64, 96, 4), (2, 4, 128, 128, 8), (16, 48, 128, 128, 4), (1, 2, 256, 256, 16)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_verify_commit_all_prefixes_and_guards(geometry, dtype, index_dtype):
    nk, heads, kd, vd, window = geometry
    lengths = [window, 1, max(1, window - 1), 0, 1, 1]
    slots = [5, 2, 7, 3, 0, -1]
    state_raw, state = _padded((9, heads, vd, kd))
    record_raw, records = _padded((9, heads, window, vd + kd + 1), offset=13, padding=53)
    torch.manual_seed(81)
    initial = torch.randn(state.shape) * 0.03
    state.copy_(initial)
    state_before, record_before = state_raw.cpu(), record_raw.cpu()
    inputs = _inputs(lengths, nk, heads, kd, vd, dtype)
    # Include token-axis padding without changing the supported inner layout.
    device_inputs = []
    for tensor in inputs:
        _, view = _padded(tensor.shape, dtype=tensor.dtype, padding=9)
        view.copy_(tensor)
        device_inputs.append(view)
    ids = torch.tensor(slots, dtype=index_dtype, device="npu")
    offsets = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=index_dtype, device="npu")
    output_ref, snapshots = _oracle(initial, inputs, slots, lengths, kd**-0.5)
    output = gdn_recover_verify(*device_inputs, state, records, ids, offsets)
    torch.npu.synchronize()
    torch.testing.assert_close(
        output.cpu(), output_ref.to(dtype), rtol=0.002, atol=0.002 if dtype == torch.bfloat16 else 2e-5
    )
    assert torch.equal(state_raw.cpu(), state_before), "verify modified checkpoint or padding"
    record_after = record_raw.cpu()
    changed = torch.zeros(record_raw.numel(), dtype=torch.bool)
    for slot, length in zip(slots, lengths):
        if slot > 0:
            for h in range(heads):
                begin = records.storage_offset() + slot * records.stride(0) + h * window * (vd + kd + 1)
                changed[begin : begin + length * (vd + kd + 1)] = True
    assert torch.equal(record_after[~changed], record_before[~changed])
    # Each trial starts from the original checkpoint and commits only once.
    for count in range(-1, window + 2):
        state.copy_(initial)
        prefixes = torch.full((len(slots),), count, dtype=index_dtype, device="npu")
        gdn_recover_commit(state, records, ids, offsets, prefixes)
        expected = initial.clone()
        selected = []
        for row, (slot, length) in enumerate(zip(slots, lengths)):
            accepted = min(max(count, 0), length)
            if slot > 0 and accepted:
                expected[slot] = snapshots[row][accepted]
                selected.append(slot)
        torch.testing.assert_close(state.cpu(), expected, rtol=5e-5, atol=2e-6)
        mask = _write_mask(state_raw, state, selected)
        assert torch.equal(state_raw.cpu()[~mask], state_before[~mask])
        assert torch.equal(record_raw.cpu(), record_after), "commit modified replay records"


@pytest.mark.parametrize("history,window,channels", [(1, 2, 19), (3, 4, 257), (5, 8, 10240)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_conv_save_and_every_prefix(history, window, channels, dtype):
    raw, conv = _padded((6, history + window - 1, channels), dtype=dtype)
    saved_raw, saved = _padded((6, history, channels), dtype=dtype, offset=17)
    torch.manual_seed(111)
    before = torch.randn(conv.shape).to(dtype)
    conv.copy_(before)
    saved_before = saved_raw.cpu()
    ids = torch.tensor([4, 2, 1, 0, -1], dtype=torch.int64, device="npu")
    lengths = [window, 1, 0, 1, 1]
    offsets = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32, device="npu")
    gdn_save_conv_history(conv, saved, ids, offsets)
    torch.testing.assert_close(saved[4].cpu(), before[4, :history], rtol=0, atol=0)
    torch.testing.assert_close(saved[2].cpu(), before[2, :history], rtol=0, atol=0)
    selected_mask = _write_mask(saved_raw, saved, [4, 2])
    assert torch.equal(saved_raw.cpu()[~selected_mask], saved_before[~selected_mask])
    saved_after = saved_raw.cpu()
    updated = torch.randn(conv.shape).to(dtype)
    for count in range(-1, window + 2):
        conv.copy_(updated)
        raw_before = raw.cpu()
        prefixes = torch.full((5,), count, dtype=torch.int32, device="npu")
        gdn_commit_conv_history(conv, saved, ids, offsets, prefixes)
        expected = updated.clone()
        for slot, length in zip(ids.cpu().tolist(), lengths):
            if slot > 0 and length:
                prefix = min(max(count, 0), length)
                expected[slot, :history] = (
                    updated[slot, prefix - 1 : prefix - 1 + history] if prefix else before[slot, :history]
                )
        torch.testing.assert_close(conv.cpu(), expected, rtol=0, atol=0)
        mask = _write_mask(raw, conv, [4, 2])
        assert torch.equal(raw.cpu()[~mask], raw_before[~mask])
        assert torch.equal(saved_raw.cpu(), saved_after)


def test_graph_replay_reorders_rows_and_reuses_slots():
    heads, kd, vd, window = 6, 33, 17, 4
    _, state = _padded((8, heads, vd, kd))
    _, records = _padded((8, heads, window, vd + kd + 1))
    inputs = _inputs([4, 2], 2, heads, kd, vd, torch.bfloat16)
    device_inputs = [x.npu() for x in inputs]
    ids = torch.tensor([5, 2], dtype=torch.int32, device="npu")
    offsets = torch.tensor([0, 4, 6], dtype=torch.int32, device="npu")
    prefixes = torch.tensor([1, 2], dtype=torch.int32, device="npu")
    out = torch.empty_like(device_inputs[2])
    torch.manual_seed(301)
    initial = torch.randn(state.shape) * 0.03
    state.copy_(initial)
    # Compile before capture; the graph owns no Python pending-commit callbacks.
    for _ in range(3):
        gdn_recover_verify(*device_inputs, state, records, ids, offsets, out=out)
        gdn_recover_commit(state, records, ids, offsets, prefixes)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        gdn_recover_verify(*device_inputs, state, records, ids, offsets, out=out)
        gdn_recover_commit(state, records, ids, offsets, prefixes)
    # Keep addresses fixed but change IDs, offsets, lengths and counts each replay.
    for slots, lengths, counts in [([5, 2], [4, 2], [0, 2]), ([2, 5], [2, 4], [2, 1]), ([7, 1], [3, 3], [3, 3])]:
        state.copy_(initial)
        ids.copy_(torch.tensor(slots, dtype=torch.int32))
        offsets.copy_(torch.tensor([0, lengths[0], 6], dtype=torch.int32))
        prefixes.copy_(torch.tensor(counts, dtype=torch.int32))
        graph.replay()
        expected_output, snapshots = _oracle(initial, inputs, slots, lengths, kd**-0.5)
        expected_state = initial.clone()
        for row, slot in enumerate(slots):
            expected_state[slot] = snapshots[row][counts[row]]
        torch.testing.assert_close(out.cpu(), expected_output.bfloat16(), rtol=0.002, atol=0.002)
        torch.testing.assert_close(state.cpu(), expected_state, rtol=5e-5, atol=2e-6)


def test_shared_storage_offset_beyond_eight_gib():
    # Only occupied regions are initialized/read back; the large gap is not a canary.
    offset = 7
    shape = (2, 2, 17, 33)
    state_size = 2 * 17 * 33
    record_size = 2 * 4 * (17 + 33 + 1)
    # slot * stride itself exceeds signed int32; a high base pointer alone
    # would not exercise multiplication overflow in the device kernel.
    stride = (8 * 1024**3) // 4 + 33
    raw = torch.empty(offset + stride + state_size + record_size + 29, dtype=torch.float32, device="npu")
    occupied = raw[offset + stride :]
    null_region = raw[offset : offset + state_size + record_size + 29]
    null_region.fill_(0.125)
    occupied.fill_(0.125)
    state = raw.as_strided(shape, (stride, 17 * 33, 33, 1), offset)
    records = raw.as_strided((2, 2, 4, 51), (stride, 4 * 51, 51, 1), offset + state_size)
    inputs = _inputs([4], 1, 2, 33, 17, torch.float32)
    initial = torch.full(shape, 0.125)
    ids = torch.tensor([1], dtype=torch.int64, device="npu")
    offsets = torch.tensor([0, 4], dtype=torch.int64, device="npu")
    prefix = torch.tensor([3], dtype=torch.int64, device="npu")
    guard_before = occupied.cpu()
    output = gdn_recover_verify(*[x.npu() for x in inputs], state, records, ids, offsets)
    _, snapshots = _oracle(initial, inputs, [1], [4], 33**-0.5)
    gdn_recover_commit(state, records, ids, offsets, prefix)
    expected = initial.clone()
    expected[1] = snapshots[0][3]
    torch.testing.assert_close(state.cpu(), expected, rtol=5e-5, atol=2e-6)
    output_ref, _ = _oracle(initial, inputs, [1], [4], 33**-0.5)
    torch.testing.assert_close(output.cpu(), output_ref, rtol=5e-5, atol=2e-6)
    changed = torch.zeros(occupied.numel(), dtype=torch.bool)
    changed[: state_size + record_size] = True
    assert torch.equal(occupied.cpu()[~changed], guard_before[~changed])
    assert torch.equal(null_region.cpu(), torch.full((null_region.numel(),), 0.125))


def test_empty_batch_and_validation():
    _, state = _padded((3, 2, 17, 33))
    _, records = _padded((3, 2, 4, 51))
    ids = torch.empty(0, dtype=torch.int32, device="npu")
    offsets = torch.zeros(1, dtype=torch.int32, device="npu")
    inputs = _inputs([], 1, 2, 33, 17, torch.float32)
    out = gdn_recover_verify(*[x.npu() for x in inputs], state, records, ids, offsets)
    assert out.shape == (0, 2, 17)
    gdn_recover_commit(state, records, ids, offsets, ids)
    with pytest.raises(ValueError, match="FP32"):
        gdn_recover_commit(state.bfloat16(), records, ids, offsets, ids)
    with pytest.raises(ValueError, match="records must hold"):
        gdn_recover_commit(state, records[..., :-1], ids, offsets, ids)


def test_consecutive_rounds_with_slot_reuse():
    _, state = _padded((8, 6, 17, 33))
    _, records = _padded((8, 6, 4, 51))
    torch.manual_seed(25)
    expected = torch.randn(state.shape) * 0.03
    state.copy_(expected)
    plans = [([5, 2], [4, 2], [4, 2]), ([2, 5], [1, 3], [1, 1]), ([5, 7], [2, 4], [0, 4])]
    for iteration in range(24):
        slots, lengths, counts = plans[iteration % len(plans)]
        if iteration == 12:
            # A completed request releases slot 5; a new request starts at zero.
            state[5].zero_()
            expected[5].zero_()
        inputs = _inputs(lengths, 2, 6, 33, 17, torch.float32)
        ids = torch.tensor(slots, dtype=torch.int32, device="npu")
        offsets = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32, device="npu")
        prefixes = torch.tensor(counts, dtype=torch.int32, device="npu")
        output_ref, snapshots = _oracle(expected, inputs, slots, lengths, 33**-0.5)
        before = state.cpu()
        output = gdn_recover_verify(*[x.npu() for x in inputs], state, records, ids, offsets)
        torch.testing.assert_close(output.cpu(), output_ref, rtol=5e-5, atol=2e-6)
        assert torch.equal(state.cpu(), before)
        gdn_recover_commit(state, records, ids, offsets, prefixes)
        for row, slot in enumerate(slots):
            expected[slot] = snapshots[row][counts[row]]
        torch.testing.assert_close(state.cpu(), expected, rtol=5e-5, atol=2e-6)


def test_conv_graph_replay_changes_prefix_and_slot_ids():
    history, window, channels = 5, 4, 257
    _, conv = _padded((8, history + window - 1, channels), dtype=torch.bfloat16)
    _, saved = _padded((8, history, channels), dtype=torch.bfloat16)
    torch.manual_seed(19)
    initial = torch.randn(conv.shape).bfloat16()
    expanded = torch.randn(conv.shape).bfloat16()
    expanded_device = expanded.npu()
    ids = torch.tensor([5, 2], dtype=torch.int32, device="npu")
    offsets = torch.tensor([0, 4, 6], dtype=torch.int32, device="npu")
    prefixes = torch.tensor([1, 2], dtype=torch.int32, device="npu")
    conv.copy_(initial)
    for _ in range(3):
        gdn_save_conv_history(conv, saved, ids, offsets)
        conv.copy_(expanded_device)
        gdn_commit_conv_history(conv, saved, ids, offsets, prefixes)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        gdn_save_conv_history(conv, saved, ids, offsets)
        # Emulate convolution's expanded-state writes independently of SSM.
        conv.copy_(expanded_device)
        gdn_commit_conv_history(conv, saved, ids, offsets, prefixes)
    for slots, lengths, counts in [([5, 2], [4, 2], [0, 2]), ([2, 5], [2, 4], [2, 1]), ([7, 1], [3, 3], [3, 3])]:
        conv.copy_(initial)
        ids.copy_(torch.tensor(slots, dtype=torch.int32))
        offsets.copy_(torch.tensor([0, lengths[0], 6], dtype=torch.int32))
        prefixes.copy_(torch.tensor(counts, dtype=torch.int32))
        graph.replay()
        expected = expanded.clone()
        for slot, count in zip(slots, counts):
            expected[slot, :history] = (
                expanded[slot, count - 1 : count - 1 + history] if count else initial[slot, :history]
            )
            torch.testing.assert_close(saved[slot].cpu(), initial[slot, :history], rtol=0, atol=0)
        torch.testing.assert_close(conv.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "g_dtype,beta_dtype",
    [(torch.float32, torch.bfloat16), (torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)],
)
def test_mixed_gate_dtypes(g_dtype, beta_dtype):
    _, state = _padded((8, 6, 17, 33))
    _, records = _padded((8, 6, 4, 51))
    torch.manual_seed(61)
    initial = torch.randn(state.shape) * 0.03
    state.copy_(initial)
    inputs = list(_inputs([2, 4], 2, 6, 33, 17, torch.bfloat16))
    inputs[3] = inputs[3].to(g_dtype)
    inputs[4] = inputs[4].to(beta_dtype)
    ids = torch.tensor([5, 2], dtype=torch.int32, device="npu")
    offsets = torch.tensor([0, 2, 6], dtype=torch.int32, device="npu")
    prefixes = torch.tensor([1, 4], dtype=torch.int32, device="npu")
    output_ref, snapshots = _oracle(initial, inputs, [5, 2], [2, 4], 33**-0.5)
    output = gdn_recover_verify(*[x.npu() for x in inputs], state, records, ids, offsets)
    gdn_recover_commit(state, records, ids, offsets, prefixes)
    expected = initial.clone()
    expected[5], expected[2] = snapshots[0][1], snapshots[1][4]
    torch.testing.assert_close(output.cpu(), output_ref.bfloat16(), rtol=0.002, atol=0.002)
    torch.testing.assert_close(state.cpu(), expected, rtol=5e-5, atol=2e-6)
