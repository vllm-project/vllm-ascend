# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-operator GDN RecoverSSM commit tests."""

import math

import pytest
import torch

pytest.importorskip("torch_npu")

from tests.e2e.nightly.single_node.ops.singlecard_ops.triton.gdn_recoverssm_test_utils import (  # noqa: E402
    WINDOW_CASES,
    ascendc_reference,
    check_accuracy,
    cpu_reference,
    model_inputs,
    npu_device,
    packed_layer,
    padded_layer,
    replay_commit,
    trace_launches,
)
from vllm_ascend.ops.triton.gdn import gdn_recoverssm_commit as gdn  # noqa: E402

npu = pytest.fixture(npu_device, name="npu")


def state_address_table(state, replay):
    return torch.tensor(
        [[[state.data_ptr()], [state.stride(0)]], [[replay.data_ptr()], [replay.stride(0)]]],
        dtype=torch.int64,
        device=state.device,
    )


def commit_state(state, replay, indices, count, spec_query_len):
    counts = torch.full_like(indices, count)
    no_boundary = torch.zeros_like(indices)
    gdn.gdn_recoverssm_commit(
        state,
        replay,
        state_address_table(state, replay),
        indices,
        counts,
        indices,
        no_boundary,
        no_boundary,
        spec_query_len=spec_query_len,
        align_mode=False,
    )


def load_replay(replay, records):
    """Populate FP32 records independently of the verify operator under test."""
    for page, expected in records.items():
        replay[page, :, : expected.shape[1]].copy_(expected.to(dtype=replay.dtype))


def test_commit_preserves_head_padding(npu, record_property):
    hk, hv, key, value, spec_query_len = 4, 8, 128, 128, 3
    layer, checkpoint_storage, replay_storage = padded_layer(npu, hv, key, value, spec_query_len)
    _, checkpoint, replay = layer.kv_cache
    indices = torch.tensor([1, 0], dtype=torch.int32, device=npu)[::2]
    inputs = model_inputs(npu, spec_query_len, hk, hv, key, value)
    _, histories, records = cpu_reference(inputs, checkpoint, [0, spec_query_len], [1])
    load_replay(replay, records)
    before = checkpoint_storage.clone()
    replay_before = replay_storage.clone()

    commit_state(checkpoint, replay, indices, 2, spec_query_len)

    check_accuracy(checkpoint[1], histories[0][2], record_property, "head_padding_commit")
    assert (checkpoint_storage[..., -16:] == -5).all()
    torch.testing.assert_close(replay_storage, replay_before, atol=0, rtol=0)
    torch.testing.assert_close(checkpoint_storage[[0, 2]], before[[0, 2]], atol=0, rtol=0)


@pytest.mark.parametrize(
    "key,value,amplitude,normalize",
    [
        (7, 5, 1, False),
        (7, 8, 1, True),
        (8, 5, 1, True),
        (128, 128, 1e-3, True),
        (128, 128, 1e-8, True),
        (128, 128, 0, True),
    ],
)
def test_commit_ragged_tiles_and_small_norms(npu, record_property, key, value, amplitude, normalize):
    hk, hv, spec_query_len = 2, 6, 4
    layer, _ = packed_layer(npu, 3, hk, hv, key, value, spec_query_len, torch.float32, False)
    _, state, replay = layer.kv_cache
    a_log, a, b, bias, q, k, v = model_inputs(npu, spec_query_len, hk, hv, key, value)
    a[0], a[1] = -8, 8
    inputs = (a_log, a, b, bias, q * amplitude, k * amplitude, v)
    pages = torch.tensor([1], device=npu, dtype=torch.int64)
    _, states, records = cpu_reference(
        inputs,
        state,
        [0, spec_query_len],
        [1],
        normalize=normalize,
        sp_beta=0.5,
        threshold=2.0,
        scale=0.25,
    )
    before = state.clone()
    load_replay(replay, records)
    for length in (1, spec_query_len):
        state.copy_(before)
        commit_state(state, replay, pages, length, spec_query_len)
        check_accuracy(state[1], states[0][length], record_property, f"prefix{length}")


@pytest.mark.parametrize("amplitude", [1.0, 1e-3, 1e-8, 0.0])
def test_commit_small_norms_against_ascendc(npu, record_property, amplitude):
    inputs, layer, _, page, _, prefixes = ascendc_reference(npu, amplitude)
    _, checkpoint, replay = layer.kv_cache
    spec_query_len = replay.shape[2]
    _, _, records = cpu_reference(inputs, checkpoint, [0, spec_query_len], [1], normalize=False)
    load_replay(replay, records)
    before = checkpoint.clone()
    for count in (1, 3, spec_query_len):
        checkpoint.copy_(before)
        commit_state(checkpoint, replay, page, count, spec_query_len)
        check_accuracy(checkpoint[1], prefixes[count - 1], record_property, f"ascendc_prefix{count}")


@pytest.mark.parametrize("page", [0, -1])
def test_commit_null_rows(npu, page):
    layer, backing = packed_layer(npu, 3, 2, 4, 16, 16, 3, torch.float32, True)
    _, state, replay = layer.kv_cache
    before = backing.clone()
    indices = torch.tensor([page], dtype=torch.int32, device=npu)

    commit_state(state, replay, indices, 3, 3)

    torch.testing.assert_close(backing, before, atol=0, rtol=0)


@pytest.mark.parametrize("accepted,final_page", [(2, 2), (4, 3)], ids=["exact", "cross"])
def test_commit_wrapper_writes_boundary_and_final_states(npu, accepted, final_page):
    spec_query_len = 4
    state = torch.full((4, 4, 128, 128), 8.0, dtype=torch.float32, device=npu)
    replay = torch.zeros((4, 4, spec_query_len, 257), dtype=torch.float32, device=npu)
    replay[1, :, :, -1] = 0.5
    source = torch.tensor([1], dtype=torch.int32, device=npu)
    boundary_page = torch.full_like(source, 2)
    boundary_length = torch.full_like(source, 2)
    expected = state.clone()
    expected[2].fill_(8.0 * 0.5**2)
    expected[final_page].fill_(8.0 * 0.5**accepted)

    gdn.gdn_recoverssm_commit(
        state,
        replay,
        state_address_table(state, replay),
        source,
        torch.full_like(source, accepted),
        torch.full_like(source, final_page),
        boundary_page,
        boundary_length,
        spec_query_len=spec_query_len,
        align_mode=True,
    )

    torch.testing.assert_close(state, expected, atol=0, rtol=0)


@pytest.mark.parametrize("hk,hv,key,value,spec_query_len,state_dtype,graph_mode", WINDOW_CASES)
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_commit_configured_window_dispatch(
    npu, monkeypatch, record_property, hk, hv, key, value, spec_query_len, state_dtype, graph_mode
):
    batch = max(3, math.ceil(96 / hv)) if key == value == 128 and hv % 2 == 0 else 3
    layer, backing = packed_layer(npu, batch + 1, hk, hv, key, value, spec_query_len, state_dtype, True)
    _, state, replay = layer.kv_cache
    pages = list(range(1, batch + 1))
    offsets = [row * spec_query_len for row in range(batch + 1)]
    indices = torch.tensor(pages, dtype=torch.int32, device=npu)
    counts = torch.full_like(indices, spec_query_len)
    inputs = model_inputs(npu, offsets[-1], hk, hv, key, value)
    ordinary = "_commit_gdn_state_kernel"
    grouped = "_commit_gdn_state_grouped_kernel"
    launches = trace_launches(monkeypatch, gdn, ordinary, grouped)
    _, histories, records = cpu_reference(inputs, state, offsets, pages)
    load_replay(replay, records)
    expected_state = state.clone()
    for row, page in enumerate(pages):
        expected_state[page].copy_(histories[row][spec_query_len])
    table = state_address_table(state, replay)
    no_boundary = torch.zeros_like(indices)

    def launch():
        gdn.gdn_recoverssm_commit(
            state,
            replay,
            table,
            indices,
            counts,
            indices,
            no_boundary,
            no_boundary,
            spec_query_len=spec_query_len,
            align_mode=False,
        )

    replay_commit(launch, [backing], graph_mode)
    a5 = gdn.is_950()
    selected = grouped if a5 and key == value == 128 and hv % 2 == 0 else ordinary
    assert {name for name, _, _ in launches} == {selected}
    for name, grid, args in launches:
        tiles = math.ceil(value / args["BV"])
        assert args["SPEC_QUERY_LEN"] == spec_query_len
        heads_per_program = 2 if name == grouped else 1
        assert grid == (tiles * (hv // heads_per_program) * batch,)
        assert args["BV"] * args["BK"] * heads_per_program * 2 * 4 <= gdn.get_ub_size_bytes()
        if name == ordinary:
            assert args["UNROLL_COMMIT"] == a5
    check_accuracy(state, expected_state, record_property, "dispatch_commit")
    record_property("commit_kernel", selected)
