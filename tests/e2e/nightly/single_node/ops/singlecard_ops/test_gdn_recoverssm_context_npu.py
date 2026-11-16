# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-card tests for GDN RecoverSSM cache binding and commit orchestration."""

from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("torch_npu")

from tests.e2e.nightly.single_node.ops.singlecard_ops.triton.gdn_recoverssm_test_utils import (  # noqa: E402
    HISTORY,
    QWEN_27B,
    QWEN_35B,
    check_accuracy,
    check_records,
    cpu_reference,
    model_inputs,
    packed_layer,
    replay_commit,
    verify,
)
from vllm_ascend.attention import recoverssm_gdn as gdn_context  # noqa: E402


@pytest.fixture
def npu():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.manual_seed(419)
    return torch.device("npu")


@pytest.fixture
def mocked_commit(monkeypatch):
    layer, _ = packed_layer(torch.device("cpu"), 3, 1, 2, 8, 8, 3, torch.float32, True)
    monkeypatch.setattr(gdn_context, "is_conv_state_dim_first", lambda: True)
    context = gdn_context.GDNRecoverSSMCommitContext.create([layer], spec_query_len=3, max_num_reqs=1)
    launches = Mock()
    for name in ("prepare_commit_plan", "compact_conv_state", "gdn_recoverssm_commit"):
        monkeypatch.setattr(gdn_context, name, getattr(launches, name))
    return context, launches


@pytest.mark.parametrize("commit_conv", [True, False])
def test_context_reuses_plan_and_respects_commit_conv(mocked_commit, commit_conv):
    context, launches = mocked_commit
    indices = torch.tensor([1], dtype=torch.int32)
    offsets = torch.tensor([0, 3], dtype=torch.int32)
    counts = torch.tensor([2], dtype=torch.int32)
    plan = context.commit_plan

    context.commit(counts, indices, offsets, commit_conv=commit_conv)

    expected_order = ["prepare_commit_plan", "gdn_recoverssm_commit"]
    if commit_conv:
        expected_order.insert(1, "compact_conv_state")
        conv = launches.compact_conv_state.call_args.args
        assert conv[6] is plan.commit_lens
        assert conv[7] is plan.final_state_indices
    assert [entry[0] for entry in launches.mock_calls] == expected_order
    prepared = launches.prepare_commit_plan.call_args.args
    assert prepared[0] is counts
    assert prepared[1] is indices
    assert prepared[2] is offsets
    assert prepared[3] is plan.commit_lens
    kernel_args = launches.gdn_recoverssm_commit.call_args
    assert kernel_args.args[3] is indices
    assert kernel_args.args[4] is plan.commit_lens
    assert kernel_args.args[5] is plan.final_state_indices
    assert kernel_args.args[6] is plan.boundary_state_indices
    assert kernel_args.args[7] is plan.boundary_recovery_lens
    assert kernel_args.kwargs == {"spec_query_len": 3, "align_mode": False}


def test_context_does_not_launch_for_empty_batch_or_invalid_align(mocked_commit):
    context, launches = mocked_commit
    empty = torch.empty(0, dtype=torch.int32)
    context.commit(empty, empty, torch.tensor([0], dtype=torch.int32))
    assert launches.mock_calls == []
    indices = torch.tensor([1], dtype=torch.int32)
    with pytest.raises(ValueError, match="align metadata is incomplete"):
        context.commit(indices, indices, torch.tensor([0, 1]), block_table=indices.reshape(1, 1))
    assert launches.mock_calls == []


def logical_conv(layer, channel_first):
    conv = layer.kv_cache[0]
    return conv if channel_first else conv.transpose(-2, -1)


@pytest.mark.parametrize(
    "global_hv,tp,spec_query_len,state_dtype,channel_first",
    [
        pytest.param(QWEN_35B, 4, 5, torch.float32, True, id="qwen36-tp4-mtp4"),
        pytest.param(QWEN_27B, 2, 9, torch.bfloat16, False, id="qwen38-tp2-dflash8"),
    ],
)
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_multilayer_verify_and_commit_across_steps(
    npu, monkeypatch, record_property, global_hv, tp, spec_query_len, state_dtype, channel_first
):
    monkeypatch.setattr(gdn_context, "is_conv_state_dim_first", lambda: channel_first)
    hk, hv = 16 // tp, global_hv // tp
    pairs = [
        packed_layer(npu, 5, hk, hv, 128, 128, spec_query_len, state_dtype, channel_first, gap) for gap in (256, 512)
    ]
    layers, backings = zip(*pairs)
    for layer in layers:
        layer.num_k_heads, layer.tp_size = 16, tp
    context = gdn_context.GDNRecoverSSMCommitContext.create(layers, spec_query_len=spec_query_len, max_num_reqs=4)
    query_start_offsets = [0, spec_query_len, spec_query_len + 2, spec_query_len + 3, spec_query_len + 3]
    query_start_loc = torch.tensor(query_start_offsets, dtype=torch.int64, device=npu)
    spec_to_sampler_rows = torch.tensor([2, 0, 1, 3], dtype=torch.int32, device=npu)
    # Spec row 1 reads sampler row 0 and clamps 99 to its two-token query span.
    sampler_accepted_counts = torch.tensor([99, 1, spec_query_len - 1, -1], dtype=torch.int32, device=npu)
    for step, pages in enumerate(([3, 1, 0, 2], [1, 3, 0, 2])):
        indices = torch.tensor(pages, dtype=torch.int32, device=npu)
        expected_layers = []
        for layer_no, layer in enumerate(layers):
            label = f"step{step}/layer{layer_no}"
            _, state, replay = layer.kv_cache
            initial = state.clone()
            conv = logical_conv(layer, channel_first)
            original_conv = conv.clone()
            inputs = model_inputs(npu, query_start_offsets[-1], hk, hv, 128, 128)
            expected, histories, records = cpu_reference(inputs, state, query_start_offsets, pages)
            out = torch.empty_like(inputs[-1], memory_format=torch.contiguous_format)
            assert verify(inputs, layer, query_start_loc, indices, spec_query_len, out=out) is out
            check_accuracy(out, expected, record_property, f"{label}/verify")
            check_records(replay, records, 128, record_property, label)
            torch.testing.assert_close(state, initial, atol=0, rtol=0)
            torch.testing.assert_close(out[:, -1], torch.zeros_like(out[:, -1]), atol=0, rtol=0)
            expected_state, expected_conv = initial.clone(), original_conv.clone()
            for row, count in enumerate((spec_query_len - 1, 2)):
                page = pages[row]
                expected_state[page].copy_(histories[row][count])
                expected_conv[page, :, :HISTORY] = original_conv[page, :, count - 1 : count - 1 + HISTORY]
                replay[page, :, count:].fill_(float("nan"))
            expected_layers.append((expected_state, expected_conv))
        context.commit(sampler_accepted_counts, indices, query_start_loc, request_indices=spec_to_sampler_rows)
        for layer_no, (layer, backing, (state, conv)) in enumerate(zip(layers, backings, expected_layers)):
            check_accuracy(layer.kv_cache[1], state, record_property, f"step{step}/layer{layer_no}/commit")
            torch.testing.assert_close(layer.kv_cache[1][[0, 2, 4]], state[[0, 2, 4]], atol=0, rtol=0)
            torch.testing.assert_close(logical_conv(layer, channel_first), conv, atol=0, rtol=0)
            assert torch.all(backing[:, -(256 if layer_no == 0 else 512) :] == 165)


@pytest.mark.parametrize("window,capacity", [(0, 2), (4, 0), (4, 2)])
def test_context_rejects_invalid_capacity_or_no_layers(window, capacity):
    layers = [object()] if min(window, capacity) == 0 else []
    with pytest.raises(ValueError, match="requires layers and positive capacities"):
        gdn_context.GDNRecoverSSMCommitContext.create(layers, spec_query_len=window, max_num_reqs=capacity)


@pytest.mark.parametrize(
    "prior,count,table,graph_mode,state_dtype",
    [
        pytest.param(6, 4, [3, 4], False, torch.float32, id="cross"),
        pytest.param(6, 2, [3, 4], True, torch.bfloat16, id="exact"),
        pytest.param(1, 2, [3, 4], False, torch.bfloat16, id="relocate"),
        pytest.param(1, 2, [1, 4], True, torch.float32, id="in-place"),
        pytest.param(6, 0, [3, 4], False, torch.float32, id="no-acceptance"),
    ],
)
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_align_final_and_boundary_routes(
    npu, monkeypatch, record_property, prior, count, table, graph_mode, state_dtype
):
    hk, hv, dim, spec_query_len, block_size = 4, 12, 128, 5, 8
    monkeypatch.setattr(gdn_context, "is_conv_state_dim_first", lambda: True)
    pairs = [packed_layer(npu, 8, hk, hv, dim, dim, spec_query_len, state_dtype, layout) for layout in (True, False)]
    layers, backings = zip(*pairs)
    layers[1].kv_cache = (layers[1].kv_cache[0].transpose(-1, -2), *layers[1].kv_cache[1:])
    context = gdn_context.GDNRecoverSSMCommitContext.create(layers, spec_query_len=spec_query_len, max_num_reqs=3)
    assert context.conv_shared_dim_stride is None
    pages, offsets = [1, 2, 0], [0, 5, 7, 8]
    starts = torch.tensor([0, -1, 5, -1, 7, -1, 8], dtype=torch.int64, device=npu)[::2]
    indices = torch.tensor(pages, dtype=torch.int32, device=npu)
    counts = torch.tensor([9, -1, 7, -1, count, -1], dtype=torch.int32, device=npu)[::2]
    mapping = torch.tensor([2, -1, 0, -1, 1, -1], dtype=torch.int32, device=npu)[::2]
    computed = torch.tensor([1, 0, prior], dtype=torch.int32, device=npu)
    table_data = [[6, 7], [0, 0], table]
    block_table = torch.tensor(table_data, dtype=torch.int64, device=npu).repeat_interleave(2, dim=1)[:, ::2]
    expected_layers = []
    for layer_no, layer in enumerate(layers):
        conv, state, replay = layer.kv_cache
        inputs = model_inputs(npu, offsets[-1], hk, hv, dim, dim)
        golden, history, records = cpu_reference(inputs, state, offsets, pages)
        output = verify(inputs, layer, starts, indices, spec_query_len)
        check_accuracy(output, golden, record_property, f"align_verify{layer_no}")
        check_records(replay, records, dim, record_property, f"align_replay{layer_no}")
        expected_state, expected_conv = state.clone(), conv.clone()
        untouched = set(range(state.shape[0]))
        for row, (previous, accepted, destinations) in enumerate(((prior, count, table), (1, 2, [6, 7]))):
            if not accepted:
                continue
            writes = [(destinations[(previous + accepted - 1) // block_size], accepted)]
            boundary = block_size - previous % block_size
            if accepted >= boundary:
                writes.append((destinations[previous // block_size], boundary))
            for destination, length in writes:
                expected_state[destination].copy_(history[row][length])
                expected_conv[destination, :, :HISTORY] = conv[pages[row], :, length - 1 : length - 1 + HISTORY]
                untouched.discard(destination)
        expected_layers.append((expected_state, expected_conv, sorted(untouched)))

    def launch():
        context.commit(counts, indices, starts, mapping, block_table, computed, block_size)

    replay_commit(launch, backings, graph_mode)
    for layer_no, (layer, (state, conv, untouched_pages)) in enumerate(zip(layers, expected_layers)):
        check_accuracy(layer.kv_cache[1], state, record_property, f"align_commit{layer_no}")
        torch.testing.assert_close(layer.kv_cache[1][untouched_pages], state[untouched_pages], atol=0, rtol=0)
        torch.testing.assert_close(layer.kv_cache[0], conv, atol=0, rtol=0)


@pytest.mark.parametrize(
    "global_hv,tp,spec_query_len,batch,layer_count,graph_mode",
    [
        pytest.param(QWEN_35B, 4, 5, 11, 30, False, id="qwen36-mtp4"),
        pytest.param(QWEN_27B, 4, 9, 12, 48, True, id="qwen38-dflash8"),
    ],
)
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_model_packed_page_commit(
    npu, monkeypatch, record_property, global_hv, tp, spec_query_len, batch, layer_count, graph_mode
):
    monkeypatch.setattr(gdn_context, "is_conv_state_dim_first", lambda: False)
    hk, hv = 16 // tp, global_hv // tp
    pairs = [
        packed_layer(npu, batch + 2, hk, hv, 128, 128, spec_query_len, torch.float32, False) for _ in range(layer_count)
    ]
    layers, backings = zip(*pairs)
    count_cpu = torch.arange(batch, dtype=torch.int32) % spec_query_len + 1
    starts = torch.arange(batch + 1, dtype=torch.int32, device=npu) * spec_query_len
    indices = torch.arange(1, batch + 1, dtype=torch.int32, device=npu)
    counts = count_cpu.to(npu)
    for layer_no, layer in enumerate(layers):
        conv, state, replay = layer.kv_cache
        state.fill_(0.02 * (1 + layer_no % 5))
        replay[..., :-1].zero_()
        replay[..., -1].fill_(0.75)
        conv.copy_(torch.arange(spec_query_len + HISTORY - 1, device=npu, dtype=torch.bfloat16)[None, :, None])

    context = gdn_context.GDNRecoverSSMCommitContext.create(layers, spec_query_len=spec_query_len, max_num_reqs=batch)

    def launch():
        context.commit(counts, indices, starts)

    replay_commit(launch, backings, graph_mode)
    # Zero effective key and correction reduce recovery to scalar decay.
    factors = torch.ones(batch + 2, dtype=torch.float64)
    factors[1:-1] = 0.75 ** count_cpu.double()
    for layer_no, (layer, raw) in enumerate(zip(layers, backings)):
        conv, state, replay = layer.kv_cache
        initial = torch.tensor(0.02 * (1 + layer_no % 5), dtype=torch.float32).double()
        golden = (initial * factors)[:, None, None, None].expand(state.shape)
        check_accuracy(state, golden, record_property, f"packed_layer{layer_no}")
        torch.testing.assert_close(state[[0, batch + 1]].cpu().double(), golden[[0, batch + 1]], atol=0, rtol=0)
        sequence = torch.arange(spec_query_len + HISTORY - 1, dtype=torch.bfloat16)
        expected_conv = sequence.expand(batch + 2, -1).clone()
        for page, count in enumerate(count_cpu.tolist(), start=1):
            expected_conv[page, :HISTORY] = sequence[count - 1 : count - 1 + HISTORY]
        torch.testing.assert_close(conv.cpu(), expected_conv[..., None].expand(conv.shape), atol=0, rtol=0)
        assert replay[..., :-1].count_nonzero() == 0
        assert (replay[..., -1] == 0.75).all()
        assert (raw[:, -256:] == 165).all()
