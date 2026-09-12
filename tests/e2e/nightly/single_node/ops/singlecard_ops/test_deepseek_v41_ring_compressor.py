# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred NPU checks for the actual four-slot FP32 circular state views."""

import pytest
import torch
import torch_npu  # noqa: F401

from tests.deepseek_v41_cache_utils import allocate_cache_views, make_cache_config
from vllm_ascend.models.deepseek_v41.compressor import DeepseekV41RMSNorm
from vllm_ascend.ops.triton.compressor.compressor_triton import _cube_core_num, compressor_from_projected


def _reference(kv, scores, state, metadata):
    out = torch.zeros_like(kv, dtype=torch.bfloat16)
    # Snapshot residuals before updating: chunks may span several ring wraps.
    for start, used, _, base, block in metadata.T.tolist():
        if not used or not 0 < block < state.shape[0]:
            continue
        for group in range(start // 2, (start + used) // 2):
            rows = []
            for position in (2 * group, 2 * group + 1):
                if position < start:
                    rows.append(state[block, position % 32].clone())
                else:
                    index = base + position - start
                    rows.append(torch.cat((kv[index], scores[index])))
            pair = torch.stack(rows)
            width = kv.shape[1]
            out[base + 2 * group + 1 - start] = (
                (pair[:, :width] * pair[:, width:].softmax(dim=0)).sum(dim=0).to(torch.bfloat16)
            )
        for index in range(max(0, used - 32), used):
            state[block, (start + index) % 32] = torch.cat((kv[base + index], scores[base + index]))
    return out


def _inputs(length, start):
    torch.manual_seed(41)
    _, caches = allocate_cache_views(make_cache_config(13), "npu")
    state = caches["model.layers.2.self_attn.compressor.state_cache"].squeeze(-2)
    assert state.shape == (13, 32, 1024) and state.stride() == (32768, 1024, 1)
    # Nonzero stale pages detect unexpected null/inactive writes and BF16 casts.
    initial = torch.randn(state.shape, dtype=torch.float32)
    initial[:, :, 0] = 1.0001
    state.copy_(initial)
    kv_cpu = torch.randn(length + 2, 512, dtype=torch.float32)
    kv_cpu[:, 0] = 1.0001
    scores_cpu = torch.randn_like(kv_cpu)
    meta_cpu = torch.tensor(
        [[start, start + 1, 0], [length, 1, 0], [0, length, length + 1], [0, length, length + 1], [3, 7, 0]],
        dtype=torch.int32,
    )
    return state, initial, kv_cpu, scores_cpu, meta_cpu


@pytest.mark.parametrize("length", [0, 1, 2, 15, 16, 17, 31, 32, 33, 129])
@pytest.mark.parametrize("start", [0, 1, 31, 32, 33])
def test_projected_ring_matches_fp32_reference(length, start):
    state, initial, kv_cpu, scores_cpu, meta_cpu = _inputs(length, start)
    expected_state = initial.clone()
    expected = _reference(kv_cpu, scores_cpu, expected_state, meta_cpu)
    kv, scores, meta = kv_cpu.npu(), scores_cpu.npu(), meta_cpu.npu()
    out = torch.empty_like(kv, dtype=torch.bfloat16)
    compressor_from_projected(kv, scores, state, meta, out, max_query_len=max(length, 1), num_cores=_cube_core_num())
    torch.testing.assert_close(out.cpu(), expected, rtol=0.016, atol=1e-5)
    torch.testing.assert_close(state.cpu(), expected_state, rtol=0, atol=0)
    norm = DeepseekV41RMSNorm(512, 1e-6)
    expected_norm = expected.float() * torch.rsqrt(expected.float().square().mean(-1, keepdim=True) + norm.eps)
    expected_norm = expected_norm.to(expected.dtype) * norm.weight
    torch.testing.assert_close(norm.npu()(out).cpu(), expected_norm, rtol=0.016, atol=1e-5)


def test_projected_ring_graph_replay_uses_new_metadata_and_request_ids():
    state, initial, kv_cpu, scores_cpu, meta_cpu = _inputs(1, 0)
    kv, scores, meta = kv_cpu.npu(), scores_cpu.npu(), meta_cpu.npu()
    out = torch.empty_like(kv, dtype=torch.bfloat16)
    cores = _cube_core_num()

    def run():
        compressor_from_projected(kv, scores, state, meta, out, max_query_len=1, num_cores=cores)

    run()  # Compile outside capture.
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        run()
    pointers = (meta.data_ptr(), state.data_ptr(), out.data_ptr())
    for start in (0, 1, 31, 32, 33, 127, 128, 129):
        controls = meta_cpu.clone()
        controls[0] = torch.tensor([start, start + 1, start + 2])
        controls[1] = torch.tensor([1, 0, 1]) if start % 2 else torch.tensor([1, 1, 0])
        controls[4] = torch.tensor([7, 0, 3]) if start % 2 else torch.tensor([3, 7, 0])
        state.copy_(initial)
        meta.copy_(controls)
        expected_state = initial.clone()
        expected = _reference(kv_cpu, scores_cpu, expected_state, controls)
        graph.replay()
        torch.npu.synchronize()
        assert pointers == (meta.data_ptr(), state.data_ptr(), out.data_ptr())
        torch.testing.assert_close(out.cpu(), expected, rtol=0.016, atol=1e-5)
        torch.testing.assert_close(state.cpu(), expected_state, rtol=0, atol=0)


def test_empty_projected_batch_keeps_ring_untouched():
    state, initial, _, _, _ = _inputs(0, 0)
    kv = torch.empty(0, 512, dtype=torch.float32, device="npu")
    out = torch.empty_like(kv, dtype=torch.bfloat16)
    controls = torch.empty(5, 0, dtype=torch.int32, device="npu")
    compressor_from_projected(kv, kv, state, controls, out, max_query_len=0, num_cores=1)
    torch.testing.assert_close(state.cpu(), initial, rtol=0, atol=0)


@pytest.mark.parametrize("start", [0, 1, 31, 32, 127, 128])
@pytest.mark.parametrize("accepted", [0, 1, 15, 30, 31])
def test_ring_residual_after_dspark_rejection_matches_verified_prefix(start, accepted):
    _, caches = allocate_cache_views(make_cache_config(17, draft_layers=3), "npu")
    state = caches["model.layers.2.self_attn.compressor.state_cache"].squeeze(-2)
    state[7].fill_(17)
    kv = torch.randn(32, 512, dtype=torch.float32, device="npu")
    scores = torch.randn_like(kv)
    metadata = torch.tensor([[start], [32], [0], [0], [7]], dtype=torch.int32, device="npu")
    out = torch.empty_like(kv, dtype=torch.bfloat16)
    cores = _cube_core_num()
    compressor_from_projected(kv, scores, state, metadata, out, max_query_len=32, num_cores=cores)
    next_start = start + accepted + 1
    next_kv = torch.randn(2, 512, dtype=torch.float32, device="npu")
    next_scores = torch.randn_like(next_kv)
    controls = torch.tensor([[next_start], [2], [0], [0], [7]], dtype=torch.int32, device="npu")
    next_out = torch.empty_like(next_kv, dtype=torch.bfloat16)
    if next_start % 2:
        pair_kv = torch.stack((kv[accepted], next_kv[0]))
        pair_scores = torch.stack((scores[accepted], next_scores[0]))
        output_index = 0
    else:
        pair_kv, pair_scores = next_kv, next_scores
        output_index = 1
    expected = (pair_kv * pair_scores.softmax(0)).sum(0).bfloat16()
    compressor_from_projected(next_kv, next_scores, state, controls, next_out, max_query_len=2, num_cores=cores)
    torch.testing.assert_close(next_out[output_index].cpu(), expected.cpu(), rtol=0.016, atol=1e-5)
    assert not next_out[1 - output_index].any()
    # G12's separate live ID is untouched by target state updates.
    assert not caches["mtp.0.self_attn.swa_cache"][13].any()


@pytest.mark.parametrize("invalid", ["state_dtype", "projection_dtype", "strided_state", "ring_rows", "output_dtype"])
def test_projected_ring_rejects_incompatible_views(invalid):
    state, _, kv_cpu, scores_cpu, controls = _inputs(1, 0)
    kv, scores, meta = kv_cpu.npu(), scores_cpu.npu(), controls.npu()
    out = torch.empty_like(kv, dtype=torch.bfloat16)
    if invalid == "state_dtype":
        state = state.bfloat16()
    elif invalid == "projection_dtype":
        kv = kv.bfloat16()
    elif invalid == "strided_state":
        state = torch.empty(13, 64, 1024, dtype=torch.float32, device="npu")[:, ::2]
    elif invalid == "ring_rows":
        state = state[:, :16]
    else:
        out = out.float()
    with pytest.raises(ValueError):
        compressor_from_projected(kv, scores, state, meta, out, max_query_len=1, num_cores=1)


@pytest.mark.parametrize("graph_mode", [False, True])
def test_mixed_prefill_first_residual_uses_ring_at_projection_boundary(graph_mode):
    """First request history must not read row -1 of packed projections."""
    torch.manual_seed(1301)
    kv_cpu = torch.randn(3968, 512)
    scores_cpu = torch.randn_like(kv_cpu)
    # Allocate projections before other large tensors to exercise their boundary.
    kv, scores = kv_cpu.npu(), scores_cpu.npu()
    initial = torch.zeros(5954, 32, 1024, dtype=torch.float32)
    blocks = [12, 145, 278, 411, 544, 677]
    for block in blocks:
        initial[block] = torch.randn(32, 1024)
    state = initial.npu()
    controls = torch.tensor(
        [
            [1301, 1301, 1366, 0, 0, 0],
            [6, 6, 15, 1338, 1300, 1303],
            [0, 6, 12, 27, 1365, 2665],
            [0, 6, 12, 27, 1365, 2665],
            blocks,
        ],
        dtype=torch.int32,
    )
    meta = controls.npu()
    out = torch.empty_like(kv, dtype=torch.bfloat16)
    cores = _cube_core_num()
    expected_state = initial.clone()
    expected = _reference(kv_cpu, scores_cpu, expected_state, controls)

    def run():
        compressor_from_projected(kv, scores, state, meta, out, max_query_len=1338, num_cores=cores)

    run()
    if graph_mode:
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
            run()
        state.copy_(initial)
        graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), expected, rtol=0.016, atol=1e-5)
    torch.testing.assert_close(state.cpu(), expected_state, rtol=0, atol=0)
