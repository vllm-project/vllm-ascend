# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Current-token LoRA increments over native CANN pools; one request per call."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.glm5_next_cann_keypool import apply_keypool_lora


@torch.inference_mode()
@pytest.mark.parametrize("start,count", [(0, 1), (0, 13), (3, 1), (3, 2), (3, 13), (7, 1)])
@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("use_graph", [False, True])
def test_current_delta_preserves_history_padding_and_gate_state(start, count, zero, use_graph):
    torch.npu.set_device(0)
    torch.manual_seed(5319)
    device, pool, dim = torch.device("npu:0"), 4, 128
    total = start + count
    complete = total // pool - start // pool
    rows = count // pool + 1
    # Native output and metadata can include unused rows and padded strides.
    backing = torch.randn(rows, dim + 16, device=device).bfloat16()
    pooled = backing[:, :dim]
    state = torch.randn(3, pool, 2 * dim, device=device)
    delta = torch.randn(count, dim, device=device) * 0.1
    if zero:
        delta.zero_()
        pooled[0, 0] = -0.0
        state[:, :, 0] = -0.0
    gates = torch.randn_like(delta).bfloat16().float()
    ape = torch.randn(pool, dim, device=device)
    writes = torch.full((1, pool + 2), -1, dtype=torch.int64, device=device)[:, :pool]
    sources = torch.full((1, pool + 2), 0, dtype=torch.int64, device=device)[:, :pool]
    for offset in range(total % pool):
        writes[0, offset] = offset
        sources[0, offset] = (1 + int(start % pool + count >= pool)) * pool + offset
    metadata = SimpleNamespace(
        compress_ratio=pool,
        cum_query_lens=torch.tensor([count], dtype=torch.int64, device=device),
        raw_seq_lens=torch.tensor([total], dtype=torch.int32, device=device),
    )
    tail_metadata = SimpleNamespace(
        cann_inputs=SimpleNamespace(write_indices=writes, state_indices=sources, retained_indices=None)
    )
    original, old_state = backing.cpu(), state.cpu()
    expected, expected_state = original.clone(), old_state.clone()
    for row in range(complete):
        positions = torch.arange((start // pool + row) * pool, (start // pool + row + 1) * pool)
        current = positions >= start
        scores = old_state[1, :, dim:].clone()
        scores[current] = gates.cpu()[positions[current] - start]
        weights = (scores.double() + ape.cpu().double()).softmax(0).bfloat16().double()
        changes = torch.zeros(pool, dim, dtype=torch.float64)
        changes[current] = delta.cpu().double()[positions[current] - start]
        expected[row, :dim] = (original[row, :dim].double() + (weights * changes).sum(0)).bfloat16()
    for offset in range(total % pool):
        position = total - total % pool + offset
        if position >= start:
            source = int(sources.cpu()[0, offset])
            expected_state.view(-1, 2 * dim)[source, :dim] += delta.cpu()[position - start]

    def run():
        apply_keypool_lora(pooled, state, delta, gates, ape, metadata, tail_metadata)

    if use_graph:
        # Capture executes the in-place update, so restore inputs before replay.
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
        backing.copy_(original)
        state.copy_(old_state)
        graph.replay()
    else:
        run()
    torch.npu.synchronize()
    actual = backing.cpu()
    torch.testing.assert_close(state.cpu(), expected_state, atol=0, rtol=0)
    torch.testing.assert_close(actual[:, dim:], original[:, dim:], atol=0, rtol=0)
    torch.testing.assert_close(actual[complete:], original[complete:], atol=0, rtol=0)
    if zero:
        # Equality of values alone would miss changes to signed zero.
        torch.testing.assert_close(actual.view(torch.int16), original.view(torch.int16), atol=0, rtol=0)
        torch.testing.assert_close(state.cpu().view(torch.int32), old_state.view(torch.int32), atol=0, rtol=0)
    else:
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-2)
