# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu
from vllm import envs

from vllm_ascend.models.common.ops import sequence_parallel as sp
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


@torch.inference_mode()
@pytest.mark.parametrize("skip_padding", [False, True], ids=["skip_off", "skip_on"])
@pytest.mark.parametrize(
    "tp_size,tp_rank",
    [pytest.param(1, 0, id="tp1")] + [pytest.param(8, rank, id=f"tp8_rank{rank}") for rank in range(8)],
)
@pytest.mark.parametrize(
    "steps",
    [
        pytest.param(((6, 8), (8, 8), (1, 8), (6, 8)), id="aligned_8"),
        pytest.param(((15, 16), (16, 16), (1, 16), (15, 16)), id="aligned_16"),
        pytest.param(((13, 13), (1, 13), (12, 13), (13, 13)), id="unaligned_13"),
        pytest.param(((6, 8), (15, 16), (1, 8), (16, 16), (8, 8)), id="switch_buckets"),
    ],
)
def test_graph_reads_refreshed_padding_mask(monkeypatch, skip_padding, tp_size, tp_rank, steps):
    monkeypatch.setattr(envs, "VLLM_MOE_SKIP_PADDING", skip_padding)
    graph_sizes = sorted({padded for _, padded in steps})
    max_tokens = max(graph_sizes)
    buffers = AscendInputBuffers(4, max_tokens, torch.device("npu"))
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.input_buffers = buffers
    mask_ptr = buffers.is_padding.data_ptr()
    # Distinct row values also check that SP masks and hidden states stay aligned.
    hidden_states = torch.arange(1, max_tokens + 1, dtype=torch.float32, device="npu").unsqueeze(1)
    # Simulate the SP rank on one device. The SP helper owns TP tail padding.
    monkeypatch.setattr(sp, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(sp, "get_tensor_model_parallel_rank", lambda: tp_rank)

    # Capture all buckets against the same persistent input storage before replay.
    graphs = {}
    for padded in graph_sizes:
        mask = runner.prepare_padding_mask(0, padded)
        states = hidden_states[:padded]
        local_padding = sp.sp_padding_mask(mask, states)
        local_states = sp.sp_shard(states)
        local_states.masked_fill(local_padding.unsqueeze(1), 0)
        torch.npu.synchronize()
        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph):
            local_padding = sp.sp_padding_mask(mask, states)
            local_states = sp.sp_shard(states)
            masked_states = local_states.masked_fill(local_padding.unsqueeze(1), 0)
        graphs[padded] = graph, local_padding, local_states, masked_states

    for valid, padded in steps:
        batch_mask = runner.prepare_padding_mask(valid, padded)
        assert batch_mask.shape == (padded,)
        assert batch_mask.data_ptr() == mask_ptr
        graph, local_padding, local_states, masked_states = graphs[padded]
        graph.replay()
        torch.testing.assert_close(batch_mask.cpu(), torch.arange(padded, device="cpu") >= valid)
        chunk = (padded + tp_size - 1) // tp_size
        rows = torch.arange(tp_rank * chunk, (tp_rank + 1) * chunk, device="cpu")
        row_values = (rows + 1).to(torch.float32).unsqueeze(1)
        torch.testing.assert_close(local_padding.cpu(), rows >= valid)
        torch.testing.assert_close(local_states.cpu(), row_values.masked_fill((rows >= padded).unsqueeze(1), 0))
        torch.testing.assert_close(masked_states.cpu(), row_values.masked_fill((rows >= valid).unsqueeze(1), 0))
