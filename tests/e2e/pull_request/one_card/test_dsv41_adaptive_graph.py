# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 MRV2 metadata and C2 consumer under one reused FULL graph."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config import CUDAGraphMode
from vllm.v1.kv_cache_interface import CircularBufferSpec

from vllm_ascend.attention.dsa_v41 import AscendDSAV41StateMetadataBuilder
from vllm_ascend.models.deepseek_v41.cache_config import STATE_RING_ROWS
from vllm_ascend.ops.triton.compressor.compressor_triton import _cube_core_num, compressor_from_projected
from vllm_ascend.worker.v2.model_states import default


@pytest.mark.parametrize("max_query_len", [6, STATE_RING_ROWS])
def test_v41_runner_full_graph_replays_variable_budgets(monkeypatch, max_query_len):
    # No checkpoint or draft allocator is needed: the inputs are the per-request
    # boundaries produced by adaptive verification, with deliberately stale CPU
    # lengths. Exercise the real runner -> common metadata -> V4.1 state builder.
    device = torch.device("npu:0")
    tokens, requests, width = 2 * max_query_len, 4, 512
    logical_tokens = max_query_len + 1
    capture_lengths = [(logical_tokens + 1) // 2, logical_tokens // 2]
    capture_starts = [0, capture_lengths[0], logical_tokens, logical_tokens, logical_tokens]
    config: Any = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=tokens, max_num_seqs=requests),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(head_dim=width, qk_rope_head_dim=64, sliding_window=128)
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, tensor_parallel_size=1),
    )
    spec = CircularBufferSpec(block_size=STATE_RING_ROWS, num_kv_heads=1, head_size=2 * width, dtype=torch.float32)
    name = "model.layers.2.self_attn.compressor.state_cache"
    builder = AscendDSAV41StateMetadataBuilder(spec, [name], config, device)
    # Static source RoPE is a model-loading fixture; no model forward is mocked.
    rope = torch.ones(1024, 1, 1, 64, dtype=torch.float32, device=device)
    assert builder._c2_source_cos.shape[-1] == rope.shape[-1]
    builder._c2_full_source_rope = (rope, rope)
    groups: Any = [[SimpleNamespace(layer_names=[name], get_metadata_builder=lambda _: builder)]]
    cache_config: Any = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)])
    runner_state: Any = default.AscendModelState.__new__(default.AscendModelState)
    runner_state.vllm_config = config
    runner_state.model_config = SimpleNamespace(max_model_len=1024)
    runner_state.decoder_replay = None
    monkeypatch.setattr(default, "get_ascend_config", lambda: SimpleNamespace(xlite_graph_config=None))
    batch: Any = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=requests,
        num_tokens=logical_tokens,
        num_tokens_after_padding=tokens,
        # Capture a balanced split, then replay unequal per-request budgets
        # with the same logical total and padded physical carrier.
        query_start_loc_np=np.array(capture_starts, dtype=np.int32),
        query_start_loc=torch.tensor(capture_starts, dtype=torch.int32, device=device),
        num_scheduled_tokens=torch.tensor(capture_lengths),
        max_query_len=max_query_len,
        is_prefilling_np=np.zeros(requests, dtype=bool),
        seq_lens=torch.tensor(
            [121 + capture_lengths[0], 255 + capture_lengths[1], 999, 999], dtype=torch.int32, device=device
        ),
        seq_lens_np=np.array([121 + capture_lengths[0], 255 + capture_lengths[1], 0, 0], dtype=np.int32),
        dcp_local_seq_lens=None,
        positions=torch.tensor(
            [*range(121, 121 + capture_lengths[0]), *range(255, 255 + capture_lengths[1])]
            + [299] * (tokens - logical_tokens),
            dtype=torch.int64,
            device=device,
        ),
        attn_state=None,
    )
    blocks = [torch.tensor([[2], [5], [0], [0]], dtype=torch.int32, device=device)]
    slots = torch.full((1, tokens), -1, dtype=torch.int64, device=device)
    generator = torch.Generator().manual_seed(19)
    initial = torch.randn(8, STATE_RING_ROWS, 2 * width, generator=generator) * 0.1
    state = initial.to(device)
    kv = torch.zeros(tokens, width, dtype=torch.float32, device=device)
    scores = torch.zeros_like(kv)
    output = torch.empty_like(kv, dtype=torch.bfloat16)
    cores = _cube_core_num()

    def forward(*, for_capture=False):
        metadata = runner_state.prepare_attn(
            batch,
            CUDAGraphMode.FULL,
            block_tables=tuple(blocks),
            slot_mappings=slots,
            attn_groups=groups,
            kv_cache_config=cache_config,
            for_capture=for_capture,
        )[name]
        assert metadata.max_query_len == max_query_len
        compressor_from_projected(
            kv,
            scores,
            state,
            metadata.c2_ring_metadata,
            output,
            max_query_len=metadata.max_query_len,
            num_cores=cores,
            prev_rows=None,
            prev_row_indices=None,
        )

    stream = torch.npu.Stream()
    graph = torch.npu.NPUGraph()
    with torch.npu.stream(stream):
        stream.wait_stream(torch.npu.default_stream())
        forward()
        with torch.npu.graph(graph, stream=stream):
            forward(for_capture=True)
    torch.npu.synchronize()
    # Reuse this graph throughout; accept only the anchor between iterations,
    # so the next residual can reference a row preceding a rejected suffix.
    reference_state = initial.clone()
    state.copy_(initial)
    histories = [121, 255]
    for lengths in (
        [max_query_len, 1],
        [1, max_query_len],
        capture_lengths,
        [2, max_query_len - 1],
        [max_query_len, 1],
    ):
        assert sum(lengths) == batch.num_tokens
        starts = [0, lengths[0], sum(lengths), sum(lengths), sum(lengths)]
        kv_cpu = torch.randn(tokens, width, generator=generator) * 0.1
        scores_cpu = torch.randn(tokens, width, generator=generator) * 0.1
        expected = torch.zeros(tokens, width, dtype=torch.bfloat16)
        positions = torch.full((tokens,), 299, dtype=torch.int64)
        for req, (history, count, page) in enumerate(zip(histories, lengths, [2, 5])):
            for local in range(count):
                row, pos = starts[req] + local, history + local
                positions[row] = pos
                reference_state[page, pos % STATE_RING_ROWS] = torch.cat((kv_cpu[row], scores_cpu[row]))
                if pos % 2:
                    pair = reference_state[page, [(pos - 1) % STATE_RING_ROWS, pos % STATE_RING_ROWS]]
                    expected[row] = (pair[:, :width] * pair[:, width:].softmax(0)).sum(0).bfloat16()
        with torch.npu.stream(stream):
            stream.wait_stream(torch.npu.default_stream())
            kv.copy_(kv_cpu)
            scores.copy_(scores_cpu)
            batch.query_start_loc.copy_(torch.tensor(starts, dtype=torch.int32))
            batch.seq_lens.copy_(torch.tensor([h + n for h, n in zip(histories, lengths)] + [999, 999]))
            batch.positions.copy_(positions)
            output.fill_(float("nan"))
            graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(state.cpu(), reference_state, rtol=0, atol=0)
        metadata = runner_state.attn_metadata[name]
        assert metadata.c2_ring_metadata[1].cpu().tolist() == lengths + [0, 0]
        assert metadata.c2_ring_metadata[4].cpu().tolist() == [2, 5, 0, 0]
        assert not metadata.c2_complete_mask[sum(lengths) :].any().item()
        histories = [h + 1 for h in histories]
