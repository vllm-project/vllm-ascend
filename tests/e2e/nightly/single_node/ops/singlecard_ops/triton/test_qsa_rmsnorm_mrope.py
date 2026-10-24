# SPDX-License-Identifier: Apache-2.0
"""Numerical and FULL-graph replay coverage for QSA indexer preparation."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.linearnorm import qsa_rmsnorm_mrope  # noqa: F401

HEAD_DIM = 128
ROPE_DIM = 64
MROPE_SECTION = [11, 11, 10]
EPS = 1e-6
CACHE_POSITIONS = 37
REPLAY_ROUNDS = 4


def _reference(tensor, weight, cache, positions):
    values = tensor.float()
    reciprocal_std = torch.rsqrt(values.square().mean(-1, keepdim=True) + EPS)
    normalized = (values * reciprocal_std * (1 + weight.float())).to(tensor.dtype)
    gathered = cache[positions.clamp(0, cache.shape[0] - 1)]
    cos, sin = gathered.chunk(2, dim=-1)
    if positions.ndim == 2:
        merged_cos, merged_sin = cos[0].clone(), sin[0].clone()
        merged_cos[..., 1 : MROPE_SECTION[1] * 3 : 3] = cos[1, ..., 1 : MROPE_SECTION[1] * 3 : 3]
        merged_cos[..., 2 : MROPE_SECTION[2] * 3 : 3] = cos[2, ..., 2 : MROPE_SECTION[2] * 3 : 3]
        merged_sin[..., 1 : MROPE_SECTION[1] * 3 : 3] = sin[1, ..., 1 : MROPE_SECTION[1] * 3 : 3]
        merged_sin[..., 2 : MROPE_SECTION[2] * 3 : 3] = sin[2, ..., 2 : MROPE_SECTION[2] * 3 : 3]
        cos, sin = merged_cos, merged_sin
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    first, second = normalized[..., : ROPE_DIM // 2], normalized[..., ROPE_DIM // 2 : ROPE_DIM]
    return torch.cat(
        (first * cos - second * sin, second * cos + first * sin, normalized[..., ROPE_DIM:]),
        dim=-1,
    )


def _call(tensor, weight, cache, positions):
    return torch.ops.vllm.triton_qsa_rmsnorm_mrope_from_cache(
        tensor, weight, cache, positions, EPS, MROPE_SECTION, ROPE_DIM
    )


def _inputs(tokens, heads, axes):
    # Projection rows and compressed-key positions both commonly have gaps.
    storage = torch.randn((tokens, heads + 1, HEAD_DIM), dtype=torch.bfloat16, device="npu")
    tensor = storage[:, :heads]
    weight = torch.randn((HEAD_DIM,), dtype=torch.bfloat16, device="npu") * 0.1
    cache_storage = torch.randn((CACHE_POSITIONS * 2, ROPE_DIM), dtype=torch.bfloat16, device="npu")
    cache = cache_storage[::2]
    positions = torch.arange(tokens * axes, dtype=torch.int64).reshape(tokens, axes).transpose(0, 1)
    if axes == 1:
        positions = positions[0]
    if tokens:
        positions[..., 0] = -1
        positions[..., -1] = 2**35
    return tensor, weight, cache, positions.to("npu")


@pytest.mark.parametrize("tokens", [0, 1, 2, 8])
@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("axes", [1, 3])
@torch.inference_mode()
def test_qsa_cache_lookup_matches_reference_and_pregathered_op(tokens, heads, axes):
    tensor, weight, cache, positions = _inputs(tokens, heads, axes)
    actual = _call(tensor, weight, cache, positions)
    expected = _reference(tensor, weight, cache, positions)
    gathered = cache[positions.clamp(0, CACHE_POSITIONS - 1)]
    previous = torch.ops.vllm.triton_qsa_rmsnorm_mrope(tensor, weight, gathered, EPS, MROPE_SECTION, ROPE_DIM)
    torch.npu.synchronize()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual, previous, rtol=0, atol=0)


@pytest.mark.parametrize("heads", [1, 4])
@torch.inference_mode()
def test_qsa_cache_lookup_replays_updated_positions_across_graph_buckets(heads):
    """Replay padded C2/C1 and larger buckets without reusing gathered inputs."""
    buckets = {}
    for tokens in (4, 8):
        tensor, weight, cache, positions = _inputs(tokens, heads, axes=3)
        # Explicitly retain the noncontiguous compressed-key position layout.
        position_storage = torch.empty((tokens, 3), dtype=torch.int64, device="npu")
        position_view = position_storage.transpose(0, 1)
        position_view.copy_(positions)
        _call(tensor, weight, cache, position_view)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
            output = _call(tensor, weight, cache, position_view)
        buckets[tokens] = (graph, tensor, weight, cache, position_view, output)

    sequence = ((8, 8), (4, 2), (8, 5), (4, 1))
    for iteration, (tokens, active_tokens) in enumerate(sequence * REPLAY_ROUNDS):
        graph, tensor, weight, cache, positions, output = buckets[tokens]
        tensor.copy_(torch.randn_like(tensor))
        tensor[active_tokens:].zero_()
        changed_positions = torch.arange(tokens * 3, dtype=torch.int64).reshape(tokens, 3).transpose(0, 1)
        changed_positions += iteration * 7
        changed_positions[:, active_tokens:] = -1
        changed_positions[0, 0] = 2**35
        positions.copy_(changed_positions.to("npu"))

        # Pressure the small-allocation pool between replays. Correct graph
        # results must not depend on a temporary gathered tensor staying live.
        for _ in range(REPLAY_ROUNDS):
            scratch = torch.full((3, tokens, ROPE_DIM), float("nan"), dtype=torch.bfloat16, device="npu")
            del scratch
        graph.replay()
        torch.npu.synchronize()
        assert torch.isfinite(output).all()
        expected = _reference(tensor, weight, cache, positions)
        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
