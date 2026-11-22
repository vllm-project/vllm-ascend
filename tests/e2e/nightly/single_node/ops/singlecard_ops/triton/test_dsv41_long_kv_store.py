# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu

from vllm_ascend.attention.dsa_v41 import scatter_cache_sk
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.models.deepseek_v41.cache_config import STATE_RING_ROWS
from vllm_ascend.ops.triton.compressor.compressor_triton import compressor_from_projected
from vllm_ascend.ops.triton.dsv41_long_kv_store import HEAD_DIM, NOPE_DIM, ROPE_DIM, LongKVStore, normalize_and_store_c1

pytestmark = pytest.mark.skipif(
    not get_current_hardware_profile().supports(HardwareCapability.DSV41_LONG_KV_STORE),
    reason="Requires Ascend A2",
)


def _assert_bf16_close(actual, expected):
    diff = (actual.float() - expected.float()).abs()
    assert (diff <= (1 + expected.float().abs()) * 2**-6).float().mean() >= 0.99
    assert diff.max() <= 1.0


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_long_kv_store_matches_unfused(ratio, use_graph):
    torch.manual_seed(531)
    num_tokens = 6
    page_size = 128 // ratio
    eps = 1e-6
    projected = torch.randn(num_tokens, HEAD_DIM, device="npu", dtype=torch.bfloat16 if ratio == 1 else torch.float32)
    weight = torch.randn(HEAD_DIM, device="npu", dtype=torch.bfloat16)
    angles = torch.randn(num_tokens, ROPE_DIM // 2, device="npu")
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    # The backing page has a gap, as with cache views into shared allocation.
    backing = torch.full((4, page_size + 1, 1, HEAD_DIM), 17, device="npu", dtype=torch.bfloat16)
    reference = backing.clone()
    cache = backing[:, :page_size]
    coords = [[3, page_size - 1], [-1, -1], [1, 0], [-1, -1], [2, 1], [-1, -1]]
    slots = torch.tensor(coords, dtype=torch.int64, device="npu")
    store = LongKVStore(weight, eps, cache, slots, cos, sin)

    if ratio == 2:
        # An odd start reads an old ring row. The three completions occur at
        # token rows 0, 2, and 4, while row 5 is graph padding.
        metadata = torch.tensor([[31, 0], [3, 2], [0, 3], [0, 3], [1, 2]], dtype=torch.int32, device="npu")
        initial_ring = torch.randn(3, STATE_RING_ROWS, 2 * HEAD_DIM, device="npu")
        ring = initial_ring.clone()
        reference_ring = initial_ring.clone()
        scores = torch.randn_like(projected)
        out = torch.empty_like(projected, dtype=torch.bfloat16)
        reference_out = torch.empty_like(out)

    def compress(state=None, output=None, cache_store=None):
        if ratio == 1:
            return normalize_and_store_c1(projected, cache_store) if cache_store else projected
        return compressor_from_projected(
            projected,
            scores,
            state,
            metadata,
            output,
            max_query_len=3,
            num_cores=8,
            prev_rows=None,
            prev_row_indices=None,
            cache_store=cache_store,
        )

    graph = None
    if use_graph:
        compress(ring if ratio == 2 else None, out if ratio == 2 else None, store)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = compress(ring if ratio == 2 else None, out if ratio == 2 else None, store)

    for _ in range(2):
        backing.fill_(17)
        reference.fill_(17)
        if ratio == 2:
            ring.copy_(initial_ring)
            reference_ring.copy_(initial_ring)
            pooled = compress(reference_ring, reference_out)
        else:
            pooled = projected
        latent, _ = torch_npu.npu_rms_norm(pooled, weight, eps)
        rotated = latent.clone().unsqueeze(1)
        DeviceOperator.apply_partial_rotary_inplace(
            rotated, cos[:, None, None, :], sin[:, None, None, :], start=NOPE_DIM, end=HEAD_DIM
        )
        scatter_cache_sk(reference[:, :page_size], slots, rotated.squeeze(1))
        if graph is not None:
            graph.replay()
        else:
            actual = compress(ring if ratio == 2 else None, out if ratio == 2 else None, store)
        _assert_bf16_close(actual.cpu(), latent.cpu())
        written = torch.zeros(backing.shape[:2], dtype=torch.bool)
        for block, offset in coords:
            if block >= 0:
                written[block, offset] = True
        actual_cache, expected_cache = backing.cpu(), reference.cpu()
        _assert_bf16_close(actual_cache[written], expected_cache[written])
        assert torch.equal(actual_cache[~written], expected_cache[~written])
        if ratio == 2:
            assert torch.equal(ring.cpu(), reference_ring.cpu())

        # Replay uses the same addresses with updated source data and slots.
        projected.mul_(-1)
        sin.mul_(-1)
        coords[0], coords[4] = coords[4], coords[0]
        slots.copy_(torch.tensor(coords, dtype=slots.dtype))
