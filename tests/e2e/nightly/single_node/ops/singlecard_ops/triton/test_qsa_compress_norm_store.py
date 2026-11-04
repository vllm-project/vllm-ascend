# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.linearnorm.qsa_rmsnorm_mrope import triton_qsa_rmsnorm_mrope
from vllm_ascend.ops.triton.qwen4_exp.qsa import qsa_compress_groups_with_ratio, qsa_store_cache_rows

pytestmark = [
    pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="Requires Ascend A3"),
    pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available"),
]


def _inputs(with_positions: bool, chunk: int):
    torch.manual_seed(42)
    requests, ring_size, width = 3, 4, 128
    rows = requests * chunk
    raw = torch.randn(rows, 1, width + 32, dtype=torch.bfloat16, device="npu")[:, :, :width]
    ring = torch.randn(requests, ring_size, 1, width + 32, dtype=torch.bfloat16, device="npu")[:, :, :, :width]
    # Different request positions exercise old-state reads, wraparound and mixed groups.
    end = (torch.tensor([7, 19, 27], device="npu")[:, None] + torch.arange(chunk, device="npu")[None, :]).reshape(-1)
    position_rows = torch.stack((end, end + 11, end + 23), dim=-1).view(rows, 1, 3)
    rope = torch.randint(-3, 140, (requests, ring_size, 1, 6), dtype=torch.int64, device="npu")[:, :, :, ::2]
    table = torch.tensor([[2, 99], [0, 99], [1, 99]], dtype=torch.int32, device="npu")[:, :1]
    owners = torch.arange(requests, dtype=torch.int32, device="npu").repeat_interleave(chunk)
    starts = torch.arange(0, rows + 1, chunk, dtype=torch.int32, device="npu")
    slots = torch.where((end + 1) % 4 == 0, torch.arange(rows, device="npu"), -1)
    weight = torch.randn(width, dtype=torch.bfloat16, device="npu")
    rotary = torch.randn(129, 80, dtype=torch.bfloat16, device="npu")[:, :64]
    cache = torch.full((rows + 1, 2, 1, width + 16), 17, dtype=torch.bfloat16, device="npu")[:, :, :, :width]
    args = (raw, position_rows, ring, table, owners, starts, end, slots, 4, rope if with_positions else None)
    return args, weight, rotary, cache


def _unfused(args, weight, rotary, cache):
    pooled, first = qsa_compress_groups_with_ratio(*args)
    gathered = rotary[first.T.clamp(0, rotary.shape[0] - 1)]
    normalized = triton_qsa_rmsnorm_mrope(pooled, weight, gathered, 1e-6, [11, 11, 10], 64)
    qsa_store_cache_rows(cache, args[7], normalized)


@pytest.mark.parametrize("with_positions", [False, True])
@pytest.mark.parametrize("chunk", [1, 4, 7])
@torch.inference_mode()
def test_compress_norm_store_matches_decomposition(with_positions, chunk):
    args, weight, rotary, cache = _inputs(with_positions, chunk)
    expected = cache.clone()
    old_ring = args[2].clone()
    _unfused(args, weight, rotary, expected)
    qsa_compress_groups_with_ratio(*args, norm_weight=weight, rotary_cache=rotary, compressed_cache=cache)
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)
    torch.testing.assert_close(args[2], old_ring, rtol=0, atol=0)


@torch.inference_mode()
def test_compressed_cache_replay_and_masked_slots():
    args, weight, rotary, cache = _inputs(True, 4)

    def run():
        qsa_compress_groups_with_ratio(*args, norm_weight=weight, rotary_cache=rotary, compressed_cache=cache)

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    for offset in (0, 4, 8):
        # Same pointers and shapes, different key contents, positions and slot ownership.
        args[0].add_(0.125)
        args[2].mul_(0.875)
        args[6].add_(4)
        args[1].add_(offset)
        args[7][0] = -1
        args[7][4] = cache.shape[0] * cache.shape[1]  # Out-of-range store is masked.
        args[7][8] = offset // 4
        cache.fill_(17)
        expected = cache.clone()
        _unfused(args, weight, rotary, expected)
        graph.replay()
        torch.testing.assert_close(cache, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_compressed_cache_empty_rows_and_invalid_contract():
    args, weight, rotary, cache = _inputs(True, 1)
    empty = list(args)
    for index in (0, 1, 4, 6, 7):
        empty[index] = empty[index][:0]
    result = qsa_compress_groups_with_ratio(*empty, norm_weight=weight, rotary_cache=rotary, compressed_cache=cache)
    assert result is None
    with pytest.raises(ValueError, match="norm weight"):
        qsa_compress_groups_with_ratio(*args, rotary_cache=rotary, compressed_cache=cache)
    with pytest.raises(ValueError, match="rotary cache"):
        qsa_compress_groups_with_ratio(*args, norm_weight=weight, rotary_cache=rotary[:, :32], compressed_cache=cache)


@torch.inference_mode()
def test_compression_transaction_preserves_ring_across_steps():
    args, weight, rotary, cache = _inputs(True, 1)
    reference_args = list(args)
    reference_args[2] = args[2].clone()
    reference_args[9] = args[9].clone()
    expected = cache.clone()
    for step in range(12):
        args[0].add_(0.125)
        args[6].add_(1)
        args[1].add_(1)
        args[7].copy_(torch.where((args[6] + 1) % 4 == 0, torch.arange(3, device="npu") * 2, -1))
        # Both paths must read the previous state before any row overwrites it.
        _unfused(reference_args, weight, rotary, expected)
        qsa_compress_groups_with_ratio(*args, norm_weight=weight, rotary_cache=rotary, compressed_cache=cache)
        raw_slots = args[3][:, 0].long() * args[2].shape[1] + args[6] % args[2].shape[1]
        for state in (args, reference_args):
            qsa_store_cache_rows(state[2], raw_slots, state[0])
            qsa_store_cache_rows(state[9], raw_slots, state[1])
        torch.testing.assert_close(cache, expected, rtol=0, atol=0, msg=f"compressed state at step {step}")
        torch.testing.assert_close(args[2], reference_args[2], rtol=0, atol=0)
        torch.testing.assert_close(args[9], reference_args[9], rtol=0, atol=0)
