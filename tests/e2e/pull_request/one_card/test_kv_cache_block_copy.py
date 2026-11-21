# SPDX-License-Identifier: Apache-2.0
"""Raw CoW transfers preserve the existing CPU copy contract byte for byte."""

import pytest
import torch
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy

from vllm_ascend.utils import enable_custom_op
from vllm_ascend.worker.utils import copy_kv_cache_blocks_inplace


@pytest.fixture(scope="module", autouse=True)
def _enable_ops():
    assert enable_custom_op()
    assert hasattr(torch.ops._C_ascend, "swap_blocks_batch")


def _case(kind, device):
    base = torch.arange(2048, dtype=torch.float32).to(device)
    if kind == "dense":
        views = [base[:512].view(8, 64)]
    elif kind == "padded":
        views = [torch.as_strided(base, (8, 2, 3), (17, 3, 1), storage_offset=7)]
    elif kind == "strided_payload":
        views = [torch.as_strided(base, (8, 2, 3), (17, 3, 1), storage_offset=7).transpose(1, 2)]
    elif kind == "kernel_blocks":
        views = [base[:512].view(16, 32)]
    elif kind == "strided_kernel_blocks":
        views = [torch.as_strided(base, (16, 2, 3), (17, 3, 1), storage_offset=7)]
    elif kind == "segmented":
        views = [[base[:128].view(8, 16), base[128:640].view(8, 64)]]
    elif kind == "overlapping_segments":
        views = [base[:512].view(8, 64), base[256:768].view(8, 64)]
    elif kind == "mixed_segments":
        views = [base[:512].view(8, 64), base[256:768].view(8, 8, 8).transpose(1, 2), base[640:1152].view(8, 64)]
    elif kind == "paired_alias":
        parent = base[:1024].view(16, 64)
        rows = base[:1024].view(8, 128)
        views = [parent, rows[:, :32], rows[:, 32:96], parent]
    elif kind == "combined_kv":
        views = [base[:1024].view(2, 8, 8, 1, 8)]
    elif kind == "fp8":
        views = [base[:512].view(torch.float8_e4m3fn).view(8, -1)]
    else:
        raise AssertionError(kind)
    return base, views


@pytest.mark.parametrize(
    "kind",
    [
        "dense",
        "padded",
        "strided_payload",
        "kernel_blocks",
        "strided_kernel_blocks",
        "segmented",
        "overlapping_segments",
        "mixed_segments",
        "paired_alias",
        "combined_kv",
        "fp8",
    ],
)
@pytest.mark.parametrize(
    "pairs", [[(0, 4), (1, 5), (0, 6)], [(0, 1), (1, 2), (2, 0)], [(0, 2), (1, 2)], [(0, 1), (1, 1)], [(-1, 0)]]
)
def test_npu_copy_matches_existing_cpu_contract(kind, pairs):
    cpu, cpu_views = _case(kind, "cpu")
    npu, npu_views = _case(kind, "npu")
    copies = [KVCacheBlockCopy(src_block_id=src, dst_block_id=dst) for src, dst in pairs]
    copy_kv_cache_blocks_inplace(cpu_views, 8, copies)
    copy_kv_cache_blocks_inplace(npu_views, 8, copies)
    torch.npu.synchronize()
    assert torch.equal(npu.cpu().view(torch.uint8), cpu.view(torch.uint8))


def test_npu_copy_on_another_stream_retains_snapshot_lifetime():
    cpu, cpu_views = _case("paired_alias", "cpu")
    npu, npu_views = _case("paired_alias", "npu")
    copies = [KVCacheBlockCopy(src_block_id=0, dst_block_id=1), KVCacheBlockCopy(src_block_id=1, dst_block_id=0)]
    copy_kv_cache_blocks_inplace(cpu_views, 8, copies)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        copy_kv_cache_blocks_inplace(npu_views, 8, copies)
        # Reuse the allocator while the native copies may still be pending.
        temporary = torch.empty(8192, dtype=torch.uint8, device="npu")
        temporary.fill_(201)
    torch.npu.synchronize()
    assert torch.equal(npu.cpu().view(torch.uint8), cpu.view(torch.uint8))


@pytest.mark.parametrize("inference", [False, True])
def test_native_copy_submits_dense_segments_together(inference, monkeypatch):
    calls = []
    swap_blocks_batch = torch.ops._C_ascend.swap_blocks_batch

    def count_copies(src, dst, sizes, direction):
        calls.append(sizes.numel())
        return swap_blocks_batch(src, dst, sizes, direction)

    monkeypatch.setattr(torch.ops._C_ascend, "swap_blocks_batch", count_copies)
    with torch.inference_mode(inference):
        cpu, cpu_views = _case("segmented", "cpu")
        npu, npu_views = _case("segmented", "npu")
        copies = [KVCacheBlockCopy(src_block_id=0, dst_block_id=1), KVCacheBlockCopy(src_block_id=1, dst_block_id=0)]
        copy_kv_cache_blocks_inplace(cpu_views, 8, copies)
        copy_kv_cache_blocks_inplace(npu_views, 8, copies)
        torch.npu.synchronize()
        assert calls == [8]
        torch.testing.assert_close(npu.cpu(), cpu, rtol=0, atol=0)
