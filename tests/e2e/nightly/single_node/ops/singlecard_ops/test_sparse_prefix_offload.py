# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from vllm.v1.kv_offload.base import (  # noqa: E402
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec  # noqa: E402

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.sparse_worker import (  # noqa: E402
    SparsePrefixOffloadingWorker,
)
from vllm_ascend.utils import enable_custom_op  # noqa: E402


@pytest.mark.parametrize("blocks_per_chunk", [1, 2, 4])
@pytest.mark.parametrize("shared_group", [False, True])
def test_paired_host_main_and_quantized_indexer_dma_round_trip(blocks_per_chunk, shared_group):
    if not enable_custom_op() or not hasattr(torch.ops._C_ascend, "swap_blocks_batch"):
        pytest.skip("Ascend batched DMA op is unavailable")
    main_k = torch.randint(-128, 127, (8, 128), dtype=torch.int8)
    main_v = torch.randint(-128, 127, (8, 64), dtype=torch.int8)
    indexer = torch.randint(-128, 127, (8, 128), dtype=torch.int8).npu()
    scales = torch.randn((8, 16), dtype=torch.float16).npu()
    parts = [main_k, main_v, indexer, scales.view(torch.int8)]
    expected = [part.cpu().clone() for part in parts]
    caches = CanonicalKVCaches(
        [CanonicalKVCacheTensor(part, part.shape[1]) for part in parts],
        [
            [CanonicalKVCacheRef(0, 128), CanonicalKVCacheRef(1, 64)],
            [CanonicalKVCacheRef(2, 128), CanonicalKVCacheRef(3, 32)],
        ],
    )
    # Three logical blocks starting at index 1 can straddle chunk boundaries.
    chunks = (1 % blocks_per_chunk + 3 + blocks_per_chunk - 1) // blocks_per_chunk
    gpu = GPULoadStoreSpec([2, 4, 5, 2, 4, 5], [3, 3], [1, 1])
    cpu = CPULoadStoreSpec(list(range(chunks)) + list(range(4, 4 + chunks)))
    if shared_group:
        caches.group_data_refs = [sum(caches.group_data_refs, [])]
        gpu = GPULoadStoreSpec([2, 4, 5], [3], [1])
        cpu = CPULoadStoreSpec(list(range(chunks)))
    worker = SparsePrefixOffloadingWorker(caches, blocks_per_chunk, 8, tp_rank=0)
    try:
        assert worker.submit_store(1, gpu, cpu)
        worker.wait({1})
        assert [result.job_id for result in worker.get_finished()] == [1]
        for part in parts:
            part.zero_()
        torch.npu.synchronize()
        assert worker.submit_load(2, cpu, gpu)
        worker.wait({2})
        assert [result.job_id for result in worker.get_finished()] == [2]
        for actual, original in zip(parts, expected):
            assert torch.equal(actual.cpu()[[2, 4, 5]], original[[2, 4, 5]])
            assert torch.count_nonzero(actual.cpu()[[0, 1, 3, 6, 7]]) == 0
    finally:
        worker.shutdown()
