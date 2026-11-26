# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check batched recovery on real Ascend VMM memory and report warm latency.

Run with pytest -s to compare the previous eager loop and the batched path.
Timing includes host dispatch and synchronization, since recovery runs eagerly.
"""

from time import perf_counter
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_host_pool import (
    HostPoolTopology,
    MooncakeHostPool,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import SparseKVOffloadManager

HOST_TOKENS = 512
ROW_TOKENS = 256
KEY_DIM = 512
ROPE_DIM = 64
ALIGNMENT = 512
WARMUP_STEPS = 3
BENCHMARK_STEPS = 10


@pytest.mark.parametrize("layers", [1, 60])
def test_vmm_batched_recovery(layers, record_property):
    shared_segment = pytest.importorskip("mooncake.shared_segment")
    if not shared_segment.shared_segment_supported(mmap=False, host_register=False):
        pytest.skip("Ascend VMM shared segments are unavailable")
    torch.npu.set_device(0)
    topology = HostPoolTopology(tp_rank=0, tp_size=1)
    sizes = [HOST_TOKENS * dim * torch.bfloat16.itemsize for _ in range(layers) for dim in (KEY_DIM, ROPE_DIM)]
    pool = MooncakeHostPool.allocate(
        size_bytes=sum(sizes) + len(sizes) * ALIGNMENT, alignment=ALIGNMENT, topology=topology
    )
    try:
        views = pool.allocate_tensors(sizes, ALIGNMENT)
        manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
        manager._host_kv_allocator = pool
        manager.max_num_reqs = 2
        manager.k_caches_cpu = []
        manager.v_caches_cpu = []
        manager.topk_buffers_k = []
        manager.topk_buffers_v = []
        expected = []
        sources = torch.tensor([[256, 0], [128, -1]], dtype=torch.int64, device="npu")
        destinations = torch.tensor([[0, 129], [256, -1]], dtype=torch.int64, device="npu")
        lengths = torch.tensor([[1, 127], [128, 0]], dtype=torch.int64, device="npu")
        for layer in range(layers):
            for component, dim in enumerate((KEY_DIM, ROPE_DIM)):
                host = views[2 * layer + component].view(torch.bfloat16).view(4, 128, 1, dim)
                host.copy_(torch.randn(host.shape, dtype=torch.bfloat16, device="npu"))
                target = torch.zeros((2, ROW_TOKENS, 1, dim), dtype=torch.bfloat16, device="npu")
                reference = torch.zeros_like(target, device="cpu").view(-1, dim)
                host_reference = host.cpu().view(-1, dim)
                reference[0:1] = host_reference[256:257]
                reference[129:256] = host_reference[0:127]
                reference[256:384] = host_reference[128:256]
                expected.append(reference)
                (manager.k_caches_cpu if component == 0 else manager.v_caches_cpu).append(host)
                (manager.topk_buffers_k if component == 0 else manager.topk_buffers_v).append(target)
        manager._bind_copy_sfa_bases()
        metadata = SimpleNamespace(
            copy_sfa_copy_src_offsets=sources,
            copy_sfa_copy_dst_offsets=destinations,
            copy_sfa_copy_lengths=lengths,
            copy_sfa_copy_count=torch.tensor([8], dtype=torch.int32, device="npu"),
            copy_sfa_tail_src=sources,
            copy_sfa_tail_dst=destinations,
            copy_sfa_tail_lengths=lengths,
        )

        def batched():
            manager.restore_copy_sfa_tails(metadata)

        def previous_eager():
            descriptors = torch.stack((sources, destinations, lengths), dim=-1).reshape(-1, 3).cpu().tolist()
            for source, destination, length in descriptors:
                if length:
                    for host_k, host_v, device_k, device_v in zip(
                        manager.k_caches_cpu, manager.v_caches_cpu, manager.topk_buffers_k, manager.topk_buffers_v
                    ):
                        for host, target in ((host_k, device_k), (host_v, device_v)):
                            dim = target.shape[-1]
                            target.view(-1, dim)[destination : destination + length].copy_(
                                host.view(-1, dim)[source : source + length]
                            )

        batched()
        for layer in range(layers):
            for component, targets in enumerate((manager.topk_buffers_k, manager.topk_buffers_v)):
                target = targets[layer]
                torch.testing.assert_close(
                    target.cpu().view(-1, target.shape[-1]), expected[2 * layer + component], rtol=0, atol=0
                )
        # Descriptor values may change without recompilation. A zero-length
        # restore must leave even the invalid-address padded range untouched.
        lengths.zero_()
        for target in (*manager.topk_buffers_k, *manager.topk_buffers_v):
            target.fill_(7)
        batched()
        for target in (*manager.topk_buffers_k, *manager.topk_buffers_v):
            assert bool((target == 7).all().item())
        lengths.copy_(torch.tensor([[1, 127], [128, 0]], dtype=torch.int64, device="npu"))
        for name, run in (("previous_eager", previous_eager), ("batched", batched)):
            for _ in range(WARMUP_STEPS):
                run()
            torch.npu.synchronize()
            start = perf_counter()
            for _ in range(BENCHMARK_STEPS):
                run()
            torch.npu.synchronize()
            latency_ms = (perf_counter() - start) * 1000 / BENCHMARK_STEPS
            record_property(f"{name}_ms", latency_ms)
            print(f"layers={layers} {name}: {latency_ms:.3f} ms/restore")
    finally:
        torch.npu.synchronize()
        pool.close()
