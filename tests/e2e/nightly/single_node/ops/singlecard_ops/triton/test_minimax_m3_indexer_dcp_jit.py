# SPDX-License-Identifier: Apache-2.0
"""Real dispatch-cache reuse across request sizes, independent of disk cache."""

from importlib import import_module
from unittest.mock import Mock

import pytest
import torch
from vllm.triton_utils import triton

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.models.minimax_m3.ops.indexer_dcp import index_shard_scores, scatter_index_shard


def variant_count(kernel):
    return sum(len(entry[0]) for entry in kernel.device_caches.values())


@pytest.fixture
def isolated_kernels(monkeypatch):
    module = import_module("vllm_ascend.models.minimax_m3.ops.indexer_dcp")
    kernels = []
    for name in ("_scatter_index_shard", "_gather_index_shard_scores"):
        original = getattr(module, name)
        kernel = triton.jit(original.fn, do_not_specialize=original.do_not_specialize)
        resolution_spy = Mock(wraps=kernel._do_compile)
        monkeypatch.setattr(kernel, "_do_compile", resolution_spy)
        monkeypatch.setattr(module, name, kernel)
        kernels.append((kernel, resolution_spy))
    return kernels


@pytest.mark.parametrize("size", [2, 4])
@torch.inference_mode()
def test_request_sizes_and_pool_capacity_reuse_variants(size, isolated_kernels, record_property):
    # Deployment geometry and layout remain fixed; vary runtime indexing bounds.
    # Width 1/7/17/33 also crosses scalar value-one and alignment classes.
    for requests, query_len, width, pool_extra in [(1, 1, 1, 2), (3, 1, 17, 3), (2, 8, 33, 7), (3, 17, 7, 5)]:
        cache = torch.zeros(width + pool_extra, 128 // size, 128, dtype=torch.bfloat16, device="npu")
        n = requests * query_len
        updates = torch.ones(n, 128, dtype=cache.dtype, device="npu")
        slots = torch.arange(n, dtype=torch.int64, device="npu")
        scatter_index_shard(cache, updates, slots, size, 0)
        table = torch.zeros(requests, width, dtype=torch.int32, device="npu")
        lengths = torch.full((requests,), 128, dtype=torch.int32, device="npu")
        query = torch.ones(n, 4, 128, dtype=cache.dtype, device="npu")
        scores = index_shard_scores(query, cache, table, lengths, query_len, size, 0)
        torch.npu.synchronize()
        assert scores.shape == (4, n, triton.cdiv(width, 16) * 16)
        assert torch.all(scores[..., 0] == 128)
        assert torch.all(scores[..., 1:] == -torch.inf)
        # One in-process resolution may compile or load from the persistent
        # cache; do not describe this counter as backend compilation count.
        for kernel, spy in isolated_kernels:
            assert variant_count(kernel) == 1, (kernel.__name__, requests, query_len, width)
            assert spy.call_count == 1
    record_property("dispatch_variants_per_kernel", 1)
    record_property("jit_resolutions_per_kernel", 1)
