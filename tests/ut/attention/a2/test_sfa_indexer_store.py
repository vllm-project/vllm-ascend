# SPDX-License-Identifier: Apache-2.0
"""Native quantization byte oracle for final packed indexer K/scale writes."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.ops.triton.sfa_indexer_store import can_fuse_store, store_indexer_key_scale  # noqa: E402


def test_native_quantized_cache_bytes_and_changing_graph_slots():
    torch.npu.set_device(0)
    retained = []
    capacity, width = 256, 128
    for rows in (1, 6, 12):
        x = torch.randn(rows, width, dtype=torch.bfloat16, device="npu")
        keys, scales = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
        slots = torch.arange(rows, dtype=torch.int64, device="npu") + 7
        backing = torch.full((capacity * (width + 2) + 64,), 19, dtype=torch.int8, device="npu")
        cache = backing[: capacity * width].view(2, 128, 1, width)
        scale_cache = backing[capacity * width : -64].view(torch.float16).view(2, 128, 1, 1)
        assert can_fuse_store(cache, scale_cache, slots, rows)
        for _ in range(5):
            store_indexer_key_scale(keys, scales, slots, cache, scale_cache)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            store_indexer_key_scale(keys, scales, slots, cache, scale_cache)
        for step in range(16):
            fresh_keys, fresh_scales = torch_npu.npu_dynamic_quant(torch.randn_like(x), dst_type=torch.int8)
            keys.copy_(fresh_keys)
            scales.copy_(fresh_scales)
            selected = torch.arange(rows, dtype=torch.int64) + 3 + step
            if step % 2:
                selected[-1] = -1
            slots.copy_(selected)
            backing.fill_(19)
            scale_cache.fill_(3.25)
            graph.replay()
            torch.npu.synchronize()
            expected_keys = torch.full((capacity, width), 19, dtype=torch.int8)
            expected_scales = torch.full((capacity, 1), 3.25, dtype=torch.float16)
            valid = selected >= 0
            expected_keys[selected[valid]] = fresh_keys.cpu()[valid]
            expected_scales[selected[valid], 0] = fresh_scales.to(torch.float16).cpu()[valid]
            assert torch.equal(cache.view(capacity, width).cpu(), expected_keys)
            assert torch.equal(scale_cache.view(capacity, 1).view(torch.int16).cpu(), expected_scales.view(torch.int16))
            assert bool((backing[-64:] == 19).all())
        retained.append((graph, backing, keys, scales, slots))
    torch.npu.synchronize()
