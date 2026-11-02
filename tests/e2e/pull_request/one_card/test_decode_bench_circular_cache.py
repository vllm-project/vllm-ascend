# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""One-card NPU integration smoke for fixed-block DecodeBench state."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec
from vllm_ascend.distributed.kv_transfer.decode_bench_connector import AscendDecodeBenchConnector
from vllm_ascend.models.glm5next.cache_views import view_kpool_tail_cache


@pytest.mark.parametrize("startup_fill", [False, True])
@torch.inference_mode()
def test_decode_bench_circular_state_on_npu(startup_fill):
    spec = AscendIndexerKPoolTailSpec(
        block_size=7,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float32,
        sliding_window=4,
        compress_ratio=4,
        page_size_padded=256,
    )
    cache = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["tail"], kv_cache_spec=spec)],
    )
    extra = {"startup_fill": startup_fill, "fill_mean": 0.015, "fill_std": 0.0}
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        kv_transfer_config=SimpleNamespace(get_from_extra_config=lambda name, default: extra.get(name, default)),
    )
    scheduler = AscendDecodeBenchConnector(config, KVConnectorRole.SCHEDULER, cache)
    worker = AscendDecodeBenchConnector(config, KVConnectorRole.WORKER, cache)
    raw = torch.full((cache.num_blocks * spec.page_size_bytes,), 42, dtype=torch.uint8, device="npu")
    views = view_kpool_tail_cache("tail", spec, raw, cache.num_blocks)
    worker.register_kv_caches({"tail": views})
    assert torch.equal(raw.cpu(), torch.full_like(raw.cpu(), 42))

    for request_id, block_id in (("first", 1), ("second", 2)):
        blocks = SimpleNamespace(blocks=([SimpleNamespace(block_id=block_id, is_null=False)],))
        scheduler.update_state_after_alloc(SimpleNamespace(request_id=request_id, num_tokens=6000), blocks, 5999)
        metadata = scheduler.build_connector_meta(None)
        assert metadata.reqs_to_fill[request_id] == (([block_id],), 5999)
        before = raw.cpu()
        worker.bind_connector_metadata(metadata)
        worker.start_load_kv(None)
        expected = before.clone()
        start = block_id * spec.page_size_bytes
        expected[start : start + spec.unpadded_page_size_bytes] = 0
        torch.testing.assert_close(raw.cpu(), expected, rtol=0, atol=0)
        views[0][block_id].fill_(1.0)
