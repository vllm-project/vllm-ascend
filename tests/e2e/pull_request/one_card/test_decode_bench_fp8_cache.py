# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real NPU FP8 writes for both request fills and startup fill."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import DecodeBenchConnectorWorker
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVQuantMode

from vllm_ascend.distributed.kv_transfer.decode_bench_connector import (
    AscendDecodeBenchConnector,
    AscendDecodeBenchConnectorWorker,
)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("byte_storage", [False, True])
@pytest.mark.parametrize("std", [0.0, 0.2])
@pytest.mark.parametrize("startup_fill", [False, True])
@torch.inference_mode()
def test_decode_bench_fp8_cache_on_npu(dtype, byte_storage, std, startup_fill):
    quant = KVQuantMode.FP8_PER_TENSOR if byte_storage else KVQuantMode.NONE
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=4, dtype=torch.uint8 if byte_storage else dtype, kv_quant_mode=quant
    )
    cache = KVCacheConfig(
        num_blocks=6,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["attention"], kv_cache_spec=spec)],
    )
    extra = {"startup_fill": startup_fill, "fill_mean": 0.015, "fill_std": std}
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(cache_dtype="fp8_e5m2" if dtype == torch.float8_e5m2 else "fp8_e4m3"),
        kv_transfer_config=SimpleNamespace(get_from_extra_config=lambda name, default: extra.get(name, default)),
    )
    scheduler = AscendDecodeBenchConnector(config, KVConnectorRole.SCHEDULER, cache)
    worker = AscendDecodeBenchConnector(config, KVConnectorRole.WORKER, cache)
    assert type(worker.connector_worker) is AscendDecodeBenchConnectorWorker
    # The extra rows/columns are padding. Native FP8 and uint8 storage share
    # these exact bytes; no FP8 arithmetic/fill is used to set the sentinels.
    raw = torch.full((6, 6, 10), 42, dtype=torch.uint8, device="npu")
    view = raw[:, :4, :8]
    kv = view if byte_storage else view.view(dtype)
    original = raw.cpu()
    reference = object.__new__(DecodeBenchConnectorWorker)
    selected = slice(None) if startup_fill else [1, 4]
    shape = (6, 4, 8) if startup_fill else (2, 4, 8)
    torch.manual_seed(1024)
    values = reference._make_fill_values(shape, torch.device("cpu"), dtype, 0.015, std).view(torch.uint8)
    expected = original.clone()
    expected[selected, :4, :8] = values
    torch.manual_seed(1024)
    worker.register_kv_caches({"attention": kv})
    if not startup_fill:
        assert torch.equal(raw.cpu(), original)
    blocks = SimpleNamespace(blocks=([SimpleNamespace(block_id=i, is_null=False) for i in [1, 4]],))
    scheduler.update_state_after_alloc(SimpleNamespace(request_id="fp8", num_tokens=9), blocks, 8)
    worker.bind_connector_metadata(scheduler.build_connector_meta(None))
    worker.start_load_kv(None)
    torch.npu.synchronize()
    assert torch.equal(raw.cpu(), expected)
    assert kv.dtype == (torch.uint8 if byte_storage else dtype)
    decoded = raw.cpu()[selected, :4, :8].contiguous().view(dtype).float()
    assert torch.isfinite(decoded).all()
