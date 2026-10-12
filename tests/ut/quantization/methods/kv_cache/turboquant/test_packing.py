# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.quantization.methods.kv_cache.turboquant import cache


def _spec(rows, width, dtype, tq=False):
    return SimpleNamespace(
        page_size_bytes=rows * width * torch.empty((), dtype=dtype).element_size(),
        dtype=dtype,
        head_size=width,
        scale_dim=0,
        cache_dtype_str="turboquant_4bit_nc" if tq else "bfloat16",
    )


def _group(specs):
    return SimpleNamespace(layer_names=list(specs), kv_cache_spec=SimpleNamespace(kv_cache_specs=specs))


@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_packed_pages_never_alias_other_block_ids(block_size, monkeypatch):
    groups = [
        _group({f"c4.{i}": _spec(block_size, 258, torch.uint8, True) for i in range(11)}),
        _group({f"swa.{i}": _spec(block_size, 512, torch.bfloat16) for i in range(3)}),
        _group({f"state.{i}": _spec(block_size // 16, 2048, torch.float32) for i in range(4)}),
    ]
    # Isolate descriptor arithmetic from the installed vLLM version. The V2
    # integration test exercises the real upstream dataclass and allocator.
    monkeypatch.setattr(cache, "KVCacheTensor", SimpleNamespace)
    config = SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks_override=None))
    bytes_per_block = cache.pool_bytes_per_block(groups)
    num_blocks, descriptors = cache.cache_config(config, groups, bytes_per_block * 3 + 17)
    assert num_blocks == 3
    assert {d.size for d in descriptors} == {bytes_per_block * 3}
    descriptors = {d.layers[0]: d for d in descriptors}
    specs = {name: spec for g in groups for name, spec in g.kv_cache_spec.kv_cache_specs.items()}
    # Any two simultaneously live (group, block) pairs must be disjoint.
    for group_id, group in enumerate(groups):
        for name in group.layer_names:
            d = descriptors[name]
            spec = specs[name]
            assert d.block_stride >= spec.page_size_bytes
            assert d.offset % torch.empty((), dtype=spec.dtype).element_size() == 0
            if spec.dtype == torch.uint8:
                assert d.block_stride % 258 == 0
            for block in range(num_blocks):
                lo, hi = d.offset + block * d.block_stride, d.offset + block * d.block_stride + spec.page_size_bytes
                assert 0 <= lo < hi <= d.size
                for other_group_id, other_group in enumerate(groups):
                    for other in other_group.layer_names:
                        for other_block in range(num_blocks):
                            if other_group_id != group_id and other_block == block:
                                continue  # The same block ID is owned by only one group.
                            if other == name and other_block == block:
                                continue
                            od = descriptors[other]
                            olo = od.offset + other_block * od.block_stride
                            ohi = olo + specs[other].page_size_bytes
                            assert hi <= olo or ohi <= lo


def test_quantized_group_pool_is_smaller_than_bf16():
    quantized = [_group({f"c4.{i}": _spec(128, 258, torch.uint8, True) for i in range(21)})]
    baseline = [_group({f"c4.{i}": _spec(128, 512, torch.bfloat16) for i in range(21)})]
    assert cache.pool_bytes_per_block(baseline) / cache.pool_bytes_per_block(quantized) == pytest.approx(1024 / 258)
