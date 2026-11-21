# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, SlidingWindowSpec

from vllm_ascend.worker import model_runner_v1 as runner_module
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def _reshape(monkeypatch, spec, raw, kernel_block_size, hybrid=True):
    """Run the real V1 reshape path with a CPU attention backend boundary."""
    name = "model.layers.0.self_attn.attn"
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.hybrid_with_attn_and_mamba = hybrid
    runner.use_hybrid_blocks = hybrid
    runner.use_sparse = False
    runner.sparse_kv_offload_enabled = False
    runner.runner_only_attn_layers = set()
    runner.shared_kv_cache_layers = {}
    runner._is_c8_mxfp_kv_cache = lambda _: False
    runner._uses_page_strided_kv_layout = lambda _: False
    runner._uses_single_raw_mla_cache = lambda *_: False
    runner._get_layer_kv_cache_specs = lambda _: {name: spec}
    runner.vllm_config = SimpleNamespace()
    backend = SimpleNamespace(get_kv_cache_shape=lambda n, block, heads, dim: (2, n, block, heads, dim))
    runner._kv_cache_spec_attn_group_iterator = lambda: [
        SimpleNamespace(kv_cache_group_id=0, kv_cache_spec=spec, backend=backend, layer_names=[name])
    ]
    monkeypatch.setattr(runner_module, "enable_fa_quant", lambda *_: False)
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec)],
    )
    return runner._reshape_kv_cache_tensors(config, {name: raw}, [kernel_block_size])[name]


@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kernel_block_size", [2, 4])
@pytest.mark.parametrize("value_head_size", [2, 4])
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("base_offset", [0, 16])
def test_padded_pages_preserve_block_identity(
    monkeypatch, sliding, dtype, kernel_block_size, value_head_size, hybrid, base_offset
):
    spec_type = SlidingWindowSpec if sliding else FullAttentionSpec
    kwargs = {"sliding_window": 8} if sliding else {}
    payload = 4 * 2 * (4 + value_head_size) * dtype.itemsize
    spec = spec_type(
        block_size=4,
        num_kv_heads=2,
        head_size=4,
        head_size_v=value_head_size,
        dtype=dtype,
        page_size_padded=payload + 64,
        **kwargs,
    )
    page_bytes = spec.page_size_bytes
    backing = torch.full((base_offset + 3 * page_bytes + 16,), 7, dtype=torch.int8)
    raw = backing[base_offset : base_offset + 3 * page_bytes]
    before = backing.clone()
    key, value = _reshape(monkeypatch, spec, raw, kernel_block_size, hybrid)
    chunks = spec.block_size // kernel_block_size
    assert key.shape == (3 * chunks, kernel_block_size, 2, 4)
    assert value.shape == (3 * chunks, kernel_block_size, 2, value_head_size)
    assert key.data_ptr() == raw.data_ptr()
    assert key.stride(0) * key.element_size() * chunks == page_bytes
    assert value.stride(0) * value.element_size() * chunks == page_bytes
    key_bytes = kernel_block_size * 2 * 4 * dtype.itemsize
    assert value.data_ptr() - key.data_ptr() == key_bytes

    # Write both kernel blocks (if split) of manager page 1. Page 0, page 2,
    # and the enclosing allocation must remain byte-identical.
    key[chunks : 2 * chunks].fill_(3)
    value[chunks : 2 * chunks].fill_(5)
    torch.testing.assert_close(backing[: base_offset + page_bytes], before[: base_offset + page_bytes])
    torch.testing.assert_close(backing[base_offset + 2 * page_bytes :], before[base_offset + 2 * page_bytes :])
    kernel_page_bytes = page_bytes // chunks
    for part in range(chunks):
        offset = page_bytes + part * kernel_page_bytes
        used = payload // chunks
        expected_k = torch.full(key.shape[1:], 3, dtype=dtype).view(torch.int8).flatten()
        expected_v = torch.full(value.shape[1:], 5, dtype=dtype).view(torch.int8).flatten()
        torch.testing.assert_close(raw[offset : offset + key_bytes], expected_k)
        torch.testing.assert_close(raw[offset + key_bytes : offset + used], expected_v)
        assert torch.all(raw[offset + used : offset + kernel_page_bytes] == 7)

    # Another cache may reuse a different physical page in the shared pool.
    key_saved, value_saved = key[chunks : 2 * chunks].clone(), value[chunks : 2 * chunks].clone()
    raw[2 * page_bytes :].fill_(11)
    torch.testing.assert_close(key[chunks : 2 * chunks], key_saved)
    torch.testing.assert_close(value[chunks : 2 * chunks], value_saved)


@pytest.mark.parametrize("combined", [False, True])
def test_unpadded_attention_keeps_existing_values(monkeypatch, combined):
    spec = FullAttentionSpec(block_size=4, num_kv_heads=2, head_size=4, dtype=torch.float16)
    shape = (3, 4, 2, 4)
    if combined:
        data = torch.arange(3 * 2 * 4 * 2 * 4, dtype=torch.float16).view(3, 2, 4, 2, 4)
        raw = data.view(torch.int8).flatten()
        expected = data[:, 0], data[:, 1]
    else:
        expected = (torch.ones(shape, dtype=spec.dtype), torch.full(shape, 2, dtype=spec.dtype))
        raw = tuple(t.view(torch.int8).flatten() for t in expected)
    key, value = _reshape(monkeypatch, spec, raw, spec.block_size, hybrid=combined)
    torch.testing.assert_close(key, expected[0])
    torch.testing.assert_close(value, expected[1])


def test_rejects_non_aligned_kernel_pages(monkeypatch):
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=2, dtype=torch.float16, page_size_padded=34)
    raw = torch.zeros(3 * spec.page_size_bytes, dtype=torch.int8)
    with pytest.raises(ValueError, match="dtype-aligned kernel blocks"):
        _reshape(monkeypatch, spec, raw, 2)


def test_sliding_draft_geometry_preserves_shared_page_identity(monkeypatch):
    # The observed hybrid draft geometry has a much larger physical page than
    # its attention payload. Reproduce it without model weights or patient data.
    physical_page_bytes = 917504
    spec = SlidingWindowSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.bfloat16,
        page_size_padded=physical_page_bytes,
        sliding_window=8192,
    )
    raw = torch.zeros(3 * physical_page_bytes, dtype=torch.int8)
    key, value = _reshape(monkeypatch, spec, raw, spec.block_size)
    assert key.stride(0) * key.element_size() == physical_page_bytes
    assert value.stride(0) * value.element_size() == physical_page_bytes
    key[1].fill_(3)
    value[1].fill_(5)
    assert torch.count_nonzero(raw[:physical_page_bytes]) == 0
    assert torch.count_nonzero(raw[2 * physical_page_bytes :]) == 0
