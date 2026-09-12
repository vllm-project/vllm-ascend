# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 The vllm-ascend contributors

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import vllm.v1.worker.utils as upstream_utils
from vllm.model_executor.layers.attention import MLAAttention
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MLAAttentionSpec,
)

import vllm_ascend.worker.model_runner_v1 as model_runner_v1
from vllm_ascend.patch.worker.patch_copy_kv_cache import (
    _component_page_view,
    _is_component_pair,
    copy_kv_cache_blocks_inplace,
)
from vllm_ascend.worker.mla_component_cache_v1 import (
    build_mla_component_cache,
    use_mla_component_cache,
)


class _FakeMLABackend:
    @classmethod
    def customize_spec(cls, spec):
        return spec

    @classmethod
    def supported_kv_cache_layouts(cls):
        return (KVCacheLayout.LBNHC,)

    @classmethod
    def get_supported_kernel_block_sizes(cls):
        return [128]


class _FakeMLAAttention(MLAAttention):
    def __init__(self, *, spec, nope_dim, rope_dim):
        self._spec = spec
        self.kv_lora_rank = nope_dim
        self.qk_rope_head_dim = rope_dim
        self.head_size = nope_dim + rope_dim
        self.num_kv_heads = spec.num_kv_heads
        self.indexer = None
        self.kv_sharing_target_layer_name = None
        self.impl = SimpleNamespace(fa_quant_layer=False, enable_mlapo=False)
        self.attn_backend = _FakeMLABackend

    def get_kv_cache_spec(self, _vllm_config):
        return self._spec


def _make_spec(*, block_size, num_kv_heads=1, nope_dim=512, rope_dim=64, dtype=torch.bfloat16):
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=nope_dim + rope_dim,
        head_size_v=0,
        dtype=dtype,
    )


def _make_vllm_config(layers, *, use_v2_model_runner=False):
    return SimpleNamespace(
        use_v2_model_runner=use_v2_model_runner,
        kv_transfer_config=None,
        compilation_config=SimpleNamespace(static_forward_context=layers),
    )


def _install_cpu_h2d(monkeypatch):
    monkeypatch.setattr(
        upstream_utils,
        "async_tensor_h2d",
        lambda array, device: torch.from_numpy(array).to(device),
    )


def _page_payload(page_bytes: int, seed: int) -> torch.Tensor:
    return ((torch.arange(page_bytes, dtype=torch.int64) * 37 + seed * 101) % 251).to(torch.uint8)


@pytest.mark.parametrize(
    ("manager_block_size", "physical_page", "ratio"),
    [(128, 147456, 1), (384, 488448, 3), (768, 976896, 6)],
)
def test_build_mla_component_cache_kernel_slot_geometry(manager_block_size, physical_page, ratio):
    logical_spec = _make_spec(block_size=manager_block_size)
    padded_spec = replace(logical_spec, page_size_padded=physical_page)
    layer = _FakeMLAAttention(spec=logical_spec, nope_dim=512, rope_dim=64)
    num_blocks = 3
    raw = torch.zeros(num_blocks * physical_page, dtype=torch.int8)
    nope, rope = build_mla_component_cache(
        raw,
        layer=layer,
        spec=padded_spec,
        kernel_block_size=128,
        num_blocks=3,
    )

    slot_bytes = physical_page // ratio
    slot_elements = slot_bytes // 2
    assert nope.shape == (num_blocks * ratio, 128, 1, 512)
    assert rope.shape == (num_blocks * ratio, 128, 1, 64)
    assert nope.stride() == (slot_elements, 512, 512, 1)
    assert rope.stride() == (slot_elements, 64, 64, 1)
    assert rope.data_ptr() == nope.data_ptr() + 128 * 512 * 2
    assert nope.stride(0) > 128 * 512
    assert rope.stride(0) > 128 * 64
    assert _is_component_pair((nope, rope))


def test_component_cow_copies_complete_kernel_slot(monkeypatch):
    logical_spec = _make_spec(block_size=384)
    spec = replace(logical_spec, page_size_padded=488448)
    layer = _FakeMLAAttention(spec=logical_spec, nope_dim=512, rope_dim=64)
    raw = torch.zeros(3 * 488448, dtype=torch.int8)
    nope, rope = build_mla_component_cache(raw, layer=layer, spec=spec, kernel_block_size=128, num_blocks=3)

    slot_bytes = 488448 // 3
    page_view = _component_page_view(nope, rope)
    payload = _page_payload(slot_bytes, 7)
    payload[-1] = 0xA5
    payload[128 * 512 * 2] = 0x5A
    page_view[7].copy_(payload)

    _install_cpu_h2d(monkeypatch)
    copy_kv_cache_blocks_inplace(
        [(nope, rope)],
        3,
        [KVCacheBlockCopy(src_block_id=2, dst_block_id=0)],
    )
    torch.testing.assert_close(page_view[0], payload)
    torch.testing.assert_close(page_view[3], torch.zeros_like(page_view[3]))
    torch.testing.assert_close(page_view[7], payload)


def test_component_cache_is_default_for_v1():
    assert use_mla_component_cache(_make_vllm_config({}))


def test_component_cache_is_v1_only():
    assert not use_mla_component_cache(_make_vllm_config({}, use_v2_model_runner=True))


def test_model_runner_keeps_exact_mla_spec_for_all_mla_models(monkeypatch):
    spec = _make_spec(block_size=128)
    layer = _FakeMLAAttention(spec=spec, nope_dim=512, rope_dim=64)
    runner = model_runner_v1.NPUModelRunner.__new__(model_runner_v1.NPUModelRunner)
    runner.vllm_config = _make_vllm_config({"layer": layer})
    runner.shared_kv_cache_layers = {}
    runner.use_compress = False
    runner.use_sparse = False
    runner.sparse_kv_offload_enabled = False
    runner._use_mla_component_cache = True

    monkeypatch.setattr(model_runner_v1, "has_ec_transfer", lambda: False)
    assert runner.get_kv_cache_spec() == {"layer": spec}


def test_non_component_cow_uses_upstream_copy(monkeypatch):
    original_copy = MagicMock()
    monkeypatch.setattr(upstream_utils, "_orig_copy_kv_cache_blocks_inplace", original_copy)
    copies = [KVCacheBlockCopy(src_block_id=1, dst_block_id=0)]
    copy_kv_cache_blocks_inplace([], 2, copies)
    original_copy.assert_called_once_with([], 2, copies)


def test_mla_component_views_are_built_in_normal_reshape_path():
    logical_spec = _make_spec(block_size=384)
    spec = replace(logical_spec, page_size_padded=488448)
    layer = _FakeMLAAttention(spec=logical_spec, nope_dim=512, rope_dim=64)
    layers = {"layer": layer}
    num_blocks = 3
    layer_stride = 488448 * num_blocks
    descriptor = KVCacheTensor(
        size=layer_stride,
        layers=["layer"],
        layer_stride=layer_stride,
        block_stride=488448,
        offset=0,
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[descriptor],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
        kv_cache_layout=KVCacheLayout.LBNHC.name,
    )

    runner = model_runner_v1.NPUModelRunner.__new__(model_runner_v1.NPUModelRunner)
    runner._use_mla_component_cache = True
    runner.runner_only_attn_layers = set()
    runner.kernel_block_sizes = [[128]]
    runner.compilation_config = SimpleNamespace(static_forward_context=layers)
    runner._kv_cache_spec_attn_group_iterator = lambda: [
        SimpleNamespace(
            backend=_FakeMLABackend,
            kv_cache_spec=spec,
            layer_names=["layer"],
            kv_cache_group_id=0,
        )
    ]

    caches = runner._reshape_kv_cache_tensors(config, {"layer": torch.zeros(layer_stride, dtype=torch.int8)})
    nope, rope = caches["layer"]
    assert nope.shape == (9, 128, 1, 512)
    assert rope.shape == (9, 128, 1, 64)
    assert _is_component_pair((nope, rope))


def test_mla_nope_uses_empty_rope_component_view():
    spec = _make_spec(block_size=128, rope_dim=0)
    layer = _FakeMLAAttention(spec=spec, nope_dim=512, rope_dim=0)
    nope, rope = build_mla_component_cache(
        torch.zeros(2 * spec.page_size_bytes, dtype=torch.int8),
        layer=layer,
        spec=spec,
        kernel_block_size=128,
        num_blocks=2,
    )
    assert nope.shape == (2, 128, 1, 512)
    assert rope.shape == (2, 128, 1, 0)
    assert _is_component_pair((nope, rope))


def test_special_mla_spec_falls_back_to_legacy_rebuild(monkeypatch):
    logical_spec = replace(_make_spec(block_size=128), model_version="deepseek_v4")
    layer = _FakeMLAAttention(spec=logical_spec, nope_dim=512, rope_dim=64)
    runner = model_runner_v1.NPUModelRunner.__new__(model_runner_v1.NPUModelRunner)
    runner.vllm_config = _make_vllm_config({"layer": layer})
    runner.shared_kv_cache_layers = {}
    runner.use_compress = False
    runner.use_sparse = False
    runner.sparse_kv_offload_enabled = False
    runner._use_mla_component_cache = True

    monkeypatch.setattr(model_runner_v1, "has_ec_transfer", lambda: False)
    rebuilt = runner.get_kv_cache_spec()["layer"]
    assert type(rebuilt) is model_runner_v1.AscendMLAAttentionSpec
    assert rebuilt.model_version == "deepseek_v4"


def test_build_rejects_raw_size_mismatch():
    spec = _make_spec(block_size=128)
    layer = _FakeMLAAttention(spec=spec, nope_dim=512, rope_dim=64)
    with pytest.raises(ValueError, match="raw cache size"):
        build_mla_component_cache(
            torch.zeros(spec.page_size_bytes + 1, dtype=torch.int8),
            layer=layer,
            spec=spec,
            kernel_block_size=128,
            num_blocks=1,
        )


def test_build_rejects_misaligned_kernel_slot():
    logical_spec = _make_spec(block_size=256)
    spec = replace(logical_spec, page_size_padded=12)
    layer = _FakeMLAAttention(spec=logical_spec, nope_dim=512, rope_dim=64)
    with pytest.raises(ValueError, match="not aligned to dtype size"):
        build_mla_component_cache(
            torch.zeros(12, dtype=torch.int8),
            layer=layer,
            spec=spec,
            kernel_block_size=128,
            num_blocks=1,
        )
