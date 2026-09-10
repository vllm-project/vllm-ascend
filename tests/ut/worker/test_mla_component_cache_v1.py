# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 The vllm-ascend contributors

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import vllm.v1.worker.utils as upstream_utils
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
)

import vllm_ascend.worker.mla_component_cache_v1 as mla_component_cache_v1
import vllm_ascend.worker.model_runner_v1 as model_runner_v1
from vllm_ascend.patch.worker.patch_copy_kv_cache import (
    _component_page_view,
    _is_component_pair,
    copy_kv_cache_blocks_inplace,
)
from vllm_ascend.worker.mla_component_cache_v1 import (
    _typed_empty_like_storage,
    allocate_mla_component_cache,
    get_mla_component_cache_capability,
    materialize_hybrid_mla_component_cache,
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


def _make_spec(*, block_size, num_kv_heads, nope_dim, rope_dim, dtype):
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=nope_dim + rope_dim,
        head_size_v=0,
        dtype=dtype,
    )


def _make_vllm_config(
    layers,
    *,
    use_v2_model_runner=False,
    compress_ratios=None,
):
    hf_text_config = SimpleNamespace()
    if compress_ratios is not None:
        hf_text_config.compress_ratios = compress_ratios
    return SimpleNamespace(
        use_v2_model_runner=use_v2_model_runner,
        kv_transfer_config=None,
        model_config=SimpleNamespace(hf_text_config=hf_text_config),
        compilation_config=SimpleNamespace(static_forward_context=layers),
    )


def _make_plan(layers, spec, *, num_blocks=5):
    page_bytes = spec.page_size_bytes
    layer_names = list(layers)
    layer_stride = page_bytes * num_blocks
    descriptor = KVCacheTensor(
        size=len(layer_names) * layer_stride,
        layers=layer_names,
        layer_stride=layer_stride,
        block_stride=page_bytes,
        offset=0,
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[descriptor],
        kv_cache_groups=[KVCacheGroupSpec(layer_names, spec)],
        kv_cache_layout=KVCacheLayout.LBNHC.name,
    )
    return config


class _FakeKDALayer(Attention):
    def __init__(self, spec):
        self._spec = spec
        self.kv_sharing_target_layer_name = None

    def get_kv_cache_spec(self, _vllm_config):
        return self._spec


def _make_k3_hybrid_plan(*, num_blocks=3, manager_block_size=384):
    kernel_block_size = 128
    element_size = torch.empty((), dtype=torch.bfloat16).element_size()
    logical_spec = _make_spec(
        block_size=manager_block_size,
        num_kv_heads=1,
        nope_dim=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    physical_page = 488448 if manager_block_size == 384 else 976896
    padded_spec = replace(logical_spec, page_size_padded=physical_page)

    mla_layers = {
        f"language_model.model.layers.{index}.self_attn": _FakeMLAAttention(
            spec=logical_spec,
            nope_dim=512,
            rope_dim=64,
        )
        for index in range(2)
    }
    kda_spec = MambaSpec(
        block_size=manager_block_size,
        shapes=((1,),),
        dtypes=(torch.float32,),
        page_size_padded=physical_page,
    )
    kda_layers = {f"language_model.model.layers.{index}.linear_attn": _FakeKDALayer(kda_spec) for index in range(2, 5)}
    layers = {**mla_layers, **kda_layers}
    layer_stride = physical_page * num_blocks
    backing_size = len(layers) * layer_stride
    backing = torch.zeros(backing_size, dtype=torch.int8)
    tensors = []
    raw_tensors = {}
    for layer_idx, (layer_name, spec) in enumerate(layers.items()):
        offset = layer_idx * layer_stride
        tensors.append(
            KVCacheTensor(
                size=backing_size,
                layers=[layer_name],
                layer_stride=layer_stride,
                block_stride=physical_page,
                offset=offset,
            )
        )
        raw_tensors[layer_name] = backing[offset : offset + layer_stride]

    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=[
            KVCacheGroupSpec(list(mla_layers), padded_spec),
            KVCacheGroupSpec(list(kda_layers), kda_spec),
        ],
        kv_cache_layout=KVCacheLayout.LBNHC.name,
    )
    vllm_config = _make_vllm_config(layers)
    capability = get_mla_component_cache_capability(vllm_config)
    assert capability is not None
    assert capability.mode == "K3_HYBRID_V1"
    assert capability.kernel_block_size == kernel_block_size
    return (
        vllm_config,
        config,
        raw_tensors,
        capability,
        physical_page,
        element_size,
    )


def _install_cpu_h2d(monkeypatch):
    monkeypatch.setattr(
        upstream_utils,
        "async_tensor_h2d",
        lambda array, device: torch.from_numpy(array).to(device),
    )


def _page_payload(page_bytes: int, seed: int) -> torch.Tensor:
    payload = ((torch.arange(page_bytes, dtype=torch.int64) * 37 + seed * 101) % 251).to(torch.uint8)
    return payload


@pytest.mark.parametrize(
    ("nope_dim", "rope_dim", "num_kv_heads", "dtype"),
    [
        (5, 3, 2, torch.float32),
        (7, 2, 1, torch.float16),
        (6, 4, 3, torch.bfloat16),
    ],
)
def test_allocate_component_cache_geometry(nope_dim, rope_dim, num_kv_heads, dtype):
    block_size, num_blocks, num_layers = 8, 5, 3
    spec = _make_spec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        dtype=dtype,
    )
    layers = {
        f"model.layers.{index}.self_attn": _FakeMLAAttention(
            spec=spec,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
        )
        for index in range(num_layers)
    }
    config = _make_plan(layers, spec, num_blocks=num_blocks)
    caches = allocate_mla_component_cache(
        vllm_config=_make_vllm_config(layers),
        kv_cache_config=config,
        static_forward_context=layers,
        device=torch.device("cpu"),
        kernel_block_sizes=[[block_size]],
    )

    element_size = torch.tensor((), dtype=dtype).element_size()
    page_elements = spec.page_size_bytes // element_size
    nope_segment_bytes = block_size * num_kv_heads * nope_dim * element_size
    storages = set()
    for layer_idx, (layer_name, (nope, rope)) in enumerate(caches.items()):
        expected_shape = (num_blocks, block_size, num_kv_heads)
        assert nope.shape == (*expected_shape, nope_dim)
        assert rope.shape == (*expected_shape, rope_dim)
        assert nope.stride() == (
            page_elements,
            num_kv_heads * nope_dim,
            nope_dim,
            1,
        )
        assert rope.stride() == (
            page_elements,
            num_kv_heads * rope_dim,
            rope_dim,
            1,
        )
        assert nope.storage_offset() == layer_idx * num_blocks * page_elements
        assert rope.storage_offset() == nope.storage_offset() + nope_segment_bytes // element_size
        assert rope.data_ptr() == nope.data_ptr() + nope_segment_bytes
        assert not nope.is_contiguous()
        assert not rope.is_contiguous()
        storages.add(nope.untyped_storage().data_ptr())

    assert len(storages) == 1


def test_typed_storage_view_preserves_nonzero_offset():
    backing = torch.zeros(24, dtype=torch.int8)
    raw = backing[8:]
    typed = _typed_empty_like_storage(raw, torch.float32)
    assert typed.storage_offset() == 2
    assert typed.numel() == 4
    assert typed.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()


def test_allocator_rejects_invalid_descriptor_stride():
    spec = _make_spec(
        block_size=8,
        num_kv_heads=1,
        nope_dim=5,
        rope_dim=3,
        dtype=torch.float32,
    )
    layers = {"layer": _FakeMLAAttention(spec=spec, nope_dim=5, rope_dim=3)}
    config = _make_plan(layers, spec)
    config.kv_cache_tensors[0].block_stride = spec.page_size_bytes + 1
    with pytest.raises(ValueError, match="block stride"):
        allocate_mla_component_cache(
            vllm_config=_make_vllm_config(layers),
            kv_cache_config=config,
            static_forward_context=layers,
            device=torch.device("cpu"),
            kernel_block_sizes=[[8]],
        )


def test_allocator_rejects_non_lbnhc_layout():
    spec = _make_spec(
        block_size=8,
        num_kv_heads=1,
        nope_dim=5,
        rope_dim=3,
        dtype=torch.float32,
    )
    layers = {"layer": _FakeMLAAttention(spec=spec, nope_dim=5, rope_dim=3)}
    config = _make_plan(layers, spec)
    config.kv_cache_layout = KVCacheLayout.LBHNC.name
    with pytest.raises(ValueError, match="LBNHC layout"):
        allocate_mla_component_cache(
            vllm_config=_make_vllm_config(layers),
            kv_cache_config=config,
            static_forward_context=layers,
            device=torch.device("cpu"),
            kernel_block_sizes=[[8]],
        )


def test_allocator_rejects_kernel_block_splitting():
    spec = _make_spec(
        block_size=8,
        num_kv_heads=1,
        nope_dim=5,
        rope_dim=3,
        dtype=torch.float32,
    )
    layers = {"layer": _FakeMLAAttention(spec=spec, nope_dim=5, rope_dim=3)}
    config = _make_plan(layers, spec)
    with pytest.raises(ValueError, match="Manager and kernel block sizes"):
        allocate_mla_component_cache(
            vllm_config=_make_vllm_config(layers),
            kv_cache_config=config,
            static_forward_context=layers,
            device=torch.device("cpu"),
            kernel_block_sizes=[[4, 4]],
        )


def test_allocator_rejects_cross_layer_kv_sharing():
    spec = _make_spec(
        block_size=8,
        num_kv_heads=1,
        nope_dim=5,
        rope_dim=3,
        dtype=torch.float32,
    )
    layer = _FakeMLAAttention(spec=spec, nope_dim=5, rope_dim=3)
    layer.kv_sharing_target_layer_name = "model.layers.0.self_attn"
    layers = {"layer": layer}
    config = _make_plan(layers, spec)
    with pytest.raises(ValueError, match="cross-layer KV sharing"):
        allocate_mla_component_cache(
            vllm_config=_make_vllm_config(layers),
            kv_cache_config=config,
            static_forward_context=layers,
            device=torch.device("cpu"),
            kernel_block_sizes=[[8]],
        )


def test_use_component_cache_capability(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_MLA_COMPONENT_CACHE", "1")
    monkeypatch.setattr(mla_component_cache_v1, "vllm_version_is", lambda _version: False)
    spec = _make_spec(
        block_size=128,
        num_kv_heads=1,
        nope_dim=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    layers = {
        "layer0": _FakeMLAAttention(spec=spec, nope_dim=512, rope_dim=64),
        "layer1": _FakeMLAAttention(spec=spec, nope_dim=512, rope_dim=64),
    }
    assert use_mla_component_cache(_make_vllm_config(layers))


def test_component_cache_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_ENABLE_MLA_COMPONENT_CACHE", raising=False)
    assert not use_mla_component_cache(_make_vllm_config({}))


def test_use_component_cache_rejects_v2(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_MLA_COMPONENT_CACHE", "1")
    with pytest.raises(ValueError, match="ModelRunner V2"):
        use_mla_component_cache(_make_vllm_config({}, use_v2_model_runner=True))


def test_use_component_cache_rejects_vllm_028(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_MLA_COMPONENT_CACHE", "1")
    monkeypatch.setattr(mla_component_cache_v1, "vllm_version_is", lambda _version: True)
    with pytest.raises(ValueError, match="vLLM main"):
        use_mla_component_cache(_make_vllm_config({}))


def test_use_component_cache_rejects_non_mla_model(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_MLA_COMPONENT_CACHE", "1")
    monkeypatch.setattr(mla_component_cache_v1, "vllm_version_is", lambda _version: False)
    with pytest.raises(ValueError, match="no MLA layers"):
        use_mla_component_cache(_make_vllm_config({}))


def test_use_component_cache_rejects_compressed_mla(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_MLA_COMPONENT_CACHE", "1")
    monkeypatch.setattr(mla_component_cache_v1, "vllm_version_is", lambda _version: False)
    with pytest.raises(ValueError, match="compressed MLA"):
        use_mla_component_cache(_make_vllm_config({}, compress_ratios=[1, 4]))


def test_component_cache_env_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_MLA_COMPONENT_CACHE", "true")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        use_mla_component_cache(_make_vllm_config({}))


def test_use_component_cache_rejects_ascend_spec_subclass(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_MLA_COMPONENT_CACHE", "1")
    monkeypatch.setattr(mla_component_cache_v1, "vllm_version_is", lambda _version: False)

    class AscendSpecSubclass(MLAAttentionSpec):
        pass

    spec = AscendSpecSubclass(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        head_size_v=0,
        dtype=torch.bfloat16,
    )
    layers = {"layer": _FakeMLAAttention(spec=spec, nope_dim=512, rope_dim=64)}
    with pytest.raises(ValueError, match="exact upstream MLAAttentionSpec"):
        use_mla_component_cache(_make_vllm_config(layers))


def test_use_component_cache_rejects_mla_nope(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_MLA_COMPONENT_CACHE", "1")
    monkeypatch.setattr(mla_component_cache_v1, "vllm_version_is", lambda _version: False)
    spec = _make_spec(
        block_size=128,
        num_kv_heads=1,
        nope_dim=512,
        rope_dim=0,
        dtype=torch.bfloat16,
    )
    layers = {"layer": _FakeMLAAttention(spec=spec, nope_dim=512, rope_dim=0)}
    with pytest.raises(ValueError, match="MLA-NoPE"):
        use_mla_component_cache(_make_vllm_config(layers))


def test_model_runner_preserves_exact_upstream_mla_spec(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_MLA_COMPONENT_CACHE", "1")
    monkeypatch.setattr(mla_component_cache_v1, "vllm_version_is", lambda _version: False)
    spec = _make_spec(
        block_size=128,
        num_kv_heads=1,
        nope_dim=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    layers = {"layer": _FakeMLAAttention(spec=spec, nope_dim=512, rope_dim=64)}
    runner = model_runner_v1.NPUModelRunner.__new__(model_runner_v1.NPUModelRunner)
    runner.vllm_config = _make_vllm_config(layers)
    runner.shared_kv_cache_layers = {}
    runner.use_compress = False
    runner.use_sparse = False
    runner.sparse_kv_offload_enabled = False

    monkeypatch.setattr(model_runner_v1, "has_ec_transfer", lambda: False)
    assert runner.get_kv_cache_spec() == {"layer": spec}
    assert runner._use_mla_component_cache


def test_component_cow_copies_whole_pages_across_layers(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_MLA_COMPONENT_CACHE", "1")
    block_size, num_blocks, num_layers = 4, 7, 3
    spec = _make_spec(
        block_size=block_size,
        num_kv_heads=2,
        nope_dim=3,
        rope_dim=2,
        dtype=torch.float32,
    )
    layers = {f"layer{index}": _FakeMLAAttention(spec=spec, nope_dim=3, rope_dim=2) for index in range(num_layers)}
    config = _make_plan(layers, spec, num_blocks=num_blocks)
    caches = allocate_mla_component_cache(
        vllm_config=_make_vllm_config(layers),
        kv_cache_config=config,
        static_forward_context=layers,
        device=torch.device("cpu"),
        kernel_block_sizes=[[block_size]],
    )

    page_views = [_component_page_view(*cache) for cache in caches.values()]
    original_pages = [
        [_page_payload(spec.page_size_bytes, layer * num_blocks + block) for block in range(num_blocks)]
        for layer in range(num_layers)
    ]
    nope_segment_bytes = block_size * 2 * 3 * 4
    for layer, page_view in enumerate(page_views):
        for block, payload in enumerate(original_pages[layer]):
            payload[-1] = 0xA5
            payload[nope_segment_bytes] = 0x5A
            page_view[block].copy_(payload)

    # Exercise first, middle, and last source blocks with non-contiguous source
    # and destination IDs. NumPy advanced indexing performs all gathers before
    # the indexed assignment, so overlapping-looking IDs remain well-defined.
    copy_map = {0: 1, 3: 4, 6: 2}
    copies = [KVCacheBlockCopy(src_block_id=src, dst_block_id=dst) for src, dst in copy_map.items()]
    _install_cpu_h2d(monkeypatch)
    copy_kv_cache_blocks_inplace(
        list(caches.values()),
        num_blocks,
        copies,
    )

    for layer, page_view in enumerate(page_views):
        for block in range(num_blocks):
            source_block = copy_map.get(block, block)
            torch.testing.assert_close(page_view[block], original_pages[layer][source_block])
            assert page_view[block][-1] == 0xA5
            assert page_view[block][nope_segment_bytes] == 0x5A


def test_component_cow_preserves_partial_tail_tokens(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_MLA_COMPONENT_CACHE", "1")
    block_size, num_blocks, num_layers = 5, 4, 2
    spec = _make_spec(
        block_size=block_size,
        num_kv_heads=2,
        nope_dim=3,
        rope_dim=2,
        dtype=torch.float32,
    )
    layers = {f"layer{index}": _FakeMLAAttention(spec=spec, nope_dim=3, rope_dim=2) for index in range(num_layers)}
    config = _make_plan(layers, spec, num_blocks=num_blocks)
    caches = allocate_mla_component_cache(
        vllm_config=_make_vllm_config(layers),
        kv_cache_config=config,
        static_forward_context=layers,
        device=torch.device("cpu"),
        kernel_block_sizes=[[block_size]],
    )

    for nope, rope in caches.values():
        nope[:, -1].fill_(1.0)
        rope[:, -1].fill_(2.0)
        nope[:, :-1].zero_()
        rope[:, :-1].zero_()

    _install_cpu_h2d(monkeypatch)
    copy_kv_cache_blocks_inplace(
        list(caches.values()),
        num_blocks,
        [KVCacheBlockCopy(src_block_id=3, dst_block_id=0)],
    )
    for nope, rope in caches.values():
        torch.testing.assert_close(nope[0, -1], torch.full_like(nope[0, -1], 1.0))
        torch.testing.assert_close(rope[0, -1], torch.full_like(rope[0, -1], 2.0))
        torch.testing.assert_close(nope[0, :-1], torch.zeros_like(nope[0, :-1]))
        torch.testing.assert_close(rope[0, :-1], torch.zeros_like(rope[0, :-1]))


def test_legacy_independent_mla_tuple_is_not_component_pair():
    nope = torch.zeros(2, 4, 1, 3)
    rope = torch.zeros(2, 4, 1, 2)
    assert not _is_component_pair((nope, rope))


def test_feature_off_cow_splits_legacy_independent_mla_tuple(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_ENABLE_MLA_COMPONENT_CACHE", raising=False)
    num_blocks, block_size, num_heads = 5, 4, 2
    nope = torch.arange(num_blocks * block_size * num_heads * 3, dtype=torch.float32)
    rope = torch.arange(num_blocks * block_size * num_heads * 2, dtype=torch.float32)
    nope = nope.reshape(num_blocks, block_size, num_heads, 3)
    rope = rope.reshape(num_blocks, block_size, num_heads, 2)
    original_nope = nope.clone()
    original_rope = rope.clone()

    _install_cpu_h2d(monkeypatch)
    copy_kv_cache_blocks_inplace(
        [(nope, rope)],
        num_blocks,
        [KVCacheBlockCopy(src_block_id=3, dst_block_id=0)],
    )
    torch.testing.assert_close(nope[0], original_nope[3])
    torch.testing.assert_close(rope[0], original_rope[3])
    torch.testing.assert_close(nope[1], original_nope[1])
    torch.testing.assert_close(rope[1], original_rope[1])


def test_cow_patch_disabled_feature_uses_upstream_copy(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_ENABLE_MLA_COMPONENT_CACHE", raising=False)
    original_copy = MagicMock()
    monkeypatch.setattr(upstream_utils, "_orig_copy_kv_cache_blocks_inplace", original_copy)
    copies = [KVCacheBlockCopy(src_block_id=1, dst_block_id=0)]
    copy_kv_cache_blocks_inplace([], 2, copies)
    original_copy.assert_called_once_with([], 2, copies)


def test_k3_capability_selects_hybrid_mode(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_MLA_COMPONENT_CACHE", "1")
    monkeypatch.setattr(mla_component_cache_v1, "vllm_version_is", lambda _version: False)
    (
        vllm_config,
        _config,
        _raw_tensors,
        capability,
        _physical_page,
        _element_size,
    ) = _make_k3_hybrid_plan()
    assert capability.mode == "K3_HYBRID_V1"
    assert capability.manager_block_size == 384
    assert capability.kernel_block_size == 128
    assert use_mla_component_cache(vllm_config)


def test_k3_capability_rejects_speculative_decoding(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_MLA_COMPONENT_CACHE", "1")
    monkeypatch.setattr(mla_component_cache_v1, "vllm_version_is", lambda _version: False)
    vllm_config, *_ = _make_k3_hybrid_plan()
    vllm_config.speculative_config = SimpleNamespace(method="dspark")
    with pytest.raises(ValueError, match="speculative decoding"):
        get_mla_component_cache_capability(vllm_config)


@pytest.mark.parametrize("manager_block_size", [384, 768])
def test_materialize_k3_component_cache_kernel_slot_geometry(manager_block_size):
    (
        _vllm_config,
        config,
        raw_tensors,
        capability,
        physical_page,
        element_size,
    ) = _make_k3_hybrid_plan(num_blocks=3, manager_block_size=manager_block_size)
    original_kda_tensors = {name: tensor.clone() for name, tensor in raw_tensors.items() if ".linear_attn" in name}
    caches = materialize_hybrid_mla_component_cache(
        raw_kv_cache_tensors=raw_tensors,
        kv_cache_config=config,
        static_forward_context=_vllm_config.compilation_config.static_forward_context,
        kernel_block_sizes=[[128], [384]],
        capability=capability,
        vllm_config=_vllm_config,
    )

    assert set(caches) == {
        "language_model.model.layers.0.self_attn",
        "language_model.model.layers.1.self_attn",
    }
    ratio = manager_block_size // 128
    slot_bytes = physical_page // ratio
    slot_elements = slot_bytes // element_size
    for nope, rope in caches.values():
        assert nope.shape == (3 * ratio, 128, 1, 512)
        assert rope.shape == (3 * ratio, 128, 1, 64)
        assert nope.stride() == (slot_elements, 512, 512, 1)
        assert rope.stride() == (slot_elements, 64, 64, 1)
        assert rope.data_ptr() == nope.data_ptr() + 128 * 1 * 512 * element_size
        assert nope.stride(0) > 128 * 1 * 512
        assert rope.stride(0) > 128 * 1 * 64

    for name, tensor in original_kda_tensors.items():
        torch.testing.assert_close(raw_tensors[name], tensor)


def test_k3_component_cow_copies_all_kernel_slots_in_manager_block(monkeypatch):
    (
        vllm_config,
        config,
        raw_tensors,
        capability,
        physical_page,
        _element_size,
    ) = _make_k3_hybrid_plan(num_blocks=3, manager_block_size=384)
    caches = materialize_hybrid_mla_component_cache(
        raw_kv_cache_tensors=raw_tensors,
        kv_cache_config=config,
        static_forward_context=vllm_config.compilation_config.static_forward_context,
        kernel_block_sizes=[[128], [384]],
        capability=capability,
        vllm_config=vllm_config,
    )

    slot_bytes = physical_page // 3
    page_views = [_component_page_view(*cache) for cache in caches.values()]
    originals = [
        [_page_payload(slot_bytes, layer * page_views[0].shape[0] + slot) for slot in range(page_views[0].shape[0])]
        for layer in range(len(page_views))
    ]
    for layer, page_view in enumerate(page_views):
        for slot, payload in enumerate(originals[layer]):
            payload[-1] = 0xA5
            payload[128 * 1 * 512 * 2] = 0x5A
            page_view[slot].copy_(payload)

    _install_cpu_h2d(monkeypatch)
    copy_kv_cache_blocks_inplace(
        list(caches.values()),
        3,
        [KVCacheBlockCopy(src_block_id=2, dst_block_id=0)],
    )
    for layer, page_view in enumerate(page_views):
        torch.testing.assert_close(page_view[:3], originals[layer][6:9])
        torch.testing.assert_close(page_view[3:6], originals[layer][3:6])
        torch.testing.assert_close(page_view[6:9], originals[layer][6:9])
