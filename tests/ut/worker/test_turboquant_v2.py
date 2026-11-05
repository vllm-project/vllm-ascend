# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.attention import dsa_v1
from vllm_ascend.attention.dsa_v1 import AscendDSAC4Backend, AscendDSAImpl
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.quantization.methods.kv_cache.turboquant import cache
from vllm_ascend.worker.v2 import attn_utils


def _runtime_views(raw, spec, num_blocks, *, block_stride=None):
    from vllm_ascend.worker.v2.attn_utils import create_runtime_component_views

    shapes = AscendDSAC4Backend.get_runtime_shapes(spec, num_blocks, None)
    dtypes = (spec.dtype, spec.scale_dtype) if getattr(spec, "scale_dim", 0) else (spec.dtype,)
    return create_runtime_component_views(
        raw,
        shapes,
        dtypes,
        spec.page_size_bytes,
        spec.page_size_bytes if block_stride is None else block_stride,
    )


@pytest.mark.parametrize("num_tokens", [None, 0, 2])
def test_non_turboquant_cache_write_preserves_scatter_call(monkeypatch, num_tokens):
    scatter = Mock()
    monkeypatch.setattr(dsa_v1, "get_dsa_attn_kv_plan", lambda _: SimpleNamespace(dsa_kv_compress_scatter=scatter))
    impl = SimpleNamespace(turboquant=None, vllm_config=object())
    raw_cache = torch.empty((1, 32, 1, 512), dtype=torch.bfloat16)
    kv = None if num_tokens is None else torch.empty((num_tokens, 1, 512), dtype=torch.bfloat16)
    slots = torch.empty((num_tokens or 0, 2), dtype=torch.int32)
    AscendDSAImpl._write_kv_cache(impl, raw_cache, kv, slots, quantize=True)
    scatter.assert_called_once_with(raw_cache, kv, slots)


def test_turboquant_cache_write_uses_padding_safe_writer(monkeypatch):
    packed = torch.empty((2, 1, 258), dtype=torch.uint8)
    transform = SimpleNamespace(compress=Mock(return_value=packed), forward=Mock())
    impl = SimpleNamespace(turboquant=transform, vllm_config=object())
    raw_cache = torch.empty((1, 32, 1, 258), dtype=torch.uint8)
    kv = torch.empty((2, 1, 512), dtype=torch.bfloat16)
    slots = torch.tensor([[0, 0], [-1, -1]], dtype=torch.int32)
    writer = Mock()
    monkeypatch.setattr(dsa_v1, "write_dsa_cache", writer)

    AscendDSAImpl._write_kv_cache(impl, raw_cache, kv, slots, quantize=True)

    transform.compress.assert_called_once_with(kv)
    writer.assert_called_once()
    cache_arg, update_arg, slot_arg = writer.call_args.args
    assert cache_arg.dtype == torch.int8
    assert update_arg.dtype == torch.int8
    assert slot_arg is slots


def test_turboquant_swa_cache_write_rotates_without_compressing(monkeypatch):
    kv = torch.empty((2, 1, 512), dtype=torch.bfloat16)
    rotated = torch.empty_like(kv)
    transform = SimpleNamespace(compress=Mock(), forward=Mock(return_value=rotated))
    impl = SimpleNamespace(turboquant=transform, vllm_config=object())
    raw_cache = torch.empty((1, 32, 1, 512), dtype=torch.bfloat16)
    slots = torch.tensor([[0, 0], [-1, -1]], dtype=torch.int32)
    writer = Mock()
    monkeypatch.setattr(dsa_v1, "write_dsa_cache", writer)

    AscendDSAImpl._write_kv_cache(impl, raw_cache, kv, slots, quantize=False)

    transform.forward.assert_called_once_with(kv)
    transform.compress.assert_not_called()
    writer.assert_called_once_with(raw_cache, rotated, slots)


def test_dsv4_cache_view_validates_backing_size_in_bytes():
    spec = AscendMLAAttentionSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        model_version="deepseek_v4",
        cache_dtype_str="bfloat16",
    )
    config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=spec)],
    )
    raw = torch.empty(
        spec.page_size_bytes // torch.empty((), dtype=torch.bfloat16).element_size(),
        dtype=torch.bfloat16,
    )

    views = _runtime_views(raw, spec, config.num_blocks)

    assert views[0].nbytes == spec.page_size_bytes


def test_non_turboquant_v2_rejects_noncontiguous_pages_before_allocation(monkeypatch):
    spec = AscendMLAAttentionSpec(
        block_size=128,
        tokens_per_state=4,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        model_version="deepseek_v4",
        cache_dtype_str="bfloat16",
    )
    descriptor = KVCacheTensor(
        size=4 * spec.page_size_bytes,
        layers=["attn"],
        offset=0,
        layer_stride=spec.page_size_bytes,
        block_stride=2 * spec.page_size_bytes,
    )
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[descriptor],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=spec)],
    )
    runtime = SimpleNamespace(
        additional_config={},
        kv_transfer_config=None,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(compress_ratios=[4])),
    )
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: runtime)
    allocate = Mock(side_effect=AssertionError("Invalid pages must fail before allocation"))
    monkeypatch.setattr(attn_utils, "_allocate_int8_cache_tensor", allocate)
    with pytest.raises(ValueError, match="requires contiguous per-layer pages"):
        attn_utils._allocate_kv_cache(config, shared_layers={}, device=torch.device("cpu"))
    allocate.assert_not_called()


def test_tq_v2_uses_one_backing_and_preserves_strided_pages(monkeypatch):
    specs = {
        f"model.layers.{i}.attn": AscendMLAAttentionSpec(
            block_size=128,
            tokens_per_state=4,
            num_kv_heads=1,
            head_size=258,
            dtype=torch.uint8,
            model_version="deepseek_v4",
            cache_dtype_str="turboquant_4bit_nc",
        )
        for i in range(3)
    }
    # A larger page in another group allows several C4 pages in the same
    # physical slot, exercising both nonzero offsets and a padded stride.
    swa = AscendMLAAttentionSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        model_version="deepseek_v4",
    )
    groups = [
        KVCacheGroupSpec(layer_names=list(specs), kv_cache_spec=UniformTypeKVCacheSpecs.from_specs(specs)),
        KVCacheGroupSpec(layer_names=["swa"], kv_cache_spec=UniformTypeKVCacheSpecs.from_specs({"swa": swa})),
    ]
    runtime = SimpleNamespace(
        additional_config={},
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        kv_transfer_config=None,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(compress_ratios=[4])),
    )
    num_blocks, descriptors = cache.cache_config(runtime, groups, cache.pool_bytes_per_block(groups) * 3)
    config = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=descriptors, kv_cache_groups=groups)
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: runtime)
    allocations = []

    def allocate(size, alignment, device):
        allocations.append(size)
        return torch.zeros(size, dtype=torch.int8)

    monkeypatch.setattr(attn_utils, "_allocate_int8_cache_tensor", allocate)
    raw = attn_utils._allocate_kv_cache(config, shared_layers={}, device=torch.device("cpu"))
    assert allocations == [descriptors[0].size]
    views = {}
    for d in descriptors:
        name = d.layers[0]
        if name == "swa":
            continue
        views[name] = _runtime_views(raw[name], specs[name], config.num_blocks, block_stride=d.block_stride)[0]
        assert views[name].shape == (3, 32, 1, 258)
        assert views[name].stride(0) == d.block_stride
    assert len({v.untyped_storage().data_ptr() for v in views.values()}) == 1
    for index, view in enumerate(views.values(), start=1):
        view[1].fill_(index)
    for index, view in enumerate(views.values(), start=1):
        assert (view[1] == index).all()
        assert (view[0] == 0).all() and (view[2] == 0).all()


@pytest.mark.parametrize("turboquant", [False, True])
@pytest.mark.parametrize("kv_transfer", [False, True])
def test_unified_dsv4_matches_existing_allocation_and_views(monkeypatch, turboquant, kv_transfer):
    """The MRV2 route preserves backing alignment, page packing and group aliases."""
    names = ["model.layers.0.attn", "model.layers.1.attn"]
    alias_name = "model.layers.2.attn"
    spec = AscendMLAAttentionSpec(
        block_size=128,
        tokens_per_state=4,
        num_kv_heads=1,
        head_size=258 if turboquant else 512,
        dtype=torch.uint8 if turboquant else torch.bfloat16,
        model_version="deepseek_v4",
        cache_dtype_str="turboquant_4bit_nc" if turboquant else "bfloat16",
    )
    blocks = 3
    stride = spec.page_size_bytes * (2 if turboquant else 1)
    layer_span = blocks * stride
    pool_size = 2 * layer_span
    config = KVCacheConfig(
        num_blocks=blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=pool_size, layers=names, offset=0, layer_stride=layer_span, block_stride=stride),
            KVCacheTensor(size=pool_size, layers=[alias_name], offset=0, layer_stride=0, block_stride=stride),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(names, UniformTypeKVCacheSpecs.from_specs(dict.fromkeys(names, spec))),
            KVCacheGroupSpec([alias_name], UniformTypeKVCacheSpecs.from_specs({alias_name: spec})),
        ],
    )
    runtime = SimpleNamespace(
        additional_config={},
        kv_transfer_config=object() if kv_transfer else None,
        cache_config=SimpleNamespace(cache_dtype=spec.cache_dtype_str),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(compress_ratios=[4]),
            hf_text_config=SimpleNamespace(model_type="deepseek_v4"),
        ),
    )
    layers = {name: SimpleNamespace(get_attn_backend=lambda: AscendDSAC4Backend) for name in (*names, alias_name)}
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: runtime)
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_args: layers)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda *_args: False)
    raw = attn_utils._allocate_kv_cache(config, {}, torch.device("cpu"))
    groups = attn_utils._get_cache_attention_groups(config, layers, {})
    expected = attn_utils.create_kv_cache_runtime_views(config, raw, groups, [128, 128], {})
    legacy_allocate = Mock(side_effect=AssertionError("Unified route must allocate its own backing"))
    monkeypatch.setattr(attn_utils, "_allocate_kv_cache", legacy_allocate)
    monkeypatch.setattr(
        attn_utils,
        "_reshape_kv_cache_v2",
        Mock(side_effect=AssertionError("V4 must not use legacy reshape")),
    )
    original_allocate = attn_utils._allocate_int8_cache_tensor
    allocated = []

    def allocate(size, alignment, device):
        buf = original_allocate(size, alignment, device)
        allocated.append(buf)
        return buf

    monkeypatch.setattr(attn_utils, "_allocate_int8_cache_tensor", allocate)
    actual = attn_utils.allocate_kv_cache_main(config, torch.device("cpu"), None, [128, 128])
    assert len(allocated) == 1 and allocated[0].numel() == pool_size
    if kv_transfer:
        assert allocated[0].data_ptr() % (2 * 1024 * 1024) == 0
    for name in actual:
        for before, after in zip(expected[name], actual[name]):
            assert before.shape == after.shape
            assert before.stride() == after.stride()
            assert before.dtype == after.dtype
            assert before.data_ptr() - raw[name].data_ptr() == after.data_ptr() - allocated[0].data_ptr() - (
                layer_span if name == names[1] else 0
            )
    assert actual[names[0]][0].data_ptr() == actual[alias_name][0].data_ptr()
    assert actual[names[1]][0].data_ptr() - actual[names[0]][0].data_ptr() == layer_span
    actual[names[0]][0][1].fill_(3)
    actual[names[1]][0][2].fill_(5)
    oracle = torch.zeros(pool_size, dtype=torch.int8)
    first = stride
    second = layer_span + 2 * stride
    oracle[first : first + spec.real_page_size_bytes].view(spec.dtype).fill_(3)
    oracle[second : second + spec.real_page_size_bytes].view(spec.dtype).fill_(5)
    torch.testing.assert_close(allocated[0], oracle)


@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.float32])
def test_dsa_runtime_shapes_are_byte_shapes_from_storage_spec(monkeypatch, scale_dtype):
    spec = AscendMLAAttentionSpec(
        block_size=128,
        tokens_per_state=4,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=scale_dtype,
        model_version="deepseek_v4",
    )
    monkeypatch.setattr(
        AscendDSAC4Backend,
        "get_kv_cache_shape",
        Mock(side_effect=AssertionError("Runtime shapes must derive from the spec")),
    )
    assert AscendDSAC4Backend.get_runtime_shapes(spec, 3, None, 128) == (
        (3, 1, 32, 128),
        (3, 1, 32, scale_dtype.itemsize),
    )
    raw = torch.zeros(3 * spec.page_size_bytes, dtype=torch.int8)
    components = _runtime_views(raw, spec, 3)
    assert components[0].shape == (3, 32, 1, 128)
    assert components[1].shape == (3, 32, 1, 1)
