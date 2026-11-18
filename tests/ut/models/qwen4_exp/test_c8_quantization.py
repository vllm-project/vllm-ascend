# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from tests.ut.models.test_qwen4_exp_cache_config import _groups
from vllm_ascend.models.qwen4_exp import qsa
from vllm_ascend.models.qwen4_exp.cache_config import (
    build_qwen4_kv_cache_layout,
    QSA_K_SCALE_REGION,
    make_qsa_c8_kernel_page_cache,
)
from vllm_ascend.models.qwen4_exp.quantization import (
    C8_SCALE_MIN,
    QSAC8Spec,
    QSAStaticVParameters,
    qsa_c8_indexer_cache_views,
    quantize_qsa_rows,
    quantize_qsa_values,
    uses_qsa_c8_quantization,
)


def test_exact_owner_metadata_and_draft_isolation():
    config = SimpleNamespace(
        quant_description={
            "fa_quant_type": "INT8",
            "kv_cache_type": "C8",
            "model.layers.11.self_attn.quant_type": "QK_INT8_DYNAMIC_V_INT8_PER_CHANNEL",
            "model.layers.11.self_attn.indexer.quant_type": "QK_INT8_DYNAMIC",
        }
    )
    assert uses_qsa_c8_quantization(config, "model.layers.11.self_attn")
    assert uses_qsa_c8_quantization(config, "model.layers.11.self_attn.indexer", indexer=True)
    for prefix in ("model.layers.1.self_attn", "model.mtp.layers.11.self_attn", "model.layers.11.self_attn.attn"):
        assert not uses_qsa_c8_quantization(config, prefix)
    assert not uses_qsa_c8_quantization(None, "model.layers.11.self_attn")


@pytest.mark.parametrize("mode", ["INT8_DYNAMIC", "INT8", "QK_INT8_DYNAMIC", "INT4_DYNAMIC", "", None])
def test_unsupported_metadata_fails_at_owner(mode):
    with pytest.raises(ValueError, match="model.layers.11.self_attn"):
        uses_qsa_c8_quantization(
            SimpleNamespace(quant_description={"model.layers.11.self_attn.quant_type": mode}),
            "model.layers.11.self_attn",
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dynamic_rounding_zero_rows_and_per_head_scales(dtype):
    rows = torch.tensor(
        [[[127.0, -127.0, 0.5, 1.5], [254.0, -254.0, -1.0, -3.0]], [[0.0, 0.0, 0.0, 0.0], [1e-9, 0.0, 0.0, 0.0]]],
        dtype=dtype,
    )
    codes, scales = quantize_qsa_rows(rows)
    assert codes.tolist()[0] == [[127, -127, 0, 2], [127, -127, 0, -2]]
    torch.testing.assert_close(scales[0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(scales[1], torch.full((2,), C8_SCALE_MIN))
    assert not codes[1].any()
    restored = codes.float() * scales.unsqueeze(-1)
    assert ((restored - rows.float()).abs() <= scales.unsqueeze(-1) * 0.5001).all()
    empty, empty_scale = quantize_qsa_rows(torch.empty(0, 2, 128, dtype=dtype))
    assert empty.shape == (0, 2, 128) and empty_scale.shape == (0, 2)


def test_indexer_strided_codes_and_scales_roundtrip_and_page_copy():
    raw = torch.full((5, 2, 8, 1, 132), 42, dtype=torch.int8)
    cache = raw[:, 1]
    codes, scales = qsa_c8_indexer_cache_views(cache)
    codes.copy_(torch.arange(5 * 8 * 128).view_as(codes).to(torch.int8))
    scales.copy_(torch.arange(40).view_as(scales) * 0.125)
    assert codes.stride()[1:] == (128, 128, 1)
    assert scales.stride()[1:] == (1, 1)
    assert scales.dtype == torch.float16
    expected_codes, expected_scales = codes[0].clone(), scales[0].clone()
    cache[4].copy_(cache[0])
    torch.testing.assert_close(codes[4], expected_codes)
    torch.testing.assert_close(scales[4], expected_scales)
    assert (raw[:, 0] == 42).all()


def test_c8_spec_counts_scales_and_preserves_logical_dim_through_merge():
    spec = QSAC8Spec(
        block_size=256, num_kv_heads=2, head_size=128, dtype=torch.int8, num_head_slots=2, state_content_bytes=264
    )
    assert spec.page_size_bytes == 2 * 256 * 2 * 132
    assert spec.head_size == 128
    merged = QSAC8Spec.merge([spec, replace(spec)])
    assert isinstance(merged, QSAC8Spec)
    assert qsa.AscendQSABackend.customize_spec(spec) == spec


def test_mixed_c8_bf16_layout_has_bounded_padding_and_disjoint_views():
    groups = _groups()
    main = groups[0]
    # Keep the same logical geometry as the draft, but charge C8 scales.
    c8 = QSAC8Spec(
        block_size=256, num_kv_heads=1, head_size=64, dtype=torch.int8, num_head_slots=2, state_content_bytes=68
    )
    main.kv_cache_spec = UniformTypeKVCacheSpecs(
        block_size=256,
        kv_cache_specs={name: c8 if i < 2 else main.kv_cache_spec for i, name in enumerate(main.layer_names)},
    )
    layout = build_qwen4_kv_cache_layout(groups, 3)
    assert layout is not None
    region = layout.region("r0")
    # Adding width 68 must not introduce an LCM with BF16 width 128.
    assert region.block_stride == 65536
    backing = torch.full((layout.slot_backing_size,), 37, dtype=torch.int8)
    k, v, scales = make_qsa_c8_kernel_page_cache(
        backing,
        spec=c8,
        num_blocks=3,
        storage_offset=region.offset,
        block_stride=region.block_stride,
        scale_offset=layout.region(QSA_K_SCALE_REGION).offset,
    )
    assert k.shape == v.shape == (6, 128, 1, 64)
    assert k.stride(0) == region.block_stride // 2
    assert scales.shape == (6, 1, 128) and scales.is_contiguous()
    k.fill_(3)
    scales.fill_(0.5)
    assert (v == 37).all()
    # Both subpages of one manager block retain the exact virtual page IDs.
    assert k[2].data_ptr() - k[0].data_ptr() == region.block_stride
    assert scales[-1, 0, -1] == 0.5


def test_c8_dispatch_bypasses_bf16_fia(monkeypatch):
    impl = object.__new__(qsa.AscendQSAImpl)
    layer = SimpleNamespace(
        c8_enabled=True,
        fa_v=SimpleNamespace(scale=torch.ones(1, 128)),
        c8_p_scale=torch.tensor([127.0]),
        indexer=SimpleNamespace(token_topk=2048, compress_ratio=4),
        topk_indices_buffer=torch.zeros(2, 2051, dtype=torch.int32),
    )
    cache = torch.zeros(1, 128, 1, 128, dtype=torch.int8)
    scales = torch.ones(1, 1, 128)
    output = torch.ones(3, 2, 128, dtype=torch.bfloat16)
    callback = Mock(return_value=output[:2])
    monkeypatch.setattr(qsa, "qsa_c8_attention", callback)
    monkeypatch.setattr(qsa, "dispatch_qsa_attention", Mock(side_effect=AssertionError("BF16 FIA used")))
    metadata = SimpleNamespace(num_actual_tokens=2, block_table=torch.tensor([[0]]), qsa_request_plan=object())
    impl._dispatch_qsa(
        layer,
        output,
        None,
        None,
        (cache, cache, scales),
        metadata,
        output,
        torch.zeros(3, dtype=torch.int32),
        torch.arange(3),
        torch.tensor([3]),
    )
    assert callback.call_count == 1
    assert callback.call_args.args[3].shape == (2, 515)
    assert callback.call_args.kwargs["key_scales"] is scales
    assert not output[2].any()


def test_c8_cache_update_keeps_scale_and_codes_in_one_transaction(monkeypatch):
    callback = Mock()
    monkeypatch.setattr(qsa, "store_qsa_c8_rows", callback)
    impl = object.__new__(qsa.AscendQSAImpl)
    key, value = torch.randn(2, 1, 128), torch.randn(2, 1, 128)
    cache = (torch.empty(1, 128, 1, 128, dtype=torch.int8), torch.empty(1, 128, 1, 128, dtype=torch.int8),
             torch.empty(1, 1, 128))
    slots = torch.tensor([0, -1])
    v_scale = torch.ones(1, 128)
    impl.do_kv_cache_update(
        SimpleNamespace(c8_enabled=True, fa_v=SimpleNamespace(scale=v_scale)), key, value, cache, slots
    )
    assert callback.call_args_list[0].args == (cache[0], slots, key)
    assert callback.call_args_list[0].kwargs["key_scales"] is cache[2]
    assert callback.call_args_list[1].args == (cache[1], slots, value)
    assert callback.call_args_list[1].kwargs["v_scale"] is v_scale
    with pytest.raises(ValueError, match="requires K, V and K scales"):
        impl.do_kv_cache_update(SimpleNamespace(c8_enabled=True), key, value, cache[:2], slots)



@pytest.mark.parametrize("all_c8", [False, True])
@pytest.mark.parametrize("manager_size", [256, 384, 768, 896])
def test_c8_runner_allocation_binding_and_pool_budget(monkeypatch, all_c8, manager_size):
    from tests.ut.models.test_qwen4_exp_cache_config import _allocate_mrv2_kv_cache
    from vllm_ascend.models.qwen4_exp.cache_config import (
        get_qwen4_exp_kv_cache_config,
        get_qwen4_exp_pool_bytes_per_block,
    )
    from vllm_ascend.patch.worker.patch_bind_kv_cache import bind_kv_cache_to_layers

    groups = _groups()
    for group in groups:
        if group is not groups[1]:
            group.kv_cache_spec = replace(group.kv_cache_spec, block_size=manager_size)
    for group_id, compressed in ((0, False), (2, True)):
        group = groups[group_id]
        specs = {}
        for name in group.layer_names:
            original = group.kv_cache_spec
            if all_c8 or ".mtp." not in name:
                if compressed:
                    spec = replace(original, dtype=torch.int8, head_size=original.head_size + 4)
                else:
                    spec = QSAC8Spec(
                        block_size=manager_size,
                        num_kv_heads=original.num_kv_heads,
                        head_size=original.head_size,
                        dtype=torch.int8,
                        num_head_slots=2,
                        state_content_bytes=original.num_kv_heads * (original.head_size + 4),
                    )
                specs[name] = spec
            else:
                specs[name] = original
        group.kv_cache_spec = UniformTypeKVCacheSpecs(block_size=manager_size, kv_cache_specs=specs)
    config = get_qwen4_exp_kv_cache_config(
        SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks_override=3, prefix_cache_retention_interval=None)),
        groups,
        10**9,
    )
    raw, caches, layout = _allocate_mrv2_kv_cache(config, monkeypatch, use_main=True)
    allocations = {t.untyped_storage().data_ptr(): t.numel() for t in raw.values()}
    assert sum(allocations.values()) == 3 * get_qwen4_exp_pool_bytes_per_block(config.kv_cache_groups)
    for name in groups[0].layer_names:
        k, v = caches[name][:2]
        is_c8 = all_c8 or ".mtp." not in name
        assert k.dtype == (torch.int8 if is_c8 else torch.bfloat16)
        assert k.shape == v.shape == (3 * manager_size // 128, 128, 1, 64)
        assert len(caches[name]) == (3 if is_c8 else 2)
        owner = SimpleNamespace(kv_cache=None)
        bind_kv_cache_to_layers({name: caches[name]}, {name: owner})
        assert owner.kv_cache is caches[name]
        if is_c8:
            scales = caches[name][2]
            assert scales.shape == (k.shape[0], 1, 128)
            assert scales.dtype == torch.float32 and scales.is_contiguous()
            scales.fill_(0.25)
            k.fill_(7)
            v.fill_(9)
            assert (scales == 0.25).all()
    for name in groups[2].layer_names:
        cache = caches[name]
        is_c8 = all_c8 or ".mtp." not in name
        assert cache.shape == (3 * manager_size // 128, 32, 1, 36 if is_c8 else 32)
        assert cache.stride(0) * cache.element_size() == layout.region("r5").block_stride * 128 // manager_size
        if is_c8:
            codes, scales = qsa_c8_indexer_cache_views(cache)
            codes.fill_(4)
            scales.fill_(0.125)
            assert scales[-1, -1, 0] == 0.125


def test_modelslim_mapper_preserves_dynamic_owner_keys():
    from vllm.model_executor.models.utils import WeightsMapper

    from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig

    config = AscendModelSlimConfig(
        {
            "model.language_model.layers.11.self_attn.quant_type": "QK_INT8_DYNAMIC_V_INT8_PER_CHANNEL",
            "model.language_model.layers.11.self_attn.indexer.quant_type": "QK_INT8_DYNAMIC",
        }
    )
    config.apply_vllm_mapper(WeightsMapper(orig_to_new_prefix={"model.language_model.": "language_model.model."}))
    assert uses_qsa_c8_quantization(config, "language_model.model.layers.11.self_attn")
    assert uses_qsa_c8_quantization(config, "language_model.model.layers.11.self_attn.indexer", indexer=True)
    assert not uses_qsa_c8_quantization(config, "mtp.layers.48.self_attn")


def test_c8_owner_constructor_disables_direct_bf16_cache_prepare(monkeypatch):
    def initialize(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.head_dim, self.num_kv_heads, self.total_num_kv_heads, self.kv_cache_dtype = 128, 1, 1, "auto"
        self.use_fused_qsa_prepare = True

    monkeypatch.setattr(qsa.upstream_qsa.Qwen4ExpQSAAttention, "__init__", initialize)
    monkeypatch.setattr(qsa, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(qsa, "get_tensor_model_parallel_rank", lambda: 0)
    config = SimpleNamespace(
        quant_description={"model.layers.0.self_attn.quant_type": "QK_INT8_DYNAMIC_V_INT8_PER_CHANNEL"}
    )
    owner = qsa.AscendQwen4ExpQSAAttention(quant_config=config, prefix="model.layers.0.self_attn")
    assert owner.c8_enabled and not owner.use_fused_qsa_prepare
    spec = owner.get_kv_cache_spec(SimpleNamespace(cache_config=SimpleNamespace(block_size=256)))
    assert isinstance(spec, QSAC8Spec) and spec.page_size_bytes == 256 * 2 * 132
    draft = qsa.AscendQwen4ExpQSAAttention(quant_config=config, prefix="mtp.layers.48.self_attn")
    assert not draft.c8_enabled and draft.use_fused_qsa_prepare


def test_c8_indexer_constructor_keeps_raw_ring_bf16(monkeypatch):
    def initialize(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.index_head_dim, self.index_n_heads = 128, 4
        self.raw_key_cache = SimpleNamespace(dtype=torch.bfloat16, head_size=128)
        self.compressed_key_cache = SimpleNamespace(dtype=torch.bfloat16, head_size=128)
        self.indexer_dtype = torch.bfloat16
        self.use_fused_pre_indexer = True

    monkeypatch.setattr(qsa.upstream_indexer.QSAIndexer, "__init__", initialize)
    config = SimpleNamespace(quant_description={"model.layers.0.self_attn.indexer.quant_type": "QK_INT8_DYNAMIC"})
    indexer = qsa.AscendQSAIndexer(quant_config=config, prefix="model.layers.0.self_attn.indexer")
    assert not indexer.use_fused_pre_indexer
    assert indexer.compressed_key_cache.dtype == torch.int8
    assert indexer.compressed_key_cache.head_size == 132
    assert indexer.raw_key_cache.dtype == torch.bfloat16
    assert indexer.raw_key_cache.head_size == 128


def test_c8_device_interfaces_reject_cpu_workloads():
    from vllm_ascend.ops.triton.qwen4_exp.c8 import store_qsa_c8_rows

    cache = torch.zeros(1, 128, 1, 128, dtype=torch.int8)
    slots, rows = torch.tensor([0]), torch.zeros(1, 1, 128)
    with pytest.raises(ValueError, match="same NPU"):
        store_qsa_c8_rows(cache, slots, rows, key_scales=torch.empty(1, 1, 128))
    with pytest.raises(ValueError, match="K scales"):
        store_qsa_c8_rows(cache, slots, rows)


@pytest.mark.parametrize("total_heads,tp_size", [(8, 1), (8, 4), (2, 8), (1, 8)])
def test_static_v_tp_shards_and_replica_groups(total_heads, tp_size):
    values = torch.arange(1, total_heads * 4 + 1, dtype=torch.float32).reshape(-1, 1)
    for rank in range(tp_size):
        module = QSAStaticVParameters(total_heads, 4, tp_size, rank)
        module.scale.weight_loader(module.scale, values)
        module.offset.weight_loader(module.offset, torch.zeros_like(values, dtype=torch.int8))
        module.process_weights_after_loading()
        local = max(1, total_heads // tp_size)
        start = (rank // max(1, tp_size // total_heads)) * local
        torch.testing.assert_close(module.scale, values.reshape(total_heads, 4)[start : start + local])
        assert module.scale.dtype == torch.float32


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
def test_static_v_rejects_invalid_or_missing_scales(bad):
    module = QSAStaticVParameters(1, 4, 1, 0)
    with pytest.raises(ValueError, match="Missing"):
        module.process_weights_after_loading()
    with pytest.raises(ValueError, match="finite and positive"):
        module.load_scale(module.scale, torch.full((4, 1), bad))
    with pytest.raises(ValueError, match="zero offset"):
        module.load_offset(module.offset, torch.ones(4, 1, dtype=torch.int8))
    with pytest.raises(ValueError, match="global KV"):
        module.load_scale(module.scale, torch.ones(3))
    with pytest.raises(ValueError, match="FP32"):
        module.load_scale(module.scale, torch.ones(4, dtype=torch.uint8))


def test_static_v_quantization_uses_channel_scales_and_saturates():
    rows = torch.tensor([[[0.5, 3.0, -512.0, 512.0]], [[1.5, -3.0, 0.0, 0.0]]])
    scale = torch.tensor([[1.0, 2.0, 2.0, 2.0]])
    assert quantize_qsa_values(rows, scale).tolist() == [[[0, 2, -128, 127]], [[2, -2, 0, 0]]]


def test_modelslim_static_v_mapper_and_actual_auto_loader():
    from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

    from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig

    config = AscendModelSlimConfig(
        {
            "fa_quant_type": "FAKQuant",
            "model.language_model.layers.3.self_attn.quant_type": "QK_INT8_DYNAMIC_V_INT8_PER_CHANNEL",
        }
    )
    config.apply_vllm_mapper(WeightsMapper(orig_to_new_prefix={"model.language_model.": "layers_root."}))
    name = "layers_root.layers.3.self_attn.fa_v.scale"
    assert config.get_cache_scale_mapper().map_name(name) == name
    assert not config.enable_fa_quant
    model = torch.nn.Module()
    model.quant_config = config
    model.add_module("self_attn", torch.nn.Module())
    model.self_attn.add_module("fa_v", QSAStaticVParameters(2, 4, 4, 3))
    loaded = AutoWeightsLoader(model).load_weights(
        [
            ("self_attn.fa_v.scale", torch.arange(1, 9, dtype=torch.float32).reshape(8, 1)),
            ("self_attn.fa_v.offset", torch.zeros(8, 1, dtype=torch.int8)),
        ]
    )
    assert loaded == {"self_attn.fa_v.scale", "self_attn.fa_v.offset"}
    model.self_attn.fa_v.process_weights_after_loading()
    torch.testing.assert_close(model.self_attn.fa_v.scale, torch.tensor([[5.0, 6.0, 7.0, 8.0]]))
