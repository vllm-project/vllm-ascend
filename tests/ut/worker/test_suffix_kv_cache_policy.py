# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm import envs as vllm_envs
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    KVQuantMode,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention import utils as attention_utils
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendC8AttentionBackendImpl,
)
from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionDCPImpl
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.worker.v2 import attn_utils


@pytest.fixture
def suffix_domain(monkeypatch):
    config = SimpleNamespace(
        use_v2_model_runner=True,
        additional_config={},
        speculative_config=SimpleNamespace(
            method="suffix",
            num_speculative_tokens=4,
            enable_adaptive_verification=False,
            num_speculative_tokens_per_batch_size=None,
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="qwen3"),
            hf_config=SimpleNamespace(model_type="qwen3"),
            runner_type="generate",
            dtype=torch.bfloat16,
            quantization=None,
            use_mla=False,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_expert_parallel=False,
        ),
        cache_config=SimpleNamespace(
            block_size=128,
            cache_dtype="auto",
            kv_offloading_size=None,
            kv_sharing_fast_prefill=False,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        quant_config=None,
        kv_transfer_config=None,
    )
    ascend = SimpleNamespace(
        sparse_kv_offload_config=SimpleNamespace(enabled=False),
        dynamic_spec_config=SimpleNamespace(method=None),
        pa_shape_list=[],
    )
    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl.vllm_config = config
    impl.attn_type = AttentionType.DECODER
    impl.sliding_window = None
    impl.alibi_slopes = None
    impl.sinks = None
    impl.enable_c8_quant = False
    impl.kv_cache_dtype = torch.bfloat16
    impl.use_pcp = False
    impl.is_pcp_decode_sharded = False
    impl.kv_sharing_target_layer_name = None
    impl.kv_cache_layout = None
    impl.num_heads, impl.num_kv_heads, impl.head_size = 32, 8, 128
    layer = SimpleNamespace(
        layer_name="model.layers.0.self_attn.attn",
        kv_sharing_target_layer_name=None,
        num_heads=32,
        impl=impl,
        get_attn_backend=lambda: AscendAttentionBackend,
    )
    spec = FullAttentionSpec(block_size=128, num_kv_heads=8, head_size=128, dtype=torch.bfloat16)
    monkeypatch.setattr(attention_utils, "get_ascend_config", lambda: ascend)
    monkeypatch.setattr(
        attention_utils, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A3)
    )
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    return SimpleNamespace(config=config, ascend=ascend, impl=impl, layer=layer, spec=spec)


def _field_owner(domain, path):
    *parts, field = path.split(".")
    owner = domain
    for part in parts:
        owner = getattr(owner, part)
    return owner, field


@pytest.mark.parametrize("layout", [None, "NHD", "HND", "LBNHC", "LBHNC"])
@pytest.mark.parametrize("cache_dtype", ["auto", "bfloat16"])
def test_suffix_policy_selects_only_fia_not_pa(suffix_domain, layout, cache_dtype):
    domain = suffix_domain
    domain.impl.kv_cache_layout = layout
    domain.config.cache_config.cache_dtype = cache_dtype
    assert attention_utils.is_suffix_fia_cache_enabled(domain.config, domain.impl)
    assert attention_utils.requires_contiguous_suffix_kv_cache(domain.layer, domain.config, domain.spec)
    assert not attention_utils.using_paged_attention(None, domain.config, domain.spec.head_size)
    assert not attention_utils.using_paged_attention(32, domain.config, domain.spec.head_size)
    assert not attention_utils.requires_contiguous_pa_kv_cache(domain.layer, domain.config, domain.spec)


@pytest.mark.parametrize("device_type", [AscendDeviceType.A2, AscendDeviceType.A5, AscendDeviceType._310P])
def test_suffix_policy_rejects_unqualified_hardware(suffix_domain, monkeypatch, device_type):
    monkeypatch.setattr(attention_utils, "get_current_hardware_profile", lambda: get_hardware_profile(device_type))
    assert not attention_utils.is_suffix_fia_cache_enabled(suffix_domain.config, suffix_domain.impl)


def test_suffix_policy_falls_back_to_hf_config(suffix_domain):
    del suffix_domain.config.model_config.hf_text_config
    assert attention_utils.is_suffix_fia_cache_enabled(suffix_domain.config, suffix_domain.impl)
    del suffix_domain.config.model_config.hf_config
    assert not attention_utils.is_suffix_fia_cache_enabled(suffix_domain.config, suffix_domain.impl)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("config.use_v2_model_runner", False),
        ("config.speculative_config", None),
        ("config.speculative_config.method", "eagle"),
        ("config.speculative_config.num_speculative_tokens", 3),
        ("config.speculative_config.enable_adaptive_verification", True),
        ("config.speculative_config.num_speculative_tokens_per_batch_size", [(1, 128, 4)]),
        ("config.speculative_config.num_speculative_tokens_per_batch_size", []),
        ("config.model_config.hf_text_config.model_type", "qwen3_moe"),
        ("config.model_config.hf_text_config.index_topk", 2048),
        ("config.model_config.runner_type", "pooling"),
        ("config.model_config.dtype", torch.float16),
        ("config.model_config.use_mla", True),
        ("config.model_config.quantization", "ascend"),
        ("config.quant_config", SimpleNamespace(enable_c8_quant=True)),
        ("config.compilation_config", None),
        ("config.kv_transfer_config", SimpleNamespace()),
        ("config.offload_config.uva.cpu_offload_gb", 1),
        ("config.offload_config.prefetch.offload_group_size", 1),
        ("config.parallel_config.tensor_parallel_size", 2),
        ("config.parallel_config.pipeline_parallel_size", 2),
        ("config.parallel_config.data_parallel_size", 2),
        ("config.parallel_config.decode_context_parallel_size", 2),
        ("config.parallel_config.prefill_context_parallel_size", 2),
        ("config.parallel_config.enable_expert_parallel", True),
        ("config.cache_config.block_size", 64),
        ("config.cache_config.cache_dtype", "fp8"),
        ("config.cache_config.cache_dtype", "mxfp8"),
        ("config.cache_config.cache_dtype", "int8"),
        ("config.cache_config.kv_offloading_size", 1),
        ("config.cache_config.kv_sharing_fast_prefill", True),
        ("ascend.sparse_kv_offload_config.enabled", True),
        ("ascend.dynamic_spec_config.method", "dspark"),
        ("ascend.pa_shape_list", [32]),
        ("ascend.pa_shape_list", None),
        ("impl.attn_type", AttentionType.ENCODER_DECODER),
        ("impl.sliding_window", 4096),
        ("impl.alibi_slopes", torch.zeros(32)),
        ("impl.sinks", torch.zeros(32)),
        ("impl.enable_c8_quant", True),
        ("impl.kv_cache_dtype", torch.float16),
        ("impl.use_pcp", True),
        ("impl.is_pcp_decode_sharded", True),
        ("impl.kv_sharing_target_layer_name", "model.layers.1.self_attn.attn"),
        ("impl.kv_cache_layout", "BLHNC"),
        ("impl.num_heads", 16),
        ("impl.num_kv_heads", 4),
        ("impl.head_size", 64),
    ],
)
def test_suffix_policy_rejects_unsupported_domain(suffix_domain, path, value):
    owner, field = _field_owner(suffix_domain, path)
    setattr(owner, field, value)
    assert not attention_utils.is_suffix_fia_cache_enabled(suffix_domain.config, suffix_domain.impl)
    assert not attention_utils.requires_contiguous_suffix_kv_cache(
        suffix_domain.layer, suffix_domain.config, suffix_domain.spec
    )


@pytest.mark.parametrize(
    "path",
    [
        "config.use_v2_model_runner",
        "config.speculative_config.enable_adaptive_verification",
        "config.speculative_config.num_speculative_tokens_per_batch_size",
        "config.model_config.hf_text_config.model_type",
        "config.model_config.quantization",
        "config.quant_config",
        "config.kv_transfer_config",
        "config.offload_config",
        "config.parallel_config.tensor_parallel_size",
        "config.parallel_config.pipeline_parallel_size",
        "config.parallel_config.data_parallel_size",
        "config.parallel_config.decode_context_parallel_size",
        "config.parallel_config.prefill_context_parallel_size",
        "config.parallel_config.enable_expert_parallel",
        "config.cache_config.kv_offloading_size",
        "ascend.sparse_kv_offload_config.enabled",
        "ascend.dynamic_spec_config.method",
        "ascend.pa_shape_list",
        "impl.sliding_window",
        "impl.kv_sharing_target_layer_name",
        "impl.kv_cache_layout",
    ],
)
def test_suffix_policy_missing_prerequisite_is_not_eligible(suffix_domain, path):
    owner, field = _field_owner(suffix_domain, path)
    delattr(owner, field)
    assert not attention_utils.is_suffix_fia_cache_enabled(suffix_domain.config, suffix_domain.impl)


@pytest.mark.parametrize("impl_type", [AscendC8AttentionBackendImpl, AscendAttentionDCPImpl])
def test_suffix_policy_does_not_select_specialized_implementations(suffix_domain, impl_type):
    impl = impl_type.__new__(impl_type)
    impl.__dict__.update(vars(suffix_domain.impl))
    assert not attention_utils.is_suffix_fia_cache_enabled(suffix_domain.config, impl)


def test_suffix_policy_unreadable_quantization_is_not_eligible(suffix_domain):
    class UnreadableModel(SimpleNamespace):
        @property
        def quantization(self):
            raise RuntimeError("quantization not initialized")

    fields = vars(suffix_domain.config.model_config).copy()
    del fields["quantization"]
    suffix_domain.config.model_config = UnreadableModel(**fields)
    assert not attention_utils.is_suffix_fia_cache_enabled(suffix_domain.config, suffix_domain.impl)


@pytest.mark.parametrize(
    "changes",
    [
        {"dtype": torch.float16},
        {"block_size": 64},
        {"num_kv_heads": 4},
        {"head_size": 64},
        {"head_size_v": 64},
        {"num_head_slots": 16},
        {"kv_quant_mode": KVQuantMode.FP8_PER_TENSOR},
        {"tokens_per_state": 2},
        {"state_content_bytes": 512},
        {"sliding_window": 4096},
        {"attention_chunk_size": 4096},
        {"non_causal": True},
        {"page_size_padded": 2 * 128 * 8 * 128 * 2 + 256},
    ],
)
def test_suffix_policy_rejects_incompatible_full_attention_spec(suffix_domain, changes):
    spec = replace(suffix_domain.spec, **changes)
    assert not attention_utils.requires_contiguous_suffix_kv_cache(suffix_domain.layer, suffix_domain.config, spec)


@pytest.mark.parametrize("spec_type", [MLAAttentionSpec, SlidingWindowSpec])
def test_suffix_policy_rejects_other_spec_types(suffix_domain, spec_type):
    kwargs = {"sliding_window": 4096} if spec_type is SlidingWindowSpec else {}
    spec = spec_type(block_size=128, num_kv_heads=8, head_size=128, dtype=torch.bfloat16, **kwargs)
    assert not attention_utils.requires_contiguous_suffix_kv_cache(suffix_domain.layer, suffix_domain.config, spec)


@pytest.mark.parametrize("target", ["model.layers.1.self_attn.attn", "model.layers.0.self_attn.attn"])
def test_suffix_allocation_policy_rejects_shared_layer(suffix_domain, target):
    suffix_domain.layer.kv_sharing_target_layer_name = target
    assert not attention_utils.requires_contiguous_suffix_kv_cache(
        suffix_domain.layer, suffix_domain.config, suffix_domain.spec
    )


def _cache_plan(domain, monkeypatch):
    spec = domain.spec
    names = [domain.layer.layer_name, "model.layers.1.self_attn.attn"]
    layers = {name: SimpleNamespace(**{**vars(domain.layer), "layer_name": name}) for name in names}
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_args: layers)
    num_blocks = 3
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=len(names) * num_blocks * spec.page_size_bytes,
                layers=names,
                layer_stride=num_blocks * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
                offset=0,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=names, kv_cache_spec=spec)],
    )
    group = AttentionGroup(
        backend=AscendAttentionBackend, layer_names=names, kv_cache_spec=spec, kv_cache_group_id=0
    )
    return config, group


@pytest.mark.parametrize("layout", [None, "NHD", "HND", "LBNHC", "LBHNC"])
def test_suffix_allocation_and_views_share_exact_raw_storage(suffix_domain, monkeypatch, layout):
    domain = suffix_domain
    domain.impl.kv_cache_layout = layout
    monkeypatch.setattr(vllm_envs, "VLLM_KV_CACHE_LAYOUT", layout)
    plan, group = _cache_plan(domain, monkeypatch)
    raw = attn_utils._allocate_kv_cache(plan, shared_layers={}, device=torch.device("cpu"))
    caches = attn_utils._reshape_kv_cache_v2([group], raw, "auto", [128], {}, kv_cache_config=plan)
    shape = (3, 8, 128, 128) if layout in ("HND", "LBHNC") else (3, 128, 8, 128)
    side_bytes = plan.num_blocks * domain.spec.page_size_bytes // 2
    pointers = set()
    for name in group.layer_names:
        assert isinstance(raw[name], tuple)
        for backing, view in zip(raw[name], caches[name]):
            assert backing.dtype == torch.int8
            assert backing.numel() == side_bytes
            assert backing.untyped_storage().nbytes() == side_bytes
            assert view.shape == shape
            assert view.dtype == torch.bfloat16
            assert view.is_contiguous()
            assert view.data_ptr() == backing.data_ptr()
            assert view.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
            assert view.numel() * view.element_size() == side_bytes
            pointers.add(backing.data_ptr())
    assert len(pointers) == 2 * len(group.layer_names)

    name, other_name = group.layer_names
    key, value = caches[name]
    # Check token/head physical ordering as well as zero-copy writes, not just
    # a reshape's element count. A swapped NHD/HND shape hits a different cell.
    index = (1, 3, 7, 11) if layout in ("HND", "LBHNC") else (1, 7, 3, 11)
    linear_index = sum(position * stride for position, stride in zip(index, key.stride()))
    key[index] = 17
    assert raw[name][0].view(torch.bfloat16)[linear_index] == 17
    assert torch.count_nonzero(key) == 1
    assert torch.count_nonzero(value) == 0
    value[2].fill_(23)
    assert torch.all(value[2] == 23)
    assert torch.count_nonzero(value[:2]) == 0
    assert torch.count_nonzero(key) == 1
    assert all(torch.count_nonzero(component) == 0 for component in caches[other_name])


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("config.speculative_config.method", "ngram"),
        ("config.speculative_config.num_speculative_tokens", 2),
        ("config.speculative_config.enable_adaptive_verification", True),
        ("config.model_config.hf_text_config.model_type", "qwen2"),
        ("config.parallel_config.tensor_parallel_size", 2),
        ("config.cache_config.cache_dtype", "fp8"),
        ("ascend.pa_shape_list", [32]),
    ],
)
def test_ineligible_suffix_keeps_combined_page_allocation(suffix_domain, monkeypatch, path, value):
    owner, field = _field_owner(suffix_domain, path)
    setattr(owner, field, value)
    monkeypatch.setattr(vllm_envs, "VLLM_KV_CACHE_LAYOUT", None)
    plan, group = _cache_plan(suffix_domain, monkeypatch)
    raw = attn_utils._allocate_kv_cache(plan, shared_layers={}, device=torch.device("cpu"))
    caches = attn_utils._reshape_kv_cache_v2([group], raw, "auto", [128], {}, kv_cache_config=plan)
    block_elements = 128 * 8 * 128
    for name in group.layer_names:
        assert isinstance(raw[name], torch.Tensor)
        assert raw[name].numel() == plan.num_blocks * suffix_domain.spec.page_size_bytes
        key, value_cache = caches[name]
        assert not key.is_contiguous()
        assert not value_cache.is_contiguous()
        assert key.stride(0) == value_cache.stride(0) == 2 * block_elements
        assert key.untyped_storage().data_ptr() == value_cache.untyped_storage().data_ptr() == raw[name].data_ptr()
        assert value_cache.storage_offset() - key.storage_offset() == block_elements
        key[1].fill_(5)
        assert torch.count_nonzero(key[0]) == 0
        assert torch.count_nonzero(key[2]) == 0
        assert torch.count_nonzero(value_cache) == 0


@pytest.mark.parametrize("layout", [None, "HND"])
def test_suffix_main_allocation_entry_uses_split_views(suffix_domain, monkeypatch, layout):
    suffix_domain.impl.kv_cache_layout = layout
    monkeypatch.setattr(vllm_envs, "VLLM_KV_CACHE_LAYOUT", layout)
    plan, group = _cache_plan(suffix_domain, monkeypatch)
    caches = attn_utils.allocate_kv_cache_main(plan, torch.device("cpu"), None, [128])
    expected_shape = (3, 8, 128, 128) if layout == "HND" else (3, 128, 8, 128)
    for name in group.layer_names:
        key, value = caches[name]
        assert key.shape == value.shape == expected_shape
        assert key.is_contiguous() and value.is_contiguous()
        assert key.dtype == value.dtype == torch.bfloat16
        assert key.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()


def test_suffix_shared_pool_keeps_combined_owner_and_alias(suffix_domain, monkeypatch):
    monkeypatch.setattr(vllm_envs, "VLLM_KV_CACHE_LAYOUT", None)
    plan, group = _cache_plan(suffix_domain, monkeypatch)
    owner, alias = group.layer_names
    layers = attn_utils.get_layers_from_vllm_config(suffix_domain.config)
    layers[alias].kv_sharing_target_layer_name = owner
    # The planner allocates only the owner; the alias is rebound after reshape.
    plan = KVCacheConfig(
        num_blocks=plan.num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=plan.num_blocks * suffix_domain.spec.page_size_bytes,
                layers=[owner],
                layer_stride=plan.num_blocks * suffix_domain.spec.page_size_bytes,
                block_stride=suffix_domain.spec.page_size_bytes,
                offset=0,
            )
        ],
        kv_cache_groups=plan.kv_cache_groups,
    )
    caches = attn_utils.allocate_kv_cache_main(plan, torch.device("cpu"), None, [128])
    assert caches[owner] is caches[alias]
    key, value = caches[owner]
    assert not key.is_contiguous() and not value.is_contiguous()
    assert key.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
    key[1].fill_(7)
    assert torch.all(caches[alias][0][1] == 7)
    assert torch.count_nonzero(key[0]) == 0
    assert torch.count_nonzero(key[2]) == 0
    assert torch.count_nonzero(value) == 0
