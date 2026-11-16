# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.config import KVTransferConfig
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs

from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import (
    LayerwiseCachePlanner,
    plan_layerwise_cache_reuse,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
    get_cache_views,
    get_layerwise_reuse_backing_bytes,
    get_layerwise_reuse_layout,
)
from vllm_ascend.models.deepseek_v4.cache_transfer import transfer_cache_views


def make_layout(model="flash", *, block_size=32, cache_format="a3", buffers=2, independent=None):
    count = 43 if model == "flash" else 61
    ratios = ([0, 0] if model == "flash" else [128, 128]) + [4 if i % 2 == 0 else 128 for i in range(2, count)]
    hf = SimpleNamespace(model_type="deepseek_v4", num_hidden_layers=count)
    extra = {"backend": "memcache", "use_layerwise": True, "layerwise_num_shared_buffers": buffers}
    if independent is not None:
        extra["layerwise_independent_layers"] = independent
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf,
            hf_text_config=hf,
            get_num_layers=lambda _: count,
            get_total_num_hidden_layers=lambda: count,
            enforce_eager=True,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            tensor_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
            enable_extensible_kv_cache=False,
            gpu_memory_utilization=0.9,
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="AscendStoreConnector", kv_role="kv_producer", kv_connector_extra_config=extra
        ),
        speculative_config=None,
        use_v2_model_runner=True,
    )
    a5 = cache_format != "a3"
    packed = cache_format == "a5_packed"
    main_dim = 608 if packed else 512
    main_dtype = torch.float8_e4m3fn if packed else torch.bfloat16
    large = block_size * (608 if packed else 1024)
    small = block_size * (132 if a5 else 130)
    specs = {}
    for layer, ratio in enumerate(ratios):
        prefix = f"model.layers.{layer}.self_attn"
        if ratio > 1:
            state_rows = block_size // (16 if ratio == 4 else (8 if a5 else 4))
            specs[prefix + ".compressor.state_cache"] = AscendSlidingWindowMLASpec(
                block_size=state_rows,
                num_kv_heads=1,
                head_size=2048 if ratio == 4 else 1024,
                dtype=torch.float32,
                sliding_window=8 if ratio == 4 else 128,
                page_size_padded=large,
            )
            if ratio == 4:
                specs[prefix + ".indexer.k_cache"] = AscendMLAAttentionSpec(
                    block_size=block_size * ratio,
                    tokens_per_state=ratio,
                    num_kv_heads=1,
                    head_size=128,
                    dtype=torch.float8_e4m3fn if a5 else torch.int8,
                    scale_dim=1,
                    scale_dtype=torch.float32 if a5 else torch.float16,
                    model_version="deepseek_v4",
                )
                specs[prefix + ".indexer.compressor.state_cache"] = AscendSlidingWindowMLASpec(
                    block_size=state_rows,
                    num_kv_heads=1,
                    head_size=512,
                    dtype=torch.float32,
                    sliding_window=8,
                    page_size_padded=small,
                )
        specs[prefix + ".swa_cache"] = AscendSlidingWindowMLASpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=main_dim,
            dtype=main_dtype,
            sliding_window=128,
            model_version="deepseek_v4",
        )
        if ratio > 1:
            specs[prefix + ".attn"] = AscendMLAAttentionSpec(
                block_size=block_size * ratio,
                tokens_per_state=ratio,
                num_kv_heads=1,
                head_size=main_dim,
                dtype=main_dtype,
                model_version="deepseek_v4",
            )
    groups = AscendKVCacheConfigBuilder().get_kv_cache_groups(runtime, specs)
    return runtime, groups, specs


def make_worker_config(runtime, groups, num_blocks):
    native = AscendKVCacheConfigBuilder().get_kv_cache_config_from_groups(runtime, groups, num_blocks=num_blocks)
    layout = plan_layerwise_cache_reuse(native, runtime)
    if layout is None:
        return native
    return apply_layerwise_kv_cache_plan(native, runtime.kv_transfer_config, layout)


@pytest.fixture(autouse=True)
def register_specs():
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()


@pytest.mark.parametrize("model", ["flash", "pro"])
@pytest.mark.parametrize("cache_format", ["a3", "a5_bf16", "a5_packed"])
@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_capacity_is_linear_and_saves_the_large_page_bucket(model, cache_format, block_size):
    runtime, groups, specs = make_layout(
        model, cache_format=cache_format, block_size=block_size, buffers=3, independent=[0, 1]
    )
    builder = AscendKVCacheConfigBuilder()
    cost = builder._get_pool_bytes_per_block(runtime, groups)
    for blocks in (1, 7, 32):
        config = make_worker_config(runtime, groups, blocks)
        assert config.num_blocks == blocks
        assert {tensor.size for tensor in config.kv_cache_tensors} == {blocks * cost}
        assert get_layerwise_reuse_layout(config).has_layer_reuse
        assert any(name.endswith(".attn") for name in get_layerwise_reuse_layout(config).reused_cache_names)
        assert len({name for t in config.kv_cache_tensors for name in t.layers}) == len(specs)
    runtime.kv_transfer_config = None
    old_cost = builder._get_pool_bytes_per_block(runtime, groups)
    assert cost < old_cost // 2
    if cache_format == "a3" and block_size == 32:
        assert old_cost == (812416 if model == "flash" else 1144768)
        assert cost > 0


@pytest.mark.parametrize("model", ["flash", "pro"])
@pytest.mark.parametrize("independent", [[], [0, 1], [0, 1, 4], [-1], [0, 3, 9]])
@pytest.mark.parametrize("buffers", [1, 2, 3, 4])
def test_every_same_group_overwrite_has_a_forward_dependency(model, independent, buffers):
    runtime, groups, specs = make_layout(model, buffers=buffers, independent=independent)
    config = make_worker_config(runtime, groups, num_blocks=8)
    layout = get_layerwise_reuse_layout(config)
    views = get_cache_views(config, specs, layout.cache_view_sizes)
    group_ids = {name: gid for gid, group in enumerate(groups) for name in group.layer_names}
    owners = {}
    for name, (_, start, end) in views.items():
        owners.setdefault((group_ids[name], start, end), set()).add(layout.cache_layer_indices[name])
    expected = {}
    for layers in owners.values():
        if set(layout.independent_layers) & layers:
            assert len(layers) == 1
        for before, after in zip(sorted(layers), sorted(layers)[1:]):
            expected.setdefault(after, set()).add(before)
    assert layout.prefetch_layer_map == {layer: tuple(sorted(values)) for layer, values in expected.items()}
    assert all(before < after for after, values in layout.prefetch_layer_map.items() for before in values)


@pytest.mark.parametrize("independent,buffers", [("all", 1), ([], 100), ([0, 1, 4], 100)])
def test_no_reuse_dependencies_keep_native_descriptors(independent, buffers):
    runtime, groups, _ = make_layout("pro", buffers=buffers, independent=independent)
    builder = AscendKVCacheConfigBuilder()
    plan = builder.get_kv_cache_config_from_groups(runtime, groups, num_blocks=17)
    runtime.kv_transfer_config = None
    native = builder.get_kv_cache_config_from_groups(runtime, groups, num_blocks=17)
    assert plan.kv_cache_tensors == native.kv_cache_tensors
    assert plan_layerwise_cache_reuse(plan, runtime) is None


@pytest.mark.parametrize("cache_format", ["a3", "a5_bf16", "a5_packed"])
def test_real_allocator_and_typed_views_keep_block_ownership(cache_format, monkeypatch):
    from vllm_ascend.device.hardware import AscendDeviceType
    from vllm_ascend.device.hardware_profile import get_hardware_profile
    from vllm_ascend.worker.v2 import attn_utils

    runtime, groups, specs = make_layout(cache_format=cache_format, independent=[0, 1, 4])
    config = make_worker_config(runtime, groups, num_blocks=16)
    allocations = []

    def allocate(size, alignment, device):
        allocations.append(size)
        return torch.zeros(size, dtype=torch.int8, device=device)

    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: runtime)
    monkeypatch.setattr(attn_utils.KVPPConfig, "from_vllm_config", lambda _: SimpleNamespace(size=1))
    monkeypatch.setattr(attn_utils, "_allocate_int8_cache_tensor", allocate)
    hardware = AscendDeviceType.A3 if cache_format == "a3" else AscendDeviceType.A5
    monkeypatch.setattr(attn_utils, "get_current_hardware_profile", lambda: get_hardware_profile(hardware))
    raw = attn_utils._allocate_kv_cache(config, {}, torch.device("cpu"))
    assert allocations == [config.kv_cache_tensors[0].size]
    assert len({value.untyped_storage().data_ptr() for value in raw.values()}) == 1
    backend = SimpleNamespace(get_kv_cache_shape=lambda n, b, h, d: (n, b, h, d))
    caches = {}
    for descriptor in config.kv_cache_tensors:
        for name in descriptor.layers:
            caches[name] = attn_utils._view_dsv4_cache(raw[name], specs[name], backend, config, descriptor.block_stride)
    transferred = transfer_cache_views(caches)
    # Independent group IDs, as assigned from one global block pool. Check
    # simultaneous components of one layer; later layers deliberately alias.
    for layer in (0, 1, 2, 3, 4, 5, 42):
        written = []
        for gid, group in enumerate(groups):
            for name in group.layer_names:
                if f"layers.{layer}." not in name:
                    continue
                for component in transferred[name]:
                    block = component[gid + 1]
                    block.fill_(gid + 1)
                    written.append((block, gid + 1))
                    assert component.stride(0) * component.element_size() >= block.numel() * block.element_size()
        assert all(torch.all(block.float() == value) for block, value in written)


def test_wire_format_matches_resident_and_reused_but_not_other_quantization():
    from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MemcacheBackend

    runtime, groups, _ = make_layout(independent=[0, 1])
    builder = AscendKVCacheConfigBuilder()
    reused = builder.get_kv_cache_config_from_groups(runtime, groups, num_blocks=8)
    runtime.kv_transfer_config.kv_connector_extra_config["layerwise_independent_layers"] = "all"
    resident = builder.get_kv_cache_config_from_groups(runtime, groups, num_blocks=17)
    scheduler = generate_scheduler_kv_cache_config([reused])
    assert reused.dsv4_cache_format == resident.dsv4_cache_format == scheduler.dsv4_cache_format
    before = reused.dsv4_cache_format
    runtime_a5, groups_a5, _ = make_layout(cache_format="a5_packed")
    quantized = builder.get_kv_cache_config_from_groups(runtime_a5, groups_a5, num_blocks=8)
    assert quantized.dsv4_cache_format != before
    keys = []
    for config in (reused, resident, scheduler, quantized):
        protocol = MemcacheBackend.bind_layerwise_keys(
            vllm_config=runtime,
            kv_cache_config=config,
            model_name="dsv4",
            use_hybrid=True,
            grouped_block_size=[128] * len(groups),
        )
        keys.append(protocol.make_hit_check_keys(0, "hash", 1))
    assert keys[0] == keys[1] == keys[2]
    assert keys[0] != keys[3]


def test_unsupported_dsv4_commit_protocol_is_rejected():
    runtime, groups, _ = make_layout()
    runtime.speculative_config = SimpleNamespace(method="deepseek_mtp")
    with pytest.raises(NotImplementedError, match="speculative state commits"):
        AscendKVCacheConfigBuilder().get_kv_cache_config_from_groups(runtime, groups, num_blocks=8)


def test_turboquant_rejected_by_dsv4_guard():
    from vllm_ascend.models.deepseek_v4.cache_reuse import validate_deepseek_v4_layerwise_config
    from vllm_ascend.quantization.methods.kv_cache.turboquant import TURBOQUANT_CACHE_DTYPE

    runtime, _, _ = make_layout()
    runtime.cache_config.cache_dtype = TURBOQUANT_CACHE_DTYPE
    with pytest.raises(NotImplementedError, match="TurboQuant"):
        validate_deepseek_v4_layerwise_config(runtime)


@pytest.mark.parametrize("mode", ["FULL", "FULL_AND_PIECEWISE"])
def test_full_graph_cannot_skip_reused_page_restore_hooks(mode):
    from vllm.config import CUDAGraphMode

    runtime, groups, _ = make_layout()
    runtime.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode[mode])
    with pytest.raises(NotImplementedError, match="per-layer transfer hooks"):
        AscendKVCacheConfigBuilder().get_kv_cache_config_from_groups(runtime, groups, num_blocks=8)


def test_piecewise_graph_preserves_the_layerwise_plan():
    from vllm.config import CUDAGraphMode

    runtime, groups, _ = make_layout()
    runtime.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE)
    plan = make_worker_config(runtime, groups, 8)
    assert get_layerwise_reuse_layout(plan) is not None


def test_no_new_model_planner_class_is_needed():
    from vllm.v1.kv_cache_interface import KVCacheSpec

    from vllm_ascend.models.deepseek_v4.cache_reuse import get_spec_reuse_rules

    runtime, groups, specs = make_layout()
    assert all(LayerwiseCachePlanner.can_reuse_spec(spec, get_spec_reuse_rules()) for spec in specs.values())
    assert LayerwiseCachePlanner.can_reuse_spec(KVCacheSpec(block_size=32), get_spec_reuse_rules()) is False


@pytest.mark.parametrize("model", ["flash", "pro"])
@pytest.mark.parametrize("resident", ["state", "swa", "one_member"])
def test_excluded_members_do_not_force_other_roles_to_reside(model, resident):
    from vllm_ascend.models.deepseek_v4.cache_reuse import get_spec_reuse_rules

    runtime, groups, specs = make_layout(model, buffers=3, independent=[0, 1])
    extra = runtime.kv_transfer_config.kv_connector_extra_config
    runtime.kv_transfer_config = None
    native = AscendKVCacheConfigBuilder().get_kv_cache_config_from_groups(runtime, groups, num_blocks=8)
    excluded = {
        name
        for name in specs
        if (resident == "state" and name.endswith(".state_cache"))
        or (resident == "swa" and name.endswith(".swa_cache"))
        or (resident == "one_member" and name == "model.layers.4.self_attn.indexer.k_cache")
    }
    excluded_ids = {id(specs[name]) for name in excluded}
    layout = LayerwiseCachePlanner(
        native,
        runtime,
        extra,
        spec_reuse_rules={spec_type: lambda spec: id(spec) not in excluded_ids for spec_type in get_spec_reuse_rules()},
    ).build()
    assert not excluded & layout.reused_cache_names
    assert layout.has_layer_reuse
    assert "model.layers.4.self_attn.attn" in layout.reused_cache_names
    expected_suffix = ".swa_cache" if resident == "state" else ".state_cache"
    assert any(name.endswith(expected_suffix) for name in layout.reused_cache_names)


def test_glm_native_alias_packs_and_resident_indexers_remain_supported():
    from dataclasses import replace

    from vllm.v1.kv_cache_interface import MambaSpec

    from tests.ut.models.test_glm5next_cache_config import make_specs

    runtime, _, _ = make_layout(buffers=3, independent=[0, 1, 2, 3])
    hf = SimpleNamespace(model_type="glm5_next_text", num_hidden_layers=24)
    runtime.model_config.hf_config = runtime.model_config.hf_text_config = hf
    runtime.model_config.get_num_layers = lambda _: 24
    runtime.model_config.get_total_num_hidden_layers = lambda: 24
    runtime.cache_config.mamba_cache_mode = "align"
    specs = {}
    for cell in range(6):
        for name, spec in make_specs(pool=4).items():
            parts = name.split(".")
            parts[2] = str(int(parts[2]) + 4 * cell)
            specs[".".join(parts)] = replace(spec, mamba_cache_mode="align") if isinstance(spec, MambaSpec) else spec
    builder = AscendKVCacheConfigBuilder()
    groups = builder.get_kv_cache_groups(runtime, specs)
    cost = builder._get_pool_bytes_per_block(runtime, groups)
    config = make_worker_config(runtime, groups, 8)
    assert {tensor.size for tensor in config.kv_cache_tensors} == {8 * cost}
    names = get_layerwise_reuse_layout(config).reused_cache_names
    assert any(name.endswith(".attn") for name in names)
    assert any(name.endswith(".linear_attn") for name in names)
    assert not any("indexer" in name for name in names)
    assert all(get_layerwise_reuse_layout(config).cache_layer_indices[name] >= 4 for name in names)


def test_core_converges_worker_budgets_using_the_reused_physical_cost(monkeypatch):
    from vllm.v1.core import kv_cache_config_builder, kv_cache_planning

    runtime, groups, specs = make_layout(buffers=3, independent=[0, 1])
    runtime.model_config.original_max_model_len = runtime.model_config.max_model_len = 128
    runtime.attention_config = SimpleNamespace(hisparse_config=None)
    runtime.max_in_flight_tokens = 32
    runtime.num_prefill_lookahead_tokens = 0
    builder = AscendKVCacheConfigBuilder()
    monkeypatch.setattr(kv_cache_config_builder, "get_kv_cache_config_builder", lambda _: builder)
    budgets = [64 << 20, 80 << 20]
    plans = kv_cache_planning.get_kv_cache_configs(runtime, [specs, specs], budgets, groups)
    expected = min(budgets) // builder._get_pool_bytes_per_block(runtime, groups)
    assert [plan.num_blocks for plan in plans] == [expected, expected]
    assert plans[0].kv_cache_tensors == plans[1].kv_cache_tensors
    assert get_layerwise_reuse_backing_bytes(plans[0]) <= min(budgets)


def test_raw_pages_roundtrip_across_requests_layers_and_steps():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
        allocate_cache_backing,
    )

    runtime, groups, specs = make_layout(buffers=2, independent=[0, 1, 4])
    config = make_worker_config(runtime, groups, num_blocks=32)
    raw = allocate_cache_backing(
        config,
        specs,
        lambda size: torch.zeros(size, dtype=torch.uint8),
        get_layerwise_reuse_layout(config).cache_view_sizes,
    )
    descriptors = {name: descriptor for descriptor in config.kv_cache_tensors for name in descriptor.layers}
    host = {}
    for step in range(3):
        for layer in range(43):
            live_pages = []
            for group_id, group in enumerate(groups):
                for name in group.layer_names:
                    if f"layers.{layer}." not in name:
                        continue
                    stride = descriptors[name].block_stride
                    for request_id in range(3):
                        block_id = 1 + group_id + len(groups) * request_id
                        page = raw[name][block_id * stride : (block_id + 1) * stride]
                        key = (request_id, name)
                        if step:
                            page.copy_(host[key])
                            assert torch.all(page == (layer + request_id + step - 1) % 251)
                        page.fill_((layer + request_id + step) % 251)
                        live_pages.append((key, page))
            # All components needed concurrently by this layer must remain
            # intact. The next layer may legally reuse their backing region.
            for key, page in live_pages:
                assert torch.all(page == (layer + key[0] + step) % 251)
                host[key] = page.clone()


def test_expensive_but_requested_reuse_is_not_silently_disabled():
    runtime, groups, specs = make_layout("flash", buffers=3, independent=[0, 1])
    extra = runtime.kv_transfer_config.kv_connector_extra_config
    runtime.kv_transfer_config = None
    builder = AscendKVCacheConfigBuilder()
    native = builder.get_kv_cache_config_from_groups(runtime, groups, num_blocks=8)
    layout = LayerwiseCachePlanner(
        native,
        runtime,
        extra,
        spec_reuse_rules={
            AscendMLAAttentionSpec: True,
            AscendSlidingWindowMLASpec: lambda spec: spec.model_version == "deepseek_v4",
        },
    ).build()
    assert layout.has_layer_reuse
    assert layout.kv_cache_tensors[0].size >= native.kv_cache_tensors[0].size


@pytest.mark.parametrize("model", ["flash", "pro"])
def test_documented_configuration_plans_reuse_and_keeps_native_core_descriptors(model):
    guide = Path(__file__).resolve().parents[3] / (
        "docs/source/user_guide/feature_guide/deepseek_v4_layerwise_offload.md"
    )
    config_json = guide.read_text().split("```json\n", 1)[1].split("```", 1)[0]
    runtime, groups, _ = make_layout(model)
    runtime.kv_transfer_config = KVTransferConfig(**json.loads(config_json))
    builder = AscendKVCacheConfigBuilder()
    native = builder.get_kv_cache_config_from_groups(runtime, groups, num_blocks=8)
    worker = make_worker_config(runtime, groups, num_blocks=8)
    layout = get_layerwise_reuse_layout(worker)
    assert layout is not None and layout.independent_layers == [0, 1]
    assert get_layerwise_reuse_layout(native) is None
    assert get_layerwise_reuse_backing_bytes(native) == worker.kv_cache_tensors[0].size
    assert worker.kv_cache_groups == native.kv_cache_groups
    assert worker.kv_cache_tensors != native.kv_cache_tensors


@pytest.mark.parametrize("model", ["flash", "pro"])
def test_reference_attention_matches_after_rotated_page_restore(model):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
        allocate_cache_backing,
    )

    runtime, groups, specs = make_layout(model, buffers=1, independent=[0, 1])
    config = make_worker_config(runtime, groups, num_blocks=16)
    layout = get_layerwise_reuse_layout(config)
    raw = allocate_cache_backing(
        config, specs, lambda size: torch.zeros(size, dtype=torch.uint8), layout.cache_view_sizes
    )
    descriptors = {name: tensor for tensor in config.kv_cache_tensors for name in tensor.layers}
    # Two C4 main-history layers deliberately share bytes. Use a small BF16
    # matrix inside an allocated page and compare attention to a resident copy.
    first, later = (f"model.layers.{layer}.self_attn.attn" for layer in (2, 4))
    assert raw[first].data_ptr() == raw[later].data_ptr()
    stride = descriptors[first].block_stride
    keys = raw[first][stride : stride + 64].view(torch.bfloat16).reshape(4, 8)
    resident = torch.arange(32, dtype=torch.float32).reshape(4, 8).div(32).to(torch.bfloat16)
    query = torch.arange(8, dtype=torch.float32).reshape(1, 8).div(8)
    expected = torch.softmax(query @ resident.float().T, dim=-1) @ resident.float()
    keys.copy_(resident)
    saved = raw[first][stride : stride + 64].clone()
    raw[later][stride : stride + 64].fill_(255)
    raw[first][stride : stride + 64].copy_(saved)
    actual = torch.softmax(query @ keys.float().T, dim=-1) @ keys.float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
