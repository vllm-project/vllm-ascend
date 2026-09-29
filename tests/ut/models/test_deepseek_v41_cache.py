# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

pytest.importorskip(
    "vllm.transformers_utils.configs.deepseek_v41",
    reason="DeepSeek V4.1 is unavailable on this vLLM release",
)

import torch
import torch_npu
from vllm.config import set_current_vllm_config
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_cache_interface import CircularBufferSpec

from tests.deepseek_v41_utils import (
    allocate_cache_views,
    build_v41_cache_specs,
    compressor_ratio2_reference,
    gather_cache_rows,
    make_cache_config,
    scatter_cache,
    select_candidate_blocks,
    select_index_topk,
)
from vllm_ascend.attention import dsa_v41
from vllm_ascend.attention.dsa_v41 import (
    AscendDSAV41Impl,
    AscendDSAV41MetadataBuilder,
    DeepseekV41CacheLayer,
    compressed_slot_mapping,
    pad_sparse_indices,
    scatter_cache_sk,
)
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    get_storage_block_size,
)
from vllm_ascend.models.deepseek_v41.cache_config import (
    MAX_VALIDATED_SLOT_BYTES,
    get_deepseek_v41_kv_cache_config,
    get_deepseek_v41_pool_bytes_per_block,
    get_layer_tuples,
    group_cache_specs,
    is_deepseek_v41_cache,
    make_cache_groups,
)
from vllm_ascend.models.deepseek_v41.compressor import DeepseekV41Compressor
from vllm_ascend.models.deepseek_v41.model import build_layer_plan
from vllm_ascend.worker.device_metadata import DeviceMetadataStage


@pytest.fixture(autouse=True)
def mock_npu_rms_norm(monkeypatch):
    # Keep cache/state tests on CPU; operator accuracy is covered on NPU.
    def rms_norm(x, gamma, epsilon=1e-6):
        rstd = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + epsilon)
        return (x.float() * rstd).to(x.dtype) * gamma, rstd

    monkeypatch.setattr(torch_npu, "npu_rms_norm", rms_norm)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    vllm_config = MagicMock()
    vllm_config.compilation_config.custom_ops = ["all"]
    vllm_config.quant_config = None
    with set_current_vllm_config(vllm_config):
        yield


@pytest.fixture
def config():
    # Deliberately small parameter dimensions; source topology matches the backbone.
    return SimpleNamespace(
        num_hidden_layers=40,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        candidate_topk_blocks=16,
        candidate_block_size=8,
        index_topk=8,
        engram_layer_ids=[1, 14],
        sliding_window=128,
        head_dim=8,
        index_head_dim=4,
        hidden_size=16,
        num_attention_heads=4,
        index_n_heads=2,
        q_lora_rank=8,
        o_lora_rank=4,
        o_groups=2,
        rms_norm_eps=1e-6,
    )


@pytest.fixture
def runtime(config):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config, enforce_eager=True),
        cache_config=SimpleNamespace(
            block_size=64,
            enable_prefix_caching=False,
            cache_dtype="auto",
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
        ),
        speculative_config=None,
        kv_transfer_config=None,
        use_v2_model_runner=False,
        # vLLM main reads attention_config.hisparse_config in the KV cache
        # config helpers; Ascend does not enable HiSparse.
        attention_config=SimpleNamespace(hisparse_config=None),
    )


def collect_specs(runtime, prefix="model"):
    return build_v41_cache_specs(runtime.model_config.hf_text_config, runtime, prefix)


def test_compressor_registers_state_cache_and_preserves_norm_weights(config, runtime):
    runtime.scheduler_config.max_num_batched_tokens = 8
    compressor = DeepseekV41Compressor(config, 2, runtime, prefix="compressor")
    state = compressor.state_cache
    assert type(state) is DeepseekV41CacheLayer
    assert runtime.compilation_config.static_forward_context["compressor.state_cache"] is state
    spec = state.get_kv_cache_spec(runtime)
    assert spec.dtype == torch.float32
    assert spec.block_size == 32
    assert spec.head_size == 2 * config.head_dim
    assert compressor.norm.weight.dtype == torch.bfloat16
    assert set(compressor.state_dict()) == {"wkv.weight", "wgate.weight", "norm.weight"}


def test_owner_counts_nested_config_and_source_resolution(config, runtime):
    topology = build_layer_plan(DeepseekV41Config(text_config=vars(config)))
    specs = collect_specs(runtime)
    assert len(specs) == 51
    assert is_deepseek_v41_cache(specs)
    assert topology.kv_consumers(2) == tuple(range(2, 8))
    assert topology.kv_consumers(20) == tuple(range(20, 40))
    assert topology.layer(26).kv_source_layer == 20
    assert topology.layer(26).index_source_layer == 24
    assert specs["model.layers.20.self_attn.long_kv_cache"].storage_block_size == 64
    assert specs["model.layers.2.self_attn.long_kv_cache"].storage_block_size == 32
    assert "model.layers.20.self_attn.compressor.state_cache" not in specs
    assert type(specs["model.layers.2.self_attn.long_kv_cache"]) is AscendMLAAttentionSpec
    assert type(specs["model.layers.2.self_attn.indexer.k_cache"]) is AscendMLAAttentionSpec
    assert type(specs["model.layers.2.self_attn.swa_cache"]) is AscendSlidingWindowMLASpec
    assert type(specs["model.layers.2.self_attn.compressor.state_cache"]) is CircularBufferSpec


def test_twelve_groups_share_four_layer_slots(config, runtime):
    original = collect_specs(runtime)
    uniform = group_cache_specs(original)
    assert [len(g.kv_cache_specs) for g in uniform] == [8, 3] + [4] * 10
    assert [g.block_size for g in uniform] == [64, 32] + [64] * 10
    groups = make_cache_groups(uniform)
    assert is_deepseek_v41_cache(groups)
    for row, group in enumerate(groups[2:]):
        assert group.layer_names == [
            f"model.layers.{layer}.self_attn.swa_cache" for layer in range(row * 4, row * 4 + 4)
        ]
    specs = {n: s for g in uniform for n, s in g.kv_cache_specs.items()}
    assert all(s.page_size_padded is None for s in original.values())
    assert group_cache_specs(specs) == uniform  # Replanning cannot accumulate padding.
    assert group_cache_specs(dict(reversed(list(original.items())))) == uniform
    page_sizes, layer_tuples = get_layer_tuples(specs)
    cache_config = get_deepseek_v41_kv_cache_config(
        runtime,
        groups,
        get_deepseek_v41_pool_bytes_per_block(groups) * 10 + 1,
    )
    blocks = cache_config.num_blocks
    allocations = cache_config.kv_cache_tensors
    assert blocks == 10 and len(allocations) == 4
    raw, caches = allocate_cache_views(cache_config)
    assert len({t.data_ptr() for t in raw}) == 4
    assert sum(t.numel() for t in raw) == blocks * get_deepseek_v41_pool_bytes_per_block(groups)
    assert set(caches) == set(original)
    for backing, allocation, page_size, layer_tuple in zip(raw, allocations, page_sizes, layer_tuples):
        assert allocation.offset == 0 and allocation.block_stride == page_size
        assert allocation.size == blocks * page_size
        assert allocation.layers == list(layer_tuple)
        for name in layer_tuple:
            spec = specs[name]
            cache = caches[name]
            views = cache if isinstance(cache, tuple) else (cache,)
            storage_block_size = get_storage_block_size(spec)
            assert views[0].shape == (blocks, storage_block_size, 1, spec.head_size)
            is_index = isinstance(spec, AscendMLAAttentionSpec) and spec.scale_dim
            expected_offset = specs[layer_tuple[0]].unpadded_page_size_bytes if is_index else 0
            assert views[0].data_ptr() == backing.data_ptr() + expected_offset
            assert all(v.stride(0) * v.element_size() == page_size for v in views)
            if is_index:
                key, scale = cache
                assert key.dtype == torch.int8 and scale.dtype == torch.float16
                assert scale.data_ptr() - key.data_ptr() == storage_block_size * spec.head_size
                assert scale.shape == (blocks, storage_block_size, 1, 1)


def test_production_layout_matches_design(config, runtime):
    runtime.cache_config.block_size = 128
    specs = build_v41_cache_specs(SimpleNamespace(**(vars(config) | {"head_dim": 512, "index_head_dim": 128})), runtime)
    groups = make_cache_groups(group_cache_specs(specs))
    assert len(groups) == 12
    assert [g.kv_cache_spec.page_size_bytes for g in groups] == [540928, 393216] + [540928] * 10
    assert get_deepseek_v41_pool_bytes_per_block(groups) == 540928
    padded = {name: spec for group in groups for name, spec in group.kv_cache_spec.kv_cache_specs.items()}
    page_sizes, layer_tuples = get_layer_tuples(padded)
    assert page_sizes == [131072] * 3 + [147712]
    assert [len(layer_tuple) for layer_tuple in layer_tuples] == [13, 13, 13, 12]
    for i, layer_tuple in enumerate(layer_tuples):
        tuple_specs = [padded[name] for name in layer_tuple]
        assert sum(isinstance(spec, AscendMLAAttentionSpec) and not spec.scale_dim for spec in tuple_specs) == 1
        assert sum(isinstance(spec, AscendMLAAttentionSpec) and spec.scale_dim for spec in tuple_specs) == 1
        assert sum(isinstance(spec, CircularBufferSpec) for spec in tuple_specs) == int(i < 3)
        assert sum(isinstance(spec, AscendSlidingWindowMLASpec) for spec in tuple_specs) == 10
        assert padded[layer_tuple[0]].unpadded_page_size_bytes == (65536 if i < 3 else 131072)
        assert padded[layer_tuple[1]].page_size_bytes == (65536 if i < 3 else 16640)
    swa_padding = [
        s.page_size_bytes - s.real_page_size_bytes
        for n, s in padded.items()
        if n.endswith(".swa_cache") and ".mtp." not in f".{n}"
    ]
    assert swa_padding.count(0) == 30 and swa_padding.count(16640) == 10
    cache_config = get_deepseek_v41_kv_cache_config(runtime, groups, 540928 * 3)
    assert cache_config.num_blocks == 3
    _, caches = allocate_cache_views(cache_config)
    assert sum(caches[n].is_contiguous() for n in padded if n.endswith(".swa_cache") and ".mtp." not in f".{n}") == 30


def _force_compressed_cache_soc(monkeypatch, supported):
    # builds_scatter_nd_update_sk() reads the profile through its defining
    # module, so patch it there rather than at the call site.
    monkeypatch.setattr(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        lambda: SimpleNamespace(supports=lambda capability: supported),
    )


def test_addressing_clamp_bounds_slot_tensors(config, runtime, monkeypatch):
    runtime.cache_config.block_size = 128
    overrides = {"head_dim": 512, "index_head_dim": 128}
    specs = build_v41_cache_specs(SimpleNamespace(**(vars(config) | overrides)), runtime)
    groups = make_cache_groups(group_cache_specs(specs))
    padded = {name: spec for group in groups for name, spec in group.kv_cache_spec.kv_cache_specs.items()}

    # SoCs that build scatter_nd_update_sk address the whole pool, so they keep it.
    _force_compressed_cache_soc(monkeypatch, False)
    page_sizes, _ = get_layer_tuples(padded)
    safe_blocks = MAX_VALIDATED_SLOT_BYTES // max(page_sizes)
    unclamped = safe_blocks * 4
    memory = sum(page_sizes) * unclamped
    assert get_deepseek_v41_kv_cache_config(runtime, groups, memory).num_blocks == unclamped

    # The dense store pads each page up for its own row alignment, so the bound
    # has to be measured under the profile that applies it.
    _force_compressed_cache_soc(monkeypatch, True)
    dense_page_sizes, _ = get_layer_tuples(padded)
    dense_safe_blocks = MAX_VALIDATED_SLOT_BYTES // max(dense_page_sizes)
    clamped = get_deepseek_v41_kv_cache_config(runtime, groups, memory)
    assert clamped.num_blocks == dense_safe_blocks
    assert max(t.size for t in clamped.kv_cache_tensors) <= MAX_VALIDATED_SLOT_BYTES

    # A pool that already fits stays untouched.
    fits = sum(dense_page_sizes) * (dense_safe_blocks - 1)
    assert get_deepseek_v41_kv_cache_config(runtime, groups, fits).num_blocks == dense_safe_blocks - 1

    # An override cannot raise the pool past the addressable bound either.
    runtime.cache_config.num_gpu_blocks_override = dense_safe_blocks * 2
    assert get_deepseek_v41_kv_cache_config(runtime, groups, memory).num_blocks == dense_safe_blocks
    runtime.cache_config.num_gpu_blocks_override = dense_safe_blocks // 2
    assert get_deepseek_v41_kv_cache_config(runtime, groups, memory).num_blocks == dense_safe_blocks // 2


def test_shared_slots_isolate_groups_and_recycled_ids(config, runtime):
    groups = make_cache_groups(group_cache_specs(collect_specs(runtime)))
    count = len(groups) + 1
    cfg = get_deepseek_v41_kv_cache_config(
        runtime,
        groups,
        get_deepseek_v41_pool_bytes_per_block(groups) * count,
    )
    _, caches = allocate_cache_views(cfg)
    expected = []
    for group_idx, group in enumerate(groups):
        block_id = group_idx + 1
        for resource_idx, name in enumerate(group.layer_names):
            cache = caches[name]
            for plane_idx, view in enumerate(cache if isinstance(cache, tuple) else (cache,)):
                slots = block_id * view.shape[1] + torch.arange(view.shape[1])
                value = torch.full(
                    (view.shape[1], view.shape[-1]), 1 + group_idx + resource_idx + plane_idx, dtype=view.dtype
                )
                scatter_cache(view, slots, value)
                expected.append((group_idx, view, slots, value))
    for _, view, slots, value in expected:
        torch.testing.assert_close(gather_cache_rows(view, slots), value)
        assert not view[0].any()
    # Simulate release of group 0's ID and reassignment to a SWA group.
    # The released full-context views are no longer valid; all other IDs remain intact.
    for name in groups[2].layer_names:
        caches[name][1].fill_(99)
    for group_idx, view, slots, value in expected:
        if group_idx != 0:
            torch.testing.assert_close(gather_cache_rows(view, slots), value)


def test_view_with_nonzero_backing_storage_offset():
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    spec = AscendMLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=4, dtype=torch.bfloat16)
    backing = torch.zeros(16 + 2 * 256, dtype=torch.uint8)
    raw = backing[16:]
    cache = NPUModelRunner._adjust_kv_layout(None, raw, [(2, 16, 1, 4)], [spec.dtype], 256, initial_offset_bytes=32)[0]
    cache[1].fill_(7)
    assert cache.data_ptr() == backing.data_ptr() + 48
    torch.testing.assert_close(backing[304:432].view(torch.bfloat16), torch.full((64,), 7, dtype=torch.bfloat16))
    assert not backing[:48].any()


def test_view_accepts_latest_vllm_int8_backing_storage():
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    spec = AscendMLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=4, dtype=torch.bfloat16)
    raw = torch.zeros(2 * 256, dtype=torch.int8)
    cache = NPUModelRunner._adjust_kv_layout(None, raw, [(2, 16, 1, 4)], [spec.dtype], 256, initial_offset_bytes=32)[0]
    assert cache.shape == (2, 16, 1, 4)


def test_request_accounting_counts_merged_full_context_once(runtime):
    runtime.model_config.max_model_len = 1024
    runtime.max_in_flight_tokens = 128
    groups = make_cache_groups(group_cache_specs(collect_specs(runtime)))
    bounded = sum(
        max(s.max_memory_usage_bytes(runtime) // s.page_size_bytes for s in g.kv_cache_spec.kv_cache_specs.values())
        for g in groups[1:]
    )
    page = get_deepseek_v41_pool_bytes_per_block(groups)
    required = kv_cache_utils._max_memory_usage_bytes_from_groups(runtime, groups)
    assert required // page == 1024 // 64 + bounded


def test_safe_override_capacity(runtime):
    groups = make_cache_groups(group_cache_specs(collect_specs(runtime)))
    page = get_deepseek_v41_pool_bytes_per_block(groups)
    runtime.cache_config.num_gpu_blocks_override = 3
    config = get_deepseek_v41_kv_cache_config(runtime, groups, 5 * page + 1)
    assert config.num_blocks == 3 and sum(t.size for t in config.kv_cache_tensors) == 3 * page


def test_upstream_entrypoint_and_admission_use_slot_reservation(runtime):
    runtime.model_config.max_model_len = 1024
    runtime.max_in_flight_tokens = 128
    groups = make_cache_groups(group_cache_specs(collect_specs(runtime)))
    page = get_deepseek_v41_pool_bytes_per_block(groups)
    config = kv_cache_utils.get_kv_cache_config_from_groups(runtime, groups, 100 * page)
    assert config.num_blocks == 100 and len(config.kv_cache_tensors) == 4
    assert sum(t.size for t in config.kv_cache_tensors) == 100 * page
    demand = kv_cache_utils._max_memory_usage_bytes_from_groups(runtime, groups) // page
    assert kv_cache_utils._pool_bytes_per_block(groups) == page
    assert kv_cache_utils._max_memory_usage_bytes_from_groups(runtime, groups) == demand * page
    assert kv_cache_utils.get_max_concurrency_for_kv_cache_config(runtime, config) == 100 / demand
    scheduler_config = kv_cache_utils.generate_scheduler_kv_cache_config([config])
    assert kv_cache_utils.get_max_concurrency_for_kv_cache_config(runtime, scheduler_config) == 100 / demand


def test_model_registration_and_binding(runtime):
    specs = collect_specs(runtime, "language_model.model")
    context = runtime.compilation_config.static_forward_context
    modules = torch.nn.ModuleDict()
    for index, (name, spec) in enumerate(specs.items()):
        modules[str(index)] = DeepseekV41CacheLayer(runtime, name, spec)
    assert len(context) == 51
    assert all(module.kv_cache[0].numel() == 0 for module in context.values())
    state = context["language_model.model.layers.2.self_attn.compressor.state_cache"]
    assert state is context["language_model.model.layers.2.self_attn.compressor.state_cache"]
    assert not state.spec.prefix_cacheable
    assert get_storage_block_size(state.spec) == 32
    owned_names = [name for name, module in modules.named_modules() if hasattr(module, "kv_cache")]
    assert len(owned_names) == 51


def test_dspark_is_one_additional_group_in_existing_slots(runtime):
    runtime.model_config.max_model_len = 4096
    runtime.max_in_flight_tokens = 256
    target = make_cache_config(17)
    draft = make_cache_config(17, draft_layers=3)
    groups = draft.kv_cache_groups
    assert len(groups) == 13 and sum(len(g.layer_names) for g in groups) == 54
    assert groups[:12] == target.kv_cache_groups
    assert groups[12].layer_names == [f"mtp.{i}.self_attn.swa_cache" for i in range(3)]
    assert all(isinstance(s, AscendSlidingWindowMLASpec) for s in groups[12].kv_cache_spec.kv_cache_specs.values())
    assert len(draft.kv_cache_tensors) == 4
    assert [t.size for t in draft.kv_cache_tensors] == [t.size for t in target.kv_cache_tensors]
    assert get_deepseek_v41_pool_bytes_per_block(groups) == 540928
    added = (
        kv_cache_utils._max_memory_usage_bytes_from_groups(runtime, groups)
        - kv_cache_utils._max_memory_usage_bytes_from_groups(runtime, target.kv_cache_groups)
    ) // get_deepseek_v41_pool_bytes_per_block(groups)
    spec = next(iter(groups[12].kv_cache_spec.kv_cache_specs.values()))
    assert added == (spec.max_memory_usage_bytes(runtime) + spec.page_size_bytes - 1) // spec.page_size_bytes
    padded = {n: s for g in groups for n, s in g.kv_cache_spec.kv_cache_specs.items()}
    assert get_layer_tuples(padded) == get_layer_tuples(dict(reversed(list(padded.items()))))
    backings, views = allocate_cache_views(draft)
    assert sum(b.numel() for b in backings) == 17 * 540928
    for stage in range(3):
        name = f"mtp.{stage}.self_attn.swa_cache"
        assert name in draft.kv_cache_tensors[stage].layers
        assert views[name].data_ptr() == backings[stage].data_ptr()
        assert views[name].shape == (17, 128, 1, 512)
        assert views[name].stride() == (65536, 512, 512, 1)


def test_dspark_slots_isolate_groups_and_reuse_released_ids():
    cfg = make_cache_config(17, draft_layers=3)
    _, views = allocate_cache_views(cfg)
    # Each group owns a different live global ID, including the draft group.
    for gid, group in enumerate(cfg.kv_cache_groups):
        for name in group.layer_names:
            planes = views[name] if isinstance(views[name], tuple) else (views[name],)
            for plane in planes:
                plane[gid + 1].fill_(gid + 1)
    for gid, group in enumerate(cfg.kv_cache_groups):
        for name in group.layer_names:
            planes = views[name] if isinstance(views[name], tuple) else (views[name],)
            for plane in planes:
                assert (plane[gid + 1] == gid + 1).all()
                assert (plane[0] == 0).all()
    # After target G0 releases ID 1, G12 may use it without touching live IDs.
    for stage in range(3):
        view = views[f"mtp.{stage}.self_attn.swa_cache"]
        view[1].fill_(7)
        assert (view[13] == 13).all()
        assert (view[0] == 0).all()


@pytest.mark.parametrize("full_graph_mode", [False, True])
def test_c2_builder_keeps_fixed_rows_for_mixed_parity_and_padding(config, runtime, full_graph_mode):
    spec = collect_specs(runtime)["model.layers.2.self_attn.compressor.state_cache"]
    builder = AscendDSAV41MetadataBuilder(spec, [], runtime, torch.device("cpu"))
    common = SimpleNamespace(
        slot_mapping=torch.tensor([10, 11, -1]),
        block_table_tensor=torch.tensor([[1], [2], [0]]),
        query_start_loc=torch.tensor([0, 1, 2, 3]),
        query_start_loc_cpu=torch.tensor([0, 1, 2, 3]),
        seq_lens=torch.tensor([3, 4, 9]),
        seq_lens_cpu=torch.tensor([3, 4, 9]),
        positions=torch.tensor([2, 3, 0]),
        num_reqs=3,
        num_actual_tokens=2,
        num_input_tokens=3,
        max_query_len=1,
        max_seq_len=9,
        is_prefilling=torch.tensor([False, False, False]),
    )

    metadata = builder.build(0, common, num_actual_reqs=2, full_graph_mode=full_graph_mode)

    assert metadata.num_actual_reqs == 2
    assert metadata.seq_lens.tolist() == [3, 4, 0]
    assert metadata.c2_complete_mask.tolist() == [False, True, False]
    assert metadata.c2_ring_metadata.tolist() == [[2, 3, 0], [1, 1, 0], [0, 1, 2], [0, 1, 2], [1, 2, 0]]
    assert metadata.slot_mapping.tolist() == [-1, -1, -1]
    assert metadata.c2_source_positions.tolist() == [0, 2, 0]
    assert metadata.c2_metadata_group_id == id(builder._c2_complete_mask)
    pointer = metadata.c2_ring_metadata.data_ptr()
    common.seq_lens = torch.tensor([4, 5, 8])
    common.seq_lens_cpu = common.seq_lens
    common.positions = torch.tensor([3, 4, 0])
    common.block_table_tensor = torch.tensor([[7], [3], [0]])
    replay = builder.build(0, common, num_actual_reqs=2, full_graph_mode=full_graph_mode)
    assert replay.c2_ring_metadata.data_ptr() == pointer
    assert replay.c2_complete_mask.tolist() == [True, False, False]
    assert replay.c2_ring_metadata[4].tolist() == [7, 3, 0]
    idle = builder.build(0, common, num_actual_reqs=2, skip_ring_state_update=True)
    assert idle.c2_ring_metadata[1].tolist() == [0, 0, 0]
    assert idle.c2_ring_metadata[4].tolist() == [0, 0, 0]
    assert not idle.c2_complete_mask.any()


@pytest.mark.parametrize("full_graph_mode", [False, True])
@pytest.mark.parametrize("index_first", [False, True])
@pytest.mark.parametrize(
    "num_actual_reqs,num_actual_tokens,stored_rows",
    [(3, 5, [1, 2, 3, 4]), (2, 5, [1, 2]), (3, 3, [1, 2]), (0, 5, []), (3, 0, [])],
)
def test_c2_builder_prepares_shared_store_mask(
    runtime, full_graph_mode, index_first, num_actual_reqs, num_actual_tokens, stored_rows
):
    specs = collect_specs(runtime)
    names = ["model.layers.2.self_attn.long_kv_cache", "model.layers.2.self_attn.indexer.k_cache"]
    if index_first:
        names.reverse()
    builders = [AscendDSAV41MetadataBuilder(specs[name], [name], runtime, torch.device("cpu")) for name in names]
    common = SimpleNamespace(
        slot_mapping=torch.tensor([10, 11, 15, 129, 131]),
        block_table_tensor=torch.tensor([[1], [2], [4]]),
        query_start_loc=torch.tensor([0, 2, 3, 5]),
        query_start_loc_cpu=torch.tensor([0, 2, 3, 5]),
        seq_lens=torch.tensor([4, 6, 10]),
        seq_lens_cpu=torch.tensor([4, 6, 10]),
        positions=torch.tensor([2, 3, 5, 7, 9]),
        num_reqs=3,
        num_actual_tokens=num_actual_tokens,
        num_input_tokens=5,
        max_query_len=2,
        max_seq_len=10,
        is_prefilling=torch.tensor([True, False, True]),
    )
    original_slots = common.slot_mapping.clone()
    shared: dict[str, Any] = {}
    metadata = [
        builder.build(
            0, common, num_actual_reqs=num_actual_reqs, full_graph_mode=full_graph_mode, common_v41_metadata=shared
        )
        for builder in builders
    ]
    expected = torch.full((5, 2), -1, dtype=torch.int32)
    coordinates = torch.tensor([[-1, -1], [0, 5], [0, 7], [2, 0], [2, 1]], dtype=torch.int32)
    expected[stored_rows] = coordinates[stored_rows]
    pointer = metadata[0].slot_mapping.data_ptr()
    for result in metadata:
        assert result.slot_mapping.data_ptr() == pointer
        torch.testing.assert_close(result.slot_mapping, expected)
    torch.testing.assert_close(common.slot_mapping, original_slots)

    # New metadata must update the captured address and retain position parity
    # even when a padded slot happens to contain a valid physical coordinate.
    common.positions = torch.tensor([3, 4, 6, 8, 10])
    common.slot_mapping = torch.tensor([11, 11, 15, 129, 131])
    common.num_actual_tokens = 5
    replay = builders[0].build(0, common, full_graph_mode=full_graph_mode)
    assert replay.slot_mapping.data_ptr() == pointer
    assert replay.slot_mapping.tolist() == [[0, 5], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]
    idle = builders[0].build(0, common, skip_ring_state_update=True)
    assert idle.slot_mapping.data_ptr() == pointer
    assert idle.slot_mapping.tolist() == [[-1, -1]] * 5


def test_scatter_cache_redirects_invalid_rows_to_null_row():
    cache = torch.full((1, 8, 1, 2), -3.0)
    values = torch.tensor([[9.0, 9.0], [7.0, 8.0]])

    scatter_cache(cache, torch.tensor([-1, 3]), values)

    assert cache[0, 0, 0].tolist() == [0.0, 0.0]
    assert cache[0, 3, 0].tolist() == [7.0, 8.0]


def test_scatter_cache_sk_consumes_prepared_coordinates_and_preserves_stride(
    monkeypatch,
):
    backing = torch.zeros(3 * 128, dtype=torch.uint8)
    cache = torch.as_strided(
        backing.view(torch.float32),
        size=(3, 4, 1, 2),
        stride=(32, 2, 2, 1),
    )
    values = torch.tensor([[9.0, 9.0], [7.0, 8.0]])
    indices = torch.tensor([[-1, -1], [1, 3]], dtype=torch.int32)
    calls = []

    def scatter(var, indices, updates):
        calls.append((var, indices, updates))

    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_scatter_nd_update_sk",
        scatter,
        raising=False,
    )
    scatter_cache_sk(cache, indices, values)

    var, actual_indices, updates = calls[0]
    assert var.shape == (3, 4, 2)
    assert var.stride() == (32, 2, 1)
    assert actual_indices.data_ptr() == indices.data_ptr()
    torch.testing.assert_close(actual_indices, indices)
    assert updates.tolist() == [[9.0, 9.0], [7.0, 8.0]]


def test_compression_slot_mapping():
    slots = torch.tensor([-1, 0, 1, 62, 63, 320, 321, 383])
    assert compressed_slot_mapping(slots, 2).tolist() == [-1, -1, 0, -1, 31, -1, 160, 191]
    assert torch.equal(compressed_slot_mapping(slots, 1), slots)


def test_candidate_blocks_pin_partial_tail_and_drop_unreachable_blocks():
    scores = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, -torch.inf, -torch.inf]])
    # With two candidate blocks, the best old block and the partially filled
    # newest block must survive. The unreachable final block must not.
    mask = select_candidate_blocks(scores, torch.tensor([[6]]), topk_blocks=2, block_size=2)
    assert mask.tolist() == [[True, True, False, False, True, True, False, False]]


def test_index_topk_is_chronological_and_marks_unreachable_slots():
    scores = torch.tensor([[1.0, 7.0, 3.0, -torch.inf, -torch.inf]])
    selected = select_index_topk(scores, torch.tensor([[3]]), index_topk=4)
    assert selected.tolist() == [[0, 1, 2, -1]]


def test_sparse_indices_are_padded_for_native_mla():
    selected = torch.tensor([[2, 7], [1, -1]], dtype=torch.int32)
    padded = pad_sparse_indices(selected, 4)
    assert padded.shape == (2, 1, 4)
    assert padded.tolist() == [[[2, 7, -1, -1]], [[1, -1, -1, -1]]]


def test_state_metadata_disables_ordinary_token_slots(config, runtime):
    specs = collect_specs(runtime)
    spec = specs["model.layers.2.self_attn.compressor.state_cache"]
    builder = AscendDSAV41MetadataBuilder(spec, [], runtime, torch.device("cpu"))
    slots = torch.tensor([7 * 16 + 15, 3 * 16, -1])
    common = SimpleNamespace(
        slot_mapping=slots,
        positions=None,
        block_table_tensor=torch.tensor([[7, 3]]),
        query_start_loc=torch.tensor([0, 2]),
        query_start_loc_cpu=torch.tensor([0, 2]),
        seq_lens=torch.tensor([17]),
        seq_lens_cpu=torch.tensor([17]),
        num_reqs=1,
        num_actual_tokens=2,
        num_input_tokens=2,
        max_query_len=2,
        max_seq_len=17,
        is_prefilling=torch.tensor([True]),
    )
    metadata = builder.build(0, common)
    assert metadata.is_compressor_state
    assert (metadata.slot_mapping == -1).all()
    assert metadata.compress_ratio == 1
    assert metadata.storage_block_size == 32
    assert metadata.max_query_len == 2
    assert metadata.max_seq_len == 17
    assert metadata.query_start_loc.tolist() == [0, 2]
    assert metadata.cache_seq_lens.tolist() == [17]
    assert metadata.cache_seq_lens is metadata.seq_lens
    assert metadata.num_prefills == 1
    assert metadata.num_prefill_tokens == 2


def test_slot_mapping_is_shared_per_compatible_cache_group(config, runtime):
    specs = collect_specs(runtime)
    common = SimpleNamespace(
        slot_mapping=torch.tensor([1, 2, 65, -1]),
        positions=None,
        block_table_tensor=torch.tensor([[5, 7]]),
        query_start_loc=torch.tensor([0, 4]),
        query_start_loc_cpu=torch.tensor([0, 4]),
        seq_lens=torch.tensor([4]),
        seq_lens_cpu=torch.tensor([4]),
        num_reqs=1,
        num_actual_tokens=3,
        num_input_tokens=4,
        max_query_len=4,
        max_seq_len=4,
        is_prefilling=torch.tensor([True]),
    )
    full_group_metadata: dict[str, Any] = {}
    long_metadata = AscendDSAV41MetadataBuilder(
        specs["model.layers.2.self_attn.long_kv_cache"],
        ["model.layers.2.self_attn.long_kv_cache"],
        runtime,
        torch.device("cpu"),
    ).build(0, common, common_v41_metadata=full_group_metadata)
    index_metadata = AscendDSAV41MetadataBuilder(
        specs["model.layers.2.self_attn.indexer.k_cache"],
        ["model.layers.2.self_attn.indexer.k_cache"],
        runtime,
        torch.device("cpu"),
    ).build(0, common, common_v41_metadata=full_group_metadata)

    assert long_metadata.slot_mapping.data_ptr() == index_metadata.slot_mapping.data_ptr()
    assert long_metadata.slot_mapping.tolist() == [
        [0, 0],
        [-1, -1],
        [1, 0],
        [-1, -1],
    ]

    # The SWA builder receives a different per-group publication dictionary,
    # so it owns an independent mapping computed from that group's flat slots.
    swa_metadata = AscendDSAV41MetadataBuilder(
        specs["model.layers.3.self_attn.swa_cache"],
        ["model.layers.3.self_attn.swa_cache"],
        runtime,
        torch.device("cpu"),
    ).build(0, common, common_v41_metadata={})
    assert swa_metadata.slot_mapping.data_ptr() != long_metadata.slot_mapping.data_ptr()
    assert swa_metadata.slot_mapping.tolist() == [
        [0, 1],
        [0, 2],
        [1, 1],
        [-1, -1],
    ]


def test_compressed_metadata_exposes_original_and_cache_coordinates(config, runtime):
    specs = collect_specs(runtime)
    spec = specs["model.layers.2.self_attn.long_kv_cache"]
    builder = AscendDSAV41MetadataBuilder(spec, [], runtime, torch.device("cpu"))
    # Request 0 starts halfway through a compression pair; request 1 ends
    # with an incomplete pair. Only completed pairs become cache rows.
    common = SimpleNamespace(
        slot_mapping=torch.tensor([1, 2, 3, 65, 66]),
        positions=torch.tensor([1, 2, 3, 1, 2]),
        block_table_tensor=torch.tensor([[5, 7], [9, 0]]),
        query_start_loc=torch.tensor([0, 3, 5]),
        query_start_loc_cpu=torch.tensor([0, 3, 5]),
        seq_lens=torch.tensor([4, 3]),
        seq_lens_cpu=torch.tensor([4, 3]),
        num_reqs=2,
        num_actual_tokens=5,
        num_input_tokens=5,
        max_query_len=3,
        max_seq_len=4,
        is_prefilling=torch.tensor([True, False]),
    )
    metadata = builder.build(0, common)
    assert metadata.seq_lens.tolist() == [4, 3]
    assert metadata.query_start_loc.tolist() == [0, 3, 5]
    assert metadata.cache_seq_lens.tolist() == [2, 1]
    assert metadata.cmp_residual.tolist() == [0, 1]
    assert metadata.max_cache_seq_len == 2
    assert metadata.slot_mapping.tolist() == [
        [0, 0],
        [-1, -1],
        [0, 1],
        [1, 0],
        [-1, -1],
    ]
    assert metadata.num_prefills == 1
    assert metadata.num_prefill_tokens == 3
    assert metadata.num_decodes == 1
    assert metadata.num_decode_tokens == 2


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("query_len", [1, 3])
def test_batch_metadata_reuses_work_and_keeps_group_slots_separate(runtime, monkeypatch, deferred, query_len):
    groups = make_cache_config(17).kv_cache_groups
    builders = []
    for group in groups:
        layers_by_spec: dict[Any, list[str]] = {}
        for name, spec in group.kv_cache_spec.kv_cache_specs.items():
            layers_by_spec.setdefault(spec, []).append(name)
        builders.append(
            [
                AscendDSAV41MetadataBuilder(spec, names, runtime, torch.device("cpu"))
                for spec, names in layers_by_spec.items()
            ]
        )
    assert sum(map(len, builders)) == 25
    counts = Mock(wraps=dsa_v41._request_counts)
    compressed_slots = Mock(wraps=dsa_v41.compressed_slot_mapping)
    rope = Mock(side_effect=lambda positions, **kwargs: (positions.float().clone(), -positions.float()))
    monkeypatch.setattr(dsa_v41, "_request_counts", counts)
    monkeypatch.setattr(dsa_v41, "compressed_slot_mapping", compressed_slots)
    monkeypatch.setattr(dsa_v41, "get_cos_and_sin_dsa", rope)

    def native_metadata(*args, **kwargs):
        lengths = kwargs.get("seqused_ori_kv", kwargs.get("seqused_k"))
        return torch.full(
            (dsa_v41.V41_METADATA_BUFFER_SIZE,), int(lengths.sum()) + kwargs["cmp_ratio"], dtype=torch.int32
        )

    smla = Mock(side_effect=native_metadata)
    qli = Mock(side_effect=native_metadata)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_mla_metadata", smla, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2_metadata", qli, raising=False)
    for group_builders in builders:
        for builder in group_builders:
            # Only native operator dispatch is mocked; all coordinates use CPU torch.
            builder._supports_device_ops = builder._cache_kind != "compressor_state"
            builder._device_metadata_enabled = deferred

    def build_batch(lengths, block_offset=0, idle=False):
        batch_shared: dict[str, Any] = {}
        results, tasks = [], []
        positions = torch.tensor(
            [*range(lengths[0] - query_len, lengths[0]), *range(lengths[1] - query_len, lengths[1]), 0]
        )
        query_start_loc = torch.tensor([0, query_len, 2 * query_len, 2 * query_len + 1], dtype=torch.int32)
        for gid, group_builders in enumerate(builders):
            block_ids = torch.tensor([gid + 1 + block_offset, gid + 2 + block_offset, 0], dtype=torch.int32)
            flat_slots = torch.cat(
                (
                    block_ids[0] * 128 + positions[:query_len],
                    block_ids[1] * 128 + positions[query_len : 2 * query_len],
                    torch.tensor([-1]),
                )
            )
            common = SimpleNamespace(
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc,
                seq_lens=torch.tensor([*lengths, 999], dtype=torch.int32),
                seq_lens_cpu=None,
                _seq_lens_cpu=torch.tensor([*lengths, 999], dtype=torch.int32),
                positions=positions,
                slot_mapping=flat_slots,
                block_table_tensor=block_ids[:, None],
                num_reqs=3,
                num_input_tokens=len(positions),
                num_actual_tokens=2 * query_len,
                max_query_len=query_len,
                max_seq_len=max(lengths),
                is_prefilling=torch.tensor([query_len > 1, query_len > 1, False]),
            )
            group_shared: dict[str, Any] = {}
            group_results = []
            for builder in group_builders:
                metadata = builder.build(
                    0,
                    common,
                    num_actual_reqs=2,
                    skip_ring_state_update=idle,
                    full_graph_mode=query_len == 1,
                    common_v41_metadata=group_shared,
                    common_v41_batch_metadata=batch_shared,
                )
                group_results.append(metadata)
                tasks.extend(builder.take_device_metadata_tasks())
            results.append(group_results)
            # Building later groups must never modify an earlier group's slots.
            if gid >= 2:
                expected = torch.stack((flat_slots.clamp_min(0) // 128, flat_slots.clamp_min(0) % 128), dim=1).int()
                expected[-1] = -1
                torch.testing.assert_close(group_results[0].slot_mapping, expected)
        for task in sorted(tasks, key=lambda task: task.stage):
            task.run()
        if deferred:
            assert [task.stage for task in tasks].count(DeviceMetadataStage.ATTENTION) == 3
            assert [task.stage for task in tasks].count(DeviceMetadataStage.INDEXER) == 2
            assert [task.stage for task in tasks].count(DeviceMetadataStage.COMPRESSOR) == 1
        else:
            assert not tasks
        return results, tuple((task.stage, task.group_id) for task in tasks)

    previous_pointers = previous_frontiers = None
    for iteration, (lengths, idle) in enumerate([([7, 8], False), ([10, 11], False), ([10, 11], True)]):
        results, frontiers = build_batch(lengths, block_offset=iteration, idle=idle)
        all_metadata = [metadata for group_results in results for metadata in group_results]
        assert counts.call_count == iteration + 1
        assert rope.call_count == iteration + 1
        assert compressed_slots.call_count == iteration + 1
        assert smla.call_count == 3 * (iteration + 1)
        assert qli.call_count == 2 * (iteration + 1)
        for metadata in all_metadata:
            assert metadata.seq_lens.tolist() == [*lengths, 0]
            assert metadata.seq_lens is all_metadata[0].seq_lens
        c2 = [metadata for metadata in results[0] if metadata.compress_ratio == 2]
        assert c2[0].cache_seq_lens is c2[1].cache_seq_lens
        assert c2[0].cmp_residual is c2[1].cmp_residual
        assert c2[0].cache_seq_lens.tolist() == [n // 2 for n in lengths] + [0]
        assert c2[0].cmp_residual.tolist() == [n % 2 for n in lengths] + [0]
        assert c2[0].max_cache_seq_len == max(lengths) // 2
        for metadata in results[0]:
            if metadata.compress_ratio == 1:
                assert metadata.cache_seq_lens is metadata.seq_lens
        swa = [metadata for group_results in results[2:] for metadata in group_results]
        assert all(metadata.cos is swa[0].cos and metadata.sin is swa[0].sin for metadata in swa)
        assert all(metadata.smla_metadata is swa[0].smla_metadata for metadata in swa)
        assert int(swa[0].smla_metadata[0]) == sum(lengths)
        assert len({group_results[0].slot_mapping.data_ptr() for group_results in results[2:]}) == 10
        for group_results in results[2:]:
            assert group_results[0].slot_mapping is group_results[1].slot_mapping
        if idle:
            assert (c2[0].slot_mapping == -1).all()
            assert (results[1][0].c2_ring_metadata[1] == 0).all()
        pointers = tuple(
            (
                metadata.seq_lens.data_ptr(),
                metadata.cache_seq_lens.data_ptr(),
                metadata.slot_mapping.data_ptr(),
                None if metadata.smla_metadata is None else metadata.smla_metadata.data_ptr(),
            )
            for metadata in all_metadata
        )
        if previous_pointers is not None:
            assert pointers == previous_pointers
            assert frontiers == previous_frontiers
        previous_pointers, previous_frontiers = pointers, frontiers


@pytest.mark.parametrize("end", [127, 128, 129, 255, 256, 257])
def test_merged_metadata_preserves_nonconsecutive_block_ids(runtime, end):
    runtime.cache_config.block_size = 128
    group = group_cache_specs(collect_specs(runtime))[0]
    table = torch.tensor([[7, 19, 3]], dtype=torch.int32)
    positions = torch.arange(end - 3, end)
    original_slots = table[0, positions // 128] * 128 + positions % 128
    common = SimpleNamespace(
        slot_mapping=original_slots,
        positions=positions,
        block_table_tensor=table,
        query_start_loc=torch.tensor([0, 3]),
        query_start_loc_cpu=torch.tensor([0, 3]),
        seq_lens=torch.tensor([end]),
        seq_lens_cpu=torch.tensor([end]),
        num_reqs=1,
        num_actual_tokens=3,
        num_input_tokens=3,
        max_query_len=3,
        max_seq_len=end,
        is_prefilling=torch.tensor([True]),
    )
    for name, spec in group.kv_cache_specs.items():
        metadata = AscendDSAV41MetadataBuilder(spec, [name], runtime, torch.device("cpu")).build(0, common)
        ratio = spec.tokens_per_state
        rows = 128 // ratio
        expected = table[0, positions // 128] * rows + (positions % 128) // ratio
        expected = torch.where((positions + 1) % ratio == 0, expected, -1)
        valid = expected >= 0
        physical = expected.clamp_min(0)
        expected_2d = torch.stack(
            (physical // get_storage_block_size(spec), physical % get_storage_block_size(spec)),
            dim=-1,
        ).to(torch.int32)
        expected_2d[~valid] = -1
        torch.testing.assert_close(metadata.slot_mapping, expected_2d)
        assert metadata.logical_block_size == 128
        assert metadata.storage_block_size == rows
        assert metadata.cache_seq_lens.tolist() == [end // ratio]
        torch.testing.assert_close(metadata.block_table, table)
    torch.testing.assert_close(common.slot_mapping, original_slots)


@pytest.mark.parametrize("end", [15, 16, 17, 31, 32, 33, 127, 128, 129, 255, 256, 257])
def test_state_boundary_mapping_with_padded_pages(runtime, end):
    group = group_cache_specs(collect_specs(runtime))[1]
    positions = torch.arange(end - 2, end)
    common = SimpleNamespace(
        slot_mapping=torch.full((2,), -1),
        block_table_tensor=torch.tensor([[7]], dtype=torch.int32),
        positions=positions,
        query_start_loc=torch.tensor([0, 2]),
        query_start_loc_cpu=torch.tensor([0, 2]),
        seq_lens=torch.tensor([end]),
        seq_lens_cpu=torch.tensor([end]),
        num_reqs=1,
        num_actual_tokens=2,
        num_input_tokens=2,
        max_query_len=2,
        max_seq_len=end,
        is_prefilling=torch.tensor([True]),
    )
    spec = next(iter(group.kv_cache_specs.values()))
    metadata = AscendDSAV41MetadataBuilder(spec, [], runtime, torch.device("cpu")).build(0, common)
    assert metadata.slot_mapping.tolist() == [-1, -1]
    assert metadata.storage_block_size == metadata.logical_block_size == 32
    assert metadata.c2_ring_metadata[:, 0].tolist() == [end - 2, 2, 0, 0, 7]
    assert metadata.c2_source_positions.tolist() == [int(p - 1) if p % 2 else 0 for p in positions]


def test_actual_attention_parameter_ownership(config, runtime):
    topology = build_layer_plan(config)
    assert not topology.layer(0).has_long_context
    assert topology.layer(2).is_kv_source and topology.layer(2).is_index_source
    assert topology.layer(20).is_kv_source and topology.layer(20).compress_ratio == 1
    assert topology.layer(24).is_index_source and not topology.layer(24).is_kv_source
    assert not topology.layer(26).is_index_source
    assert topology.layer(26).kv_source_layer == 20


@pytest.mark.parametrize("chunks", [(1, 1, 1, 2, 2), (3, 4), (2, 2, 3), (7,)])
@torch.inference_mode()
def test_compressor_chunk_boundary_matches_vector_reference(config, chunks):
    torch.manual_seed(7)
    compressor = DeepseekV41Compressor(config, 2)
    x = torch.randn(7, 16, dtype=torch.bfloat16)
    kv = compressor.wkv(x.float())[:6].reshape(3, 2, 8)
    gate = compressor.wgate(x.float())[:6].reshape(3, 2, 8)
    expected = compressor.norm((kv * gate.softmax(dim=1)).sum(dim=1).to(x.dtype))
    state = torch.full((6, 32, 16), float("nan"), dtype=torch.float32)
    block_table = [4]
    actual = []
    start = 0
    for size in chunks:
        actual.append(compressor_ratio2_reference(compressor, x[start : start + size], start, state, block_table))
        start += size
    torch.testing.assert_close(torch.cat(actual), expected)
    torch.testing.assert_close(state[4, 6, :8], compressor.wkv(x[-1:].float())[0])


@pytest.mark.parametrize(
    "num_tokens,start,dtype",
    [
        (1, 0, torch.float32),
        (1, 1, torch.bfloat16),
        (2, 0, torch.float16),
        (3, 1, torch.float32),
        (5, 0, torch.bfloat16),
    ],
)
def test_ring_source_reuses_prepared_store_coordinates(monkeypatch, num_tokens, start, dtype):
    from vllm_ascend.attention import dsa_v41

    positions = torch.arange(start, start + num_tokens)
    completed = positions.remainder(2) == 1
    slots = torch.tensor([[7, 63], [19, 0], [19, 1], [3, 0], [3, 1]], dtype=torch.int32)[:num_tokens]
    slots[~completed] = -1
    original_slots = slots.clone()
    rope = torch.zeros(num_tokens, 1, 2)
    state = SimpleNamespace(
        c2_ring_metadata=torch.zeros(5, 1, dtype=torch.int32),
        c2_metadata_group_id="ring",
        c2_source_cos=rope,
        c2_source_sin=rope,
    )
    events = []
    hidden_states = torch.randn(num_tokens, 8, dtype=dtype)
    original_hidden_states = hidden_states.clone()

    def pool(kv, score, metadata):
        assert kv.dtype == score.dtype == torch.float32
        # Identity projections must receive the same FP32 conversion.
        assert kv is score
        torch.testing.assert_close(kv, hidden_states.float(), rtol=0, atol=0)
        if dtype == torch.float32:
            assert kv is hidden_states
        assert metadata is state
        events.append("pool")
        return kv.to(torch.bfloat16)

    expected = slots.clone()

    def update_keys(latent, coordinates, cos, sin):
        events.append("index")
        assert coordinates.data_ptr() == slots.data_ptr()
        torch.testing.assert_close(coordinates, expected)

    def store(cache, coordinates, values):
        events.append("kv")
        assert coordinates.data_ptr() == slots.data_ptr()
        torch.testing.assert_close(coordinates, expected)

    monkeypatch.setattr(dsa_v41, "wait_for_device_metadata", lambda *args: events.append("wait"))
    monkeypatch.setattr(dsa_v41, "scatter_cache_sk", store)
    monkeypatch.setattr(torch.ops._C_ascend, "inplace_partial_rotary_mul", lambda *args, **kwargs: None, raising=False)
    attn = SimpleNamespace(
        compressor=SimpleNamespace(wkv=lambda x: x, wgate=lambda x: x, pool_projected=pool),
        indexer=SimpleNamespace(update_keys=update_keys),
        long_kv_cache=SimpleNamespace(kv_cache=[torch.empty(0)]),
        head_dim=8,
        nope_head_dim=6,
    )
    cache = SimpleNamespace(slot_mapping=slots)
    metadata = SimpleNamespace(
        compressor=SimpleNamespace(cache=cache, state=state),
        indexer=SimpleNamespace(cache=cache),
    )
    AscendDSAV41Impl._write_compressed_source(
        SimpleNamespace(role=SimpleNamespace(compress_ratio=2)),
        attn,
        hidden_states,
        positions,
        rope,
        rope,
        metadata,
    )
    assert events == ["wait", "pool", "index", "kv"]
    torch.testing.assert_close(slots, original_slots)
    torch.testing.assert_close(hidden_states, original_hidden_states, rtol=0, atol=0)


def test_state_uses_one_ring_page_and_block_table_entry(config, runtime):
    from vllm.v1.kv_cache_interface import CircularBufferSpec

    spec = collect_specs(runtime)["model.layers.2.self_attn.compressor.state_cache"]
    assert isinstance(spec, CircularBufferSpec)
    assert spec.tokens_per_state == 1 and not spec.prefix_cacheable
    assert get_storage_block_size(spec) == 32
    assert spec.page_size_bytes == 32 * 16 * 4
    assert spec.max_num_blocks_per_req(runtime, 1024) == 1
    assert spec.max_memory_usage_bytes(runtime) == spec.page_size_bytes


def test_projected_model_entry_keeps_fp32_state_and_existing_norm(config, monkeypatch):
    compressor = DeepseekV41Compressor(config, 2)
    compressor.register_buffer("_ring_pooled", torch.empty(4, 8, dtype=torch.bfloat16), persistent=False)
    compressor._ring_num_cores = 1
    state = torch.zeros(3, 32, 1, 16, dtype=torch.float32)
    compressor.state_cache = SimpleNamespace(kv_cache=[state])
    metadata = SimpleNamespace(c2_ring_metadata=torch.zeros(5, 1, dtype=torch.int32), max_query_len=2)
    pooled = torch.randn(2, 8, dtype=torch.bfloat16)
    expected = compressor.norm(pooled).clone()
    pointer = compressor._ring_pooled.data_ptr()

    def kernel(kv, scores, state_view, controls, out, **kwargs):
        assert kv.dtype == scores.dtype == state_view.dtype == torch.float32
        assert state_view.data_ptr() == state.data_ptr()
        assert controls is metadata.c2_ring_metadata
        assert out.data_ptr() == pointer and out.dtype == torch.bfloat16
        out.copy_(pooled)
        return out

    monkeypatch.setattr("vllm_ascend.ops.triton.compressor.compressor_triton.compressor_from_projected", kernel)
    hidden = torch.randn(2, 16, dtype=torch.bfloat16)
    actual = compressor.pool_projected(compressor.wkv(hidden.float()), compressor.wgate(hidden.float()), metadata)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert compressor.wkv.weight.dtype == compressor.wgate.weight.dtype == torch.float32


@torch.inference_mode()
def test_compressor_rejects_missing_previous_state_page(config):
    compressor = DeepseekV41Compressor(config, 2)
    state = torch.full((3, 32, 16), float("nan"), dtype=torch.float32)
    with pytest.raises(ValueError, match="absent/null"):
        compressor_ratio2_reference(
            compressor,
            torch.zeros(1, 16, dtype=torch.bfloat16),
            1,
            state,
            [0],
        )


@torch.inference_mode()
def test_state_page_reuse_does_not_require_request_reset(config):
    compressor = DeepseekV41Compressor(config, 2)
    state = torch.full((3, 32, 16), float("nan"), dtype=torch.float32)
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    expected = compressor_ratio2_reference(compressor, x, 0, state, [1]).clone()
    state[1].fill_(12345)
    actual = compressor_ratio2_reference(compressor, x, 0, state, [1])
    torch.testing.assert_close(actual, expected)


def test_state_registers_circular_manager():
    from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    assert (
        KVCacheSpecRegistry.get_manager_class(
            CircularBufferSpec(block_size=32, num_kv_heads=1, head_size=16, head_size_v=0, dtype=torch.float32)
        )
        is CircularBufferManager
    )


@torch.inference_mode()
def test_interleaved_request_state_isolation(config):
    compressor = DeepseekV41Compressor(config, 2)
    state = torch.full((3, 32, 16), float("nan"), dtype=torch.float32)
    first = torch.randn(2, 16, dtype=torch.bfloat16)
    second = torch.randn(2, 16, dtype=torch.bfloat16)
    compressor_ratio2_reference(compressor, first[:1], 0, state, [1])
    saved = state[1, 0].clone()
    compressor_ratio2_reference(compressor, second, 0, state, [2])
    torch.testing.assert_close(state[1, 0], saved)
    actual = compressor_ratio2_reference(compressor, first[1:], 1, state, [1])
    expected = compressor_ratio2_reference(compressor, first, 0, state, [1])
    torch.testing.assert_close(actual, expected)


class _CPCommon(SimpleNamespace):
    def replace(self, **kwargs):
        return type(self)(**(vars(self) | kwargs))


def _cp_common():
    # The second request resumes in the middle of a ratio-2 pair.
    return _CPCommon(
        slot_mapping=torch.tensor([0, 1, 2, 68]),
        block_table_tensor=torch.tensor([[0, 1], [1, 2]]),
        query_start_loc=torch.tensor([0, 3, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 3, 4], dtype=torch.int32),
        seq_lens=torch.tensor([3, 5], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([3, 5], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=4,
        num_input_tokens=4,
        max_query_len=3,
        max_seq_len=5,
        positions=torch.tensor([0, 1, 2, 4]),
        is_prefilling=torch.tensor([True, True]),
        causal=True,
    )


@pytest.mark.parametrize(
    "rank,size,query_offsets,seq_lens,positions",
    [
        (0, 2, [0, 2, 2], [2, 0], [0, 1]),
        (1, 2, [0, 1, 2], [3, 5], [2, 4]),
        (5, 8, [0, 0, 0], [0, 0], []),
    ],
)
def test_v41_cp_metadata_preserves_global_compression_and_local_causality(
    runtime, monkeypatch, rank, size, query_offsets, seq_lens, positions
):
    from vllm_ascend.attention.context_parallel.dsa_v41_cp import AscendDSAV41CPMetadataBuilder

    monkeypatch.setattr(
        "vllm_ascend.attention.context_parallel.dsa_cp.get_tp_group",
        lambda: SimpleNamespace(world_size=size, rank_in_group=rank),
    )
    spec = collect_specs(runtime)["model.layers.2.self_attn.long_kv_cache"]
    builder = AscendDSAV41CPMetadataBuilder(spec, [], runtime, torch.device("cpu"))
    metadata = builder.build(0, _cp_common(), common_v41_metadata={}, common_v41_batch_metadata={})
    assert metadata.query_start_loc.tolist() == query_offsets
    assert metadata.query_start_loc.dtype == torch.int32
    pointer = metadata.query_start_loc.data_ptr()
    assert builder.build(0, _cp_common()).query_start_loc.data_ptr() == pointer
    assert metadata.seq_lens.tolist() == seq_lens
    assert metadata.positions.tolist() == positions
    assert metadata.global_metadata.seq_lens.tolist() == [3, 5]
    assert metadata.global_metadata.cache_seq_lens.tolist() == [1, 2]
    assert metadata.global_metadata.slot_mapping.tolist() == [[-1, -1], [0, 0], [-1, -1], [-1, -1]]
    assert metadata.num_actual_tokens == len(positions)


@pytest.mark.parametrize("rank", [0, 1])
def test_v41_cp_uses_device_seq_lens_when_cpu_mirror_is_upper_bound(runtime, monkeypatch, rank):
    from vllm_ascend.attention.context_parallel.dsa_v41_cp import AscendDSAV41CPMetadataBuilder

    monkeypatch.setattr(
        "vllm_ascend.attention.context_parallel.dsa_cp.get_tp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=rank),
    )
    # A speculative rejection has corrected device lengths and positions;
    # the host mirror still describes the optimistic upper bound.
    common = _cp_common().replace(
        seq_lens=torch.tensor([7, 9], dtype=torch.int32),
        seq_lens_cpu=None,
        _seq_lens_cpu=torch.tensor([9, 11], dtype=torch.int32),
        positions=torch.tensor([4, 5, 6, 8]),
        max_seq_len=11,
    )
    spec = collect_specs(runtime)["model.layers.2.self_attn.long_kv_cache"]
    builder = AscendDSAV41CPMetadataBuilder(spec, [], runtime, torch.device("cpu"))
    metadata = builder.build(0, common)
    expected = [6, 0] if rank == 0 else [7, 9]
    assert metadata.global_metadata.seq_lens.tolist() == [7, 9]
    assert metadata.seq_lens.tolist() == expected
    assert metadata.cache_seq_lens.tolist() == [n // 2 for n in expected]
    assert metadata.cmp_residual.tolist() == [n % 2 for n in expected]


@pytest.mark.parametrize("rank", [0, 1, 3, 7])
@pytest.mark.parametrize("prefill", [False, True])
def test_v41_cp_rope_preserves_global_rows_across_builds(runtime, monkeypatch, rank, prefill):
    from vllm_ascend.attention.context_parallel.dsa_v41_cp import AscendDSAV41CPMetadataBuilder
    from vllm_ascend.ops import rope_dsv4 as rope

    monkeypatch.setattr(
        "vllm_ascend.attention.context_parallel.dsa_cp.get_tp_group",
        lambda: SimpleNamespace(world_size=8, rank_in_group=rank),
    )
    state = rope.RopeGlobalState()
    full = torch.arange(128, dtype=torch.float32).reshape(128, 1, 1, 1)
    state.full_rope_cache["test"] = (full, full + 1000)
    state.runtime_buffer["test"] = {"default": (torch.zeros(8, 1, 1, 1), torch.zeros(8, 1, 1, 1))}
    state.registry_summary["test"] = {"default"}
    state.layer_info["test.layer"] = ("test", ["default"])
    monkeypatch.setattr(rope, "_ROPE_STATE", state)
    calls = []

    def gather(positions, **kwargs):
        calls.append(positions.clone())
        return rope.get_cos_and_sin_dsa(positions, **kwargs)

    monkeypatch.setattr("vllm_ascend.attention.dsa_v41.get_cos_and_sin_dsa", gather)
    spec = collect_specs(runtime)["model.layers.0.self_attn.swa_cache"]
    builder = AscendDSAV41CPMetadataBuilder(spec, [], runtime, torch.device("cpu"))
    pointer = None
    for step in (0, 3):
        positions = torch.tensor([10, 20, 30, 40]) + step
        offsets = torch.arange(5, dtype=torch.int32)
        common = _cp_common().replace(
            positions=positions,
            num_reqs=4,
            query_start_loc=offsets,
            query_start_loc_cpu=offsets,
            seq_lens=(positions + 1).int(),
            seq_lens_cpu=(positions + 1).int(),
            max_query_len=1,
            max_seq_len=int(positions.max()) + 1,
            block_table_tensor=torch.zeros(4, 2, dtype=torch.int32),
            is_prefilling=torch.full((4,), prefill),
        )
        metadata = builder.build(0, common)
        global_cos = metadata.global_metadata.cos["test.layer"]
        local_cos = metadata.cos["test.layer"]
        local_sin = metadata.sin["test.layer"]
        expected = positions[rank : rank + 1].float()
        torch.testing.assert_close(global_cos.flatten(), positions.float())
        torch.testing.assert_close(local_cos.flatten(), expected)
        torch.testing.assert_close(local_sin.flatten(), expected + 1000)
        if not prefill:
            if pointer is not None:
                assert global_cos.data_ptr() == pointer
            pointer = global_cos.data_ptr()
        if expected.numel():
            assert local_cos.data_ptr() == global_cos.data_ptr() + rank * global_cos.element_size()
    assert len(calls) == 2  # One global gather per build, including empty local ranks.


@pytest.mark.parametrize(
    "cache_name,field,stage",
    [
        ("model.layers.0.self_attn.swa_cache", "smla_metadata", DeviceMetadataStage.ATTENTION),
        ("model.layers.2.self_attn.indexer.k_cache", "qli_metadata", DeviceMetadataStage.INDEXER),
        ("model.layers.2.self_attn.compressor.state_cache", "c2_ring_metadata", DeviceMetadataStage.COMPRESSOR),
    ],
)
def test_v41_cp_builds_device_controls_only_on_consuming_side(runtime, monkeypatch, cache_name, field, stage):
    from vllm_ascend.attention.context_parallel.dsa_v41_cp import AscendDSAV41CPMetadataBuilder

    monkeypatch.setattr(
        "vllm_ascend.attention.context_parallel.dsa_cp.get_tp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=0),
    )
    builder = AscendDSAV41CPMetadataBuilder(collect_specs(runtime)[cache_name], [], runtime, torch.device("cpu"))
    global_builder = builder._global_builder
    compressor = stage == DeviceMetadataStage.COMPRESSOR
    for side in (builder, global_builder):
        # Queue native metadata operations without invoking NPU kernels on CPU.
        side._device_metadata_enabled = True
        side._supports_device_ops = not compressor
    assert global_builder._smla_metadata.numel() == 0
    assert global_builder._qli_metadata.numel() == 0
    assert builder._c2_ring_metadata.numel() == 0
    assert builder._c2_complete_mask.numel() == 0
    assert builder._c2_source_positions.numel() == 0
    assert builder._c2_source_cos.numel() == 0
    assert builder._c2_source_sin.numel() == 0
    for _ in range(2):
        metadata = builder.build(0, _cp_common(), common_v41_metadata={}, common_v41_batch_metadata={})
        owner, unused = (metadata.global_metadata, metadata) if compressor else (metadata, metadata.global_metadata)
        assert getattr(owner, field) is not None
        assert getattr(unused, field) is None
        tasks = builder.take_device_metadata_tasks()
        assert len(tasks) == 1
        assert tasks[0].stage == stage
        assert builder.take_device_metadata_tasks() == ()


@pytest.mark.parametrize("local_tokens", [0, 1, 2])
@pytest.mark.parametrize("num_tokens", [3, 4])
def test_v41_cp_output_exchange_only_pads_partial_ranks(monkeypatch, local_tokens, num_tokens):
    from vllm_ascend.attention.context_parallel.dsa_v41_cp import AscendDSAV41CPImpl

    impl = AscendDSAV41CPImpl("layer", SimpleNamespace(is_kv_source=False), None, None, None)
    calls = []
    monkeypatch.setattr("vllm_ascend.attention.context_parallel.dsa_v41_cp.get_tp_group", lambda: None)

    def exchange(tensor, group):
        calls.append(tensor)
        return torch.ones((4, 2, 3))

    monkeypatch.setattr("vllm_ascend.attention.context_parallel.dsa_v41_cp.restore_tp_heads", exchange)

    def project(tensor, output):
        assert output is destination
        assert tensor.shape == (num_tokens, 2, 3)
        output.copy_(tensor.flatten(1))

    projection = SimpleNamespace(_forward_o_proj=project)
    attn = SimpleNamespace(dsa_attn=SimpleNamespace(dsa_attn=SimpleNamespace(impl=projection)))
    destination = torch.empty((num_tokens, 6))
    local_output = torch.ones((local_tokens, 4, 3))
    output = impl._project_output(
        attn,
        local_output,
        torch.empty((num_tokens, 6)),
        SimpleNamespace(swa=SimpleNamespace(cp_token_range=(0, 2, 2, 4))),
        projected=destination,
    )
    assert output.data_ptr() == destination.data_ptr()
    assert len(calls) == 1
    assert calls[0].shape == (2, 4, 3)
    assert (calls[0] is local_output) == (local_tokens == 2)
    torch.testing.assert_close(calls[0][:local_tokens], local_output)
    assert torch.count_nonzero(calls[0][local_tokens:]) == 0
    assert output.shape == (num_tokens, 6)
    torch.testing.assert_close(output, torch.ones_like(destination))


def test_v41_cp_consumers_reuse_local_topk_and_candidates():
    from vllm_ascend.attention.context_parallel.dsa_v41_cp import AscendDSAV41CPImpl

    impl = AscendDSAV41CPImpl(
        "layer", SimpleNamespace(is_kv_source=False, has_long_context=True, is_index_source=False), None, None, None
    )
    indices = torch.tensor([[0, 2], [1, 3]])
    candidates = torch.tensor([[True, False]])
    shared = SimpleNamespace(topk_indices=indices, candidates=candidates)
    actual = impl._select_sparse_indices(
        SimpleNamespace(shared_state=shared),
        torch.empty(16, 1),
        torch.empty(2, 1),
        None,
        None,
        None,
        None,
    )
    assert actual.data_ptr() == indices.data_ptr()
    torch.testing.assert_close(actual, indices)
    assert shared.candidates is candidates


@pytest.mark.parametrize("cp", [False, True])
def test_v41_backend_routes_metadata_and_execution_together(monkeypatch, cp):
    from vllm_ascend.attention.context_parallel import dsa_v41_cp
    from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheBackend

    monkeypatch.setattr(dsa_v41_cp, "enable_dsa_cp", lambda: cp)
    builder, impl = dsa_v41_cp.get_v41_cp_classes()
    assert DeepseekV41CacheBackend.get_builder_cls() is builder
    assert not DeepseekV41CacheBackend.supports_pcp()
    if cp:
        assert issubclass(impl, dsa_v41_cp.AscendDSAV41CPImpl)


def test_v41_cp_accepts_async_seq_lens_mirror(runtime, monkeypatch):
    from vllm_ascend.attention.context_parallel.dsa_v41_cp import AscendDSAV41CPMetadataBuilder

    monkeypatch.setattr(
        "vllm_ascend.attention.context_parallel.dsa_cp.get_tp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=1),
    )
    common = _cp_common()
    common._seq_lens_cpu = common.seq_lens_cpu
    common.seq_lens_cpu = None
    spec = collect_specs(runtime)["model.layers.2.self_attn.long_kv_cache"]
    metadata = AscendDSAV41CPMetadataBuilder(spec, [], runtime, torch.device("cpu")).build(0, common)
    assert metadata.seq_lens.tolist() == [3, 5]
    assert metadata.global_metadata.cache_seq_lens.tolist() == [1, 2]


def test_v41_cp_resolves_own_planes_with_native_draft_metadata_present():
    from vllm_ascend.attention.context_parallel.dsa_v41_cp import AscendDSAV41CPImpl

    impl = AscendDSAV41CPImpl(
        prefix="model.layers.0.self_attn",
        role=SimpleNamespace(is_kv_source=False, compress_ratio=0),
        topology=None,
        long_kv_source_prefix=None,
        index_k_source_prefix=None,
    )
    global_swa = object()
    metadata = {
        impl.swa_prefix: SimpleNamespace(global_metadata=global_swa),
        "mtp.0.self_attn.swa_cache": SimpleNamespace(seq_lens=torch.tensor([4])),
    }
    assert impl._global_layer_metadata(metadata).swa is global_swa


@pytest.mark.parametrize("overlap", [False, True])
def test_v41_query_preparation_uses_multistream(monkeypatch, overlap):
    from unittest.mock import Mock

    from vllm_ascend.attention import dsa_v41 as dsa_v41_module
    from vllm_ascend.attention.dsa_v41 import AscendDSAV41Impl

    impl = AscendDSAV41Impl.__new__(AscendDSAV41Impl)
    impl.role = SimpleNamespace(is_kv_source=True)
    impl.multistream_preprocess = Mock(return_value=("q", "qr"))
    impl.preprocess = Mock(return_value=("q", "qr"))
    impl._write_compressed_source = Mock()
    # The overlap decision comes from the ascend config, not from the layer.
    monkeypatch.setattr(dsa_v41_module, "v41_multistream_preprocess_enabled", lambda: overlap)
    attn = SimpleNamespace(
        dsa_attn=SimpleNamespace(dsa_attn=SimpleNamespace(impl=SimpleNamespace(multistream_dsv4_dsa_overlap=overlap)))
    )
    metadata = SimpleNamespace(swa=SimpleNamespace(num_actual_tokens=6))
    assert impl._prepare_queries(attn, "hidden", "positions", "cos", "sin", metadata) == ("q", "qr")
    chosen = impl.multistream_preprocess if overlap else impl.preprocess
    other = impl.preprocess if overlap else impl.multistream_preprocess
    chosen.assert_called_once_with(attn, "hidden", "cos", "sin", metadata.swa)
    other.assert_not_called()
    impl._write_compressed_source.assert_called_once_with(attn, "hidden", "positions", "cos", "sin", metadata)


@pytest.mark.parametrize("overlap", [False, True])
def test_v41_cp_query_preparation_uses_full_inputs(overlap):
    from unittest.mock import Mock

    from vllm_ascend.attention.context_parallel.dsa_v41_cp import AscendDSAV41CPImpl

    impl = AscendDSAV41CPImpl.__new__(AscendDSAV41CPImpl)
    impl.multistream_preprocess = Mock(return_value=("q", "qr"))
    impl._write_compressed_source = Mock()
    attn = SimpleNamespace(
        dsa_attn=SimpleNamespace(dsa_attn=SimpleNamespace(impl=SimpleNamespace(multistream_dsv4_dsa_overlap=overlap)))
    )
    metadata = SimpleNamespace(swa=SimpleNamespace(num_actual_tokens=2, cp_token_range=(2, 4, 2, 6)))
    assert impl._prepare_queries(attn, "abcdef", "positions", "cos", "sin", metadata) == ("q", "qr")
    impl.multistream_preprocess.assert_called_once_with(attn, "abcdef", "cos", "sin", metadata.swa)
    impl._write_compressed_source.assert_not_called()


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("local_tokens", [0, 2])
def test_v41_cp_input_preparation_updates_empty_rank_cache(overlap, local_tokens):
    from unittest.mock import Mock

    from vllm_ascend.attention.context_parallel.dsa_v41_cp import AscendDSAV41CPImpl

    impl = AscendDSAV41CPImpl.__new__(AscendDSAV41CPImpl)
    full = torch.arange(24).reshape(6, 4)
    global_metadata = SimpleNamespace(swa=SimpleNamespace(num_actual_tokens=5))
    impl._global_layer_metadata = Mock(return_value=global_metadata)
    impl._update_caches = Mock()
    attn = SimpleNamespace(
        dsa_attn=SimpleNamespace(dsa_attn=SimpleNamespace(impl=SimpleNamespace(multistream_dsv4_dsa_overlap=overlap)))
    )
    metadata = SimpleNamespace(swa=SimpleNamespace(cp_token_range=(3, 6, 3, 6), num_actual_tokens=local_tokens))
    assert impl._prepare_inputs_and_caches(attn, full, metadata, {}) is None
    if local_tokens == 0:
        impl._update_caches.assert_called_once()
        assert torch.equal(impl._update_caches.call_args.args[1], full[:5])
        assert impl._update_caches.call_args.args[2] is global_metadata
    else:
        impl._update_caches.assert_not_called()


def test_v41_cp_inherits_forward():
    from vllm_ascend.attention.context_parallel.dsa_v41_cp import AscendDSAV41CPImpl
    from vllm_ascend.attention.dsa_v41 import AscendDSAV41Impl

    assert AscendDSAV41CPImpl.forward is AscendDSAV41Impl.forward


@pytest.mark.parametrize("rank", [None, 0, 1, 7])
@pytest.mark.parametrize("first_seq_len", [3, 260])
def test_dspark_v41_noncausal_metadata_preserves_full_visible_block(runtime, monkeypatch, rank, first_seq_len):
    from vllm_ascend.attention.context_parallel.dsa_v41_cp import AscendDSAV41CPMetadataBuilder

    runtime.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    spec = AscendSlidingWindowMLASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        sliding_window=128,
        cache_dtype_str="bfloat16",
        model_version="deepseek_v41",
    )
    common = _cp_common().replace(
        causal=False,
        # Deliberately non-identity pages: logical positions must not be mapped
        # here because SparseFlashMLA performs the physical lookup itself.
        block_table_tensor=torch.tensor([[7, 3, 9], [5, 2, 8]], dtype=torch.int32),
        seq_lens=torch.tensor([first_seq_len, 5], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([first_seq_len, 5], dtype=torch.int32),
        max_seq_len=max(first_seq_len, 5),
        positions=torch.tensor([first_seq_len - 3, first_seq_len - 2, first_seq_len - 1, 4]),
    )
    native = Mock(return_value=torch.zeros(dsa_v41.V41_METADATA_BUFFER_SIZE, dtype=torch.int32))
    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_mla_metadata", native, raising=False)
    builder = AscendDSAV41MetadataBuilder(spec, [], runtime, torch.device("cpu"))
    builder._supports_device_ops = True
    full = builder.build_for_drafting(common, 1)
    torch.testing.assert_close(
        native.call_args.kwargs["ori_topk_length"],
        (full.ori_sparse_indices >= 0).sum(-1, dtype=torch.int32),
    )
    assert full.ori_topk_length is native.call_args.kwargs["ori_topk_length"]
    assert full.ori_mask_mode == 0
    assert full.ori_sparse_indices.shape[0] == 4
    # Every query of the first request can see its complete draft block.
    torch.testing.assert_close(full.ori_sparse_indices[0], full.ori_sparse_indices[2])
    expected = list(range(max(0, first_seq_len - 3 - 128), first_seq_len))
    assert full.ori_sparse_indices[0, 0, : len(expected)].tolist() == expected
    assert torch.all(full.ori_sparse_indices[0, 0, len(expected) :] == -1)
    assert full.ori_sparse_indices[3, 0, :5].tolist() == list(range(5))
    assert full.ori_topk_length[:, 0].tolist() == [len(expected)] * 3 + [5]
    if rank is None:
        return
    monkeypatch.setattr(
        "vllm_ascend.attention.context_parallel.dsa_cp.get_tp_group",
        lambda: SimpleNamespace(world_size=8, rank_in_group=rank),
    )
    native.reset_mock()
    local_builder = AscendDSAV41CPMetadataBuilder(spec, [], runtime, torch.device("cpu"))
    local_builder._supports_device_ops = True
    local = local_builder.build_for_drafting(common, 1)
    if rank >= full.num_actual_tokens:
        native.assert_not_called()
        assert local.smla_metadata is local_builder._smla_metadata
        assert torch.count_nonzero(local.smla_metadata) == 0
    else:
        torch.testing.assert_close(
            native.call_args.kwargs["ori_topk_length"],
            (local.ori_sparse_indices >= 0).sum(-1, dtype=torch.int32),
        )
    torch.testing.assert_close(local.ori_sparse_indices, full.ori_sparse_indices[rank : rank + 1])
    assert local.seq_lens.tolist() == ([first_seq_len, 0] if rank < 3 else [0, 0])
    assert local.ori_mask_mode == 0
