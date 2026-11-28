# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode, KVTransferConfig
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs

from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import (
    plan_layerwise_cache_reuse,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
    get_layerwise_reuse_backing_bytes,
    get_layerwise_reuse_layout,
)


def make_layout(model="flash", *, block_size=32, cache_format="a3", buffers=2, independent=None, mtp=False):
    count = 43 if model == "flash" else 61
    ratios = ([0, 0] if model == "flash" else [128, 128]) + [4 if i % 2 == 0 else 128 for i in range(2, count)]
    hf = SimpleNamespace(model_type="deepseek_v4", num_hidden_layers=count, compress_ratios=ratios)
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
    if mtp:
        runtime.speculative_config = SimpleNamespace(method="mtp", num_speculative_tokens=1, use_eagle=lambda: True)
        runtime.num_hash_lookahead_tokens = 0
        specs["mtp.layers.0.self_attn.swa_cache"] = replace(specs["model.layers.0.self_attn.swa_cache"], is_draft=True)
    groups = AscendKVCacheConfigBuilder().get_kv_cache_groups(runtime, specs)
    # The core planner marks these after get_kv_cache_groups; mirror that step
    # so the coordinator applies draft lookahead only to the owning group(s).
    for group in groups:
        group.is_eagle_group = mtp and any(specs[name].is_draft for name in group.layer_names)
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


def allocate_test_views(monkeypatch, runtime, config, specs):
    from vllm_ascend.attention.dsa_v1 import AscendDSABackend
    from vllm_ascend.worker.v2 import attn_utils

    allocations = []

    def allocate(size, alignment, device):
        allocations.append(size)
        return torch.zeros(size, dtype=torch.int8, device=device)

    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: runtime)
    monkeypatch.setattr(attn_utils, "_allocate_int8_cache_tensor", allocate)
    # DSV4 attention, SWA, state and indexer backends inherit the same runtime
    # view method. Supply the layer registry and use the normal MRv2 entry.
    layer = SimpleNamespace(get_attn_backend=lambda: AscendDSABackend)
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_: dict.fromkeys(specs, layer))
    monkeypatch.setattr(attn_utils, "get_shared_kv_cache_layers", lambda _: {})
    kernel_block_sizes = [group.kv_cache_spec.block_size for group in config.kv_cache_groups]
    caches = attn_utils.allocate_kv_cache_main(config, torch.device("cpu"), None, kernel_block_sizes)
    return caches, allocations


@pytest.mark.parametrize("cache_format,model", [("a3", "flash"), ("a5_packed", "pro")])
def test_real_allocator_and_typed_views_keep_block_ownership(cache_format, model, monkeypatch):
    runtime, groups, specs = make_layout(model, cache_format=cache_format, independent=[0, 1, 4], mtp=True)
    runtime.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE)
    num_blocks = 2 * len(groups) + 1
    config = make_worker_config(runtime, groups, num_blocks=num_blocks)
    layout = get_layerwise_reuse_layout(config)
    assert layout is not None and layout.has_layer_reuse
    builder = AscendKVCacheConfigBuilder()
    cost = builder._get_pool_bytes_per_block(runtime, groups)
    assert 0 < cost < builder._get_native_pool_bytes_per_block(runtime, groups)
    caches, allocations = allocate_test_views(monkeypatch, runtime, config, specs)
    assert allocations == [num_blocks * cost]
    assert len({components[0].untyped_storage().data_ptr() for components in caches.values()}) == 1
    draft_group = next(gid for gid, group in enumerate(groups) if group.is_eagle_group)
    for component in caches["mtp.layers.0.self_attn.swa_cache"]:
        component[draft_group + 1].fill_(64)
    # Independent group IDs, as assigned from one global block pool. Check
    # simultaneous components of one layer; later layers deliberately alias.
    for layer in (0, 1, 2, 3, 4, 5, 42):
        written: list[tuple[torch.Tensor, int]] = []
        for gid, group in enumerate(groups):
            for name in group.layer_names:
                if f"model.layers.{layer}." not in name:
                    continue
                for component in caches[name]:
                    assert component.stride(0) * component.element_size() >= component[0].nbytes
                    # Two requests in this group have distinct global block IDs.
                    for block_id in (gid + 1, gid + 1 + len(groups)):
                        block = component[block_id]
                        value = len(written) + 1
                        block.fill_(value)
                        written.append((block, value))
        assert all(torch.all(block.float() == value) for block, value in written)
    assert all(torch.all(view[draft_group + 1].float() == 64) for view in caches["mtp.layers.0.self_attn.swa_cache"])


def test_push_from_rotating_a5_to_resident_peer_preserves_indexer_bytes(monkeypatch):
    import ctypes
    from unittest.mock import Mock

    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import base_worker
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_worker import MooncakePushConnectorWorker

    runtime, groups, specs = make_layout("pro", cache_format="a5_packed", independent=[0, 1], mtp=True)
    p_plan = make_worker_config(runtime, groups, 6)
    runtime.kv_transfer_config = None
    d_plan = make_worker_config(runtime, groups, 8)
    assert get_layerwise_reuse_layout(p_plan) and not get_layerwise_reuse_layout(d_plan)
    monkeypatch.setattr(base_worker, "global_te", Mock())

    def register(plan):
        caches, _ = allocate_test_views(monkeypatch, runtime, plan, specs)
        worker = object.__new__(MooncakePushConnectorWorker)
        worker.kv_cache_config = plan
        worker.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
        worker.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1))
        worker.tp_rank = worker.pcp_rank = 0
        worker.tp_size = worker.pcp_size = worker.dcp_size = 1
        worker.block_size = 32
        worker.engine_id = "test"
        worker.te_rpc_port = worker.handshake_port = 0
        worker.side_channel_host = "localhost"
        base_worker.MooncakeBaseConnectorWorker.register_kv_caches(worker, caches)
        return worker, caches

    producer, source = register(p_plan)
    consumer, target = register(d_plan)
    local, remote = producer.transfer_metadata, consumer.transfer_metadata
    assert dict(zip(local.layer_names, local.block_lens)) == dict(zip(remote.layer_names, remote.block_lens))

    name = "model.layers.2.self_attn.indexer.k_cache"
    src, dst = source[name], target[name]
    assert len(src) == len(dst) == 2
    src[0][2].view(torch.uint8).fill_(0x37)
    src[1][2].view(torch.uint8).fill_(0x49)
    for component in dst:
        component.view(torch.uint8).fill_(0x55)
    producer._layer_indices = {name: local.layer_names.index(name)}
    rank_metadata = SimpleNamespace(
        layer_indices=range(len(remote.layer_names)),
        kv_caches_base_addr=remote.kv_caches_base_addr,
        local_ip=remote.local_ip,
        te_rpc_port=remote.te_rpc_port,
    )
    pp = SimpleNamespace(
        layer_names=remote.layer_names,
        block_lens=remote.block_lens,
        block_strides=remote.block_strides,
        metadata_by_pcp_rank={0: SimpleNamespace(metadata_by_tp_rank={0: rank_metadata})},
    )
    batch = SimpleNamespace(
        layer_name=name,
        remote_metadata=SimpleNamespace(metadata_by_pp_rank={0: pp}, tp_size=1, pcp_size=1, dcp_size=1),
        remote_rank=(0, 0, 0),
        event=Mock(),
        requests=[SimpleNamespace(local_block_ids=[2], remote_block_ids=[4])],
    )

    def write(session, sources, destinations, lengths):
        # The Push worker coalesces the contiguous data and scale segments.
        assert src[1][2].data_ptr() == src[0][2].data_ptr() + src[0][2].nbytes
        assert dst[1][4].data_ptr() == dst[0][4].data_ptr() + dst[0][4].nbytes
        assert sources == [src[0][2].data_ptr()]
        assert destinations == [dst[0][4].data_ptr()]
        assert lengths == [sum(component[2].nbytes for component in src)]
        for address, destination, length in zip(sources, destinations, lengths, strict=True):
            ctypes.memmove(destination, address, length)
        return 0

    producer.engine = Mock()
    producer.engine.batch_transfer_sync_write.side_effect = write
    assert producer._execute_layer_transfer(batch) is None
    producer.engine.batch_transfer_sync_write.assert_called_once()
    for source_component, target_component in zip(src, dst, strict=True):
        assert torch.equal(target_component[4].view(torch.uint8), source_component[2].view(torch.uint8))
        assert (target_component.view(torch.uint8)[[3, 5]] == 0x55).all()

    # Incompatible peer metadata must fail before a partial WRITE is submitted.
    remote_index = remote.layer_names.index(name)
    pp.block_lens = [list(lengths) for lengths in remote.block_lens]
    pp.block_lens[remote_index][0] -= 1
    producer.engine.batch_transfer_sync_write.reset_mock()
    assert producer._execute_layer_transfer(batch) == "P/D cache slice lengths differ"
    producer.engine.batch_transfer_sync_write.assert_not_called()


def make_capacity_inputs(monkeypatch, model):
    from vllm.v1.core import kv_cache_planning

    runtime, groups, specs = make_layout(model, buffers=3, independent=[0, 1])
    runtime.model_config.original_max_model_len = runtime.model_config.max_model_len = 128
    runtime.attention_config = SimpleNamespace(hisparse_config=None)
    runtime.max_in_flight_tokens = 32
    runtime.num_prefill_lookahead_tokens = 0
    builder = AscendKVCacheConfigBuilder()
    monkeypatch.setattr(kv_cache_planning, "get_kv_cache_config_builder", lambda _: builder)
    return builder, runtime, groups, specs


def test_core_converges_worker_budgets_using_the_reused_physical_cost(monkeypatch):
    from vllm.v1.core import kv_cache_planning

    builder, runtime, groups, specs = make_capacity_inputs(monkeypatch, "flash")
    block_cost = builder._get_pool_bytes_per_block(runtime, groups)
    required = kv_cache_planning._max_memory_usage_bytes_from_groups(builder, runtime, groups)
    budget = required + block_cost  # Block zero is reserved even for a single request.
    assert budget < required // block_cost * builder._get_native_pool_bytes_per_block(runtime, groups)
    with pytest.raises(ValueError, match="KV cache"):
        kv_cache_planning.get_kv_cache_configs(runtime, [specs], [budget - 1], groups)
    budgets = [budget + 2 * block_cost, budget]
    plans = kv_cache_planning.get_kv_cache_configs(runtime, [specs, specs], budgets, groups)
    expected = min(budgets) // builder._get_pool_bytes_per_block(runtime, groups)
    assert [plan.num_blocks for plan in plans] == [expected, expected]
    assert plans[0].kv_cache_tensors == plans[1].kv_cache_tensors
    assert get_layerwise_reuse_backing_bytes(plans[0]) <= min(budgets)


def test_mtp_members_keep_hbm_and_enter_native_store_layers(request, monkeypatch):
    from unittest import TestCase

    from tests.ut.distributed.ascend_store import test_pool_worker

    runtime, groups, specs = make_layout("flash", mtp=True)
    pool_extra = runtime.kv_transfer_config.kv_connector_extra_config
    config = make_worker_config(runtime, groups, 8)
    draft = {name for name, spec in specs.items() if spec.is_draft}
    assert draft
    layout = get_layerwise_reuse_layout(config)
    assert draft == layout.draft_entry_names
    assert layout.reused_cache_names and not draft.intersection(layout.reused_cache_names)
    case = TestCase()
    request.addfinalizer(case.doCleanups)
    target_layers = runtime.model_config.hf_config.num_hidden_layers
    create_worker = test_pool_worker.create_kv_pool_worker

    def create(config, use_layerwise, kv_cache_config):
        config.model_config.hf_text_config = runtime.model_config.hf_text_config
        config.speculative_config = runtime.speculative_config
        return create_worker(config, use_layerwise, kv_cache_config)

    monkeypatch.setattr(test_pool_worker, "create_kv_pool_worker", create)
    worker = test_pool_worker.make_worker(
        case,
        num_layers=target_layers,
        num_hidden_layers=target_layers,
        use_layerwise=True,
        extra_config=pool_extra,
        kv_cache_config=config,
    )
    assert worker.layerwise_offload
    assert not worker._dspark_draft_layer_names
    for index, group in enumerate(groups):
        for name in draft.intersection(group.layer_names):
            physical = worker._extract_physical_layer_index(name)
            assert physical >= target_layers
            assert any(gid == index for gid, _ in worker.physical_layer_to_group_layers[physical])
