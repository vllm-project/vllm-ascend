# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm.v1.hisparse.backend")
from vllm.config import KVTransferConfig
from vllm.v1.core.kv_cache_planning import (
    DefaultKVCacheConfigBuilder,
    _max_memory_usage_bytes_from_groups,
    _pool_bytes_per_block,
    _pool_bytes_per_block_from_builder,
)
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.hisparse import layout as core_layout
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheGroupRole,
    MambaSpec,
    MLAAttentionSpec,
    SparseCacheRole,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec, register_ascend_kv_cache_specs
from vllm_ascend.hisparse import supports_hybrid_hisparse_core
from vllm_ascend.hisparse.layout import get_hybrid_sfa_hisparse_groups, has_hybrid_sfa_hisparse_groups

requires_hybrid_core = pytest.mark.skipif(
    not supports_hybrid_hisparse_core(), reason="The runtime layout requires Core Mamba page fitting"
)


@pytest.fixture(autouse=True)
def registered_specs():
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()


def make_runtime(model_type="unregistered_sparse_model", layout=None):
    return SimpleNamespace(
        attention_config=SimpleNamespace(
            hisparse_config=SimpleNamespace(device_buffer_size=None, eager_host_mirror=True),
        ),
        cache_config=SimpleNamespace(
            kv_cache_layout=layout,
            mamba_cache_mode="align",
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
            get_resolved_kv_cache_layout=lambda: layout,
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type), max_model_len=2048),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        speculative_config=None,
        max_in_flight_tokens=1024,
    )


def make_specs():
    return {
        "sparse": MLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=656,
            dtype=torch.float8_e4m3fn,
            cache_role=SparseCacheRole.SPARSE,
            state_content_bytes=656,
            block_stride_alignment=128 * 656,
            is_index_group_leader=True,
            topk_buffer_width=2048,
        ),
        "indexer": AscendSFAIndexerCacheSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float8_e4m3fn,
            cache_role=SparseCacheRole.INDEXER,
            cache_sparse_li_c8=True,
            scale_dim=1,
            scale_dtype=torch.float32,
        ),
        "state": MambaSpec(
            block_size=128,
            shapes=((25_000,),),
            dtypes=(torch.bfloat16,),
        ),
    }


@pytest.mark.parametrize("pooled", [False, True])
def test_ascendstore_rejects_new_hybrid_layouts(pooled):
    runtime = make_runtime()
    runtime.kv_transfer_config = KVTransferConfig(kv_connector="AscendStoreConnector", kv_role="kv_both")
    if pooled:
        runtime.model_config.hf_text_config = SimpleNamespace(index_kpool=4)
    specs = {} if pooled else make_specs()
    with pytest.raises(ValueError, match="AscendStore does not support pooled or recurrent HiSparse"):
        AscendKVCacheConfigBuilder().get_kv_cache_groups(runtime, specs)


@pytest.mark.parametrize("model_type", ["kimi_k31", "unregistered_sparse_model"])
@requires_hybrid_core
def test_native_hybrid_retains_separate_cache_geometry(model_type):
    specs = make_specs()
    groups = get_hybrid_sfa_hisparse_groups(make_runtime(model_type), specs)

    host, indexer, resident, hot, state = groups
    assert host.host_resident and host.role is KVCacheGroupRole.HISPARSE_SOURCE
    assert host.layer_names == ["sparse"]
    assert indexer.role is KVCacheGroupRole.HISPARSE_INDEXER
    native_indexer = indexer.kv_cache_spec.kv_cache_specs["indexer"]
    assert native_indexer.block_size == native_indexer.block_table_block_size == 128
    assert native_indexer.page_size_bytes == 128 * (128 + 4)
    assert isinstance(resident.kv_cache_spec, HiSparseResidentSpec)
    assert isinstance(hot.kv_cache_spec, HiSparseHotSpec)
    assert hot.kv_cache_spec.get_block_stride_alignment() == 128 * 656
    assert state.layer_names == ["state"]
    assert state.kv_cache_spec == specs["state"]
    assert all(not group.host_resident for group in groups[1:])
    assert has_hybrid_sfa_hisparse_groups(groups)
    assert has_hybrid_sfa_hisparse_groups(groups[1:])


@requires_hybrid_core
def test_layout_feedback_fits_more_state_layers_within_same_stride(monkeypatch):
    specs = {f"{name}.{i}": spec for i in range(6) for name, spec in make_specs().items()}
    provisional = get_hybrid_sfa_hisparse_groups(make_runtime(), specs)
    runtime = make_runtime(layout=KVCacheLayout.BLHNC)
    resolved = get_hybrid_sfa_hisparse_groups(runtime, specs)

    def state_counts(groups):
        return [len(group.layer_names) for group in groups if isinstance(group.kv_cache_spec, MambaSpec)]

    assert state_counts(provisional) == [2, 2, 2]
    assert state_counts(resolved) == [3, 3]
    assert _pool_bytes_per_block(runtime, provisional) == _pool_bytes_per_block(runtime, resolved)
    builder = DefaultKVCacheConfigBuilder()
    assert _max_memory_usage_bytes_from_groups(builder, runtime, resolved) < _max_memory_usage_bytes_from_groups(
        builder, runtime, provisional
    )

    monkeypatch.setattr(core_layout, "use_shared_hisparse_host_pool", lambda _: False)
    stride = _pool_bytes_per_block(runtime, resolved)
    materialized = core_layout.get_hisparse_kv_cache_config(runtime, resolved, 64 * stride, 1 << 30)
    assert materialized.num_blocks == 64
    assert all(
        tensor.block_stride == stride and tensor.size == 64 * stride
        for tensor in materialized.kv_cache_tensors
        if not tensor.host_resident
    )


@pytest.mark.parametrize("model_type", ["kimi_k31", "unregistered_sparse_model"])
@requires_hybrid_core
def test_ascend_builder_routes_native_hybrid_to_core_placement(model_type, monkeypatch):
    from vllm_ascend.core import kv_cache_config_builder as builder_module

    runtime = make_runtime(model_type, layout=KVCacheLayout.BLHNC)
    runtime.kv_transfer_config = KVTransferConfig(
        kv_connector="HiSparseConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"host_pool_gib": 1},
    )

    def fail_paired_layout(*args):
        pytest.fail("HiSparse must not enter the ordinary paired SFA layout.")

    monkeypatch.setattr(builder_module, "paired_sfa_cache_groups", fail_paired_layout)
    monkeypatch.setattr(builder_module, "paired_sfa_cache_config", fail_paired_layout)
    monkeypatch.setattr(core_layout, "use_shared_hisparse_host_pool", lambda _: False)
    builder = builder_module.AscendKVCacheConfigBuilder()
    specs = make_specs()
    groups = builder.get_kv_cache_groups(runtime, specs)
    assert groups == get_hybrid_sfa_hisparse_groups(runtime, specs)

    stride = _pool_bytes_per_block_from_builder(builder, runtime, groups)
    assert stride == _pool_bytes_per_block(runtime, groups)
    materialized = builder.get_kv_cache_config_from_groups(runtime, groups, num_blocks=64)
    assert materialized.num_blocks == 64
    assert materialized.hisparse_host_num_blocks > 0
    assert {name for tensor in materialized.kv_cache_tensors if tensor.host_resident for name in tensor.layers} == {
        "sparse"
    }
    assert all(
        tensor.block_stride == stride and tensor.size == 64 * stride
        for tensor in materialized.kv_cache_tensors
        if not tensor.host_resident
    )


def test_native_provider_leaves_other_layouts_to_their_owners():
    runtime = make_runtime()
    specs = make_specs()
    attention_specs = {name: spec for name, spec in specs.items() if name != "state"}
    assert get_hybrid_sfa_hisparse_groups(runtime, attention_specs) is None
    runtime.attention_config.hisparse_config = None
    assert get_hybrid_sfa_hisparse_groups(runtime, specs) is None


def test_native_hybrid_requires_core_page_fitting(monkeypatch):
    from vllm_ascend import hisparse

    monkeypatch.setattr(hisparse, "supports_hybrid_hisparse_core", lambda: False)

    with pytest.raises(ValueError, match="verified vLLM main pin"):
        get_hybrid_sfa_hisparse_groups(make_runtime(), make_specs())


@pytest.mark.parametrize("draft_source", ["config", "indexer", "state"])
@requires_hybrid_core
def test_native_hybrid_rejects_speculative_and_draft_caches(draft_source):
    runtime = make_runtime()
    specs = make_specs()
    if draft_source == "config":
        runtime.speculative_config = SimpleNamespace(method="mtp")
    elif draft_source == "state":
        specs["state"] = replace(specs["state"], num_speculative_blocks=1)
    else:
        specs["indexer"] = replace(specs["indexer"], is_draft=True)
    with pytest.raises(ValueError, match="does not support speculative decoding or draft caches"):
        get_hybrid_sfa_hisparse_groups(runtime, specs)


@requires_hybrid_core
def test_native_hybrid_rejects_unsupported_attention_cache():
    specs = make_specs()
    specs["dense"] = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"Unsupported cache spec.*dense"):
        get_hybrid_sfa_hisparse_groups(make_runtime(), specs)


@pytest.mark.parametrize(("rope_dim", "row_bytes"), [(0, 528), (64, 656)])
@requires_hybrid_core
def test_worker_native_hybrid_uses_runtime_row_width(rope_dim, row_bytes, monkeypatch):
    from vllm.v1.attention.backend import AttentionBackend

    from vllm_ascend.attention.utils import get_sfa_qsfa_packed_head_dim
    from vllm_ascend.worker.v2 import attn_utils

    class Backend(AttentionBackend):
        @staticmethod
        def get_supported_kernel_block_sizes(kv_cache_spec=None):
            return [128]

    runtime = make_runtime()
    runtime.cache_config.mamba_cache_mode = "none"
    runtime.cache_config.mamba_page_size_padded = None
    specs = make_specs()
    specs["sparse"] = replace(specs["sparse"], state_content_bytes=None, block_stride_alignment=None)

    def layer_attributes(spec):
        return {
            "get_attn_backend": lambda: Backend,
            "get_kv_cache_spec": lambda _: spec,
            "is_draft_layer": False,
        }

    layers = {name: SimpleNamespace(**layer_attributes(spec)) for name, spec in specs.items()}
    sparse = attn_utils.MLAAttention.__new__(attn_utils.MLAAttention)
    torch.nn.Module.__init__(sparse)
    for name, value in layer_attributes(specs["sparse"]).items():
        setattr(sparse, name, value)
    packed_dim = get_sfa_qsfa_packed_head_dim(512, rope_dim)
    sparse.impl = SimpleNamespace(
        enable_sparse_sfa_c8=True,
        sfa_qsfa_packed_kv_head_dim=packed_dim,
        c8_cache_dtype=torch.float8_e4m3fn,
    )
    sparse.hisparse_cache = SimpleNamespace(runtime=SimpleNamespace(is_group_leader=True))
    layers["sparse"] = sparse
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_: layers)

    collected = attn_utils.get_kv_cache_spec(runtime)

    assert collected["sparse"].head_size == row_bytes
    assert collected["sparse"].block_stride_alignment == 128 * row_bytes
    [hot] = [spec for spec in collected.values() if isinstance(spec, HiSparseHotSpec)]
    assert hot.page_size_bytes == hot.block_stride_alignment == 128 * row_bytes
