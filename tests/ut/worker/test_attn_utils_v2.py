from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu import attn_utils as upstream_attn_utils
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridAttnMetadata
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention import dsa_attn_kv_plan, dsa_v1
from vllm_ascend.attention import utils as attention_utils
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionState,
    AscendC8AttentionBackendImpl,
)
from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionDCPImpl
from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder
from vllm_ascend.attention.dsa_v1 import (
    AscendDSAC4Backend,
    AscendDSAC4StateBackend,
    AscendDSAC128Backend,
    AscendDSAC128StateBackend,
    AscendDSAMetadataBuilder,
    AscendDSASWABackend,
)
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadataBuilder
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    get_storage_block_size,
)
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.models.deepseek_v4 import compressor as deepseek_v4_compressor
from vllm_ascend.models.deepseek_v4 import indexer as deepseek_v4_indexer
from vllm_ascend.models.deepseek_v4 import model as deepseek_v4_model
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _get_kv_cache_config_deepseek_v4_main,
)
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.model_states import default as model_state_module
from vllm_ascend.worker.v2.model_states.default import AscendModelState


class SparseAttentionBackend(AscendAttentionBackend):
    @classmethod
    def is_sparse(cls) -> bool:
        return True


class _RecordingGDNReuseBuilder(AscendGDNAttentionMetadataBuilder):
    def __init__(self):
        self.supports_update_block_table = True
        self.num_spec = 3
        self.use_spec_decode = True
        self.use_full_cuda_graph = False
        self.decode_cudagraph_max_bs = 64
        self.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(mamba_cache_mode="none"),
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
        )
        self.build_calls = []
        self.update_calls = []

    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        self.build_calls.append((common_attn_metadata, kwargs))
        return SimpleNamespace(
            batch=common_attn_metadata.query_start_loc,
            block_table=common_attn_metadata.block_table_tensor,
        )

    def build_for_cudagraph_capture(self, common_attn_metadata, **kwargs):
        return self.build(0, common_attn_metadata, **kwargs)

    def update_block_table(self, metadata, blk_table, slot_mapping=None, graph_state_updates=None):
        self.update_calls.append(metadata)
        if graph_state_updates is not None:
            graph_state_updates.append(blk_table)
        return SimpleNamespace(batch=metadata.batch, block_table=blk_table)


def _make_gdn_reuse_inputs(num_groups):
    spec = MambaSpec(block_size=16, shapes=((1,), (1,)), dtypes=(torch.float32,), num_speculative_blocks=3)
    builders = [_RecordingGDNReuseBuilder() for _ in range(num_groups)]
    for builder in builders:
        builder.kv_cache_spec = spec
    groups = [
        SimpleNamespace(
            kv_cache_spec=spec,
            layer_names=[f"layer{i}"],
            get_metadata_builder=lambda _index, builder=builder: builder,
        )
        for i, builder in enumerate(builders)
    ]
    query_start_loc = torch.tensor([0, 4, 8], dtype=torch.int32)
    accepted = torch.tensor([2, 3], dtype=torch.int32)
    drafts = torch.tensor([3, 3], dtype=torch.int32)
    is_prefilling = torch.zeros(2, dtype=torch.bool)
    model_specific = SimpleNamespace(
        get_extra_common_attn_kwargs=lambda _group, n: {"is_prefilling": is_prefilling[:n]},
        get_extra_attn_kwargs=lambda _builder, n: {
            "num_accepted_tokens": accepted[:n],
            "num_decode_draft_tokens_cpu": drafts[:n],
        },
    )
    kwargs = dict(
        attn_groups=[[group] for group in groups],
        num_reqs=2,
        num_tokens=8,
        query_start_loc_gpu=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        max_query_len=4,
        seq_lens=torch.tensor([24, 48], dtype=torch.int32),
        max_seq_len=48,
        block_tables=tuple(torch.arange(8, dtype=torch.int32).view(2, 4) + i * 100 for i in range(num_groups)),
        slot_mappings=torch.arange(num_groups * 8, dtype=torch.int64).view(num_groups, 8),
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        seq_lens_cpu_upper_bound=torch.tensor([24, 48], dtype=torch.int32),
        model_specific_attn_metadata=model_specific,
    )
    return builders, kwargs


@pytest.mark.parametrize("num_groups", [2, 38])
@pytest.mark.parametrize("for_capture", [False, True])
def test_standard_mamba_common_fields_follow_each_current_batch(num_groups, for_capture, monkeypatch):
    builders, kwargs = _make_gdn_reuse_inputs(num_groups)
    provider = MambaHybridAttnMetadata(
        is_prefilling=torch.tensor([False, True]),
        num_accepted_tokens=torch.tensor([2, 3], dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([3, 3], dtype=torch.int32),
    )
    kwargs["model_specific_attn_metadata"] = provider
    kwargs["for_cudagraph_capture"] = for_capture
    original = MambaHybridAttnMetadata.get_extra_common_attn_kwargs
    calls = []

    def record_common_fields(self, group, num_reqs):
        calls.append((group, num_reqs))
        return original(self, group, num_reqs)

    monkeypatch.setattr(MambaHybridAttnMetadata, "get_extra_common_attn_kwargs", record_common_fields)
    # Disable builder reuse to inspect the common fields for every cache group.
    for builder in builders:
        builder.supports_update_block_table = False
    for step, num_reqs in enumerate((2, 1, 2)):
        provider.is_prefilling = torch.tensor([bool(step % 2), bool((step + 1) % 2)])
        kwargs["num_reqs"] = num_reqs
        attn_utils.build_attn_metadata(**kwargs)
        common = [builder.build_calls[-1][0] for builder in builders]
        assert all(metadata.is_prefilling is common[0].is_prefilling for metadata in common)
        assert torch.equal(common[0].is_prefilling, provider.is_prefilling[:num_reqs])
        assert common[0].is_prefilling.data_ptr() == provider.is_prefilling.data_ptr()
    assert calls == [(0, 2), (0, 1), (0, 2)]


def test_mamba_instance_common_override_keeps_group_specific_fields():
    builders, kwargs = _make_gdn_reuse_inputs(2)
    provider = MambaHybridAttnMetadata(is_prefilling=torch.zeros(2, dtype=torch.bool))
    masks = [torch.zeros(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool)]
    fields = [{"is_prefilling": masks[i], "num_computed_tokens_cpu": torch.full((2,), i)} for i in range(2)]
    override = MagicMock(side_effect=lambda group, n: fields[group].copy())
    provider.get_extra_common_attn_kwargs = override
    kwargs["model_specific_attn_metadata"] = provider
    attn_utils.build_attn_metadata(**kwargs)
    assert [entry.args for entry in override.call_args_list] == [(0, 2), (1, 2)]
    for i, builder in enumerate(builders):
        common = builder.build_calls[0][0]
        assert common.is_prefilling is masks[i]
        assert torch.equal(common.num_computed_tokens_cpu, torch.full((2,), i))
        assert set(fields[i]) == {"is_prefilling", "num_computed_tokens_cpu"}


def test_exact_cpu_lengths_require_a_separate_mirror():
    builders, kwargs = _make_gdn_reuse_inputs(1)
    kwargs["seq_lens_cpu_is_exact"] = True
    with pytest.raises(ValueError, match="Exact CPU sequence lengths"):
        attn_utils.build_attn_metadata(**kwargs)
    assert not builders[0].build_calls


@pytest.mark.parametrize("for_capture", [False, True])
def test_custom_gdn_provider_keeps_group_builds_separate(for_capture):
    builders, kwargs = _make_gdn_reuse_inputs(24)
    kwargs["for_cudagraph_capture"] = for_capture
    for _ in range(2):
        result = attn_utils.build_attn_metadata(**kwargs)
        assert len({id(metadata) for metadata in result.values()}) == 24
        for i, metadata in enumerate(result.values()):
            assert metadata.block_table is kwargs["block_tables"][i]
            assert metadata.batch is result["layer0"].batch
    assert len(builders[0].build_calls) == 2
    assert all(len(builder.build_calls) == 2 for builder in builders)
    assert all(not builder.update_calls for builder in builders)


@pytest.mark.parametrize("for_capture", [False, True])
def test_batch_global_mamba_gdn_groups_reuse_during_capture_and_replay(for_capture, monkeypatch):
    builders, kwargs = _make_gdn_reuse_inputs(24)
    kwargs["model_specific_attn_metadata"] = MambaHybridAttnMetadata(
        is_prefilling=torch.zeros(2, dtype=torch.bool),
        num_accepted_tokens=torch.tensor([2, 3], dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([3, 3], dtype=torch.int32),
    )
    provider = kwargs["model_specific_attn_metadata"]
    extra_inputs = MagicMock(wraps=provider.get_extra_attn_kwargs)
    monkeypatch.setattr(provider, "get_extra_attn_kwargs", extra_inputs)
    common_builds = MagicMock(wraps=attn_utils.AscendCommonAttentionMetadata)
    monkeypatch.setattr(attn_utils, "AscendCommonAttentionMetadata", common_builds)
    kwargs["for_cudagraph_capture"] = for_capture
    result = attn_utils.build_attn_metadata(**kwargs)
    assert len(builders[0].build_calls) == 1
    assert all(not builder.build_calls and len(builder.update_calls) == 1 for builder in builders[1:])
    for i, metadata in enumerate(result.values()):
        assert metadata.block_table is kwargs["block_tables"][i]
    assert extra_inputs.call_count == (0 if for_capture else 1)
    assert common_builds.call_count == 1


@pytest.mark.parametrize("for_capture", [False, True])
def test_v2_flushes_group_graph_writes_once_before_return(for_capture):
    builders, kwargs = _make_gdn_reuse_inputs(24)
    kwargs["model_specific_attn_metadata"] = MambaHybridAttnMetadata(
        is_prefilling=torch.zeros(2, dtype=torch.bool),
        num_accepted_tokens=torch.tensor([2, 3], dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([3, 3], dtype=torch.int32),
    )
    for builder in builders:
        builder.graph_state_updater = MagicMock()
    result = attn_utils.build_attn_metadata(**kwargs, for_cudagraph_capture=for_capture)
    builders[0].graph_state_updater.apply.assert_called_once()
    updates = builders[0].graph_state_updater.apply.call_args.args[0]
    assert len(updates) == 23
    for i, update in enumerate(updates, 1):
        assert update is kwargs["block_tables"][i]
        assert result[f"layer{i}"].block_table is update
    assert all(not builder.graph_state_updater.apply.called for builder in builders[1:])


@pytest.mark.parametrize("shared_capture", [False, True])
def test_gdn_provider_changes_preserve_capture_contract(shared_capture):
    builders, kwargs = _make_gdn_reuse_inputs(2)
    custom_provider = kwargs["model_specific_attn_metadata"]
    standard_provider = MambaHybridAttnMetadata(
        is_prefilling=torch.zeros(2, dtype=torch.bool),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.full((2,), 3, dtype=torch.int32),
    )
    kwargs["model_specific_attn_metadata"] = standard_provider if shared_capture else custom_provider
    attn_utils.build_attn_metadata(**kwargs, for_cudagraph_capture=True)
    kwargs["model_specific_attn_metadata"] = custom_provider if shared_capture else standard_provider
    if shared_capture:
        with pytest.raises(ValueError, match="batch-global Mamba provider"):
            attn_utils.build_attn_metadata(**kwargs)
        assert len(builders[0].build_calls) == 1
    else:
        attn_utils.build_attn_metadata(**kwargs)
        assert all(len(builder.build_calls) == 2 for builder in builders)
        assert all(not builder.update_calls for builder in builders)


@pytest.mark.parametrize(
    "difference",
    [
        "disabled",
        "spec_width",
        "prefill",
        "extra_common",
        "accepted",
        "draft",
        "cache_mode",
        "extra_builder",
        "pcp",
    ],
)
def test_gdn_reuse_keeps_different_group_semantics_separate(difference):
    builders, kwargs = _make_gdn_reuse_inputs(2)
    if difference in ("disabled", "spec_width", "cache_mode", "pcp"):
        kwargs["model_specific_attn_metadata"] = MambaHybridAttnMetadata(
            is_prefilling=torch.zeros(2, dtype=torch.bool),
            num_accepted_tokens=torch.ones(2, dtype=torch.int32),
            num_decode_draft_tokens_cpu=torch.full((2,), 3, dtype=torch.int32),
        )
    if difference == "disabled":
        builders[1].supports_update_block_table = False
    elif difference == "spec_width":
        spec = replace(builders[1].kv_cache_spec, num_speculative_blocks=7)
        builders[1].kv_cache_spec = kwargs["attn_groups"][1][0].kv_cache_spec = spec
    elif difference == "prefill":
        masks = [torch.zeros(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool)]
        kwargs["model_specific_attn_metadata"].get_extra_common_attn_kwargs = lambda group, n: {
            "is_prefilling": masks[group][:n]
        }
    elif difference == "extra_common":
        kwargs["model_specific_attn_metadata"].get_extra_common_attn_kwargs = lambda group, n: {
            "num_computed_tokens_cpu": torch.full((n,), group, dtype=torch.int32)
        }
    elif difference in ("accepted", "draft"):
        accepted = [torch.tensor([1, 2]), torch.tensor([2, 3])]
        drafts = [torch.tensor([3, 3]), torch.tensor([1, 3])]
        kwargs["model_specific_attn_metadata"].get_extra_attn_kwargs = lambda builder, n: {
            "num_accepted_tokens": accepted[builders.index(builder) if difference == "accepted" else 0][:n],
            "num_decode_draft_tokens_cpu": drafts[builders.index(builder) if difference == "draft" else 0][:n],
        }
    elif difference == "cache_mode":
        spec = replace(builders[1].kv_cache_spec, mamba_cache_mode="align")
        builders[1].kv_cache_spec = kwargs["attn_groups"][1][0].kv_cache_spec = spec
    elif difference == "extra_builder":
        kwargs["model_specific_attn_metadata"].get_extra_attn_kwargs = lambda builder, n: {"group_specific": 1}
    else:
        kwargs["pcp_context"] = object()
    attn_utils.build_attn_metadata(**kwargs)
    assert all(len(builder.build_calls) == 1 for builder in builders)
    assert all(not builder.update_calls for builder in builders)


def _make_kv_cache_tensor(size: int, layer_names: list[str], page_size: int = 0) -> KVCacheTensor:
    """Build a KVCacheTensor; vLLM #51718 renamed shared_by -> layers on main."""
    return KVCacheTensor(
        size=size,
        layers=layer_names,
        layer_stride=page_size,
        block_stride=page_size,
        offset=0,
    )


def _make_dsv4_mla_spec(block_size: int, compress_ratio: int) -> AscendMLAAttentionSpec:
    """Build a DSV4 AscendMLAAttentionSpec; #51718 moved compress_ratio ->
    tokens_per_state on main."""
    ratio_kwargs = {"tokens_per_state": compress_ratio}
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        model_version="deepseek_v4",
        **ratio_kwargs,
    )


def _spec_compress_ratio(spec) -> int:
    """Compression ratio of an MLA spec on either vLLM lane."""
    return spec.tokens_per_state


@pytest.fixture(autouse=True)
def default_pa_disabled(monkeypatch):
    monkeypatch.setattr(attn_utils, "requires_contiguous_pa_kv_cache", lambda *_args, **_kwargs: False)


def _configure_pa_allocation_check(monkeypatch, vllm_config, pa_enabled):
    """Exercise PA allocation policy using actual config and Attention implementations."""
    vllm_config.speculative_config = None
    vllm_config.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    monkeypatch.setattr(attn_utils, "requires_contiguous_pa_kv_cache", attention_utils.requires_contiguous_pa_kv_cache)
    monkeypatch.setattr(
        attention_utils, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A3)
    )
    monkeypatch.setattr(
        attention_utils, "get_ascend_config", lambda: SimpleNamespace(pa_shape_list=[32] if pa_enabled else [])
    )


@pytest.mark.parametrize(("block_size", "kernel_block_size"), [(128, 128), (1152, 128), (2048, 128)])
@pytest.mark.parametrize("kv_transfer", [False, True])
@pytest.mark.parametrize(
    ("cache_kind", "pa_enabled"),
    [
        (cache_kind, pa_enabled)
        for cache_kind in ("full", "sliding_window", "sparse", "full_sparse", "mixed")
        for pa_enabled in (False, True)
    ]
    + [("c8", True), ("dcp", True)],
)
def test_main_allocator_attention_layout(
    monkeypatch, block_size, kernel_block_size, kv_transfer, cache_kind, pa_enabled
):
    layer_name = "model.layers.0.self_attn.attn"
    second_layer_name = "model.layers.1.self_attn.attn"
    spec_type = SlidingWindowSpec if cache_kind == "sliding_window" else FullAttentionSpec
    extra_args = {"sliding_window": 4096} if cache_kind == "sliding_window" else {}
    spec = spec_type(
        block_size=block_size,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
        **extra_args,
    )
    num_blocks = 3
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=2 * num_blocks * spec.page_size_bytes,
                layers=[layer_name, second_layer_name],
                layer_stride=num_blocks * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
                offset=0,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[layer_name, second_layer_name], kv_cache_spec=spec)],
    )
    impl_cls = {
        "c8": AscendC8AttentionBackendImpl,
        "dcp": AscendAttentionDCPImpl,
    }.get(cache_kind, AscendAttentionBackendImpl)
    impl = impl_cls.__new__(impl_cls)
    impl.sliding_window = extra_args.get("sliding_window")
    layer = SimpleNamespace(
        get_attn_backend=lambda: SparseAttentionBackend if cache_kind == "full_sparse" else AscendAttentionBackend,
        kv_sharing_target_layer_name=None,
        num_heads=8,
        impl=impl,
    )
    second_layer = SimpleNamespace(
        get_attn_backend=lambda: SparseAttentionBackend if cache_kind == "mixed" else layer.get_attn_backend(),
        kv_sharing_target_layer_name=None,
        num_heads=8,
        impl=impl,
    )
    vllm_config = SimpleNamespace(
        additional_config={},
        cache_config=SimpleNamespace(cache_dtype="auto"),
        kv_transfer_config=object() if kv_transfer else None,
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2 if cache_kind == "dcp" else 1),
        quant_config=None,
    )
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(
        attn_utils,
        "get_layers_from_vllm_config",
        lambda *_args, **_kwargs: {layer_name: layer, second_layer_name: second_layer},
    )
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda *_args, **_kwargs: cache_kind == "sparse")
    monkeypatch.setattr(attn_utils, "enable_fa_quant", lambda *_args, **_kwargs: False)
    _configure_pa_allocation_check(monkeypatch, vllm_config, pa_enabled)

    kv_caches = attn_utils.allocate_kv_cache_main(
        kv_cache_config,
        device=torch.device("cpu"),
        layout=None,
        kernel_block_sizes=[kernel_block_size],
    )

    key_cache, value_cache = kv_caches[layer_name]
    expected_shape = (
        num_blocks * block_size // kernel_block_size,
        kernel_block_size,
        spec.num_kv_heads,
        spec.head_size,
    )
    assert key_cache.shape == expected_shape
    assert value_cache.shape == expected_shape
    second_key, second_value = kv_caches[second_layer_name]
    assert second_key.shape == expected_shape
    assert second_value.shape == expected_shape
    block_elements = kernel_block_size * spec.num_kv_heads * spec.head_size
    if cache_kind == "c8" or (cache_kind in ("full", "mixed") and not pa_enabled):
        assert not key_cache.is_contiguous()
        assert not value_cache.is_contiguous()
        assert key_cache.stride(0) == value_cache.stride(0) == 2 * block_elements
        assert key_cache.untyped_storage().data_ptr() == value_cache.untyped_storage().data_ptr()
        assert value_cache.storage_offset() - key_cache.storage_offset() == block_elements
        assert key_cache.untyped_storage().data_ptr() != second_key.untyped_storage().data_ptr()
        expected_bytes = num_blocks * spec.page_size_bytes + (2 * 1024 * 1024 if kv_transfer else 0)
        assert key_cache.untyped_storage().nbytes() == expected_bytes
        if kv_transfer:
            assert key_cache.data_ptr() % (2 * 1024 * 1024) == 0
    else:
        assert key_cache.is_contiguous()
        assert value_cache.is_contiguous()
        assert key_cache.untyped_storage().data_ptr() != value_cache.untyped_storage().data_ptr()
    if cache_kind == "mixed":
        assert second_key.is_contiguous()
        assert second_value.is_contiguous()
        assert second_key.untyped_storage().data_ptr() != second_value.untyped_storage().data_ptr()

    key_cache[1].fill_(3)
    value_cache[2].fill_(5)
    assert torch.all(key_cache[1] == 3)
    assert torch.count_nonzero(key_cache[0]) == 0
    assert torch.count_nonzero(key_cache[2:]) == 0
    assert torch.all(value_cache[2] == 5)
    assert torch.count_nonzero(value_cache[:2]) == 0
    assert torch.count_nonzero(value_cache[3:]) == 0
    assert torch.count_nonzero(second_key) == 0
    assert torch.count_nonzero(second_value) == 0


def test_sparse_offload_allocator_uses_host_main_cache_and_device_resident_buffer(monkeypatch):
    name = "model.layers.0.self_attn.attn"
    spec = AscendMLAAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=6,
        dtype=torch.bfloat16,
        store_on_host=True,
    )
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[_make_kv_cache_tensor(3 * spec.page_size_bytes, [name], spec.page_size_bytes)],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec)],
    )
    layer = SimpleNamespace(
        get_attn_backend=lambda: SparseAttentionBackend,
        kv_sharing_target_layer_name=None,
        num_heads=8,
    )
    sparse_cfg = SimpleNamespace(keep_device_kv_cache=False)
    vllm_config = SimpleNamespace(
        additional_config={},
        cache_config=SimpleNamespace(cache_dtype="auto"),
        kv_transfer_config=object(),
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        parallel_config=SimpleNamespace(),
    )
    calls = []
    raw_cache = (None, None, torch.empty(48, dtype=torch.int8), torch.empty(24, dtype=torch.int8), 72)
    marker = object()

    def allocate(k_size, v_size, alignment, tp_rank, keep_device, allocate_npu):
        calls.append((k_size, v_size, alignment, tp_rank, keep_device))
        return raw_cache

    def reshape(raw, layer_spec, backend, tp_rank, config, offload_config):
        assert raw is raw_cache
        assert layer_spec is spec
        assert backend is SparseAttentionBackend
        assert tp_rank == 0
        assert config is vllm_config
        assert offload_config is sparse_cfg
        return marker

    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_args, **_kwargs: {name: layer})
    monkeypatch.setattr(attn_utils, "get_ascend_config", lambda: SimpleNamespace(sparse_kv_offload_config=sparse_cfg))
    monkeypatch.setattr(attn_utils, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(attn_utils, "_get_attention_kv_cache_dims", lambda *_args: (4, 2))
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda *_args: True)
    monkeypatch.setattr(attn_utils, "allocate_kv_cache_tensors_for_sparse_kv_offload", allocate)
    monkeypatch.setattr(attn_utils, "reshape_kv_cache_tensors_for_sparse_kv_offload", reshape)

    assert attn_utils.allocate_kv_cache_main(config, torch.device("cpu"), None, [2])[name] is marker
    assert calls == [(48, 24, 2 * 1024 * 1024, 0, False)]


@pytest.mark.parametrize("pa_enabled", [False, True])
def test_hybrid_allocator_keeps_shared_backing_with_pa_configured(monkeypatch, pa_enabled):
    attn_name = "model.layers.0.self_attn.attn"
    mamba_name = "model.layers.1.linear_attn"
    attn_spec = FullAttentionSpec(block_size=2, num_kv_heads=1, head_size=4, dtype=torch.float16)
    mamba_spec = MambaSpec(block_size=2, shapes=((2, 4),), dtypes=(torch.float32,))
    assert attn_spec.page_size_bytes == mamba_spec.page_size_bytes
    backing_size = 3 * attn_spec.page_size_bytes
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[
            KVCacheTensor(
                size=backing_size,
                layers=[name],
                layer_stride=backing_size,
                block_stride=spec.page_size_bytes,
                offset=0,
            )
            for name, spec in ((attn_name, attn_spec), (mamba_name, mamba_spec))
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec)
            for name, spec in ((attn_name, attn_spec), (mamba_name, mamba_spec))
        ],
    )
    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl.sliding_window = None
    layer = SimpleNamespace(impl=impl)
    vllm_config = SimpleNamespace(
        additional_config={},
        kv_transfer_config=None,
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        quant_config=None,
    )
    _configure_pa_allocation_check(monkeypatch, vllm_config, pa_enabled)
    assert attention_utils.requires_contiguous_pa_kv_cache(layer, vllm_config, attn_spec) == pa_enabled
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: vllm_config)
    raw_caches = attn_utils._allocate_kv_cache(config, shared_layers={}, device=torch.device("cpu"))

    assert all(isinstance(raw, torch.Tensor) for raw in raw_caches.values())
    assert all(raw.numel() == backing_size for raw in raw_caches.values())
    assert all(raw.untyped_storage().nbytes() == backing_size for raw in raw_caches.values())
    assert raw_caches[attn_name].data_ptr() == raw_caches[mamba_name].data_ptr()


def test_hybrid_attention_layout_preserves_padding(monkeypatch):
    name = "model.layers.0.self_attn.attn"
    spec = FullAttentionSpec(block_size=2, num_kv_heads=1, head_size=4, dtype=torch.float16, page_size_padded=64)
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec)],
    )
    group = SimpleNamespace(kv_cache_group_id=0, kv_cache_spec=spec, backend=AscendAttentionBackend, layer_names=[name])
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda *_args: False)
    raw = torch.full((3 * spec.page_size_bytes,), 7, dtype=torch.int8)
    before = raw.clone()

    def reshape():
        return attn_utils._reshape_kv_cache_v2(
            attn_groups=[group],
            kv_cache_raw_tensors={name: raw},
            cache_dtype="auto",
            kernel_block_sizes=[spec.block_size],
            shared_kv_cache_layers={},
            kv_cache_config=config,
        )

    key, value = reshape()[name]
    assert not key.is_contiguous() and not value.is_contiguous()
    key[1].fill_(3)
    value[1].fill_(5)
    torch.testing.assert_close(raw[: spec.page_size_bytes], before[: spec.page_size_bytes])
    torch.testing.assert_close(raw[2 * spec.page_size_bytes :], before[2 * spec.page_size_bytes :])
    torch.testing.assert_close(
        raw[spec.page_size_bytes + spec.real_page_size_bytes : 2 * spec.page_size_bytes],
        before[spec.page_size_bytes + spec.real_page_size_bytes : 2 * spec.page_size_bytes],
    )


@pytest.mark.parametrize("sparse_backend", [False, True])
def test_padded_full_attention_allocation_preserves_backend_behavior(monkeypatch, sparse_backend):
    name = "model.layers.0.self_attn.attn"
    spec = FullAttentionSpec(block_size=2, num_kv_heads=1, head_size=4, dtype=torch.float16, page_size_padded=64)
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[
            KVCacheTensor(
                size=3 * spec.page_size_bytes,
                layers=[name],
                layer_stride=3 * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
                offset=0,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec)],
    )
    vllm_config = SimpleNamespace(
        additional_config={},
        kv_transfer_config=None,
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        quant_config=None,
    )
    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl.sliding_window = None
    layer = SimpleNamespace(
        get_attn_backend=lambda: SparseAttentionBackend if sparse_backend else AscendAttentionBackend,
        impl=impl,
    )
    _configure_pa_allocation_check(monkeypatch, vllm_config, pa_enabled=True)
    assert attention_utils.using_paged_attention(None, vllm_config, spec.head_size)
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_args: {name: layer})
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda *_args: False)
    monkeypatch.setattr(attn_utils, "enable_fa_quant", lambda *_args: False)
    if sparse_backend:
        with pytest.raises(ValueError, match="unpadded FullAttention pages"):
            attn_utils._allocate_kv_cache(config, shared_layers={}, device=torch.device("cpu"))
    else:
        raw_caches = attn_utils._allocate_kv_cache(config, shared_layers={}, device=torch.device("cpu"))
        assert isinstance(raw_caches[name], torch.Tensor)
        assert raw_caches[name].numel() == 3 * spec.page_size_bytes


def test_mrv2_mamba_views_skip_physical_page_padding():
    spec = MambaSpec(
        block_size=1,
        shapes=((4,), (2,)),
        dtypes=(torch.float16, torch.float32),
        page_size_padded=32,
    )
    raw = torch.zeros(3 * 32, dtype=torch.int8)

    conv_state, ssm_state = attn_utils._reshape_mamba_kv_cache(raw, spec)

    assert conv_state.shape == (3, 4)
    assert ssm_state.shape == (3, 2)
    assert conv_state.stride() == (16, 1)
    assert ssm_state.stride() == (8, 1)
    assert not conv_state.is_contiguous()
    assert not ssm_state.is_contiguous()

    conv_state[1].fill_(1)
    ssm_state[2].fill_(2)
    assert torch.count_nonzero(raw[:32]) == 0
    assert torch.count_nonzero(raw[32:40]) > 0
    assert torch.count_nonzero(raw[40:64]) == 0
    assert torch.count_nonzero(raw[64:72]) == 0
    assert torch.count_nonzero(raw[72:80]) > 0
    assert torch.count_nonzero(raw[80:]) == 0


def test_mrv2_attention_views_interleave_kv_per_physical_page():
    num_blocks = 3
    block_size = 2
    num_heads = 1
    head_size = 4
    logical_page_elements = 2 * block_size * num_heads * head_size
    physical_page_elements = logical_page_elements + 8
    raw = torch.zeros(
        num_blocks * physical_page_elements * torch.float16.itemsize,
        dtype=torch.int8,
    )

    key, value = attn_utils._reshape_combined_attention_kv_cache(
        raw,
        (2, num_blocks, block_size, num_heads, head_size),
        torch.float16,
        physical_page_elements * torch.float16.itemsize,
    )

    assert key.stride() == (physical_page_elements, 4, 4, 1)
    assert value.stride() == (physical_page_elements, 4, 4, 1)
    assert not key.is_contiguous()
    assert not value.is_contiguous()

    key[1].fill_(1)
    value[2].fill_(2)
    raw_typed = raw.view(torch.float16)
    assert torch.count_nonzero(raw_typed[:physical_page_elements]) == 0
    assert torch.count_nonzero(raw_typed[physical_page_elements : 2 * physical_page_elements]) == 8
    assert torch.count_nonzero(raw_typed[2 * physical_page_elements :]) == 8


def test_main_dsv4_materializes_real_planner_geometry_once(monkeypatch):
    small_name = "model.layers.0.self_attn.attn"
    large_name = "model.layers.1.self_attn.attn"
    aliased_large_name = "model.layers.2.self_attn.attn"
    mtp_name = "model.mtp.layers.0.self_attn.attn"
    small_spec = AscendMLAAttentionSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float16,
        model_version="deepseek_v4",
        tokens_per_state=1,
    )
    large_spec = AscendMLAAttentionSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
        model_version="deepseek_v4",
        tokens_per_state=1,
    )
    full_group_spec = UniformTypeKVCacheSpecs.from_specs(
        {
            small_name: small_spec,
            large_name: large_spec,
            mtp_name: large_spec,
        }
    )
    alias_group_spec = UniformTypeKVCacheSpecs.from_specs({aliased_large_name: large_spec})
    assert full_group_spec is not None
    assert alias_group_spec is not None
    groups = [
        KVCacheGroupSpec(
            layer_names=[small_name, large_name, mtp_name],
            kv_cache_spec=full_group_spec,
        ),
        KVCacheGroupSpec(
            layer_names=[aliased_large_name],
            kv_cache_spec=alias_group_spec,
        ),
    ]
    num_blocks = 3
    tuple_stride = (small_spec.page_size_bytes + large_spec.page_size_bytes) * num_blocks
    backing_size = tuple_stride * 2
    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_kv_cache_utils.may_override_num_blocks",
        lambda _config, value: value,
    )
    planned_num_blocks, descriptors = _get_kv_cache_config_deepseek_v4_main(
        SimpleNamespace(),
        groups,
        backing_size,
    )
    assert planned_num_blocks == num_blocks
    kv_cache_config = KVCacheConfig(
        num_blocks=planned_num_blocks,
        kv_cache_tensors=descriptors,
        kv_cache_groups=groups,
    )
    vllm_config = SimpleNamespace(
        additional_config={},
        kv_transfer_config=None,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(compress_ratios=[1]),
        ),
    )
    monkeypatch.setattr(
        attn_utils,
        "get_current_vllm_config",
        lambda: vllm_config,
    )
    allocations = []

    def allocate_once(numel, _alignment, _device):
        allocations.append(numel)
        return torch.zeros(numel, dtype=torch.int8)

    monkeypatch.setattr(
        attn_utils,
        "_allocate_int8_cache_tensor",
        allocate_once,
    )

    raw_caches = attn_utils._allocate_kv_cache(
        kv_cache_config,
        shared_layers={},
        device=torch.device("cpu"),
    )

    assert allocations == [backing_size]
    tensor_raw_caches: dict[str, torch.Tensor] = {}
    for layer_name, raw in raw_caches.items():
        assert isinstance(raw, torch.Tensor)
        tensor_raw_caches[layer_name] = raw
    assert len({raw.untyped_storage().data_ptr() for raw in tensor_raw_caches.values()}) == 1
    descriptors_by_layer = {layer_name: descriptor for descriptor in descriptors for layer_name in descriptor.layers}
    base_offset = min(
        raw.storage_offset() - descriptors_by_layer[name].offset for name, raw in tensor_raw_caches.items()
    )
    for layer_name, raw in tensor_raw_caches.items():
        descriptor = descriptors_by_layer[layer_name]
        assert raw.storage_offset() == base_offset + descriptor.offset
    assert tensor_raw_caches[large_name].storage_offset() == tensor_raw_caches[aliased_large_name].storage_offset()
    assert tensor_raw_caches[small_name].storage_offset() != tensor_raw_caches[large_name].storage_offset()
    assert tensor_raw_caches[mtp_name].storage_offset() == (
        base_offset + tuple_stride + small_spec.page_size_bytes * num_blocks
    )


@pytest.mark.parametrize(
    "state_kwargs", [{}, {"attn_state": None}, {"attn_state": AscendAttentionState.ChunkedPrefill}]
)
@pytest.mark.parametrize("factory_state", [None, AscendAttentionState.ChunkedPrefill])
def test_build_attn_metadata_factory_applies_state(monkeypatch, state_kwargs, factory_state):
    captured_kwargs = {}

    def raw_build_attn_metadata(*_args, **kwargs):
        captured_kwargs.update(kwargs)
        return "metadata"

    monkeypatch.setattr(
        attn_utils._BUILD_ATTN_METADATA_MODULE,
        "build_attn_metadata",
        raw_build_attn_metadata,
    )
    positions = torch.arange(8, dtype=torch.int32)
    is_prefilling = torch.tensor([False, False])

    with attn_utils.build_attn_metadata_factory(
        positions,
        pad=5,
        is_prefilling=is_prefilling,
        attn_state=factory_state,
        offload_kwargs={"copy_sfa_draft_index": 2},
    ):
        metadata = attn_utils._BUILD_ATTN_METADATA_MODULE.build_attn_metadata(**state_kwargs)

    assert metadata == "metadata"
    torch.testing.assert_close(captured_kwargs["positions"], positions[:5])
    assert captured_kwargs["is_prefilling"] is is_prefilling
    assert captured_kwargs["attn_state"] == factory_state
    assert captured_kwargs["copy_sfa_draft_index"] == 2


def test_build_attn_metadata_factory_passes_exact_host_lengths(monkeypatch):
    raw = MagicMock(return_value="metadata")
    monkeypatch.setattr(attn_utils._BUILD_ATTN_METADATA_MODULE, "build_attn_metadata", raw)
    seq_lens_cpu = torch.tensor([73, 22666, 0, 0], dtype=torch.int32)
    with attn_utils.build_attn_metadata_factory(
        torch.arange(32),
        32,
        torch.zeros(4, dtype=torch.bool),
        seq_lens_cpu=seq_lens_cpu,
        seq_lens_cpu_is_exact=True,
    ):
        assert attn_utils._BUILD_ATTN_METADATA_MODULE.build_attn_metadata() == "metadata"
    assert raw.call_args.kwargs["seq_lens_cpu_is_exact"]
    np.testing.assert_array_equal(raw.call_args.kwargs["seq_lens_np"], seq_lens_cpu.numpy())
    assert attn_utils._BUILD_ATTN_METADATA_MODULE.build_attn_metadata is raw


@pytest.mark.parametrize("dcp_size", [2])
def test_gqa_pcp_dcp_cache_keeps_tp_local_kv_heads(monkeypatch, dcp_size):
    source_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    layer = SimpleNamespace(
        kv_sharing_target_layer_name=None,
        align_kv_cache_with_mamba=True,
        get_kv_cache_spec=lambda _config: source_spec,
    )
    model_config = SimpleNamespace(
        use_mla=False,
        dtype=torch.bfloat16,
        get_total_num_kv_heads=lambda: 4,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            prefill_context_parallel_size=2,
            decode_context_parallel_size=dcp_size,
        ),
        attention_config=SimpleNamespace(indexer_kv_dtype="auto"),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=128),
    )
    monkeypatch.setattr(
        attn_utils,
        "get_layers_from_vllm_config",
        lambda *_args, **_kwargs: {"model.layers.0.self_attn.attn": layer},
    )
    monkeypatch.setattr(attn_utils, "enable_sfa_dcp_replicated_indexer", lambda _config: False)

    specs = attn_utils.get_kv_cache_spec(vllm_config)

    assert specs["model.layers.0.self_attn.attn"].num_kv_heads == 1


@pytest.mark.parametrize(
    ("replicated_indexer", "expected_size"),
    [(False, 1), (True, 4)],
)
@pytest.mark.parametrize("li_c8", [False, True])
@pytest.mark.parametrize("li_c4", [False, True])
@pytest.mark.parametrize("owner", ["unpaired", "static_shared", "mtp", "regular"])
def test_sfa_indexer_cache_spec_runtime_ownership_and_dcp_replication(
    monkeypatch, replicated_indexer, expected_size, li_c8, li_c4, owner
):
    layer_name = "model.layers.0.self_attn.indexer.k_cache"
    indexer_module = DeepseekV32IndexerCache.__new__(DeepseekV32IndexerCache)
    torch.nn.Module.__init__(indexer_module)
    monkeypatch.setattr(
        indexer_module,
        "get_kv_cache_spec",
        lambda _config: object(),
    )
    layers = {layer_name: indexer_module}
    if owner != "unpaired":
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.has_indexer = True
        impl._is_mtp_layer = owner == "mtp"
        impl.skip_topk = owner != "regular"
        layers[layer_name.replace(".indexer.k_cache", ".attn")] = SimpleNamespace(
            impl=impl, get_kv_cache_spec=lambda _config: None
        )

    vllm_config = SimpleNamespace(
        additional_config={},
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
        cache_config=SimpleNamespace(block_size=128, cache_dtype="auto"),
        attention_config=SimpleNamespace(indexer_kv_dtype="int8", hisparse_config=None),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(index_head_dim=128),
        ),
    )
    monkeypatch.setattr(
        attn_utils,
        "get_layers_from_vllm_config",
        lambda *_args, **_kwargs: layers,
    )
    monkeypatch.setattr(
        attn_utils,
        "enable_sfa_dcp_replicated_indexer",
        lambda _config: replicated_indexer,
    )
    monkeypatch.setattr(
        attn_utils,
        "get_current_hardware_profile",
        lambda: get_hardware_profile(AscendDeviceType.A2),
    )
    monkeypatch.setattr(
        attn_utils,
        "get_ascend_config",
        lambda: SimpleNamespace(
            is_sparse_li_c8_layer=lambda _layer_name: li_c8,
            is_sparse_li_c4_layer=lambda _layer_name: li_c4,
        ),
    )

    specs = attn_utils.get_kv_cache_spec(vllm_config)
    if owner == "static_shared":
        assert layer_name not in specs
        return
    spec = specs[layer_name]

    assert isinstance(spec, AscendSFAIndexerCacheSpec)
    assert spec.sfa_dcp_replicated_indexer_size == expected_size
    if li_c4:
        assert spec.head_size == 128 // 2
        assert spec.dtype == torch.uint8
        assert spec.scale_dim == 128 // 64 * 2
        assert spec.scale_dtype == torch.float8_e8m0fnu
        assert spec.cache_sparse_li_c4 is True
    elif li_c8:
        assert spec.dtype == torch.int8
        assert spec.scale_dim == 1
    else:
        assert spec.dtype == torch.bfloat16
        assert spec.scale_dim == 0


@pytest.mark.parametrize(
    ("device_type", "cache_dtype", "scale_dtype", "component_dims"),
    [
        (
            AscendDeviceType.A2,
            torch.int8,
            torch.float16,
            (128, 1),
        ),
    ],
)
def test_mrv2_initializes_dsv4_cache_only_layer(
    monkeypatch,
    device_type,
    cache_dtype,
    scale_dtype,
    component_dims,
):
    """Exercise DSV4 discovery, allocation, reshape, and binding as one flow."""
    layer_name = "model.layers.0.self_attn.indexer.k_cache"
    cache_config = SimpleNamespace(
        block_size=32,
        cache_dtype="auto",
        # vLLM #51718: main's init_kv_cache reads the resolved KV cache layout
        # from the cache config; the Ascend DSV4 path ignores it.
        get_resolved_kv_cache_layout=lambda: None,
    )
    vllm_config = SimpleNamespace(
        additional_config={},
        attention_config=SimpleNamespace(indexer_kv_dtype="int8", hisparse_config=None),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                compress_ratios=[4],
                model_type="deepseek_v4",
            ),
        ),
        cache_config=cache_config,
        kv_transfer_config=None,
        quant_config=None,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )

    cache_layer = deepseek_v4_indexer.AscendDeepseekV4IndexerCache.__new__(
        deepseek_v4_indexer.AscendDeepseekV4IndexerCache
    )
    torch.nn.Module.__init__(cache_layer)
    cache_layer.head_dim = 128
    cache_layer.dtype = torch.int8
    cache_layer.cache_config = cache_config
    cache_layer.compress_ratio = 4
    cache_layer.kv_cache = torch.tensor([])

    monkeypatch.setattr(
        deepseek_v4_indexer,
        "get_current_hardware_profile",
        lambda: get_hardware_profile(device_type),
    )
    monkeypatch.setattr(
        attn_utils,
        "get_current_hardware_profile",
        lambda: get_hardware_profile(device_type),
    )
    monkeypatch.setattr(
        attn_utils,
        "get_layers_from_vllm_config",
        lambda *_args, **_kwargs: {layer_name: cache_layer},
    )
    monkeypatch.setattr(
        attn_utils,
        "get_current_vllm_config",
        lambda: vllm_config,
    )
    monkeypatch.setattr(
        upstream_attn_utils,
        "get_shared_kv_cache_layers",
        lambda _config: {},
    )

    discovered_specs = attn_utils.get_kv_cache_spec(vllm_config)
    assert set(discovered_specs) == {layer_name}
    spec = discovered_specs[layer_name]
    assert isinstance(spec, AscendMLAAttentionSpec)
    assert spec.block_size == cache_config.block_size * cache_layer.compress_ratio
    assert get_storage_block_size(spec) == cache_config.block_size
    merged_spec = spec.merge([spec])
    assert merged_spec.tokens_per_state == cache_layer.compress_ratio
    assert get_storage_block_size(merged_spec) == cache_config.block_size

    num_blocks = 2
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            _make_kv_cache_tensor(
                num_blocks * spec.page_size_bytes,
                [layer_name],
                spec.page_size_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[layer_name],
                kv_cache_spec=spec,
            )
        ],
    )
    attn_group = AttentionGroup(
        backend=AscendDSAC4Backend,
        layer_names=[layer_name],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
    )
    runner_kv_caches: list[Any] = []

    # vLLM #51718 reworked upstream init_kv_cache to allocate generic 4D
    # views via `allocate_kv_cache` + `create_kv_cache_views`; that layout
    # cannot express the Ascend DSV4 page-strided cache. Route the
    # allocation through the Ascend DSV4 path (the same wiring the v0.28.0
    # patch applies) so the returned structure matches on both lanes.
    def _ascend_allocate_kv_cache(
        _kv_cache_config: KVCacheConfig,
        _device: torch.device,
        _layout: Any,
        _kernel_block_sizes: list[int],
    ) -> dict[str, Any]:
        del _layout
        raw_tensors = attn_utils._allocate_kv_cache(
            _kv_cache_config,
            shared_layers={},
            device=_device,
        )
        # `_reshape_kv_cache_v2` expects the flat attention-group list,
        # matching upstream v0.28.0 `init_kv_cache`, which flattens
        # `attn_groups` before reshaping.
        return attn_utils._reshape_kv_cache_v2(
            attn_groups=[attn_group],
            kv_cache_raw_tensors=raw_tensors,
            cache_dtype=cache_config.cache_dtype,
            kernel_block_sizes=_kernel_block_sizes,
            shared_kv_cache_layers={},
            kv_cache_config=_kv_cache_config,
        )

    def _ascend_bind_kv_cache(
        kv_caches: dict[str, Any],
        forward_context: dict[str, Any],
        num_attn_module: int = 1,
        kv_cache_groups: Any = None,
    ) -> None:
        del num_attn_module, kv_cache_groups
        assert len(runner_kv_caches) == 0
        for kv_cache in kv_caches.values():
            runner_kv_caches.append(kv_cache)
        for layer_name_, kv_cache in kv_caches.items():
            forward_context[layer_name_].kv_cache = kv_cache

    monkeypatch.setattr(upstream_attn_utils, "allocate_kv_cache", _ascend_allocate_kv_cache)
    # vLLM main (#53781) routes init_kv_cache through bind_kv_cache_to_layers
    # and dropped the runner_kv_caches parameter.
    monkeypatch.setattr(upstream_attn_utils, "bind_kv_cache_to_layers", _ascend_bind_kv_cache)
    kv_caches = upstream_attn_utils.init_kv_cache(
        forward_context={layer_name: cache_layer},
        kv_cache_config=kv_cache_config,
        device=torch.device("cpu"),
        kernel_block_sizes=[spec.block_size],
        vllm_config=vllm_config,
    )

    cache_components = kv_caches[layer_name]
    assert len(runner_kv_caches) == 1
    assert runner_kv_caches[0] is cache_components
    # On main the layer cache is replaced by the freshly allocated views, so
    # the returned structure is validated by the checks below instead.
    assert [component.shape for component in cache_components] == [
        (num_blocks, get_storage_block_size(spec), 1, dim) for dim in component_dims
    ]
    assert [component.dtype for component in cache_components] == [
        cache_dtype,
        scale_dtype,
        *([cache_dtype] if device_type == AscendDeviceType.A5 else []),
    ]
    backing_storage = cache_components[0].untyped_storage().data_ptr()
    assert all(component.untyped_storage().data_ptr() == backing_storage for component in cache_components)


class _RecordingDSAMetadataBuilder(AscendDSAMetadataBuilder):
    def __init__(self, calls: list[dict[str, Any]], compressor_ratio: int):
        self.calls = calls
        self.for_cudagraph_capture = False
        self.tq_group_block_sizes = None
        self.compressor_ratio = compressor_ratio

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata,
        **kwargs,
    ):
        self.for_cudagraph_capture = True
        return super().build_for_cudagraph_capture(
            common_attn_metadata,
            **kwargs,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
        **kwargs,
    ):
        del common_prefix_len, fast_build
        self.common_ratio_to_sas_metadata = kwargs["common_ratio_to_sas_metadata"]
        call = {
            "common_attn_metadata": common_attn_metadata,
            "common_ratio_to_sas_metadata": self.common_ratio_to_sas_metadata,
            "for_cudagraph_capture": self.for_cudagraph_capture,
            "num_actual_reqs": kwargs["num_actual_reqs"],
            "pcp_context": kwargs.get("pcp_context"),
            "pcp_cache_group_idx": kwargs.get("pcp_cache_group_idx"),
            "formatted_slot_mapping": kwargs.get("formatted_slot_mapping"),
        }
        assert "block_size" not in kwargs
        self.calls.append(call)
        call["common_ratio_to_sas_metadata"].setdefault("first_group", len(self.calls) == 1)
        return SimpleNamespace(common_attn_metadata=common_attn_metadata)


class _RecordingDSACPMetadataBuilder(AscendDSACPMetadataBuilder):
    def __init__(self, calls: list[dict[str, Any]], compressor_ratio: int):
        self.calls = calls
        self.for_cudagraph_capture = False
        self.tq_group_block_sizes = None
        self.compressor_ratio = compressor_ratio

    build = _RecordingDSAMetadataBuilder.build

    def build_for_cudagraph_capture(self, common_attn_metadata, **kwargs):
        self.for_cudagraph_capture = True
        return self.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata, **kwargs)


def _make_dsa_metadata_groups(builder_cls=_RecordingDSAMetadataBuilder):
    layer_names = [
        "model.layers.0.self_attn.compressor",
        "model.layers.0.self_attn.indexer",
    ]
    specs = [
        _make_dsv4_mla_spec(storage_block_size * compress_ratio, compress_ratio)
        for storage_block_size, compress_ratio in ((32, 4), (64, 128))
    ]
    calls: list[dict[str, Any]] = []
    attn_groups = [
        [
            AttentionGroup(
                backend=(AscendDSAC4Backend if _spec_compress_ratio(spec) == 4 else AscendDSAC128Backend),
                layer_names=[layer_name],
                kv_cache_spec=spec,
                kv_cache_group_id=group_id,
                metadata_builders=[builder_cls(calls, _spec_compress_ratio(spec))],
            )
        ]
        for group_id, (layer_name, spec) in enumerate(zip(layer_names, specs))
    ]
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[layer_name],
                kv_cache_spec=spec,
            )
            for layer_name, spec in zip(layer_names, specs)
        ],
    )
    return layer_names, specs, calls, attn_groups, kv_cache_config


def test_draft_metadata_uses_per_request_cpu_upper_bounds():
    _, _, calls, attn_groups, kv_cache_config = _make_dsa_metadata_groups()
    upper_bounds = torch.tensor([13, 27], dtype=torch.int32)
    req_ids = torch.tensor([17, 23], dtype=torch.int64)
    token_to_req = torch.tensor([0, 1], dtype=torch.int32)

    attn_utils.build_attn_metadata(
        attn_groups=attn_groups,
        num_reqs=2,
        num_tokens=2,
        query_start_loc_gpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        max_query_len=1,
        seq_lens=torch.tensor([11, 25], dtype=torch.int32),
        max_seq_len=27,
        block_tables=(torch.zeros((2, 1), dtype=torch.int32),) * 2,
        slot_mappings=(torch.zeros(2, dtype=torch.int32),) * 2,
        kv_cache_config=kv_cache_config,
        seq_lens_cpu_upper_bound=upper_bounds,
        req_ids_tensor=req_ids,
        token_to_req=token_to_req,
        offload_dummy=True,
    )

    assert len(calls) == 2
    for call in calls:
        common_metadata = call["common_attn_metadata"]
        torch.testing.assert_close(common_metadata.seq_lens_cpu, upper_bounds)
        torch.testing.assert_close(common_metadata.seq_lens, torch.tensor([11, 25], dtype=torch.int32))
        assert common_metadata.req_ids_tensor is req_ids
        assert common_metadata.token_to_req is token_to_req
        assert common_metadata.offload_dummy


@pytest.mark.parametrize("for_capture", [False, True])
def test_v2_fused_offload_splits_target_and_draft_metadata_in_one_sfa_group(for_capture):
    class RecordingSFABuilder(AscendSFAMetadataBuilder):
        def __init__(self):
            self.calls = []

        def build(self, common_prefix_len, common_attn_metadata, **kwargs):
            self.calls.append(common_attn_metadata)
            return SimpleNamespace(common=common_attn_metadata)

        def build_for_cudagraph_capture(self, common_attn_metadata, **kwargs):
            return self.build(0, common_attn_metadata, **kwargs)

    builder = RecordingSFABuilder()
    group = SimpleNamespace(
        layer_names=["target.layer", "draft.layer"],
        get_metadata_builder=lambda _: builder,
    )
    metadata = attn_utils.build_attn_metadata(
        attn_groups=[[group]],
        num_reqs=1,
        num_tokens=1,
        query_start_loc_gpu=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        max_query_len=1,
        seq_lens=torch.tensor([129], dtype=torch.int32),
        max_seq_len=4096,
        block_tables=(torch.zeros((1, 1), dtype=torch.int32),),
        slot_mappings=(torch.zeros(1, dtype=torch.int32),),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace()]),
        seq_lens_np=np.array([129], dtype=np.int32),
        draft_layer_names={"draft.layer"},
        copy_sfa_restore_tails=True,
        for_cudagraph_capture=for_capture,
    )

    assert len(builder.calls) == 2
    assert builder.calls[0].copy_sfa_draft_index is None
    assert builder.calls[0].copy_sfa_restore_tails
    assert builder.calls[1].copy_sfa_draft_index == 0
    assert not builder.calls[1].copy_sfa_restore_tails
    assert metadata["target.layer"].common is builder.calls[0]
    assert metadata["draft.layer"].common is builder.calls[1]


def test_v2_fused_offload_prepares_request_ownership_and_restores_tails(monkeypatch):
    state = AscendModelState.__new__(AscendModelState)
    state.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
        cache_config=SimpleNamespace(block_size=128),
    )
    state.max_model_len = 4096
    state.pcp_manager = None
    state._offload_live_req_ids = {"request-1": 0}
    state._offload_draft_attn_groups = []
    state._offload_draft_layer_names = {"draft.layer"}
    state._offload_req_ids = SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int64))
    state._offload_token_to_req = SimpleNamespace(gpu=torch.zeros(4, dtype=torch.int32))
    state._offload_pool_slots = SimpleNamespace(np=np.zeros(4, dtype=np.int32), cpu=torch.zeros(4, dtype=torch.int32))
    state._offload_pool_active = SimpleNamespace(np=np.zeros(4, dtype=np.bool_), cpu=torch.zeros(4, dtype=torch.bool))
    state._offload_request_states = MagicMock()
    state._offload_request_states.prepare.return_value = (True, {0: (0, 128)}, ())
    metadata = SimpleNamespace(copy_sfa_copy_src_offsets=torch.zeros(1, dtype=torch.int32))
    built = {}

    def build_metadata(**kwargs):
        built.update(kwargs)
        return {"main.layer": metadata}

    manager = MagicMock()
    update = MagicMock()
    monkeypatch.setattr(model_state_module, "build_attn_metadata", build_metadata)
    monkeypatch.setattr(model_state_module, "update_sparse_kv_offload_metadata", update)
    monkeypatch.setattr(model_state_module, "get_prebound_copy_sfa_slots", lambda: {"request-1": 0})
    monkeypatch.setattr(model_state_module, "get_sparse_kv_offload_manager", lambda: manager)
    monkeypatch.setattr(
        model_state_module,
        "get_ascend_config",
        lambda: SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(topk_buffer_size=4096)),
    )
    batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=1,
        query_start_loc_np=np.array([0, 1], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        req_ids=["request-1"],
        num_computed_tokens_np=np.array([128], dtype=np.int32),
        is_prefilling_np=np.array([False]),
        num_scheduled_tokens=torch.tensor([1], dtype=torch.int32),
        seq_lens=torch.tensor([129], dtype=torch.int32),
        seq_lens_np=np.array([129], dtype=np.int32),
        dcp_local_seq_lens=None,
        positions=torch.tensor([128], dtype=torch.int32),
        attn_state=None,
        is_dummy=False,
    )
    block_table = torch.zeros((1, 1), dtype=torch.int32)
    result = state.prepare_attn(
        batch,
        CUDAGraphMode.NONE,
        (block_table,),
        torch.zeros((1, 1), dtype=torch.int32),
        [],
        SimpleNamespace(),
    )

    assert result == {"main.layer": metadata}
    update.assert_called_once()
    prepared = state._offload_request_states.prepare.call_args.kwargs
    assert prepared["prebound_slots"] == {"request-1": 0}
    assert prepared["live_req_ids"] is state._offload_live_req_ids
    assert built["copy_sfa_restore_tails"]
    assert built["draft_layer_names"] == {"draft.layer"}
    assert built["req_topk_buffer_slots"].shape == (1,)
    assert manager.dense_fill_copy_sfa_rows.call_args.kwargs["block_table"] is block_table
    manager.restore_copy_sfa_tails.assert_called_once_with(metadata)


@pytest.mark.parametrize("splits", [1, 9, 16])
def test_combined_attention_kernel_blocks_match_physical_rows(splits):
    shape = (2, 3 * splits, 128, 2, 256)
    block_elements = 128 * 2 * 256
    raw = torch.zeros(2 * shape[1] * block_elements, dtype=torch.float16)
    key, value = attn_utils._reshape_combined_attention_kv_cache(
        raw.view(torch.int8), shape, raw.dtype, splits * 2 * block_elements * 2, splits
    )
    physical = raw.view(3, splits, 2, 128, 2, 256)
    for block_id in range(shape[1]):
        key[block_id].fill_(block_id + 1)
        value[block_id].fill_(-block_id - 1)
    torch.testing.assert_close(key, physical[:, :, 0].flatten(0, 1))
    torch.testing.assert_close(value, physical[:, :, 1].flatten(0, 1))
    assert not key.is_contiguous()
    assert key.stride(0) == 2 * block_elements
    assert key.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    with pytest.raises(ValueError, match="Padded combined"):
        attn_utils._reshape_combined_attention_kv_cache(
            raw.view(torch.int8), shape, raw.dtype, 9 * 2 * block_elements * 2 + 64, 9
        )


@pytest.mark.parametrize("splits", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_padded_attention_kernel_writes_preserve_other_pages_and_padding(splits, dtype):
    num_blocks, block_size, heads, dim = 3, 128, 1, 1
    block_elements = block_size * heads * dim
    dtype_size = torch.empty((), dtype=dtype).element_size()
    page_bytes = splits * 2 * block_elements * dtype_size + 64
    kernel_stride_elements = page_bytes // splits // dtype_size
    raw = torch.full((num_blocks, splits, kernel_stride_elements), -7, dtype=dtype)
    shape = (2, num_blocks * splits, block_size, heads, dim)
    key, value = attn_utils._reshape_combined_attention_kv_cache(
        raw.view(torch.int8).flatten(), shape, dtype, page_bytes, splits
    )
    assert key.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    assert value.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    assert not key.is_contiguous()
    assert not value.is_contiguous()
    assert key.stride(0) * splits * dtype_size == page_bytes

    for block_id in range(num_blocks):
        raw.fill_(-7)
        before = raw.clone()
        start, end = block_id * splits, (block_id + 1) * splits
        key[start:end].fill_(block_id + 1)
        value[start:end].fill_(-block_id - 1)
        torch.testing.assert_close(raw[:block_id], before[:block_id])
        torch.testing.assert_close(raw[block_id + 1 :], before[block_id + 1 :])
        assert torch.all(raw[block_id, :, :block_elements] == block_id + 1)
        assert torch.all(raw[block_id, :, block_elements : 2 * block_elements] == -block_id - 1)
        torch.testing.assert_close(raw[:, :, 2 * block_elements :], before[:, :, 2 * block_elements :])


@pytest.mark.parametrize(
    ("splits", "page_bytes", "message"),
    [(0, 512, "positive"), (2, 1026, "dtype-aligned"), (2, 512, "too small")],
)
def test_combined_attention_rejects_invalid_kernel_page_geometry(splits, page_bytes, message):
    raw = torch.zeros(4096, dtype=torch.int8)
    with pytest.raises(ValueError, match=message):
        attn_utils._reshape_combined_attention_kv_cache(raw, (2, 4, 128, 1, 1), torch.float16, page_bytes, splits)


@pytest.mark.parametrize("for_capture", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_tq_batched_slots_use_builder_config_without_global_context(monkeypatch, for_capture, dtype):
    _, specs, calls, attn_groups, kv_cache_config = _make_dsa_metadata_groups()
    config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="turboquant_4bit_nc"))
    for group, attn_group in zip(kv_cache_config.kv_cache_groups, attn_groups):
        group.kv_cache_spec = replace(group.kv_cache_spec, cache_dtype_str="turboquant_4bit_nc")
        attn_group[0].get_metadata_builder(0).vllm_config = config
    plan = dsa_attn_kv_plan.get_dsa_attn_kv_plan(config, compress_ratio=4)
    get_plan = MagicMock(return_value=plan)
    get_current = MagicMock(side_effect=AssertionError("No global config during metadata preparation"))
    monkeypatch.setattr(attn_utils, "get_dsa_attn_kv_plan", get_plan)
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", get_current)
    slots = torch.tensor([[0, 65, -1, 111], [1, 130, -2, 222]], dtype=dtype)
    metadata_args = dict(
        attn_groups=attn_groups,
        num_reqs=1,
        num_tokens=3,
        query_start_loc_gpu=torch.tensor([0, 3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 3], dtype=torch.int32),
        max_query_len=3,
        seq_lens=torch.tensor([3], dtype=torch.int32),
        max_seq_len=3,
        block_tables=tuple(torch.zeros((1, 1), dtype=torch.int32) for _ in specs),
        slot_mappings=slots,
        kv_cache_config=kv_cache_config,
        for_cudagraph_capture=for_capture,
    )
    attn_utils.build_attn_metadata(**metadata_args)
    get_current.assert_not_called()
    get_plan.assert_called_once_with(config, 4)
    for call, spec, row in zip(calls, specs, slots):
        expected = plan.format_dsa_slot_mapping(row[:3], get_storage_block_size(spec))
        torch.testing.assert_close(call["formatted_slot_mapping"], expected, rtol=0, atol=0)
        assert call["for_cudagraph_capture"] == for_capture
    cached_sizes = attn_groups[0][0].get_metadata_builder(0).tq_group_block_sizes
    slots.add_(17)
    attn_utils.build_attn_metadata(**metadata_args)
    assert attn_groups[0][0].get_metadata_builder(0).tq_group_block_sizes is cached_sizes
    for call, spec, row in zip(calls[2:], specs, slots):
        expected = plan.format_dsa_slot_mapping(row[:3], get_storage_block_size(spec))
        torch.testing.assert_close(call["formatted_slot_mapping"], expected, rtol=0, atol=0)
    get_current.assert_not_called()


def test_prepare_kernel_block_sizes_uses_logical_size_for_dsv4():
    spec = _make_dsv4_mla_spec(128, 4)
    attn_groups = [
        [
            AttentionGroup(
                backend=AscendDSAC4Backend,
                layer_names=["model.layers.0.self_attn"],
                kv_cache_spec=spec,
                kv_cache_group_id=0,
            )
        ]
    ]
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.0.self_attn"],
                kv_cache_spec=spec,
            )
        ],
    )

    assert get_storage_block_size(spec) == 32
    assert upstream_attn_utils.prepare_kernel_block_sizes(kv_cache_config, attn_groups) == [spec.block_size]


@pytest.mark.parametrize(
    ("device_type", "expected_c128_state_sizes"),
    [
        (AscendDeviceType.A2, [8, 16, 32]),
        (AscendDeviceType.A5, [4, 8, 16]),
    ],
)
def test_dsv4_backends_declare_role_specific_logical_sizes(
    monkeypatch,
    device_type,
    expected_c128_state_sizes,
):
    monkeypatch.setattr(
        dsa_v1,
        "get_current_hardware_profile",
        lambda: get_hardware_profile(device_type),
    )

    assert AscendDSAC4Backend.get_supported_kernel_block_sizes() == [128, 256, 512]
    assert AscendDSAC128Backend.get_supported_kernel_block_sizes() == [4096, 8192, 16384]
    assert AscendDSASWABackend.get_supported_kernel_block_sizes() == [32, 64, 128]
    assert AscendDSAC4StateBackend.get_supported_kernel_block_sizes() == [2, 4, 8]
    assert AscendDSAC128StateBackend.get_supported_kernel_block_sizes() == expected_c128_state_sizes

    c4_cache = SimpleNamespace(compress_ratio=4)
    c128_cache = SimpleNamespace(compress_ratio=128)
    c4_indexer = cast(deepseek_v4_indexer.AscendDeepseekV4IndexerCache, c4_cache)
    c128_indexer = cast(deepseek_v4_indexer.AscendDeepseekV4IndexerCache, c128_cache)
    swa_cache = cast(deepseek_v4_model.AscendDeepseekV4SWACache, SimpleNamespace())
    c4_state = cast(deepseek_v4_compressor.AscendCompressorStateCache, c4_cache)
    c128_state = cast(deepseek_v4_compressor.AscendCompressorStateCache, c128_cache)
    assert deepseek_v4_indexer.AscendDeepseekV4IndexerCache.get_attn_backend(c4_indexer) is AscendDSAC4Backend
    assert deepseek_v4_indexer.AscendDeepseekV4IndexerCache.get_attn_backend(c128_indexer) is AscendDSAC128Backend
    assert deepseek_v4_model.AscendDeepseekV4SWACache.get_attn_backend(swa_cache) is AscendDSASWABackend
    assert deepseek_v4_compressor.AscendCompressorStateCache.get_attn_backend(c4_state) is AscendDSAC4StateBackend
    assert deepseek_v4_compressor.AscendCompressorStateCache.get_attn_backend(c128_state) is AscendDSAC128StateBackend


@pytest.mark.parametrize(
    (
        "caller",
        "cudagraph_mode",
        "for_capture",
        "pcp_size",
        "expected_input_tokens",
    ),
    [
        ("default", None, False, 1, 5),
        ("model_state", CUDAGraphMode.NONE, False, 1, 5),
        ("model_state", CUDAGraphMode.FULL, False, 1, 8),
        ("pcp_capture", CUDAGraphMode.NONE, True, 2, 8),
        ("pcp_runtime", CUDAGraphMode.NONE, False, 2, 8),
    ],
)
@pytest.mark.parametrize("builder_cls", [_RecordingDSAMetadataBuilder, _RecordingDSACPMetadataBuilder])
def test_mrv2_builds_shared_dsa_metadata_for_each_execution_mode(
    monkeypatch,
    caller,
    cudagraph_mode,
    for_capture,
    pcp_size,
    expected_input_tokens,
    builder_cls,
):
    parallel_config = SimpleNamespace(
        prefill_context_parallel_size=pcp_size,
        decode_context_parallel_size=2,
        cp_kv_cache_interleave_size=2,
    )
    monkeypatch.setattr(attn_utils, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=0))
    layer_names, specs, calls, attn_groups, kv_cache_config = _make_dsa_metadata_groups(builder_cls)
    block_tables = (
        torch.zeros((4, 1), dtype=torch.int32),
        torch.zeros((4, 1), dtype=torch.int32),
    )
    slot_mappings = torch.zeros((2, 8), dtype=torch.int32)
    dcp_local_seq_lens = torch.tensor(
        [2, 2, 0, 0],
        dtype=torch.int32,
    )
    pcp_context = object() if pcp_size > 1 else None
    pcp_manager = (
        SimpleNamespace(
            build_attention_context=MagicMock(return_value=pcp_context),
        )
        if pcp_context is not None
        else None
    )

    if caller == "default":
        metadata = attn_utils.build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=2,
            num_tokens=5,
            query_start_loc_gpu=torch.tensor([0, 2, 5], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 2, 5], dtype=torch.int32),
            max_query_len=3,
            seq_lens=torch.tensor([2, 3], dtype=torch.int32),
            max_seq_len=8,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_np=np.array([2, 3], dtype=np.int32),
            positions=torch.arange(5, dtype=torch.int32),
            dcp_local_seq_lens=dcp_local_seq_lens[:2],
            parallel_config=parallel_config,
        )
    else:
        model_state = AscendModelState.__new__(AscendModelState)
        model_state.max_model_len = 8
        model_state.vllm_config = SimpleNamespace(
            parallel_config=parallel_config,
        )
        model_state.pcp_manager = pcp_manager
        input_batch = SimpleNamespace(
            num_reqs=2,
            num_reqs_after_padding=4,
            num_tokens=5,
            num_tokens_after_padding=8,
            query_start_loc_np=np.array([0, 2, 5, 5, 5], dtype=np.int32),
            query_start_loc=torch.tensor([0, 2, 5, 5, 5], dtype=torch.int32),
            num_scheduled_tokens=torch.tensor([2, 3, 0, 0], dtype=torch.int32),
            seq_lens=torch.tensor([2, 3, 0, 0], dtype=torch.int32),
            seq_lens_np=np.array([2, 3, 0, 0], dtype=np.int32),
            is_prefilling_np=np.array([True, True, False, False]),
            dcp_local_seq_lens=dcp_local_seq_lens,
            positions=torch.arange(8, dtype=torch.int32),
            attn_state=None,
        )
        metadata = model_state.prepare_attn(
            input_batch=input_batch,
            cudagraph_mode=cudagraph_mode,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            attn_groups=attn_groups,
            kv_cache_config=kv_cache_config,
            for_capture=for_capture,
        )

    assert set(metadata) == set(layer_names)
    assert len(calls) == 2
    for call in calls:
        common_metadata = call["common_attn_metadata"]
        assert common_metadata.num_actual_tokens == 5
        assert common_metadata.num_input_tokens == expected_input_tokens
        assert call["for_cudagraph_capture"] is for_capture
        assert call["num_actual_reqs"] == 2
        assert call["pcp_context"] is pcp_context
        if caller != "default":
            assert torch.equal(
                common_metadata.is_prefilling,
                torch.tensor([True, True, False, False]),
            )
        expected_dcp_local_seq_lens = dcp_local_seq_lens[:2] if caller == "default" else dcp_local_seq_lens
        torch.testing.assert_close(common_metadata.dcp_local_seq_lens, expected_dcp_local_seq_lens)
        torch.testing.assert_close(
            common_metadata.dcp_local_seq_lens_cpu, expected_dcp_local_seq_lens[: common_metadata.num_reqs]
        )
    cache_name = "common_ratio_to_sas_metadata"
    assert calls[0][cache_name] is calls[1][cache_name]
    assert calls[1][cache_name]["first_group"] is True
    if pcp_context is not None:
        assert [call["pcp_cache_group_idx"] for call in calls] == [0, 1]
        assert pcp_manager is not None
        pcp_manager.build_attention_context.assert_called_once_with(input_batch, block_tables, slot_mappings)
    else:
        assert all(call["pcp_cache_group_idx"] is None for call in calls)


def test_mrv2_allocates_and_reshapes_hidden_state_cache(monkeypatch):
    """Keep private buffers and match the lane's upstream cache-write layout."""
    from vllm.model_executor.models.extract_hidden_states import (
        CacheOnlyAttentionBackend,
    )

    layer_name = "draft.cache_only_layers.36"
    block_size = 16
    num_kv_heads = 3
    head_size = 8
    num_blocks = 4
    dtype = torch.bfloat16
    spec = HiddenStateCacheSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
    )
    page_bytes = spec.page_size_bytes
    tensor_size = num_blocks * page_bytes

    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[_make_kv_cache_tensor(tensor_size, [layer_name], page_bytes)],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[layer_name],
                kv_cache_spec=spec,
            )
        ],
    )

    monkeypatch.setattr(
        attn_utils,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            additional_config={},
            kv_transfer_config=None,
            model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3")),
            quant_config=None,
            cache_config=SimpleNamespace(cache_dtype="auto"),
        ),
    )
    monkeypatch.setattr(attn_utils, "_is_dsv4_model", lambda _cfg: False)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _cfg: False)

    raw = attn_utils._allocate_kv_cache(kv_cache_config, shared_layers={}, device="cpu")
    assert isinstance(raw[layer_name], torch.Tensor)
    assert raw[layer_name].numel() == tensor_size

    attn_groups = [
        AttentionGroup(
            backend=CacheOnlyAttentionBackend,
            layer_names=[layer_name],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]
    reshaped = attn_utils._reshape_kv_cache_v2(
        attn_groups=attn_groups,
        kv_cache_raw_tensors=raw,
        cache_dtype="auto",
        kernel_block_sizes=[block_size],
        shared_kv_cache_layers={},
        kv_cache_config=kv_cache_config,
    )
    cache = reshaped[layer_name]
    assert isinstance(cache, torch.Tensor)
    assert cache.shape == (num_blocks, num_kv_heads, block_size, head_size)
    assert cache.dtype == dtype


class _PrefillStateBuilder:
    consumes_pcp_context = False

    def __init__(self):
        self.extra_kwargs = None

    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        assert common_prefix_len == 0
        self.extra_kwargs = kwargs
        return common_attn_metadata.is_prefilling


class _CaptureStateBuilder(_PrefillStateBuilder):
    def build_for_cudagraph_capture(self, common_attn_metadata, **kwargs):
        self.extra_kwargs = kwargs
        return common_attn_metadata.is_prefilling


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("full_graph", [False, True])
def test_v41_capture_keeps_compressed_indexer_branches(monkeypatch, capture, full_graph):
    monkeypatch.setattr(attn_utils, "AscendDSAV41MetadataBuilder", _CaptureStateBuilder)
    builder = _CaptureStateBuilder()
    group = SimpleNamespace(layer_names=["layer.0"], get_metadata_builder=lambda _: builder)
    attn_utils.build_attn_metadata(
        attn_groups=[[group]],
        num_reqs=1,
        num_tokens=1,
        query_start_loc_gpu=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        max_query_len=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        max_seq_len=1,
        block_tables=(torch.zeros((1, 1), dtype=torch.int32),),
        slot_mappings=(torch.zeros(1, dtype=torch.int64),),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())]),
        seq_lens_np=np.array([1], dtype=np.int32),
        positions=torch.tensor([0], dtype=torch.int64),
        for_cudagraph_capture=capture,
        full_graph_mode=full_graph,
    )
    assert builder.extra_kwargs is not None
    assert builder.extra_kwargs["full_graph_mode"] is (capture or full_graph)


@pytest.mark.parametrize("cache_only_backend", [False, True])
@pytest.mark.parametrize("for_cudagraph_capture", [False, True])
def test_build_attn_metadata_propagates_prefill_and_pcp_context(monkeypatch, for_cudagraph_capture, cache_only_backend):
    sfa_builder_cls = type("UnrelatedSFABuilder", (), {}) if cache_only_backend else _PrefillStateBuilder
    monkeypatch.setattr(attn_utils, "AscendSFAMetadataBuilder", sfa_builder_cls)
    builder = _CaptureStateBuilder() if for_cudagraph_capture else _PrefillStateBuilder()
    builder.consumes_pcp_context = cache_only_backend
    attn_group = SimpleNamespace(
        layer_names=["layer.0"],
        get_metadata_builder=lambda _: builder,
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())],
    )
    is_prefilling = torch.tensor([True])
    pcp_context = object()

    metadata = attn_utils.build_attn_metadata(
        attn_groups=[[attn_group]],
        num_reqs=1,
        num_tokens=1,
        query_start_loc_gpu=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        max_query_len=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        max_seq_len=1,
        block_tables=(torch.zeros((1, 1), dtype=torch.int32),),
        slot_mappings=(torch.zeros(1, dtype=torch.int64),),
        kv_cache_config=kv_cache_config,
        is_prefilling=is_prefilling,
        pcp_context=pcp_context,
        seq_lens_np=np.array([1], dtype=np.int32),
        positions=torch.tensor([0], dtype=torch.int64),
        for_cudagraph_capture=for_cudagraph_capture,
    )

    assert metadata["layer.0"] is is_prefilling
    assert builder.extra_kwargs == {
        "pcp_context": pcp_context,
        "pcp_cache_group_idx": 0,
    }


@pytest.mark.parametrize("packed", [False, True], ids=["mla", "sfa-c8"])
def test_main_entry_allocates_and_reshapes_kvpp_views(monkeypatch, packed):
    from vllm.v1.worker.gpu import model_runner as upstream_model_runner

    from tests.ut.kvpp_utils import assert_attention_cache_views, make_attention_cache_case, make_cache_config
    from vllm_ascend.core import kv_cache_placement
    from vllm_ascend.patch.worker.patch_v2 import patch_attn_utils
    from vllm_ascend.worker import kvpp_cache

    config, specs, layers = make_attention_cache_case(packed)
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_args, **_kwargs: layers)
    monkeypatch.setattr(kv_cache_placement, "get_layers_from_vllm_config", lambda *_args: layers)
    for module in (attn_utils, kv_cache_placement):
        monkeypatch.setattr(module, "enable_sfa", lambda _: packed)
        monkeypatch.setattr(module, "enable_fa_quant", lambda _: False)
    monkeypatch.setattr(kvpp_cache, "get_kvpp_group", lambda: SimpleNamespace(rank_in_group=1))
    monkeypatch.setattr(attn_utils, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: False))
    raw = {}

    def allocate(*args, **kwargs):
        result = kvpp_cache.allocate_kvpp_cache(*args, **kwargs)
        raw.update(result)
        return result

    monkeypatch.setattr(attn_utils, "allocate_kvpp_cache", allocate)
    assert upstream_model_runner.get_kv_cache_spec is patch_attn_utils.get_kv_cache_spec
    assert upstream_attn_utils.allocate_kv_cache is patch_attn_utils.allocate_kv_cache_main
    caches = upstream_attn_utils.allocate_kv_cache(
        make_cache_config(specs), device=torch.device("cpu"), layout=None, kernel_block_sizes=[2]
    )
    assert_attention_cache_views(caches, raw, packed)


def _make_mla_layer(*, fa_quant: bool = False, sparse_c8: bool = False):
    layer = attn_utils.MLAAttention.__new__(attn_utils.MLAAttention)
    layer.kv_sharing_target_layer_name = None
    layer.head_size = 64
    layer.qk_rope_head_dim = 64
    layer.kv_lora_rank = 128
    layer.impl = SimpleNamespace(
        fa_quant_layer=fa_quant,
        enable_sparse_sfa_c8=sparse_c8,
        enable_sparse_sfa_turboquant=False,
        dtype=torch.bfloat16,
    )
    layer.get_kv_cache_spec = lambda _cfg: SimpleNamespace(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
        model_version=None,
        non_causal_multi_token_decode=False,
        tokens_per_state=1,
    )
    return layer


def test_v2_sparse_offload_marks_main_mla_cache_host_resident(monkeypatch):
    layer_name = "model.layers.0.self_attn.attn"
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(block_size=16, cache_dtype="auto"),
        attention_config=SimpleNamespace(indexer_kv_dtype="int8"),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(kv_lora_rank=128, qk_rope_head_dim=64),
        ),
    )
    sparse_cfg = SimpleNamespace(enabled=True)
    monkeypatch.setattr(
        attn_utils, "get_layers_from_vllm_config", lambda *_args, **_kwargs: {layer_name: _make_mla_layer()}
    )
    monkeypatch.setattr(attn_utils, "enable_sfa_dcp_replicated_indexer", lambda _cfg: False)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _cfg: True)
    monkeypatch.setattr(attn_utils, "get_ascend_config", lambda: SimpleNamespace(sparse_kv_offload_config=sparse_cfg))

    assert attn_utils.get_kv_cache_spec(vllm_config)[layer_name].store_on_host
    sparse_cfg.enabled = False
    assert not attn_utils.get_kv_cache_spec(vllm_config)[layer_name].store_on_host


def test_sfa_indexer_allocates_and_reshapes_scale_views(monkeypatch):
    layer_name = "model.layers.0.self_attn.indexer.k_cache"
    spec = AscendSFAIndexerCacheSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        sfa_dcp_replicated_indexer_size=1,
    )
    num_blocks = 2
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            _make_kv_cache_tensor(
                num_blocks * spec.page_size_bytes,
                [layer_name],
                spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=spec)],
    )
    vllm_config = SimpleNamespace(
        additional_config={},
        kv_transfer_config=None,
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _cfg: False)

    raw = attn_utils._allocate_kv_cache(kv_cache_config, shared_layers={}, device=torch.device("cpu"))
    raw_k, raw_scale = raw[layer_name]
    assert raw_k.dtype == torch.int8
    assert raw_scale.dtype == torch.int8

    backend = SimpleNamespace(
        get_kv_cache_shape=lambda num_blocks_, block_size, num_kv_heads, head_size: (
            num_blocks_,
            block_size,
            num_kv_heads,
            head_size,
        )
    )
    caches = attn_utils._reshape_kv_cache_v2(
        attn_groups=[
            SimpleNamespace(
                kv_cache_group_id=0,
                kv_cache_spec=spec,
                layer_names=[layer_name, "alias"],
                backend=backend,
            ),
            SimpleNamespace(kv_cache_group_id=9, kv_cache_spec=spec, layer_names=[], backend=backend),
        ],
        kv_cache_raw_tensors=raw,
        cache_dtype="auto",
        kernel_block_sizes=[spec.block_size],
        shared_kv_cache_layers={"alias": layer_name},
        kv_cache_config=kv_cache_config,
    )
    indexer_k, indexer_scale = caches[layer_name]
    assert indexer_k.shape == (num_blocks, spec.block_size, spec.num_kv_heads, spec.head_size)
    assert indexer_scale.shape == (num_blocks, spec.block_size, spec.num_kv_heads, spec.scale_dim)
    assert caches["alias"] is caches[layer_name]


def test_sfa_indexer_li_c4_allocates_and_reshapes_scale_views(monkeypatch):
    layer_name = "model.layers.0.self_attn.indexer.k_cache"
    index_head_dim = 128
    spec = AscendSFAIndexerCacheSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=index_head_dim // 2,
        dtype=torch.uint8,
        scale_dim=index_head_dim // 64 * 2,
        scale_dtype=torch.float8_e8m0fnu,
        cache_sparse_li_c4=True,
        cache_sparse_li_c8=False,
        sfa_dcp_replicated_indexer_size=1,
    )
    num_blocks = 2
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            _make_kv_cache_tensor(
                num_blocks * spec.page_size_bytes,
                [layer_name],
                spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=spec)],
    )
    vllm_config = SimpleNamespace(
        additional_config={},
        kv_transfer_config=None,
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _cfg: False)

    raw = attn_utils._allocate_kv_cache(kv_cache_config, shared_layers={}, device=torch.device("cpu"))
    raw_k, raw_scale = raw[layer_name]
    assert raw_k.dtype == torch.int8
    assert raw_scale.dtype == torch.int8

    backend = SimpleNamespace(
        get_kv_cache_shape=lambda num_blocks_, block_size, num_kv_heads, head_size: (
            num_blocks_,
            block_size,
            num_kv_heads,
            head_size,
        )
    )
    caches = attn_utils._reshape_kv_cache_v2(
        attn_groups=[
            SimpleNamespace(
                kv_cache_group_id=0,
                kv_cache_spec=spec,
                layer_names=[layer_name],
                backend=backend,
            ),
        ],
        kv_cache_raw_tensors=raw,
        cache_dtype="auto",
        kernel_block_sizes=[spec.block_size],
        shared_kv_cache_layers={},
        kv_cache_config=kv_cache_config,
    )
    indexer_k, indexer_scale = caches[layer_name]
    # indexer k: (num_blocks, block_size, 1, head_size)
    assert indexer_k.shape == (num_blocks, spec.block_size, 1, index_head_dim // 2)
    assert indexer_k.dtype == torch.uint8
    assert indexer_scale.shape == (num_blocks, spec.block_size, 1, 2, 2)
    assert indexer_scale.dtype == torch.float8_e8m0fnu


def test_attn_state_mla_spec_and_metadata_wrappers(monkeypatch):
    encoder_spec = attn_utils.EncoderOnlyAttentionSpec.__new__(attn_utils.EncoderOnlyAttentionSpec)
    pooling_encoder = SimpleNamespace(
        model_config=SimpleNamespace(runner_type="pooling"),
    )
    pooling_other = SimpleNamespace(
        model_config=SimpleNamespace(runner_type="pooling"),
    )
    encoder_kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=encoder_spec)])
    other_kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())])
    mtp = SimpleNamespace(
        model_config=SimpleNamespace(runner_type="generate"),
        speculative_config=SimpleNamespace(method="mtp"),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
    )
    no_spec = SimpleNamespace(
        model_config=SimpleNamespace(runner_type="generate"),
        speculative_config=None,
        scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
    )
    eagle = SimpleNamespace(
        model_config=SimpleNamespace(runner_type="generate"),
        speculative_config=SimpleNamespace(method="eagle"),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
    )
    chunked = SimpleNamespace(
        model_config=SimpleNamespace(runner_type="generate"),
        speculative_config=None,
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )
    seq = np.array([4, 4], dtype=np.int32)
    ones = np.array([1, 1], dtype=np.int32)
    scheduled = np.array([2, 2], dtype=np.int32)
    state = attn_utils.AscendAttentionState
    assert (
        attn_utils.build_attn_state(
            pooling_encoder,
            seq,
            2,
            seq,
            seq,
            kv_cache_config=encoder_kv_cache_config,
        )
        is state.PrefillNoCache
    )
    assert (
        attn_utils.build_attn_state(
            pooling_other,
            seq,
            2,
            seq,
            seq,
            kv_cache_config=other_kv_cache_config,
        )
        is state.PrefillCacheHit
    )
    assert attn_utils.build_attn_state(no_spec, seq, 2, seq, seq) is state.PrefillNoCache
    assert attn_utils.build_attn_state(mtp, seq, 2, ones, ones) is state.DecodeOnly
    assert attn_utils.build_attn_state(no_spec, seq, 2, ones, ones) is state.DecodeOnly
    assert attn_utils.build_attn_state(mtp, seq, 2, scheduled, ones) is state.ChunkedPrefill
    assert attn_utils.build_attn_state(eagle, seq, 2, scheduled, ones) is state.ChunkedPrefill
    assert attn_utils.build_attn_state(chunked, seq, 2, scheduled, scheduled) is state.ChunkedPrefill
    assert attn_utils.build_attn_state(no_spec, seq, 2, scheduled, scheduled) is state.PrefillCacheHit

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(block_size=16, cache_dtype="auto"),
        attention_config=SimpleNamespace(indexer_kv_dtype="int8", hisparse_config=None),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(kv_lora_rank=128, qk_rope_head_dim=64),
        ),
    )
    layers = {
        "shared": SimpleNamespace(kv_sharing_target_layer_name="fa"),
        "dropped": SimpleNamespace(kv_sharing_target_layer_name=None, get_kv_cache_spec=lambda _cfg: None),
        "fa": _make_mla_layer(fa_quant=True),
        "sfa": _make_mla_layer(sparse_c8=True),
    }
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_args, **_kwargs: layers)
    monkeypatch.setattr(attn_utils, "enable_sfa_dcp_replicated_indexer", lambda _cfg: False)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _cfg: True)
    monkeypatch.setattr(
        attn_utils,
        "get_ascend_config",
        lambda: SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=False)),
    )
    monkeypatch.setattr(
        attn_utils,
        "get_current_hardware_profile",
        lambda: get_hardware_profile(AscendDeviceType.A2),
    )
    specs = attn_utils.get_kv_cache_spec(vllm_config)
    assert set(specs) == {"fa", "sfa"}
    assert specs["fa"].head_size == 128
    assert specs["sfa"].cache_sparse_sfa_c8 is True

    mla_spec = AscendMLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: vllm_config)
    assert attn_utils._get_attention_kv_cache_dims("sfa", mla_spec) == (128, 64)
    layers["sfa"] = object()
    with pytest.raises(TypeError):
        attn_utils._get_attention_kv_cache_dims("sfa", mla_spec)

    builder = _PrefillStateBuilder()
    attn_utils.build_attn_metadata(
        attn_groups=[[SimpleNamespace(layer_names=["layer.0"], get_metadata_builder=lambda _: builder)]],
        num_reqs=1,
        num_tokens=2,
        query_start_loc_gpu=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        max_query_len=2,
        seq_lens=torch.tensor([2], dtype=torch.int32),
        max_seq_len=4,
        block_tables=(torch.zeros((1, 1), dtype=torch.int32),),
        slot_mappings=(torch.zeros(1, dtype=torch.int64),),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())]),
    )

    monkeypatch.setattr(
        attn_utils,
        "get_current_vllm_config",
        lambda: SimpleNamespace(kv_transfer_config=object()),
    )
    aligned = attn_utils._allocate_int8_cache_tensor(8, 64, torch.device("cpu"))
    assert aligned.numel() == 8
    assert attn_utils._align_up(5, 4) == 8

    with pytest.raises(ValueError, match="requires KVCacheConfig"):
        attn_utils._reshape_kv_cache_v2([], {}, "auto", [], {}, None)

    def stub(**kwargs):
        return kwargs

    module = SimpleNamespace(build_attn_metadata=stub)
    monkeypatch.setattr(attn_utils, "_BUILD_ATTN_METADATA_MODULE", module)
    with attn_utils.build_attn_metadata_wrapper():
        assert module.build_attn_metadata is attn_utils.build_attn_metadata
    with attn_utils.build_attn_metadata_factory(torch.arange(4), 2, True):
        forwarded = module.build_attn_metadata()
    assert forwarded["positions"].tolist() == [0, 1]
    assert forwarded["is_prefilling"] is True
    assert module.build_attn_metadata is stub


def test_mrv2_binding_wraps_only_v41_slots():
    from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheLayer
    from vllm_ascend.patch.worker.patch_bind_kv_cache import bind_kv_cache_to_layers

    vllm_config = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context={}))
    v41_layer = DeepseekV41CacheLayer(vllm_config, "model.layers.0.self_attn.attn", object())
    v41_indexer = DeepseekV41CacheLayer(vllm_config, "model.layers.2.self_attn.indexer.k_cache", object())
    other_layer = SimpleNamespace(kv_cache=None)
    kv_caches = {
        # V4.1 reshape output: one slot tensor per layer.
        "model.layers.0.self_attn.attn": torch.zeros(4, 2),
        # Non-V4.1 Ascend allocation: a (k, v) tuple.
        "model.layers.1.self_attn.attn": (torch.zeros(2, 2), torch.zeros(2, 2)),
        # V4.1 indexer reshape output: a (kv, scale) tuple slot view.
        "model.layers.2.self_attn.indexer.k_cache": (torch.zeros(2, 2), torch.zeros(2, 1)),
    }
    forward_context = {
        "model.layers.0.self_attn.attn": v41_layer,
        "model.layers.1.self_attn.attn": other_layer,
        "model.layers.2.self_attn.indexer.k_cache": v41_indexer,
    }

    bind_kv_cache_to_layers(kv_caches, forward_context)

    # vLLM main (#53781) routes init_kv_cache through bind_kv_cache_to_layers:
    # V4.1 slots dispatch to their own bind_kv_cache (kv_cache[0] contract)
    # while other layers keep the raw (k, v) allocation.
    assert isinstance(v41_layer.kv_cache, list)
    assert v41_layer.kv_cache[0] is kv_caches["model.layers.0.self_attn.attn"]
    assert other_layer.kv_cache is kv_caches["model.layers.1.self_attn.attn"]
    # The indexer consumer unpacks kv_cache[0] into (kv, scale).
    kv_view, scale_view = v41_indexer.kv_cache[0]
    assert kv_view is kv_caches["model.layers.2.self_attn.indexer.k_cache"][0]
    assert scale_view is kv_caches["model.layers.2.self_attn.indexer.k_cache"][1]


@pytest.mark.parametrize("for_capture", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_tq_groups_without_a_dsa_builder_skip_the_formatted_mapping(for_capture, dtype):
    # Regression: TurboQuant also backs the packed SFA main cache, which has no
    # DSA metadata builder. The DSA-only slot formatting has to be skipped rather
    # than raised on -- and only the is_dsa_builder branch reads a formatted
    # mapping, so nothing on the SFA path loses one.
    layer_names, specs, _, _, kv_cache_config = _make_dsa_metadata_groups()
    for group in kv_cache_config.kv_cache_groups:
        group.kv_cache_spec = replace(group.kv_cache_spec, cache_dtype_str="turboquant_4bit_nc")
    builders = [MagicMock(tq_group_block_sizes=None) for _ in layer_names]
    attn_groups = [
        [
            AttentionGroup(
                backend=AscendDSAC4Backend if _spec_compress_ratio(spec) == 4 else AscendDSAC128Backend,
                layer_names=[layer_name],
                kv_cache_spec=spec,
                kv_cache_group_id=group_id,
                metadata_builders=[builder],
            )
        ]
        for group_id, (layer_name, spec, builder) in enumerate(zip(layer_names, specs, builders))
    ]

    attn_utils.build_attn_metadata(
        attn_groups=attn_groups,
        num_reqs=1,
        num_tokens=3,
        query_start_loc_gpu=torch.tensor([0, 3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 3], dtype=torch.int32),
        max_query_len=3,
        seq_lens=torch.tensor([3], dtype=torch.int32),
        max_seq_len=3,
        block_tables=tuple(torch.zeros((1, 1), dtype=torch.int32) for _ in specs),
        slot_mappings=torch.tensor([[0, 65, -1, 111], [1, 130, -2, 222]], dtype=dtype),
        kv_cache_config=kv_cache_config,
        for_cudagraph_capture=for_capture,
    )

    for builder in builders:
        build_call = builder.build_for_cudagraph_capture if for_capture else builder.build
        assert "formatted_slot_mapping" not in build_call.call_args.kwargs
        # A group without a DSA builder keeps its lazy geometry untouched.
        assert builder.tq_group_block_sizes is None
