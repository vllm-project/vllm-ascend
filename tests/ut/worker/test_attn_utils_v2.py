from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.worker.gpu import attn_utils as upstream_attn_utils
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention import dsa_v1
from vllm_ascend.attention.dsa_v1 import (
    AscendDSAC4Backend,
    AscendDSAC4StateBackend,
    AscendDSAC128Backend,
    AscendDSAC128StateBackend,
    AscendDSAMetadataBuilder,
    AscendDSASWABackend,
)
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
)
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.models.deepseek_v4 import compressor as deepseek_v4_compressor
from vllm_ascend.models.deepseek_v4 import indexer as deepseek_v4_indexer
from vllm_ascend.models.deepseek_v4 import model as deepseek_v4_model
from vllm_ascend.utils import AscendDeviceType, vllm_version_is
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.model_states.default import AscendModelState


def _make_kv_cache_tensor(size: int, layer_names: list[str], page_size: int = 0) -> KVCacheTensor:
    """Build a KVCacheTensor; vLLM #51718 renamed shared_by -> layers on main."""
    if vllm_version_is("0.27.1"):
        return KVCacheTensor(size=size, shared_by=layer_names)
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
    ratio_kwargs = (
        {"compress_ratio": compress_ratio} if vllm_version_is("0.27.1") else {"tokens_per_state": compress_ratio}
    )
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
    return spec.compress_ratio if vllm_version_is("0.27.1") else spec.tokens_per_state


@pytest.mark.parametrize(
    ("replicated_indexer", "expected_size"),
    [(False, 1), (True, 4)],
)
def test_sfa_indexer_cache_spec_uses_dcp_replication(monkeypatch, replicated_indexer, expected_size):
    layer_name = "model.layers.0.self_attn.indexer.k_cache"
    indexer_module = DeepseekV32IndexerCache.__new__(DeepseekV32IndexerCache)
    torch.nn.Module.__init__(indexer_module)
    monkeypatch.setattr(
        indexer_module,
        "get_kv_cache_spec",
        lambda _config: object(),
    )

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
        cache_config=SimpleNamespace(block_size=128, cache_dtype="auto"),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(index_head_dim=128),
        ),
    )
    monkeypatch.setattr(
        attn_utils,
        "get_layers_from_vllm_config",
        lambda *_args, **_kwargs: {layer_name: indexer_module},
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
        lambda: SimpleNamespace(is_sparse_li_c8_layer=lambda _layer_name: False),
    )

    spec = attn_utils.get_kv_cache_spec(vllm_config)[layer_name]

    assert isinstance(spec, AscendSFAIndexerCacheSpec)
    assert spec.sfa_dcp_replicated_indexer_size == expected_size


@pytest.mark.parametrize(
    ("device_type", "cache_dtype", "scale_dtype", "component_dims"),
    [
        (
            AscendDeviceType.A2,
            torch.int8,
            torch.float16,
            (128, 1),
        ),
        (
            AscendDeviceType.A5,
            torch.float8_e4m3fn,
            torch.float32,
            (128, 1, 132),
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
        model_config=SimpleNamespace(
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
    assert spec.storage_block_size == cache_config.block_size
    merged_spec = spec.merge([spec])
    if vllm_version_is("0.27.1"):
        assert merged_spec.compress_ratio == cache_layer.compress_ratio
    else:
        assert merged_spec.tokens_per_state == cache_layer.compress_ratio
    assert merged_spec.storage_block_size == cache_config.block_size

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

    if vllm_version_is("0.27.1"):
        kv_caches = upstream_attn_utils.init_kv_cache(
            runner_kv_caches=runner_kv_caches,
            forward_context={layer_name: cache_layer},
            kv_cache_config=kv_cache_config,
            attn_groups=[[attn_group]],
            device=torch.device("cpu"),
            cache_dtype=cache_config.cache_dtype,
            kernel_block_sizes=[spec.block_size],
            vllm_config=vllm_config,
        )
    else:
        # vLLM #51718 reworked upstream init_kv_cache to allocate generic 4D
        # views via `allocate_kv_cache` + `create_kv_cache_views`; that layout
        # cannot express the Ascend DSV4 page-strided cache. Route the
        # allocation through the Ascend DSV4 path (the same wiring the v0.27.1
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
            return attn_utils._reshape_kv_cache_v2(
                attn_groups=[[attn_group]],
                kv_cache_raw_tensors=raw_tensors,
                cache_dtype=cache_config.cache_dtype,
                kernel_block_sizes=_kernel_block_sizes,
                shared_kv_cache_layers={},
                kv_cache_config=_kv_cache_config,
            )

        def _ascend_bind_kv_cache(
            kv_caches: dict[str, Any],
            forward_context: dict[str, Any],
            runner_kv_caches_: list[Any],
            num_attn_module: int = 1,
        ) -> None:
            del num_attn_module
            assert len(runner_kv_caches_) == 0
            for kv_cache in kv_caches.values():
                runner_kv_caches_.append(kv_cache)
            for layer_name_, kv_cache in kv_caches.items():
                forward_context[layer_name_].kv_cache = kv_cache

        monkeypatch.setattr(upstream_attn_utils, "allocate_kv_cache", _ascend_allocate_kv_cache)
        monkeypatch.setattr(upstream_attn_utils, "bind_kv_cache", _ascend_bind_kv_cache)
        kv_caches = upstream_attn_utils.init_kv_cache(
            runner_kv_caches=runner_kv_caches,
            forward_context={layer_name: cache_layer},
            kv_cache_config=kv_cache_config,
            device=torch.device("cpu"),
            kernel_block_sizes=[spec.block_size],
            vllm_config=vllm_config,
        )

    cache_components = kv_caches[layer_name]
    assert len(runner_kv_caches) == 1
    assert runner_kv_caches[0] is cache_components
    if vllm_version_is("0.27.1"):
        # The v0.27.1 patch binds the pre-set layer tensor in place.
        assert cache_layer.kv_cache is cache_components
    # On main the layer cache is replaced by the freshly allocated views, so
    # the returned structure is validated by the checks below instead.
    assert [component.shape for component in cache_components] == [
        (num_blocks, spec.storage_block_size, 1, dim) for dim in component_dims
    ]
    assert [component.dtype for component in cache_components] == [
        cache_dtype,
        scale_dtype,
        *([cache_dtype] if device_type == AscendDeviceType.A5 else []),
    ]
    backing_storage = cache_components[0].untyped_storage().data_ptr()
    assert all(component.untyped_storage().data_ptr() == backing_storage for component in cache_components)


class _RecordingDSAMetadataBuilder(AscendDSAMetadataBuilder):
    def __init__(self, calls: list[dict[str, Any]]):
        self.calls = calls
        self.for_cudagraph_capture = False

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
        }
        assert "block_size" not in kwargs
        self.calls.append(call)
        call["common_ratio_to_sas_metadata"].setdefault("first_group", len(self.calls) == 1)
        return SimpleNamespace(common_attn_metadata=common_attn_metadata)


def _make_dsa_metadata_groups():
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
                metadata_builders=[_RecordingDSAMetadataBuilder(calls)],
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

    assert spec.storage_block_size == 32
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
    ],
)
def test_mrv2_builds_shared_dsa_metadata_for_each_execution_mode(
    caller,
    cudagraph_mode,
    for_capture,
    pcp_size,
    expected_input_tokens,
):
    layer_names, specs, calls, attn_groups, kv_cache_config = _make_dsa_metadata_groups()
    block_tables = (
        torch.zeros((4, 1), dtype=torch.int32),
        torch.zeros((4, 1), dtype=torch.int32),
    )
    slot_mappings = torch.zeros((2, 8), dtype=torch.int32)
    dcp_local_seq_lens = torch.tensor(
        [2, 1, 0, 0],
        dtype=torch.int32,
    )
    pcp_context = object() if caller == "pcp_capture" else None
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
        )
    else:
        model_state = AscendModelState.__new__(AscendModelState)
        model_state.max_model_len = 8
        model_state.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                prefill_context_parallel_size=pcp_size,
            ),
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
    cache_name = "common_ratio_to_sas_metadata"
    assert calls[0][cache_name] is calls[1][cache_name]
    assert calls[1][cache_name]["first_group"] is True
    if pcp_context is not None:
        assert [call["pcp_cache_group_idx"] for call in calls] == [0, 1]
        assert pcp_manager is not None
        pcp_manager.build_attention_context.assert_called_once_with()
    else:
        assert all(call["pcp_cache_group_idx"] is None for call in calls)


class _PrefillStateBuilder:
    def build(self, common_prefix_len, common_attn_metadata):
        assert common_prefix_len == 0
        return common_attn_metadata.is_prefilling


def test_build_attn_metadata_propagates_prefill_state():
    attn_group = SimpleNamespace(
        layer_names=["layer.0"],
        get_metadata_builder=lambda _: _PrefillStateBuilder(),
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())],
    )
    is_prefilling = torch.tensor([True])

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
        seq_lens_np=np.array([1], dtype=np.int32),
        positions=torch.tensor([0], dtype=torch.int64),
    )

    assert metadata["layer.0"] is is_prefilling
