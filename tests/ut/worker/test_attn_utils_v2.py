from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.worker.gpu import attn_utils as upstream_attn_utils
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.dsa_v1 import (
    AscendDSABackend,
    AscendDSAMetadataBuilder,
)
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.models import deepseek_v4
from vllm_ascend.utils import AscendDeviceType
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.model_states.default import AscendModelState


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
    cache_config = SimpleNamespace(block_size=32, cache_dtype="auto")
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
    )

    cache_layer = deepseek_v4.AscendDeepseekV4IndexerCache.__new__(deepseek_v4.AscendDeepseekV4IndexerCache)
    torch.nn.Module.__init__(cache_layer)
    cache_layer.head_dim = 128
    cache_layer.dtype = torch.int8
    cache_layer.cache_config = cache_config
    cache_layer.compress_ratio = 4
    cache_layer.kv_cache = torch.tensor([])

    monkeypatch.setattr(
        deepseek_v4,
        "get_ascend_device_type",
        lambda: device_type,
    )
    monkeypatch.setattr(
        attn_utils,
        "get_ascend_device_type",
        lambda: device_type,
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
    assert spec.storage_block_size == spec.block_size

    num_blocks = 2
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * spec.page_size_bytes,
                shared_by=[layer_name],
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[layer_name],
                kv_cache_spec=spec,
            )
        ],
    )
    attn_group = AttentionGroup(
        backend=AscendDSABackend,
        layer_names=[layer_name],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
    )
    runner_kv_caches: list[Any] = []

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

    cache_components = kv_caches[layer_name]
    assert cache_layer.kv_cache is cache_components
    assert len(runner_kv_caches) == 1
    assert runner_kv_caches[0] is cache_components
    assert [component.shape for component in cache_components] == [
        (num_blocks, spec.block_size, 1, dim) for dim in component_dims
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
            "block_size": kwargs["block_size"],
            "common_ratio_to_sas_metadata": self.common_ratio_to_sas_metadata,
        }
        self.calls.append(call)
        call["common_ratio_to_sas_metadata"].setdefault("first_group", len(self.calls) == 1)
        return SimpleNamespace(common_attn_metadata=common_attn_metadata)


def _make_dsa_metadata_groups():
    layer_names = [
        "model.layers.0.self_attn.compressor",
        "model.layers.0.self_attn.indexer",
    ]
    specs = [
        AscendMLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            compress_ratio=compress_ratio,
        )
        for block_size, compress_ratio in ((32, 2), (64, 4))
    ]
    calls: list[dict[str, Any]] = []
    attn_groups = [
        [
            AttentionGroup(
                backend=AscendDSABackend,
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


@pytest.mark.parametrize(
    ("caller", "cudagraph_mode", "expected_input_tokens"),
    [
        ("default", None, 5),
        ("model_state", CUDAGraphMode.NONE, 5),
        ("model_state", CUDAGraphMode.FULL, 8),
    ],
)
def test_mrv2_builds_shared_dsa_metadata_for_each_execution_mode(
    caller,
    cudagraph_mode,
    expected_input_tokens,
):
    layer_names, specs, calls, attn_groups, kv_cache_config = _make_dsa_metadata_groups()
    block_tables = (
        torch.zeros((4, 1), dtype=torch.int32),
        torch.zeros((4, 1), dtype=torch.int32),
    )
    slot_mappings = torch.zeros((2, 8), dtype=torch.int32)

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
        )
    else:
        model_state = AscendModelState.__new__(AscendModelState)
        model_state.max_model_len = 8
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
            dcp_local_seq_lens=None,
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
        )

    assert set(metadata) == set(layer_names)
    assert [call["block_size"] for call in calls] == [spec.block_size for spec in specs]
    assert len(calls) == 2
    for call in calls:
        common_metadata = call["common_attn_metadata"]
        assert common_metadata.num_actual_tokens == 5
        assert common_metadata.num_input_tokens == expected_input_tokens
        if caller != "default":
            assert torch.equal(
                common_metadata.is_prefilling,
                torch.tensor([True, True, False, False]),
            )
    cache_name = "common_ratio_to_sas_metadata"
    assert calls[0][cache_name] is calls[1][cache_name]
    assert calls[1][cache_name]["first_group"] is True
