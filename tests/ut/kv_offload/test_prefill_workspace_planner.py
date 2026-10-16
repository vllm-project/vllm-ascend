# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.attention.offload_capability import (
    DsaOffloadAttentionBackendCapability,
    DsaOffloadCacheMemory,
)
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.patch.platform import patch_kv_cache_utils
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _ascend_get_kv_cache_config_from_groups,
    _ascend_get_kv_cache_configs,
    _ascend_max_memory_usage_bytes_from_groups,
    _get_group_layer_kv_cache_specs,
    _get_layerwise_decode_kv_cache_config,
    _get_layerwise_prefill_max_memory_usage_bytes,
    _get_layerwise_prefill_kv_cache_config,
    is_layerwise_host_offload_decode,
    is_layerwise_host_offload_prefill,
)


class _TestDsaOffloadBackend(DsaOffloadAttentionBackendCapability):
    @classmethod
    def get_dsa_offload_cache_memory(cls, *, layer_name, kv_cache_spec):
        del layer_name
        # Match the non-C8 SFA tuple used by the merged MLA fixture:
        # latent KV + RoPE on host-visible memory and indexer K on device.
        sparse_dims = kv_cache_spec.sparse_head_dim
        assert sparse_dims is not None
        dtype_size = torch.tensor([], dtype=kv_cache_spec.dtype).element_size()
        main_bytes_per_token = (sparse_dims[0] + sparse_dims[1]) * dtype_size
        return DsaOffloadCacheMemory(
            main_bytes_per_token=main_bytes_per_token,
            device_bytes_per_block=sparse_dims[2] * dtype_size * kv_cache_spec.block_size,
        )


@pytest.fixture(autouse=True)
def _patch_dsa_offload_backend_resolution(monkeypatch):
    """Planner tests use synthetic configs without instantiated attention layers."""

    monkeypatch.setattr(
        patch_kv_cache_utils,
        "get_dsa_offload_backends",
        lambda _config, layer_names: {
            layer_name: _TestDsaOffloadBackend for layer_name in layer_names
        },
    )


def _config(*, enabled: bool = True, role: str = "kv_producer"):
    return SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None, block_size=128),
        kv_transfer_config=SimpleNamespace(
            kv_role=role,
            kv_connector_extra_config={
                "layerwise_host_kv_offload": enabled,
            },
        ),
        model_config=SimpleNamespace(
            max_model_len=1024,
            hf_text_config=SimpleNamespace(num_hidden_layers=2, index_topk=2048),
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )


def _group(*, second_head_size: int = 16):
    specs = {
        f"model.layers.{layer_id}.self_attn.attn": FullAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=second_head_size if layer_id == 1 else 16,
            dtype=torch.float16,
        )
        for layer_id in range(9)
    }
    uniform = UniformTypeKVCacheSpecs.from_specs(specs)
    assert uniform is not None
    return KVCacheGroupSpec(list(specs), uniform)


def test_layerwise_offload_requires_pd_separated_role() -> None:
    config = _config(role="kv_both")
    assert not is_layerwise_host_offload_prefill(config)
    assert not is_layerwise_host_offload_decode(config)

    config = _config(role="kv_producer")
    assert is_layerwise_host_offload_prefill(config)
    assert not is_layerwise_host_offload_decode(config)

    config = _config(role="kv_consumer")
    assert not is_layerwise_host_offload_prefill(config)
    assert is_layerwise_host_offload_decode(config)


def test_prefill_planner_shares_four_base_bundles_and_counts_mtp() -> None:
    config = _config()
    config.model_config.hf_text_config.num_hidden_layers = 8
    group = _group()
    page_size = group.kv_cache_spec.kv_cache_specs[
        "model.layers.0.self_attn.attn"
    ].page_size_bytes
    available_memory = page_size * 5 * 17

    result = _get_layerwise_prefill_kv_cache_config(
        config,
        [group],
        available_memory,
    )

    assert result.num_blocks == 17
    assert len(result.kv_cache_tensors) == 5
    for arena_id in range(4):
        assert result.kv_cache_tensors[arena_id].layers == [
            f"model.layers.{layer_id}.self_attn.attn"
            for layer_id in range(arena_id, 8, 4)
        ]
    assert result.kv_cache_tensors[4].layers == [
        "model.layers.8.self_attn.attn"
    ]
    assert sum(item.size for item in result.kv_cache_tensors) == available_memory


def test_prefill_memory_check_counts_four_base_bundles_and_mtp() -> None:
    config = _config()
    config.model_config.hf_text_config.num_hidden_layers = 8
    config.model_config.max_model_len = 1_024_000
    config.parallel_config = SimpleNamespace(
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    group = _group()
    page_size = group.kv_cache_spec.kv_cache_specs[
        "model.layers.0.self_attn.attn"
    ].page_size_bytes

    result = _get_layerwise_prefill_max_memory_usage_bytes(config, [group])

    assert result == 8000 * page_size * 5


def test_prefill_planner_reuses_arena_after_four_layers() -> None:
    config = _config()
    config.model_config.hf_text_config.num_hidden_layers = 12
    specs = {
        f"model.layers.{layer_id}.self_attn.attn": FullAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=16,
            dtype=torch.float16,
        )
        for layer_id in range(13)
    }
    uniform = UniformTypeKVCacheSpecs.from_specs(specs)
    assert uniform is not None
    group = KVCacheGroupSpec(list(specs), uniform)
    page_size = next(iter(specs.values())).page_size_bytes

    result = _get_layerwise_prefill_kv_cache_config(
        config,
        [group],
        page_size * 5 * 7,
    )

    assert result.num_blocks == 7
    assert result.kv_cache_tensors[0].layers == [
        "model.layers.0.self_attn.attn",
        "model.layers.4.self_attn.attn",
        "model.layers.8.self_attn.attn",
    ]
    assert result.kv_cache_tensors[1].layers == [
        "model.layers.1.self_attn.attn",
        "model.layers.5.self_attn.attn",
        "model.layers.9.self_attn.attn",
    ]
    assert result.kv_cache_tensors[2].layers == [
        "model.layers.2.self_attn.attn",
        "model.layers.6.self_attn.attn",
        "model.layers.10.self_attn.attn",
    ]
    assert result.kv_cache_tensors[3].layers == [
        "model.layers.3.self_attn.attn",
        "model.layers.7.self_attn.attn",
        "model.layers.11.self_attn.attn",
    ]
    assert result.kv_cache_tensors[4].layers == [
        "model.layers.12.self_attn.attn",
    ]


def test_prefill_planner_rejects_incompatible_workspace_layers() -> None:
    config = _config()
    config.model_config.hf_text_config.num_hidden_layers = 8
    with pytest.raises(NotImplementedError, match="compatible base-layer"):
        _get_layerwise_prefill_kv_cache_config(
            config,
            [_group(second_head_size=32)],
            1 << 30,
        )


def test_feature_gate_delegates_when_disabled(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_kv_cache_utils."
        "_orig_get_kv_cache_config_from_groups",
        lambda *_args: sentinel,
    )
    assert (
        _ascend_get_kv_cache_config_from_groups(
            _config(enabled=False),
            [_group()],
            1 << 30,
        )
        is sentinel
    )


def _merged_mla_group(
    num_layers: int = 2,
) -> tuple[KVCacheGroupSpec, AscendMLAAttentionSpec]:
    spec = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=28,
        sparse_head_dim=(16, 8, 4),
        dtype=torch.bfloat16,
        cache_sparse_c8=False,
        c8_k_cache_dtype=torch.int8,
        c8_k_scale_cache_dtype=torch.float16,
    )
    return KVCacheGroupSpec(
        [
            f"model.layers.{layer_id}.self_attn.attn"
            for layer_id in range(num_layers)
        ],
        spec,
    ), spec


def test_merged_mla_group_is_expanded_per_layer() -> None:
    group, merged_spec = _merged_mla_group()
    specs = _get_group_layer_kv_cache_specs(group)

    assert specs == {
        "model.layers.0.self_attn.attn": merged_spec,
        "model.layers.1.self_attn.attn": merged_spec,
    }


def test_prefill_planner_accepts_merged_ascend_mla_spec() -> None:
    config = _config()
    config.model_config.hf_text_config.num_hidden_layers = 8
    group, spec = _merged_mla_group(num_layers=8)
    result = _get_layerwise_prefill_kv_cache_config(
        config,
        [group],
        spec.page_size_bytes * 4 * 3,
    )

    assert result.num_blocks == 3
    assert len(result.kv_cache_tensors) == 4
    for arena_id in range(4):
        assert result.kv_cache_tensors[arena_id].layers == [
            group.layer_names[layer_id]
            for layer_id in range(arena_id, len(group.layer_names), 4)
        ]


@pytest.mark.parametrize("num_speculative_tokens", [None, 3])
def test_decode_planner_accepts_merged_ascend_mla_spec(
    monkeypatch,
    num_speculative_tokens,
) -> None:
    config = _config(enabled=True, role="kv_consumer")
    config.speculative_config = (
        None
        if num_speculative_tokens is None
        else SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
    )
    config.model_config.max_model_len = 1024
    config.scheduler_config = SimpleNamespace(max_num_seqs=1)
    group, spec = _merged_mla_group()

    def fake_base(_config, groups, _available_memory):
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=spec.page_size_bytes,
                    layers=groups[0].layer_names,
                    offset=64,
                    block_stride=spec.page_size_bytes + 64,
                )
            ],
            kv_cache_groups=groups,
        )

    monkeypatch.setattr(
        patch_kv_cache_utils,
        "_orig_get_kv_cache_config_from_groups",
        fake_base,
    )
    result = _get_layerwise_decode_kv_cache_config(
        config,
        [group],
        1 << 30,
    )

    assert result.num_blocks > 0
    assert result.kv_cache_tensors[0].layers == group.layer_names
    assert result.kv_cache_tensors[0].offset == 64
    assert result.kv_cache_tensors[0].block_stride == spec.page_size_bytes + 64


def test_decode_planner_accepts_zero_memory_for_minimal_profile(monkeypatch) -> None:
    config = _config(enabled=True, role="kv_consumer")
    config.cache_config.num_gpu_blocks_override = 1
    config.model_config.max_model_len = 1024
    config.scheduler_config = SimpleNamespace(max_num_seqs=1)
    group, spec = _merged_mla_group()

    def fake_base(_config, groups, available_memory):
        assert available_memory >= spec.page_size_bytes
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[
                KVCacheTensor(size=spec.page_size_bytes, layers=groups[0].layer_names)
            ],
            kv_cache_groups=groups,
        )

    monkeypatch.setattr(
        patch_kv_cache_utils,
        "_orig_get_kv_cache_config_from_groups",
        fake_base,
    )
    result = _get_layerwise_decode_kv_cache_config(
        config,
        [group],
        0,
    )

    assert result.num_blocks == 1
    assert result.kv_cache_tensors[0].layers == group.layer_names


def test_decode_planner_rejects_zero_memory_without_profile_override() -> None:
    config = _config(enabled=True, role="kv_consumer")
    config.model_config.max_model_len = 1024
    config.scheduler_config = SimpleNamespace(max_num_seqs=1)
    group, _ = _merged_mla_group()

    with pytest.raises(MemoryError, match="cannot reserve fixed DSA workspace"):
        _get_layerwise_decode_kv_cache_config(config, [group], 0)


def test_decode_admission_counts_indexer_and_dsa_workspace_not_full_main() -> None:
    config = _config(enabled=True, role="kv_consumer")
    config.model_config.max_model_len = 1_024_000
    group, spec = _merged_mla_group()

    result = _ascend_max_memory_usage_bytes_from_groups(config, [group])
    full_kv_bytes = len(group.layer_names) * spec.max_memory_usage_bytes(config)

    assert 0 < result < full_kv_bytes


def _profiled_decode_config(
    group: KVCacheGroupSpec,
    spec: AscendMLAAttentionSpec,
    num_blocks: int,
) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=spec.page_size_bytes * num_blocks,
                layers=group.layer_names,
            )
        ],
        kv_cache_groups=[group],
    )


def test_decode_override_uses_real_profiled_capacity_and_shrinks_tensors(
    monkeypatch,
) -> None:
    config = _config(enabled=True, role="kv_consumer")
    config.cache_config.num_gpu_blocks_override = 8
    group, spec = _merged_mla_group()
    profiled_memory = [987_654_321]

    def fake_get_configs(vllm_config, kv_cache_specs, available_memory):
        assert vllm_config.cache_config.num_gpu_blocks_override is None
        assert kv_cache_specs == [{group.layer_names[0]: spec}]
        assert available_memory is profiled_memory
        return [_profiled_decode_config(group, spec, 10)]

    monkeypatch.setattr(
        patch_kv_cache_utils,
        "_orig_get_kv_cache_configs",
        fake_get_configs,
    )

    result = _ascend_get_kv_cache_configs(
        config,
        [{group.layer_names[0]: spec}],
        profiled_memory,
    )

    assert config.cache_config.num_gpu_blocks_override == 8
    assert result[0].num_blocks == 8
    assert result[0].kv_cache_tensors[0].size == spec.page_size_bytes * 8


def test_decode_override_rejects_capacity_above_real_profiled_hbm(
    monkeypatch,
) -> None:
    config = _config(enabled=True, role="kv_consumer")
    config.cache_config.num_gpu_blocks_override = 11
    group, spec = _merged_mla_group()
    monkeypatch.setattr(
        patch_kv_cache_utils,
        "_orig_get_kv_cache_configs",
        lambda *_args: [_profiled_decode_config(group, spec, 10)],
    )

    with pytest.raises(MemoryError, match="real Decode Indexer/MTP HBM budget"):
        _ascend_get_kv_cache_configs(config, [{}], [1])

    assert config.cache_config.num_gpu_blocks_override == 11


def test_decode_override_rejects_less_than_one_max_length_request(
    monkeypatch,
) -> None:
    config = _config(enabled=True, role="kv_consumer")
    config.cache_config.num_gpu_blocks_override = 7
    group, spec = _merged_mla_group()
    monkeypatch.setattr(
        patch_kv_cache_utils,
        "_orig_get_kv_cache_configs",
        lambda *_args: [_profiled_decode_config(group, spec, 10)],
    )

    with pytest.raises(ValueError, match="cannot serve max_model_len"):
        _ascend_get_kv_cache_configs(config, [{}], [1])

    assert config.cache_config.num_gpu_blocks_override == 7


def test_group_spec_helper_rejects_unsupported_spec() -> None:
    group = KVCacheGroupSpec(
        ["layer"],
        FullAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=16,
            dtype=torch.float16,
        ),
    )
    with pytest.raises(NotImplementedError, match="merged MLAAttentionSpec"):
        _get_group_layer_kv_cache_specs(group)
