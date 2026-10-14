from types import SimpleNamespace
from unittest.mock import Mock

import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
)

from vllm_ascend.patch.platform import patch_kv_cache_utils


def _qsa_specs() -> dict[str, KVCacheSpec]:
    main = FullAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.bfloat16,
    )
    raw = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=32,
        head_size_v=0,
        dtype=torch.bfloat16,
    )
    compressed = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=32,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    specs: dict[str, KVCacheSpec] = {}
    for index in range(2):
        source = f"model.layers.{index * 4 + 3}.self_attn"
        specs[f"{source}.indexer.compressed_key_cache"] = compressed
        specs[f"{source}.indexer.raw_key_cache"] = raw
        specs[f"{source}.attn"] = main
    return specs


def test_qsa_grouping_delegates_to_upstream_vllm(monkeypatch) -> None:
    specs = _qsa_specs()
    vllm_config = SimpleNamespace()
    expected = [Mock(spec=KVCacheGroupSpec)]
    upstream_grouping = Mock(return_value=expected)
    monkeypatch.setattr(
        patch_kv_cache_utils,
        "_orig_get_kv_cache_groups",
        upstream_grouping,
    )

    actual = patch_kv_cache_utils._get_ascend_kv_cache_groups(
        vllm_config,
        specs,
    )

    assert actual is expected
    upstream_grouping.assert_called_once_with(vllm_config, specs)


def test_qsa_config_uses_upstream_vllm_planner(monkeypatch) -> None:
    groups = [Mock(spec=KVCacheGroupSpec)]
    expected = SimpleNamespace()
    upstream_planner = Mock(return_value=expected)
    monkeypatch.setattr(
        patch_kv_cache_utils,
        "_orig_get_kv_cache_config_from_groups",
        upstream_planner,
    )
    monkeypatch.setattr(
        patch_kv_cache_utils,
        "_is_deepseek_v4_groups",
        lambda _: False,
    )
    monkeypatch.setattr(
        patch_kv_cache_utils,
        "_get_glm5_next_cache_layout",
        lambda _: None,
    )
    vllm_config = SimpleNamespace(kv_transfer_config=None)

    actual = patch_kv_cache_utils._ascend_get_kv_cache_config_from_groups(
        vllm_config,
        groups,
        available_memory=123456,
    )

    assert actual is expected
    upstream_planner.assert_called_once_with(vllm_config, groups, 123456)
    assert actual.kv_transfer_config is None


def test_qsa_gbsa_aligns_shared_block_stride(monkeypatch) -> None:
    page_row_bytes = 1 * 256 * torch.empty((), dtype=torch.bfloat16).element_size()
    qsa_main = FullAttentionSpec(
        block_size=1536,
        num_kv_heads=1,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
        num_head_slots=2,
        state_content_bytes=page_row_bytes,
    )
    group = KVCacheGroupSpec(
        layer_names=["model.layers.3.self_attn.attn"],
        kv_cache_spec=qsa_main,
    )
    old_stride = 21_725_184
    old_num_blocks = 1060
    config = SimpleNamespace(
        num_blocks=old_num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=old_stride * old_num_blocks,
                layers=group.layer_names,
                layer_stride=qsa_main.page_size_bytes,
                block_stride=old_stride,
            )
        ],
    )
    layout = SimpleNamespace(is_block_outermost=True)
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: layout,
        )
    )
    monkeypatch.setattr(
        patch_kv_cache_utils,
        "may_override_num_blocks",
        lambda _config, num_blocks: num_blocks,
    )

    aligned = patch_kv_cache_utils._align_qsa_gbsa_block_stride(
        config,
        vllm_config,
        [group],
        available_memory=old_stride * old_num_blocks,
    )

    page_bytes = 128 * page_row_bytes
    expected_stride = 21_757_952
    assert aligned.kv_cache_tensors[0].block_stride == expected_stride
    assert expected_stride % page_bytes == 0
    assert aligned.kv_cache_tensors[0].size == (
        expected_stride * aligned.num_blocks
    )
    assert aligned.num_blocks == (old_stride * old_num_blocks) // expected_stride


def test_qsa_gbsa_alignment_recognizes_cache_role_pair() -> None:
    owner = "model.layers.3.self_attn"
    main = FullAttentionSpec(
        block_size=1536,
        num_kv_heads=1,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
    )
    compressed = MLAAttentionSpec(
        block_size=384,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    groups = [
        KVCacheGroupSpec(
            layer_names=[f"{owner}.attn"],
            kv_cache_spec=main,
        ),
        KVCacheGroupSpec(
            layer_names=[f"{owner}.indexer.compressed_key_cache"],
            kv_cache_spec=compressed,
        ),
    ]

    assert patch_kv_cache_utils._qsa_gbsa_page_alignment(groups) == 65_536


def test_non_qsa_cache_keeps_upstream_block_stride(monkeypatch) -> None:
    ordinary = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
    )
    group = KVCacheGroupSpec(
        layer_names=["model.layers.0.self_attn"],
        kv_cache_spec=ordinary,
    )
    tensor = KVCacheTensor(
        size=1000,
        layers=group.layer_names,
        layer_stride=100,
        block_stride=100,
    )
    config = SimpleNamespace(num_blocks=10, kv_cache_tensors=[tensor])
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: SimpleNamespace(
                is_block_outermost=True
            )
        )
    )

    actual = patch_kv_cache_utils._align_qsa_gbsa_block_stride(
        config,
        vllm_config,
        [group],
        available_memory=1000,
    )

    assert actual is config
    assert actual.num_blocks == 10
    assert actual.kv_cache_tensors == [tensor]
