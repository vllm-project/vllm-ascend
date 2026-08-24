from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.models.minimax_m3.minimax_m3 import _uses_aux_hidden_states
from vllm_ascend.worker.model_runner_v1 import (
    _get_dspark_aux_hidden_state_layers,
    _get_sfa_indexer_kernel_num_blocks,
    _get_unpacked_kv_cache_tensor_size,
    _normalize_sfa_indexer_cache_spec,
)


@pytest.mark.parametrize(
    ("method", "uses_dspark", "expected"),
    [
        ("eagle3", False, True),
        ("dspark", True, True),
        ("draft_model", False, False),
    ],
)
def test_minimax_m3_aux_hidden_states_are_enabled_for_dspark(
    method: str,
    uses_dspark: bool,
    expected: bool,
) -> None:
    speculative_config = SimpleNamespace(
        method=method,
        use_dspark=lambda: uses_dspark,
    )
    vllm_config = SimpleNamespace(speculative_config=speculative_config)

    assert _uses_aux_hidden_states(vllm_config) is expected


def test_minimax_m3_aux_hidden_states_are_disabled_without_speculation() -> None:
    vllm_config = SimpleNamespace(speculative_config=None)

    assert not _uses_aux_hidden_states(vllm_config)


def test_dspark_layers_are_read_from_nested_dflash_config() -> None:
    hf_config = SimpleNamespace(
        dflash_config={
            "target_layer_ids": [1, 12, 23, 35, 46, 57],
        }
    )

    assert _get_dspark_aux_hidden_state_layers(hf_config) == (
        2,
        13,
        24,
        36,
        47,
        58,
    )


@pytest.mark.parametrize(
    "hf_config",
    [
        SimpleNamespace(dspark_target_layer_ids=[1, 3]),
        SimpleNamespace(target_layer_ids=[1, 3]),
    ],
)
def test_dspark_layers_support_existing_top_level_configs(
    hf_config: SimpleNamespace,
) -> None:
    assert _get_dspark_aux_hidden_state_layers(hf_config) == (2, 4)


def test_stale_minimax_indexer_cache_spec_is_rebuilt() -> None:
    stale_spec_type = type("AscendSFAIndexerCacheSpec", (SimpleNamespace,), {})
    stale_spec = stale_spec_type(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        cache_dtype_str=None,
        scale_dim=0,
        scale_dtype=torch.int8,
        cache_sparse_li_c8=False,
        sfa_dcp_replicated_indexer_size=1,
    )

    normalized = _normalize_sfa_indexer_cache_spec(stale_spec)

    assert isinstance(normalized, AscendSFAIndexerCacheSpec)
    assert normalized.block_size == 128
    assert normalized.head_size == 128


def test_packed_kv_tensor_is_unpacked_to_its_layer_slice() -> None:
    kv_tensor = SimpleNamespace(
        size=4096,
        block_stride=1024,
        offset=256,
        shared_by=["index_cache"],
    )
    specs = {"index_cache": SimpleNamespace(page_size_bytes=128)}

    assert _get_unpacked_kv_cache_tensor_size(kv_tensor, specs) == 512


def test_packed_kv_tensor_rejects_slice_outside_block_stride() -> None:
    kv_tensor = SimpleNamespace(
        size=4096,
        block_stride=1024,
        offset=960,
        shared_by=["index_cache"],
    )
    specs = {"index_cache": SimpleNamespace(page_size_bytes=128)}

    with pytest.raises(ValueError, match="slice exceeds block stride"):
        _get_unpacked_kv_cache_tensor_size(kv_tensor, specs)


def test_sfa_indexer_expands_physical_blocks_for_kernel_block_table() -> None:
    assert _get_sfa_indexer_kernel_num_blocks(
        num_physical_blocks=2216,
        physical_block_size=2048,
        kernel_block_size=128,
        replicated_size=1,
    ) == 35456
