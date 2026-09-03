# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.indexer import (
    AscendSFAIndexerBackend,
    AscendSFAIndexerMetadata,
    AscendSFAIndexerMetadataBuilder,
)

_KERNEL_BLOCK_SIZE = 128


def _make_builder(pcp_size: int = 1) -> AscendSFAIndexerMetadataBuilder:
    kv_cache_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=160,
        dtype=torch.uint8,
    )
    layer_names = ["model.layers.0.self_attn.indexer.k_cache"]
    vllm_config = MagicMock()
    vllm_config.parallel_config.prefill_context_parallel_size = pcp_size
    with patch(
        "vllm_ascend.attention.indexer.select_common_block_size",
        return_value=_KERNEL_BLOCK_SIZE,
    ):
        return AscendSFAIndexerMetadataBuilder(
            kv_cache_spec,
            layer_names,
            vllm_config,
            torch.device("cpu"),
        )


def _make_common_metadata() -> SimpleNamespace:
    return SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=4,
        num_input_tokens=4,
        slot_mapping=torch.tensor([1, 2, 3, 4, 5]),
        positions=torch.tensor([0, 1, 0, 1, 9]),
        query_start_loc=torch.tensor([0, 2, 4]),
        seq_lens=torch.tensor([5, 6, 7]),
        block_table_tensor=torch.arange(6).view(3, 2),
        group_len=MagicMock(name="group_len"),
        group_key_idx=MagicMock(name="group_key_idx"),
        group_key_cache_idx=MagicMock(name="group_key_cache_idx"),
    )


def test_sfa_indexer_backend_contract():
    assert AscendSFAIndexerBackend.accept_output_buffer
    assert AscendSFAIndexerBackend.get_name() == "ASCEND_SFA_INDEXER"
    assert AscendSFAIndexerBackend.get_builder_cls() is AscendSFAIndexerMetadataBuilder
    assert AscendSFAIndexerBackend.get_kv_cache_shape(8, 128, 1, 160) == (
        8,
        128,
        1,
        160,
    )
    assert AscendSFAIndexerBackend.get_supported_kernel_block_sizes() == [128]


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_builds_kernel_metadata(mock_cos_sin, mock_get_ascend_config):
    mock_get_ascend_config.return_value.c8_enable_reshape_optim = False
    cos = torch.zeros(5, 1, 1, 8)
    sin = torch.zeros(5, 1, 1, 8)
    mock_cos_sin.return_value = (cos, sin)

    builder = _make_builder()
    assert builder.reorder_batch_threshold is None
    assert builder.get_cudagraph_support(MagicMock(), MagicMock()) is AttentionCGSupport.UNIFORM_BATCH

    common = _make_common_metadata()
    metadata = builder.build(0, common)

    assert isinstance(metadata, AscendSFAIndexerMetadata)
    assert metadata.num_actual_tokens == 4
    assert torch.equal(metadata.slot_mapping, common.slot_mapping[:4])
    assert torch.equal(metadata.seq_lens, common.seq_lens[:2])
    assert torch.equal(metadata.cum_query_lens, common.query_start_loc[1:3])
    assert torch.equal(metadata.block_table, common.block_table_tensor[:2])
    assert metadata.block_size == _KERNEL_BLOCK_SIZE
    assert metadata.group_len is common.group_len
    assert metadata.group_key_idx is common.group_key_idx
    assert metadata.group_key_cache_idx is common.group_key_cache_idx

    positions = mock_cos_sin.call_args.args[0]
    assert torch.equal(positions, common.positions[:4])
    assert mock_cos_sin.call_args.kwargs["use_cache"] is True
    assert torch.equal(metadata.cos, cos[:4])
    assert torch.equal(metadata.sin, sin[:4])


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_emits_full_slot_mapping_under_pcp(mock_cos_sin, mock_get_ascend_config):
    mock_get_ascend_config.return_value.c8_enable_reshape_optim = False
    mock_cos_sin.return_value = (torch.zeros(5, 1, 1, 8), torch.zeros(5, 1, 1, 8))

    builder = _make_builder(pcp_size=2)
    common = _make_common_metadata()
    metadata = builder.build(0, common)

    # Under PCP the commit writes the gathered prefill region too, so the
    # builder emits the full slot mapping instead of the input-token slice.
    assert metadata.slot_mapping is common.slot_mapping


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
@patch("vllm_ascend.attention.indexer.torch.ops._C_ascend.store_kv_block_metadata", create=True)
def test_sfa_indexer_metadata_builder_primes_reshape_optim(
    mock_store_kv_block_metadata,
    mock_cos_sin,
    mock_get_ascend_config,
):
    mock_get_ascend_config.return_value.c8_enable_reshape_optim = True
    mock_cos_sin.return_value = (torch.zeros(5, 1, 1, 8), torch.zeros(5, 1, 1, 8))

    builder = _make_builder()
    common = _make_common_metadata()
    metadata = builder.build(0, common)

    mock_store_kv_block_metadata.assert_called_once_with(
        metadata.slot_mapping,
        common.group_len,
        common.group_key_idx,
        common.group_key_cache_idx,
        _KERNEL_BLOCK_SIZE,
    )
