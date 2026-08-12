# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.context_parallel.common_cp import DCPMetadataBuilderMixin
from vllm_ascend.attention.context_parallel.sfa_cp import (
    AscendSFADCPImpl,
    AscendSFADCPMetadata,
    AscendSFADCPMetadataBuilder,
    DCPGatherContext,
)
from vllm_ascend.attention.sfa_v1 import (
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
)


def test_sfa_dcp_extends_v1_backend() -> None:
    assert issubclass(AscendSFADCPImpl, AscendSFAImpl)
    assert issubclass(
        AscendSFADCPMetadataBuilder,
        AscendSFAMetadataBuilder,
    )
    assert "dcp_context" not in {field.name for field in fields(AscendSFAMetadata)}
    assert "dcp_context" in {field.name for field in fields(AscendSFADCPMetadata)}


@pytest.mark.parametrize(
    ("max_model_len", "expected_replicated_cols"),
    [
        (1024, 16),  # cdiv(max_model_len, block_size) is divisible by dcp_size
        (1055, 20),  # padded up to the next multiple of dcp_size: 9 -> 10
    ],
)
def test_sfa_dcp_builder_sizes_replicated_view_from_padded_block_table(
    max_model_len: int,
    expected_replicated_cols: int,
) -> None:
    def fake_base_init(self, *args, **kwargs) -> None:
        self.dcp_size = 2
        self.kernel_block_size = 128

    kv_cache_spec = SimpleNamespace(block_size=128)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        scheduler_config=SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=1024,
        ),
        model_config=SimpleNamespace(max_model_len=max_model_len),
    )

    with patch.object(DCPMetadataBuilderMixin, "__init__", new=fake_base_init):
        builder = AscendSFADCPMetadataBuilder(
            kv_cache_spec,
            [],
            vllm_config,
            torch.device("cpu"),
        )

    assert builder.block_table_replicated_view_buf.shape == (5, expected_replicated_cols)
    assert builder.arange_buffer.shape == (expected_replicated_cols,)


def _make_builder(rank: int = 0) -> AscendSFADCPMetadataBuilder:
    builder = AscendSFADCPMetadataBuilder.__new__(AscendSFADCPMetadataBuilder)
    builder.dcp_size = 2
    builder.dcp_rank = rank
    builder.cp_kv_cache_interleave_size = 4
    builder.blocks_per_phys_block = 1
    builder.replicated_view_block_size = 4
    builder.device = torch.device("cpu")
    builder.block_table_replicated_view_buf = torch.empty(
        (4, 8),
        dtype=torch.int32,
    )
    builder.arange_buffer = torch.arange(8, dtype=torch.int32)
    builder.slot_mapping_replicated_view_buf = torch.empty(32, dtype=torch.int32)
    return builder


def test_sfa_dcp_local_sequence_lengths_follow_interleave_layout() -> None:
    seq_lens = torch.tensor([0, 3, 4, 5, 8, 9, 12], dtype=torch.int32)

    rank0 = _make_builder(rank=0)._get_dcp_local_seq_lens(seq_lens)
    rank1 = _make_builder(rank=1)._get_dcp_local_seq_lens(seq_lens)

    torch.testing.assert_close(rank0, torch.tensor([0, 3, 4, 4, 4, 5, 8], dtype=torch.int32))
    torch.testing.assert_close(rank1, torch.tensor([0, 0, 0, 1, 4, 4, 4], dtype=torch.int32))


def test_sfa_dcp_builds_replicated_block_table_view() -> None:
    builder = _make_builder()
    local_block_table = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    seq_lens = torch.tensor([16], dtype=torch.int32)

    replicated = builder._build_block_table_replicated_view(
        local_block_table,
        seq_lens,
    )

    torch.testing.assert_close(
        replicated,
        torch.tensor([[20, 21, 22, 23, 24, 25, 26, 27]], dtype=torch.int32),
    )


def _make_fp8_metadata(num_blocks: int = 4) -> AscendSFADCPMetadata:
    metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
    metadata.num_prefills = 1
    metadata.dcp_context = SimpleNamespace(
        kv_gather_block_ids=torch.arange(num_blocks, dtype=torch.int32),
        kv_gather_block_table=torch.arange(num_blocks, dtype=torch.int32).view(1, -1),
        gather_context=None,
    )
    return metadata


def test_sfa_dcp_fp8_kv_gather_stays_uint8_through_collective() -> None:
    impl = AscendSFADCPImpl.__new__(AscendSFADCPImpl)
    impl.enable_sparse_sfa_c8 = True
    impl.dcp_group = object()
    kv_cache = (torch.zeros(16, 1, 1, 8, dtype=torch.float8_e4m3fn),)
    metadata = _make_fp8_metadata()
    captured: dict[str, Any] = {}

    def fake_start(self, x, dim, split_sizes):
        captured["x"] = x
        captured["split_sizes"] = split_sizes
        return SimpleNamespace()

    with patch.object(
        AscendSFADCPImpl,
        "_start_dcp_gather",
        autospec=True,
        side_effect=fake_start,
    ):
        impl._record_dcp_kv_gather_context(kv_cache, metadata)

    assert captured["x"].dtype == torch.uint8
    assert captured["split_sizes"] == (8,)


def test_sfa_dcp_prefill_restores_fp8_after_kv_gather() -> None:
    impl = AscendSFADCPImpl.__new__(AscendSFADCPImpl)
    impl.enable_sparse_sfa_c8 = True
    impl.dcp_group = object()
    metadata = _make_fp8_metadata(num_blocks=2)
    metadata.dcp_context.gather_context = DCPGatherContext(
        gathered=torch.zeros(4, 1, 1, 8, dtype=torch.uint8),
        handle=None,
        restore_perm=None,
        split_sizes=(8,),
    )
    kv_cache = (torch.zeros(16, 1, 1, 8, dtype=torch.float8_e4m3fn),)
    received: dict[str, Any] = {}

    def fake_super_exec(self, ql_nope, q_pe, kv_cache, topk_indices, attn_metadata, *args, **kwargs):
        received["kv_cache"] = kv_cache
        return torch.zeros(1, dtype=torch.bfloat16)

    with patch.object(
        AscendSFAImpl,
        "_execute_sparse_flash_attention_process",
        autospec=True,
        side_effect=fake_super_exec,
    ):
        impl._execute_sparse_flash_attention_process(
            torch.zeros(2, 1, 4, dtype=torch.bfloat16),
            torch.zeros(2, 1, 2, dtype=torch.bfloat16),
            kv_cache,
            torch.zeros(2, 1, 8, dtype=torch.int32),
            metadata,
            torch.zeros(3, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
        )

    assert received["kv_cache"][0].dtype == torch.float8_e4m3fn


def test_sfa_dcp_updates_dsa_cp_local_slot_mapping_with_padding() -> None:
    builder = _make_builder()
    dsa_cp_context = SimpleNamespace(
        num_tokens_pad=6,
        local_start=2,
        local_end_with_pad=5,
        slot_mapping_cp=None,
    )
    metadata = SimpleNamespace(dsa_cp_context=dsa_cp_context)

    builder._update_dsa_cp_slot_mapping_for_dcp(
        metadata,
        dcp_slot_mapping=torch.tensor([10, 11, 12, 13], dtype=torch.int32),
        num_input_tokens=4,
    )

    torch.testing.assert_close(
        dsa_cp_context.slot_mapping_cp,
        torch.tensor([12, 13, -1], dtype=torch.int32),
    )
