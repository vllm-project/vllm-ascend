# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.context_parallel.common_cp import DCPMetadataBuilderMixin
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPMetadataBuilder
from vllm_ascend.attention.context_parallel.sfa_dcp_utils import (
    build_sfa_dcp_replicated_block_table,
    get_sfa_dcp_local_block_table,
)
from vllm_ascend.attention.indexer import AscendSFAIndexerMetadataBuilder
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.platform import _validate_sfa_dcp_kv_sp
from vllm_ascend.utils import enable_sfa_dcp_sharded_indexer
from vllm_ascend.worker.v2.block_table import AscendBlockTables


def _config(max_len, interleave):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=8,
            prefill_context_parallel_size=1,
            tensor_parallel_size=8,
            cp_kv_cache_interleave_size=interleave,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=30, max_num_batched_tokens=120),
        model_config=SimpleNamespace(max_model_len=max_len, hf_text_config=SimpleNamespace()),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
    )


def _create_builder(spec, config, indexer):
    # Exercise the real upstream factory, including its kernel-size spec copy.
    cls = AscendSFAIndexerMetadataBuilder if indexer else AscendSFADCPMetadataBuilder
    backend = SimpleNamespace(get_builder_cls=lambda: cls, get_supported_kernel_block_sizes=lambda _spec: [128])
    group = AttentionGroup(backend=backend, layer_names=["layer"], kv_cache_spec=spec, kv_cache_group_id=0)

    def init_base(self, *args, **kwargs):
        self.dcp_size = 8
        self.kernel_block_size = 128

    with (
        patch.object(DCPMetadataBuilderMixin, "__init__", init_base),
        patch("vllm_ascend.attention.context_parallel.sfa_cp.is_dcp_enabled", return_value=True),
        patch(
            "vllm_ascend.attention.context_parallel.sfa_cp.get_dcp_group",
            return_value=SimpleNamespace(ranks=list(range(8))),
        ),
        patch("vllm_ascend.attention.indexer.enable_sfa_dcp_replicated_indexer", return_value=True),
        patch("vllm_ascend.attention.indexer.enable_sfa_dcp_sharded_indexer", return_value=False),
    ):
        group.create_metadata_builders(config, torch.device("cpu"), kernel_block_size=128)
    return group.get_metadata_builder()


@pytest.mark.parametrize("indexer", [False, True])
@pytest.mark.parametrize("physical_size", [128, 768, 1536])
@pytest.mark.parametrize("max_len", [135807, 135808, 135809, 135811, 140000])
def test_dcp_factory_preserves_complete_physical_pages(indexer, physical_size, max_len):
    config = _config(max_len, physical_size)
    spec_cls = AscendSFAIndexerCacheSpec if indexer else AscendMLAAttentionSpec
    spec = spec_cls(block_size=physical_size, num_kv_heads=1, head_size=128, dtype=torch.uint8)
    builder = _create_builder(spec, config, indexer)
    physical_cols = spec.max_num_blocks_per_req(config, max_len)
    split = physical_size // 128
    assert builder.blocks_per_phys_block == split
    assert builder.max_local_block_table_cols == physical_cols * split

    table = AscendBlockTables.__new__(AscendBlockTables)
    table.blocks_per_kv_block = [split]
    storage = torch.zeros(30, physical_cols * split, dtype=torch.int32)

    def write_row(self, req_index, new_block_ids, overwrite):
        assert overwrite
        storage[req_index, : len(new_block_ids[0])] = torch.tensor(new_block_ids[0], dtype=torch.int32)

    # Distinct non-consecutive physical pages expose both lost tail columns and
    # accidental reads from the next request. Include the last batch row.
    # Exercise V2's real expansion; replace only the parent's staged device write.
    with patch.object(BlockTables, "append_block_ids", write_row):
        for row in range(30):
            pages = [1 + 2 * col + row * physical_cols * 3 for col in range(physical_cols)]
            table.append_block_ids(row, (pages,), overwrite=True)
    local = get_sfa_dcp_local_block_table(storage, 30, builder.max_local_block_table_cols)
    assert local.is_contiguous()
    assert local.shape == (30, physical_cols * split)

    # Independently assign every token to a rank, rather than checking against
    # a second copy of the capacity formula.
    global_tokens = torch.arange(max_len)
    for rank in range(8):
        owned = global_tokens[(global_tokens // physical_size) % 8 == rank]
        local_indices = torch.arange(owned.numel())
        assert local_indices.numel() <= local.shape[1] * 128
        if local_indices.numel():
            last = int(local_indices[-1])
            for row in (0, 28, 29):
                page = 1 + 2 * (last // physical_size) + row * physical_cols * 3
                slot = int(local[row, last // 128]) * 128 + last % 128
                assert slot == page * physical_size + last % physical_size


@pytest.mark.parametrize("spec_cls", [AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec])
def test_kernel_spec_copy_keeps_manager_size_without_mutating_original(spec_cls):
    spec = spec_cls(block_size=768, num_kv_heads=1, head_size=128, dtype=torch.uint8)
    kernel_spec = spec.copy_with_new_block_size(128)
    assert spec.block_size == 768
    assert spec.block_table_block_size is None
    assert kernel_spec.block_size == 128
    assert kernel_spec.block_table_block_size == 768
    assert kernel_spec.copy_with_new_block_size(128).block_table_block_size == 768


def test_report_boundary_and_replicated_indexer_addresses():
    config = _config(140000, 768)
    spec = AscendMLAAttentionSpec(block_size=768, num_kv_heads=1, head_size=128, dtype=torch.uint8)
    builder = _create_builder(spec, config, False)
    assert builder.max_local_block_table_cols == 138
    # 135807 + four verify tokens crosses local token 17536 (column 137).
    local = torch.arange(138, dtype=torch.int32).view(1, -1) + 6
    output = torch.empty(1, 138 * 8, dtype=torch.int32)
    build_sfa_dcp_replicated_block_table(local, torch.tensor([135811]), output, torch.arange(138 * 8), 8, 6)
    for token in range(135807, 135811):
        owner = (token // 768) % 8
        local_token = (token // (768 * 8)) * 768 + token % 768
        local_block = int(local[0, local_token // 128])
        expected = ((local_block // 6) * 8 + owner) * 768 + local_token % 768
        actual = int(output[0, token // 128]) * 128 + token % 128
        assert actual == expected
    assert local[0, 17536 // 128] == 143


def test_insufficient_table_is_rejected_before_indexing():
    with pytest.raises(ValueError, match="needs 138 columns"):
        get_sfa_dcp_local_block_table(torch.zeros(30, 137, dtype=torch.int32), 30, 138)


def test_wider_table_is_packed_with_operator_row_stride():
    table = torch.arange(30 * 144, dtype=torch.int32).reshape(30, 144)
    local = get_sfa_dcp_local_block_table(table, 30, 138)
    assert local.stride() == (138, 1)
    torch.testing.assert_close(local, table[:, :138])
    assert local.flatten()[29 * 138] == table[29, 0]


def test_dcp_builder_refreshes_captured_local_block_table():
    config = _config(140000, 768)
    spec = AscendMLAAttentionSpec(block_size=768, num_kv_heads=1, head_size=128, dtype=torch.uint8)
    builder = _create_builder(spec, config, False)
    backing = torch.arange(2 * 144, dtype=torch.int32).reshape(2, 144)
    captured = builder._get_dcp_local_block_table(backing, 2)
    other = backing.clone()
    independent = builder._get_dcp_local_block_table(other, 2)

    backing.add_(1000)
    current = builder._get_dcp_local_block_table(backing, 2)
    assert current.data_ptr() == captured.data_ptr()
    torch.testing.assert_close(captured, backing[:, :138])
    torch.testing.assert_close(independent, other[:, :138])
    assert current.stride() == (138, 1)


@pytest.mark.parametrize("interleave", [1, 128, 768])
def test_kimi_k31_dcp_uses_final_physical_interleave(interleave):
    config = _config(140000, interleave)
    config.model_config.architecture = "KimiK31ForConditionalGeneration"
    config.model_config.hf_text_config = SimpleNamespace(model_type="kimi_k31", index_topk=2048)
    config.cache_config = SimpleNamespace(block_size=768, cache_dtype="fp8")
    config.kv_transfer_config = None
    config.use_v2_model_runner = True
    with patch("vllm_ascend.platform.enable_sp", return_value=False):
        _validate_sfa_dcp_kv_sp(config)
    assert config.parallel_config.cp_kv_cache_interleave_size == 768
    assert enable_sfa_dcp_sharded_indexer(config)
    spec = AscendMLAAttentionSpec(block_size=768, num_kv_heads=1, head_size=128, dtype=torch.uint8)
    builder = _create_builder(spec, config, False)
    assert builder.cp_kv_cache_interleave_size == 768
    assert builder.max_local_block_table_cols == 138
