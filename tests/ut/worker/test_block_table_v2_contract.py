# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.base import PytestBase


class TestBlockTableV2Contract(PytestBase):
    @pytest.mark.parametrize("legacy", [False, True])
    def test_slot_mapping_enablement_reaches_kernel(self, legacy):
        from vllm_ascend.worker.v2.block_table import AscendBlockTables

        tables = object.__new__(AscendBlockTables)
        tables.num_kv_cache_groups = 2
        tables.slot_mappings = torch.empty((2, 8), dtype=torch.int32)
        tables.block_table_ptrs = MagicMock()
        tables.block_table_strides = MagicMock()
        tables.block_sizes_tensor = MagicMock()
        tables.kernel_block_sizes_tensor = MagicMock()
        tables.cp_rank = 0
        tables.cp_size = 1
        tables.cp_interleave = 1
        tables._block_table_pad_size = 4
        if not legacy:
            tables.slot_mapping_enabled = torch.tensor([True, False])

        with (
            patch("vllm_ascend.worker.v2.block_table.vllm_version_is", return_value=legacy),
            patch("vllm_ascend.worker.v2.block_table._compute_slot_mappings_kernel") as kernel,
        ):
            result = tables.compute_slot_mappings(
                torch.tensor([0]), torch.tensor([0, 4]), torch.arange(4), num_tokens_padded=6
            )

        kwargs = kernel.__getitem__.return_value.call_args.kwargs
        args = kernel.__getitem__.return_value.call_args.args
        assert args[6] is tables.block_sizes_tensor
        assert args[7] is tables.kernel_block_sizes_tensor
        assert kwargs["HAS_SLOT_MAPPING_ENABLED"] is not legacy
        assert kwargs["slot_mapping_enabled"] is (None if legacy else tables.slot_mapping_enabled)
        assert kwargs["USE_BLOCK_TABLE_STAGING"] is True
        assert result.shape == (2, 6)
        assert result.data_ptr() == tables.slot_mappings.data_ptr()

    @pytest.mark.parametrize("legacy", [False, True])
    def test_kernel_block_layout_refresh(self, legacy):
        from vllm.v1.worker.gpu.block_table import BlockTables

        from vllm_ascend.worker.v2.block_table import AscendBlockTables

        tables = object.__new__(AscendBlockTables)
        tables.kernel_block_sizes = [128, 4096]
        tables.block_sizes = [2048, 4096]
        tables.device = torch.device("cpu")
        main_tensor = torch.tensor([128, 4096], dtype=torch.int32)

        def parent_refresh():
            if legacy:
                tables.block_sizes_tensor = main_tensor
            else:
                tables.block_sizes_tensor = torch.tensor(tables.block_sizes, dtype=torch.int32)
                tables.kernel_block_sizes_tensor = main_tensor

        with (
            patch("vllm_ascend.worker.v2.block_table.vllm_version_is", return_value=legacy),
            patch.object(BlockTables, "init_block_table_layout_tensors", side_effect=parent_refresh) as parent,
        ):
            for _ in range(2):
                tables.kernel_block_sizes_tensor = torch.tensor([-1, -1], dtype=torch.int32)
                tables.init_block_table_layout_tensors()
                torch.testing.assert_close(tables.kernel_block_sizes_tensor, main_tensor)
                torch.testing.assert_close(tables.block_sizes_tensor, torch.tensor([2048, 4096], dtype=torch.int32))
                assert tables.kernel_block_sizes_tensor is main_tensor
        assert parent.call_count == 2
