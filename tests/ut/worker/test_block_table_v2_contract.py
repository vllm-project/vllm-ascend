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
        assert kwargs["HAS_SLOT_MAPPING_ENABLED"] is not legacy
        assert kwargs["slot_mapping_enabled"] is (None if legacy else tables.slot_mapping_enabled)
        assert result.shape == (2, 6)
        assert result.data_ptr() == tables.slot_mappings.data_ptr()
