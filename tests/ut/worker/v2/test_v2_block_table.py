# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables

from vllm_ascend.worker.v2 import block_table as block_table_mod
from vllm_ascend.worker.v2.block_table import AscendBlockTables

_TRITON_BLOCK_SIZE = 1024


def _parent_init(
    self,
    block_sizes,
    max_num_reqs,
    max_num_batched_tokens,
    max_num_blocks_per_group,
    device,
    kernel_block_sizes,
    cp_size,
    cp_rank,
    cp_interleave,
    slot_mapping_enabled=None,
    dcp_sharded=None,
):
    self.block_sizes = block_sizes
    self.kernel_block_sizes = kernel_block_sizes
    self.num_kv_cache_groups = len(block_sizes)
    self.max_num_batched_tokens = max_num_batched_tokens
    self.device = device
    self.cp_size = cp_size
    self.cp_rank = cp_rank
    self.cp_interleave = cp_interleave
    self.slot_mapping_enabled = slot_mapping_enabled
    self.dcp_sharded = dcp_sharded
    self.block_tables = [SimpleNamespace(gpu=torch.zeros(2, 8))]
    self.block_table_ptrs = MagicMock()
    self.block_table_strides = MagicMock()
    self.block_sizes_tensor = MagicMock()
    self.slot_mappings = object()


def _patch_version_is(is_v028: bool):
    if hasattr(block_table_mod, "vllm_version_is"):
        return patch.object(block_table_mod, "vllm_version_is", return_value=is_v028)
    return nullcontext()


def _init_tables(*args, is_v028=True, next_power_of_2=16, **kwargs):
    with (
        patch.object(BlockTables, "__init__", _parent_init),
        patch(
            "vllm_ascend.worker.v2.block_table.triton.next_power_of_2",
            return_value=next_power_of_2,
            create=True,
        ),
        _patch_version_is(is_v028),
    ):
        return AscendBlockTables(*args, **kwargs)


def test_init_defaults_kernel_sizes_and_rebuilds_int32_slots():
    tables = _init_tables([4], 2, 8, [4], torch.device("cpu"))
    assert tables.kernel_block_sizes == [4]
    assert tables.slot_mappings.dtype == torch.int32
    assert tables.slot_mappings.shape == (1, 8)
    if hasattr(tables, "_triton_block_size"):
        assert tables._triton_block_size == _TRITON_BLOCK_SIZE
    if hasattr(tables, "_block_table_window_size"):
        assert tables._block_table_window_size == 16
    if hasattr(tables, "_block_table_pad_size"):
        assert tables._block_table_pad_size == 16


def test_init_keeps_explicit_kernel_block_sizes():
    tables = _init_tables(
        [8],
        2,
        4,
        [2],
        torch.device("cpu"),
        kernel_block_sizes=[4],
    )
    assert tables.kernel_block_sizes == [4]


def test_init_forwards_slot_mapping_enabled_on_newer_vllm():
    if not hasattr(block_table_mod, "vllm_version_is"):
        return
    seen: dict[str, Any] = {}

    def recording_init(self, *args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        _parent_init(self, *args, **kwargs)

    enabled = [True, False]
    sharded = [True, False]
    with (
        patch.object(BlockTables, "__init__", recording_init),
        patch(
            "vllm_ascend.worker.v2.block_table.triton.next_power_of_2",
            return_value=16,
            create=True,
        ),
        _patch_version_is(False),
    ):
        AscendBlockTables([4, 4], 2, 8, [4, 8], torch.device("cpu"), slot_mapping_enabled=enabled, dcp_sharded=sharded)
    assert seen["kwargs"]["slot_mapping_enabled"] is enabled
    assert seen["kwargs"]["dcp_sharded"] is sharded


def test_init_omits_slot_mapping_enabled_on_v028():
    if not hasattr(block_table_mod, "vllm_version_is"):
        return
    seen: dict[str, Any] = {}

    def recording_init(self, *args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        _parent_init(self, *args, **kwargs)

    with (
        patch.object(BlockTables, "__init__", recording_init),
        patch(
            "vllm_ascend.worker.v2.block_table.triton.next_power_of_2",
            return_value=16,
            create=True,
        ),
        _patch_version_is(True),
    ):
        AscendBlockTables([4], 2, 8, [4], torch.device("cpu"), slot_mapping_enabled=[True])
    assert "slot_mapping_enabled" not in seen["kwargs"]


def _make_uninitialized_tables():
    tables = AscendBlockTables.__new__(AscendBlockTables)
    tables.num_kv_cache_groups = 2
    tables.slot_mappings = torch.zeros(2, 6, dtype=torch.int32)
    tables.block_table_ptrs = MagicMock(name="ptrs")
    tables.block_table_strides = MagicMock(name="strides")
    tables.block_sizes_tensor = MagicMock(name="sizes")
    tables.kernel_block_sizes_tensor = MagicMock(name="kernel_sizes")
    tables.slot_mapping_enabled = MagicMock(name="enabled")
    tables.dcp_sharded = torch.tensor([True, False])
    tables.cp_rank = 1
    tables.cp_size = 2
    tables.cp_interleave = 4
    tables._triton_block_size = _TRITON_BLOCK_SIZE
    tables._block_table_window_size = 16
    tables._block_table_pad_size = 16
    return tables


def test_compute_slot_mappings_launches_kernel_and_honors_out():
    tables = _make_uninitialized_tables()
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    positions = torch.zeros(6, dtype=torch.int64)
    custom_out = torch.full((2, 6), 7, dtype=torch.int32)
    kernel = MagicMock()

    with (
        patch("vllm_ascend.worker.v2.block_table._compute_slot_mappings_kernel", kernel),
        _patch_version_is(True),
    ):
        sliced = tables.compute_slot_mappings(idx_mapping, query_start_loc, positions, 3)
        reused = tables.compute_slot_mappings(idx_mapping, query_start_loc, positions, 4, out=custom_out)

    kernel.__getitem__.assert_called_with((2, 3))
    assert kernel.__getitem__.call_count == 2
    kwargs = kernel.__getitem__.return_value.call_args.kwargs
    assert kwargs["PAD_ID"] == PAD_SLOT_ID
    assert kwargs["CP_SIZE"] == 2
    assert kwargs["CP_INTERLEAVE"] == 4
    if "USE_BLOCK_TABLE_STAGING" in kwargs:
        assert kwargs["USE_BLOCK_TABLE_STAGING"] is True
    if "BLOCK_TABLE_WINDOW_SIZE" in kwargs:
        assert kwargs["BLOCK_TABLE_WINDOW_SIZE"] == 16
        assert kwargs["TRITON_BLOCK_SIZE"] == _TRITON_BLOCK_SIZE
        assert kwargs["HAS_SLOT_MAPPING_ENABLED"] is False
    assert sliced.shape == (2, 3)
    assert reused.shape == (2, 4)
    assert reused.data_ptr() == custom_out.data_ptr()


def test_compute_slot_mappings_passes_enabled_mask_on_newer_vllm():
    if not hasattr(block_table_mod, "vllm_version_is"):
        return
    tables = _make_uninitialized_tables()
    enabled = tables.slot_mapping_enabled
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    positions = torch.zeros(6, dtype=torch.int64)
    kernel = MagicMock()

    with (
        patch("vllm_ascend.worker.v2.block_table._compute_slot_mappings_kernel", kernel),
        _patch_version_is(False),
    ):
        tables.compute_slot_mappings(idx_mapping, query_start_loc, positions, 3)

    kwargs = kernel.__getitem__.return_value.call_args.kwargs
    assert kwargs["HAS_SLOT_MAPPING_ENABLED"] is True
    assert kwargs["HAS_DCP_SHARDED"] is True
    assert kwargs["dcp_sharded"] is tables.dcp_sharded
    assert kwargs["slot_mapping_enabled"] is enabled


def test_init_block_table_layout_tensors_normalizes_v028_sizes():
    if not hasattr(AscendBlockTables, "init_block_table_layout_tensors"):
        return
    tables = AscendBlockTables.__new__(AscendBlockTables)
    tables.block_sizes = [16, 32]
    tables.device = torch.device("cpu")
    tables.block_sizes_tensor = torch.tensor([8, 8], dtype=torch.int32)

    with (
        patch.object(BlockTables, "init_block_table_layout_tensors", lambda self: None),
        _patch_version_is(True),
    ):
        tables.init_block_table_layout_tensors()

    assert torch.equal(tables.kernel_block_sizes_tensor, torch.tensor([8, 8], dtype=torch.int32))
    assert torch.equal(tables.block_sizes_tensor, torch.tensor([16, 32], dtype=torch.int32))
    assert tables.block_sizes_tensor.device.type == "cpu"


def test_init_block_table_layout_tensors_keeps_main_contract():
    if not hasattr(AscendBlockTables, "init_block_table_layout_tensors"):
        return
    tables = AscendBlockTables.__new__(AscendBlockTables)
    original = torch.tensor([8, 8], dtype=torch.int32)
    tables.block_sizes = [16, 32]
    tables.device = torch.device("cpu")
    tables.block_sizes_tensor = original

    with (
        patch.object(BlockTables, "init_block_table_layout_tensors", lambda self: None),
        _patch_version_is(False),
    ):
        tables.init_block_table_layout_tensors()

    assert tables.block_sizes_tensor is original
    assert not hasattr(tables, "kernel_block_sizes_tensor")
