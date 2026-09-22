# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec
from vllm_ascend.ops.triton.compute_slot_mapping import compute_slot_mapping_fused_groups
from vllm_ascend.worker.block_table import BlockTable


@pytest.mark.parametrize("capacity", [4, 7])
def test_real_tail_block_table_maps_slots_instead_of_padding(capacity):
    spec = AscendIndexerKPoolTailSpec(
        block_size=capacity,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float32,
        sliding_window=4,
        compress_ratio=4,
    )
    group = SimpleNamespace(kv_cache_spec=spec)
    with patch(
        "vllm_ascend.worker.block_table.get_dcp_group", return_value=SimpleNamespace(world_size=1, rank_in_group=0)
    ):
        table = BlockTable(
            block_size=capacity,
            max_num_reqs=1,
            max_num_blocks_per_req=1,
            max_num_batched_tokens=4,
            pin_memory=False,
            device=torch.device("npu"),
            kv_cache_group=group,
        )
    table.block_table.gpu.fill_(2)
    table.compute_slot_mapping(
        1,
        torch.tensor([0, 4], dtype=torch.int32, device="npu"),
        torch.tensor([0, 1, capacity, capacity + 1], device="npu"),
    )
    assert table.slot_mapping.gpu.cpu().tolist() == [2 * capacity, 2 * capacity + 1, 2 * capacity, 2 * capacity + 1]


@pytest.mark.parametrize("lengths", [[3, 0, 5], [1025, 0, 1027], [1] * 64])
@torch.inference_mode()
def test_normal_and_mixed_fused_circular_mapping_and_graph(lengths):
    count, num_reqs, capacity = sum(lengths), len(lengths), 4
    positions_cpu = torch.cat([torch.arange(17 + req * 4096, 17 + req * 4096 + n) for req, n in enumerate(lengths)])
    req_ids = torch.repeat_interleave(torch.arange(num_reqs), torch.tensor(lengths))
    ends = torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0).int().npu()
    positions = positions_cpu.npu()
    tail_table = torch.arange(5, 5 + num_reqs, dtype=torch.int32).view(-1, 1).npu()
    ncols = (int(positions_cpu.max()) + 2) // 16 + 1
    full_table = torch.arange(100, 100 + num_reqs * ncols, dtype=torch.int32).view(num_reqs, ncols).npu()
    outputs = [torch.full((count + 19,), 12345, dtype=torch.int32, device="npu") for _ in range(2)]
    tables = [tail_table, full_table]
    table_ptrs = torch.tensor([x.data_ptr() for x in tables], dtype=torch.uint64, device="npu")
    output_ptrs = torch.tensor([x.data_ptr() for x in outputs], dtype=torch.uint64, device="npu")
    strides = torch.tensor([x.stride(0) for x in tables], dtype=torch.int64, device="npu")
    sizes = torch.tensor([capacity, 16], dtype=torch.int32, device="npu")
    circular = torch.tensor([1, 0], dtype=torch.int32, device="npu")

    def expected():
        p = positions.cpu()
        tail = tail_table.cpu()[req_ids, 0] * capacity + p % capacity
        full = full_table.cpu()[req_ids, p // 16] * 16 + p % 16
        return [torch.cat((x.int(), torch.full((19,), -1, dtype=torch.int32))) for x in [tail, full]]

    obj = BlockTable.__new__(BlockTable)
    obj.dcp_world_size, obj.dcp_rank = 1, 0
    obj.physical_block_size = obj.block_size = capacity
    obj.blocks_per_phys_block, obj.cp_kv_cache_interleave_size = 1, 1
    obj.max_num_batched_tokens = count + 19
    obj.is_circular = True
    # Circular tail addressing is not a CircularKVCacheSpec group. Newer
    # runtimes skip slot mapping for those groups (e.g. recurrent state).
    obj.is_circular_group = False
    obj.block_table = SimpleNamespace(gpu=tail_table)
    obj.slot_mapping = SimpleNamespace(gpu=outputs[0])
    obj.compute_slot_mapping(num_reqs, ends, positions)
    torch.testing.assert_close(outputs[0].cpu(), expected()[0], rtol=0, atol=0)

    def fused():
        compute_slot_mapping_fused_groups(
            2,
            num_reqs,
            count,
            count + 19,
            ends,
            positions,
            table_ptrs,
            output_ptrs,
            strides,
            sizes,
            capacity,
            pad_id=-1,
            is_circular_ptr=circular,
        )

    fused()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        fused()
    for _ in range(2):
        graph.replay()
        for actual, ref in zip(outputs, expected()):
            torch.testing.assert_close(actual.cpu(), ref, rtol=0, atol=0)
        positions.add_(1)
        tail_table.add_(3)

    # Pre-existing callers omit the optional circular descriptor entirely.
    compute_slot_mapping_fused_groups(
        1,
        num_reqs,
        count,
        count + 19,
        ends,
        positions,
        table_ptrs[1:],
        output_ptrs[1:],
        strides[1:],
        sizes[1:],
        16,
        pad_id=-1,
    )
    torch.testing.assert_close(outputs[1].cpu(), expected()[1], rtol=0, atol=0)


def test_draft_helper_preserves_circular_request_ids_and_negative_positions():
    # Address-helper regression only; speculative acceptance remains deferred.
    obj = BlockTable.__new__(BlockTable)
    obj.dcp_world_size, obj.max_num_blocks_per_req, obj.blocks_per_phys_block = 1, 1, 1
    obj.block_size, obj.kernel_sizes, obj.is_circular = 4, [4], True
    obj.is_circular_group = False
    obj.block_table = SimpleNamespace(np=np.array([[5], [9]], dtype=np.int32))
    obj.slot_mapping = SimpleNamespace(np=np.full(5, 777, dtype=np.int32), copy_to_gpu=lambda n: None)
    obj.compute_slot_mapping_draft(np.array([0, 1, 0, 1, 0]), np.array([17, 1024, -1, 1027, 100000]))
    assert obj.slot_mapping.np.tolist() == [21, 36, -1, 39, 20]
