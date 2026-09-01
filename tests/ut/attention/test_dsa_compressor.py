# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.dsa_compressor import (
    CompressorExecutor,
    CompressorSPGatherHandle,
    IndexerCompressorExecutor,
)
from vllm_ascend.device.device_op import DeviceOperator


@dataclass
class _KernelMetadata:
    query_start_loc: torch.Tensor
    start_pos: torch.Tensor
    num_compressed_tokens: int
    cache_group_key: str


# Minimal Compressor fixtures shared by executor tests.

def _make_compressor(
    compress_ratio: int,
    coff: int,
    *,
    rotate: bool = False,
) -> SimpleNamespace:
    """Create the weight/config surface consumed by CompressorExecutor."""
    return SimpleNamespace(
        compress_ratio=compress_ratio,
        coff=coff,
        rotate=rotate,
        norm_eps=1e-6,
        wkv=SimpleNamespace(weight=torch.ones((2, 2))),
        wgate=SimpleNamespace(weight=torch.ones((2, 2))),
        ape=torch.ones((compress_ratio, coff * 2)),
        norm=SimpleNamespace(weight=torch.ones(2)),
    )


# Main Compressor input and execution behavior.


def test_prepare_non_sp_input_gathers_and_unpads_hidden_states() -> None:
    executor = CompressorExecutor(
        _make_compressor(4, 2),
        rope_head_dim=1,
        tp_group=SimpleNamespace(world_size=2, device_group=object()),
    )
    hidden_states_local = torch.ones((3, 2))
    gathered_hidden_states = torch.arange(12, dtype=torch.float32).view(6, 2)

    with patch.object(
        torch.ops.vllm,
        "maybe_all_gather_and_maybe_unpad",
        return_value=gathered_hidden_states,
    ) as gather:
        compressor_input = executor.prepare_non_sp_input(
            hidden_states_local,
            num_actual_tokens=5,
            need_gather_q_kv=True,
        )

    gather.assert_called_once_with(hidden_states_local, True)
    torch.testing.assert_close(compressor_input, gathered_hidden_states[:5])


@pytest.mark.parametrize(("compress_ratio", "coff"), [(4, 2), (128, 1)])
def test_compressor_executor_uses_bound_compressor_configuration(
    compress_ratio: int,
    coff: int,
) -> None:
    compressor = _make_compressor(compress_ratio, coff)
    executor = CompressorExecutor(
        compressor,
        rope_head_dim=1,
        tp_group=SimpleNamespace(world_size=1, device_group=object()),
    )
    compressed_kv = torch.ones((1, 2))
    slot_mapping = torch.tensor([[0, 0]], dtype=torch.int32)
    cos = torch.ones((1, 1))
    sin = torch.ones((1, 1))
    metadata = _KernelMetadata(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        start_pos=torch.tensor([0], dtype=torch.int32),
        num_compressed_tokens=1,
        cache_group_key=f"c{compress_ratio}",
    )

    with (
        patch(
            "vllm_ascend.attention.dsa_compressor.get_or_compute_compressor_metadata",
            return_value=(cos, sin, slot_mapping),
        ),
        patch.object(
            torch.ops._C_ascend,
            "compressor",
            create=True,
            return_value=compressed_kv,
        ) as compressor_op,
        patch.object(DeviceOperator, "dsa_kv_compress_scatter") as scatter,
    ):
        executor.run(
            torch.ones((4, 2)),
            torch.ones((1, 4, 1, 2)),
            torch.empty((1, 4, 2)),
            metadata=metadata,
            state_block_table=torch.zeros((1, 1), dtype=torch.int32),
        )

    assert compressor_op.call_args.kwargs["cmp_ratio"] == compress_ratio
    assert compressor_op.call_args.kwargs["coff"] == coff
    assert compressor_op.call_args.args[1] is compressor.wkv.weight
    assert compressor_op.call_args.args[2] is compressor.wgate.weight
    scatter.assert_called_once()
    assert scatter.call_args.args[1] is compressed_kv
    assert scatter.call_args.args[2] is slot_mapping


# LI-specific cache epilogue behavior.


def test_indexer_compressor_executor_owns_indexer_epilog() -> None:
    compressor = _make_compressor(4, 2, rotate=True)
    executor = IndexerCompressorExecutor(
        compressor,
        rope_head_dim=1,
        tp_group=SimpleNamespace(world_size=1, device_group=object()),
    )
    compressed_kv = torch.ones((1, 2))
    rotated_kv = torch.full((1, 2), 2.0)
    kv_scale = torch.ones((1, 1))
    slot_mapping = torch.tensor([[0, 0]], dtype=torch.int32)
    output_cache = (
        torch.empty((1, 4, 2), dtype=torch.int8),
        torch.empty((1, 4, 1)),
        torch.empty((1, 4, 2)),
    )
    hadamard = torch.eye(2)
    metadata = _KernelMetadata(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        start_pos=torch.tensor([0], dtype=torch.int32),
        num_compressed_tokens=1,
        cache_group_key="indexer-c4",
    )

    with (
        patch.object(
            executor,
            "_run_kernel",
            return_value=(compressed_kv, slot_mapping),
        ),
        patch(
            "vllm_ascend.attention.dsa_compressor.rotate_activation",
            return_value=rotated_kv,
        ) as rotate,
        patch.object(
            DeviceOperator,
            "indexer_quant_scatter_part1",
            return_value=(None, kv_scale),
        ) as quant_scatter,
        patch.object(
            DeviceOperator, "dsa_indexer_scatter_scale_part3"
        ) as scale_scatter,
    ):
        executor.run(
            torch.ones((4, 2)),
            torch.ones((1, 4, 1, 2)),
            output_cache,
            metadata=metadata,
            state_block_table=torch.zeros((1, 1), dtype=torch.int32),
            hadamard=hadamard,
        )

    rotate.assert_called_once_with(compressed_kv, hadamard)
    assert quant_scatter.call_args.args == (
        rotated_kv,
        output_cache[0],
        output_cache[2],
        slot_mapping,
    )
    scale_scatter.assert_called_once_with(
        kv_scale,
        output_cache[1],
        slot_mapping,
    )


def test_indexer_compressor_sp_reuses_input_and_runs_epilog_after_reorder() -> None:
    compressor = _make_compressor(4, 2, rotate=True)
    executor = IndexerCompressorExecutor(
        compressor,
        rope_head_dim=1,
        tp_group=SimpleNamespace(world_size=2, device_group=object()),
    )
    prepared_input = torch.ones((8, 2))
    compressed_kv = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    reordered_kv = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
    rotated_kv = torch.full((2, 2), 5.0)
    kv_scale = torch.ones((2, 1))
    slot_mapping = torch.tensor([[1, 0], [1, 1]], dtype=torch.int32)
    output_cache = (
        torch.empty((1, 4, 2), dtype=torch.int8),
        torch.empty((1, 4, 1)),
        torch.empty((1, 4, 2)),
    )
    hadamard = torch.eye(2)
    metadata = _KernelMetadata(
        query_start_loc=torch.tensor([0, 8], dtype=torch.int32),
        start_pos=torch.tensor([0], dtype=torch.int32),
        num_compressed_tokens=2,
        cache_group_key="indexer-c4",
    )
    sp_metadata = SimpleNamespace(
        input_count=8,
        packed_query_start_loc=metadata.query_start_loc,
        packed_start_pos=metadata.start_pos,
        num_compressed_tokens=2,
        gathered_kv_reorder_indices=torch.tensor([1, 0]),
        global_num_compressed_tokens=2,
    )

    with (
        patch.object(
            executor,
            "_run_kernel",
            return_value=(compressed_kv, torch.empty((2, 2), dtype=torch.int32)),
        ) as run_kernel,
        patch.object(
            executor,
            "_launch_sp_output",
            return_value=CompressorSPGatherHandle(
                recv_buffer=compressed_kv,
                send_buffer=compressed_kv,
            ),
        ),
        patch.object(executor, "_sync_sp_state"),
        patch(
            "vllm_ascend.attention.dsa_compressor.get_or_compute_compressor_metadata",
            return_value=(torch.empty(0), torch.empty(0), slot_mapping),
        ),
        patch(
            "vllm_ascend.attention.dsa_compressor.rotate_activation",
            return_value=rotated_kv,
        ) as rotate,
        patch.object(
            DeviceOperator,
            "indexer_quant_scatter_part1",
            return_value=(None, kv_scale),
        ) as quant_scatter,
        patch.object(DeviceOperator, "dsa_indexer_scatter_scale_part3") as scale_scatter,
    ):
        executor.run(
            prepared_input,
            torch.empty((1, 4, 1, 4)),
            output_cache,
            metadata=metadata,
            state_block_table=torch.zeros((1, 1), dtype=torch.int32),
            sp_metadata=sp_metadata,
            hadamard=hadamard,
        )

    assert run_kernel.call_args.args[0] is prepared_input
    rotate.assert_called_once()
    torch.testing.assert_close(rotate.call_args.args[0], reordered_kv)
    assert rotate.call_args.args[1] is hadamard
    assert quant_scatter.call_args.args == (
        rotated_kv,
        output_cache[0],
        output_cache[2],
        slot_mapping,
    )
    scale_scatter.assert_called_once_with(kv_scale, output_cache[1], slot_mapping)
