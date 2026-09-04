# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate v0.27.1 DFlash semantics and the pinned-main signature shim."""

import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    _prepare_dflash_inputs_kernel_ascend,
)


def _run_prepare(*, positions: list[int], block_table_values: list[int]):
    device = torch.device("npu")
    max_num_reqs = 4
    max_num_tokens = 16
    num_speculative_steps = 3

    out_input_ids = torch.full((max_num_tokens,), -1, dtype=torch.int32, device=device)
    out_query_positions = torch.full((max_num_tokens,), -1, dtype=torch.int64, device=device)
    out_query_start_loc = torch.full((max_num_reqs + 1,), -1, dtype=torch.int32, device=device)
    out_seq_lens = torch.full((max_num_reqs,), -1, dtype=torch.int32, device=device)
    out_query_slot_mapping = torch.full((max_num_tokens,), -2, dtype=torch.int64, device=device)
    out_context_positions = torch.full((max_num_tokens,), -1, dtype=torch.int64, device=device)
    out_context_slot_mapping = torch.full((max_num_tokens,), -2, dtype=torch.int64, device=device)
    out_sample_indices = torch.full(
        (max_num_reqs * num_speculative_steps,),
        -1,
        dtype=torch.int64,
        device=device,
    )
    out_sample_pos = torch.full_like(out_sample_indices, -1)
    out_sample_idx_mapping = torch.full(
        out_sample_indices.shape,
        -1,
        dtype=torch.int32,
        device=device,
    )
    out_temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=device)
    out_seeds = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)

    target_positions = torch.tensor(positions, dtype=torch.int64, device=device)
    target_query_start_loc = torch.tensor([0, 4], dtype=torch.int32, device=device)
    idx_mapping = torch.tensor([2], dtype=torch.int32, device=device)
    last_sampled = torch.tensor([0, 0, 99, 0], dtype=torch.int64, device=device)
    next_prefill_tokens = torch.zeros_like(last_sampled)
    num_sampled = torch.tensor([1], dtype=torch.int32, device=device)
    num_rejected = torch.tensor([2], dtype=torch.int32, device=device)
    temperature = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    seeds = torch.tensor([0, 0, 17, 0], dtype=torch.int64, device=device)
    block_table = torch.tensor([block_table_values], dtype=torch.int32, device=device)

    args = (
        out_input_ids,
        out_query_positions,
        out_query_start_loc,
        out_seq_lens,
        out_query_slot_mapping,
        out_context_positions,
        out_context_slot_mapping,
        out_sample_indices,
        out_sample_pos,
        out_sample_idx_mapping,
        out_temperature,
        out_seeds,
        target_positions,
        target_query_start_loc,
        idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        temperature,
        seeds,
        block_table,
        len(block_table_values),
        123,
        4,
        num_speculative_steps,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        128,
    )
    if vllm_version_is("0.27.1"):
        _prepare_dflash_inputs_kernel_ascend[(1, 1)](
            *args,
            SAMPLE_FROM_ANCHOR=True,
            PAD_SLOT_ID=PAD_SLOT_ID,
            BLOCK_SIZE=16,
        )
    else:
        _prepare_dflash_inputs_kernel_ascend[(1, 1)](
            *args,
            0,
            SAMPLE_FROM_ANCHOR=True,
            PAD_SLOT_ID=PAD_SLOT_ID,
            CP_SIZE=1,
            CP_INTERLEAVE=1,
            BLOCK_SIZE=16,
        )
    torch.npu.synchronize()

    return {
        "input_ids": out_input_ids.cpu(),
        "query_positions": out_query_positions.cpu(),
        "query_slots": out_query_slot_mapping.cpu(),
        "context_positions": out_context_positions.cpu(),
        "context_slots": out_context_slot_mapping.cpu(),
    }


def test_rejected_context_suffix_is_inert() -> None:
    out = _run_prepare(
        positions=[10, 11, 12, 13],
        block_table_values=[0, 0, 7, 8, 9, 10, 11, 12],
    )

    assert out["context_positions"][:4].tolist() == [10, 11, 0, 0]
    assert out["context_slots"][:4].tolist() == [30, 31, PAD_SLOT_ID, PAD_SLOT_ID]
    assert out["input_ids"][:3].tolist() == [99, 123, 123]
    assert out["query_positions"][:3].tolist() == [12, 13, 14]
    assert out["query_slots"][:3].tolist() == [32, 33, 34]


def test_null_block_is_never_writable() -> None:
    out = _run_prepare(
        positions=[2, 3, 4, 5],
        block_table_values=[0, 0, 7, 8, 9, 10, 11, 12],
    )

    assert out["context_slots"][:4].tolist() == [
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
    ]
    assert out["query_slots"][:3].tolist() == [PAD_SLOT_ID, PAD_SLOT_ID, PAD_SLOT_ID]
