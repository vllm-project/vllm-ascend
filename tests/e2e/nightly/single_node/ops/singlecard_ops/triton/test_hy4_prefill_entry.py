"""Compare HY4 Prefill input fusion with the ordinary MRV2 cache mapping."""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.input_batch import prepare_pos_seq_lens
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.worker.v2.block_table import AscendBlockTables
from vllm_ascend.worker.v2.hy4_prefill_inputs import prepare_hy4_prefill_inputs


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="HY4 Prefill targets A5")
@pytest.mark.parametrize("fusion_first", [False, True])
def test_hy4_prefill_cache_mapping_matches_mrv2(fusion_first):
    device = torch.device("npu")
    page_size, page_count = 64, 16
    max_requests, capacity = 4, 128
    states = SimpleNamespace(
        next_prefill_tokens=torch.empty((1, max_requests), dtype=torch.int32, device=device),
        all_token_ids=SimpleNamespace(
            gpu=torch.arange(max_requests * 1024, dtype=torch.int32, device=device).reshape(max_requests, -1)
        ),
        prefill_len=SimpleNamespace(gpu=torch.full((max_requests,), 1024, dtype=torch.int32, device=device)),
        num_computed_tokens=SimpleNamespace(gpu=None),
    )
    tables = AscendBlockTables([page_size] * 2, max_requests, capacity, [page_count] * 2, device)
    tables.has_sliding_window_groups = True
    tables.sliding_window_reach[1] = 512
    pages = [
        [
            [1 + (req + group * max_requests) * 50 + 2 * page for page in range(page_count)]
            for req in range(max_requests)
        ]
        for group in range(2)
    ]
    for req in range(max_requests):
        tables.append_block_ids(req, (pages[0][req], pages[1][req]), overwrite=True)
    tables.apply_staged_writes()
    runner = SimpleNamespace(pcp_manager=None, block_tables=tables, req_states=states)
    buffers = SimpleNamespace(
        input_ids=torch.empty(capacity, dtype=torch.int32, device=device),
        positions=torch.empty(capacity, dtype=torch.int64, device=device),
        seq_lens=torch.empty(max_requests, dtype=torch.int32, device=device),
    )

    # Cold request, prefix hits, and a long SWA history share the first batch.
    # The second batch swaps request order and shrinks, then a continuation
    # shrinks it again. Each path runs first on one of the two test variants.
    steps = [
        ((2, 0, 1), (64, 3, 7), (127, 645, 0, 0)),
        ((0, 2), (3, 2), (130, 652, 64, 0)),
        ((2,), (1,), (133, 652, 66, 0)),
    ]
    for step, (requests, lengths, computed) in enumerate(steps):
        num_tokens = sum(lengths)
        mapping = torch.tensor(requests, dtype=torch.int32, device=device)
        starts = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device=device)
        positions = torch.empty_like(buffers.positions)
        states.num_computed_tokens.gpu = torch.tensor(computed, dtype=torch.int32, device=device)
        previous_rows = len(steps[step - 1][0]) if step else 0
        stale_rows = slice(len(requests), previous_rows)

        def prepare(fused):
            # Exercise each path starting with the same stale rows and address buffer.
            tables._num_dirty_rows = previous_rows
            for table in tables.input_block_tables:
                table[stale_rows].fill_(-7)
            tables.slot_mappings.fill_(-7)
            if fused:
                _, _, (block_tables, slots), _ = prepare_hy4_prefill_inputs(
                    buffers, states, mapping, starts, tables=tables, num_tokens=num_tokens
                )
            else:
                prepare_pos_seq_lens(mapping, starts, states.num_computed_tokens.gpu, positions, buffers.seq_lens)
                batch = SimpleNamespace(
                    idx_mapping=mapping,
                    query_start_loc=starts,
                    positions=positions,
                    num_reqs_after_padding=len(requests),
                    num_tokens_after_padding=num_tokens,
                )
                block_tables, slots = GPUModelRunner.prepare_attn(runner, batch)
            assert tables._num_dirty_rows == len(requests)
            for table in tables.input_block_tables:
                assert torch.count_nonzero(table[stale_rows].cpu()) == 0
            assert torch.all(tables.slot_mappings[:, num_tokens:].cpu() == PAD_SLOT_ID)
            return tuple(table.to("cpu", copy=True) for table in block_tables), slots.to("cpu", copy=True)

        fused = fusion_first if step % 2 == 0 else not fusion_first
        first_tables, first_slots = prepare(fused)
        second_tables, second_slots = prepare(not fused)
        for first, second in zip(first_tables, second_tables):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(first_slots, second_slots, rtol=0, atol=0)
        expected_slots = [
            [
                group_pages[req][position // page_size] * page_size + position % page_size
                for req, length in zip(requests, lengths)
                for position in range(computed[req], computed[req] + length)
            ]
            for group_pages in pages
        ]
        torch.testing.assert_close(first_slots, torch.tensor(expected_slots, dtype=torch.int32), rtol=0, atol=0)
