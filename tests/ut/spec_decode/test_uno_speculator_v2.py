# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm_ascend.worker.v2.spec_decode.uno.speculator import AscendUnoSpeculator


def _make_speculator() -> AscendUnoSpeculator:
    speculator = object.__new__(AscendUnoSpeculator)
    speculator.device = torch.device("cpu")
    speculator.num_speculative_steps = 3
    speculator.max_num_reqs = 4
    speculator.max_num_tokens = 12
    speculator.max_model_len = 32
    speculator.uno_lora_id = 7
    speculator._step = 0
    speculator.speculative_config = SimpleNamespace(
        uno_noise_seed=17,
        uno_mask_token_id=31,
    )
    speculator.query_start_loc_cpu = torch.arange(5, dtype=torch.int32) * 3
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.zeros(12, dtype=torch.int32),
        positions=torch.zeros(12, dtype=torch.int64),
        query_start_loc=torch.zeros(5, dtype=torch.int32),
        seq_lens=torch.zeros(4, dtype=torch.int32),
        seq_lens_cpu=torch.zeros(4, dtype=torch.int32),
    )
    speculator.block_tables = SimpleNamespace(
        input_block_tables=[
            torch.tensor(
                [[2, 4, 6, 8, 10, 12, 14, 16], [3, 5, 7, 9, 11, 13, 15, 17]],
                dtype=torch.int32,
            )
        ],
        kernel_block_sizes=[4],
        slot_mappings=torch.zeros(1, 12, dtype=torch.int64),
    )
    speculator._copy_request_inputs = Mock()
    return speculator


def test_prepare_inputs_uses_accepted_prefix_and_adapts_only_noise():
    speculator = _make_speculator()
    input_batch = SimpleNamespace(
        num_reqs=2,
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        positions=torch.tensor([3, 4, 5, 6, 9, 10, 11, 12]),
        idx_mapping=torch.tensor([1, 3], dtype=torch.int64),
    )
    last_sampled = torch.tensor([0, 21, 0, 23])
    next_prefill = torch.tensor([0, 25, 0, 27])
    num_queries, mapping, seq_lens_np = speculator._prepare_inputs(
        input_batch,
        num_sampled=torch.tensor([1, 0]),
        num_rejected=torch.tensor([1, 2]),
        last_sampled=last_sampled,
        next_prefill_tokens=next_prefill,
        temperature=torch.ones(4),
        seeds=torch.arange(4),
    )

    assert num_queries == 6
    assert mapping == (0, 7, 7, 0, 7, 7)
    assert speculator.input_buffers.positions[:6].tolist() == [6, 7, 8, 11, 12, 13]
    assert speculator.input_buffers.input_ids[[0, 3]].tolist() == [21, 27]
    noise = speculator.input_buffers.input_ids[torch.tensor([1, 2, 4, 5])]
    assert torch.all((noise >= 1) & (noise < 31))
    assert speculator.input_buffers.query_start_loc.tolist() == [0, 3, 6, 0, 0]
    assert seq_lens_np.tolist() == [9, 14]
    assert speculator.block_tables.slot_mappings[0, :6].tolist() == [18, 19, 24, 31, 36, 37]
    speculator._copy_request_inputs.assert_called_once()


def test_draft_lora_always_restores_base_mapping():
    speculator = _make_speculator()
    calls = []
    speculator._lora_hook = calls.append

    with nullcontext(), speculator._draft_lora((0, 7, 7)):
        pass

    assert calls == [(0, 7, 7), None]
