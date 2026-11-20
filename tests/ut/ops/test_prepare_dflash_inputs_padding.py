# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

prepare_module = importlib.import_module("vllm_ascend.ops.triton.v2.spec_decode.prepare_dflash_inputs")


@pytest.mark.parametrize("max_num_tokens", [12, 16])
def test_draft_padding_refreshes_after_capture_and_batch_resize(max_num_tokens):
    max_num_reqs = 4
    num_query_per_req = 3
    target_buffers = InputBuffers(max_num_reqs, 12, torch.device("cpu"))
    input_batch = InputBatch.make_dummy(max_num_reqs, 12, target_buffers)
    draft_buffers = InputBuffers(max_num_reqs, max_num_tokens, torch.device("cpu"))
    # Graph capture marks every draft row as padding. A real step must clear
    # the active prefix, then restore padding when the next batch shrinks.
    draft_buffers.is_padding.fill_(True)
    token_buffer = torch.zeros(max_num_tokens, dtype=torch.int64)
    request_buffer = torch.zeros(max_num_reqs, dtype=torch.int32)
    sample_buffer = torch.zeros(max_num_reqs * (num_query_per_req - 1), dtype=torch.int64)
    kernel = MagicMock()
    with (
        patch.object(prepare_module, "_prepare_dflash_inputs_kernel", kernel),
        patch.object(prepare_module, "init_device_properties_triton"),
        patch.object(prepare_module, "get_vectorcore_num", return_value=8),
        # vLLM CPU imports expose a Triton placeholder without launch helpers.
        patch.object(prepare_module.triton, "next_power_of_2", lambda n: 1 << (n - 1).bit_length(), create=True),
    ):
        for num_reqs_override, active_reqs in ((None, 4), (1, 1), (3, 3), (None, 4)):
            prepare_module.prepare_dflash_inputs_triton(
                input_buffers=draft_buffers,
                query_slot_mapping=token_buffer,
                context_positions=token_buffer,
                context_slot_mapping=token_buffer,
                sample_indices=sample_buffer,
                sample_pos=sample_buffer,
                sample_idx_mapping=sample_buffer,
                temperature=torch.zeros_like(sample_buffer, dtype=torch.float32),
                seeds=sample_buffer,
                input_batch=input_batch,
                num_sampled=request_buffer,
                num_rejected=request_buffer,
                last_sampled=request_buffer,
                next_prefill_tokens=request_buffer,
                input_temperature=torch.ones(max_num_reqs),
                input_seeds=torch.zeros(max_num_reqs, dtype=torch.int64),
                block_table=torch.zeros((max_num_reqs, 4), dtype=torch.int32),
                block_size=4,
                cp_rank=0,
                cp_size=1,
                cp_interleave=1,
                parallel_drafting_token_id=123,
                num_query_per_req=num_query_per_req,
                num_speculative_steps=num_query_per_req - 1,
                max_num_reqs=max_num_reqs,
                max_num_tokens=max_num_tokens,
                max_model_len=16,
                num_reqs_override=num_reqs_override,
                kv_cache_block_size=4,
            )
            expected = torch.arange(max_num_tokens) >= active_reqs * num_query_per_req
            torch.testing.assert_close(draft_buffers.is_padding, expected)
