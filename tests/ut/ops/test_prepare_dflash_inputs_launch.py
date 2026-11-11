#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

"""Launcher-level unit tests for the fixed DFlash context tile width.

``prepare_dflash_inputs_triton`` passes ``BLOCK_SIZE`` to the kernel as a
``tl.constexpr``. These tests pin down the two properties that make a fixed
tile width correct and cheap (see #20334):

  1. The launcher emits **exactly one** tile width — ``_MAX_CONTEXT_BLOCK_SIZE``
     — regardless of batch shape, draft shape, or the device VectorCore count.
     A dynamic width would turn each distinct value into a separate Triton
     specialization and cost a full JIT compile (2.3-3.4 s) the first time
     each value is hit during serving.
  2. Every worker's context partition still fits inside that single masked
     tile: ``workers_per_req * BLOCK_SIZE >= max_target_query_len``.

The tests do NOT require NPU hardware: the Triton kernel and the device
property helpers are mocked, so only the launcher's host-side logic runs.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm_ascend.ops.triton.spec_decode.prepare_dflash_inputs import (
    _MAX_CONTEXT_BLOCK_SIZE,
    prepare_dflash_inputs_triton,
)

MODULE = "vllm_ascend.ops.triton.spec_decode.prepare_dflash_inputs"


class _RecordingKernel:
    """Stands in for the ``@triton.jit`` kernel: records grid + kwargs."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def _launch(*args, **kwargs):
            self.calls.append(SimpleNamespace(grid=grid, args=args, kwargs=kwargs))

        return _launch


def _run_launcher(scheduled_tokens, vectorcores=48, num_query_per_req=9,
                  num_speculative_steps=8):
    """Invoke ``prepare_dflash_inputs_triton`` on CPU tensors and return the
    recorded kernel launches."""
    num_reqs = len(scheduled_tokens)
    max_num_reqs = max(8, num_reqs)
    total_tokens = int(sum(scheduled_tokens))
    max_num_tokens = max(64, total_tokens + num_reqs * num_query_per_req)

    ends = np.cumsum([0] + list(scheduled_tokens))
    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_scheduled_tokens=np.asarray(scheduled_tokens, dtype=np.int32),
        positions=torch.zeros(total_tokens, dtype=torch.int64),
        query_start_loc=torch.tensor(ends, dtype=torch.int32),
        idx_mapping=torch.zeros(max_num_reqs, dtype=torch.int32),
    )
    input_buffers = SimpleNamespace(
        input_ids=torch.zeros(max_num_tokens, dtype=torch.int32),
        positions=torch.zeros(max_num_tokens, dtype=torch.int64),
        query_start_loc=torch.zeros(max_num_reqs + 1, dtype=torch.int32),
        seq_lens=torch.zeros(max_num_reqs, dtype=torch.int32),
    )
    sample_capacity = max_num_reqs * num_speculative_steps
    outputs = SimpleNamespace(
        query_slot_mapping=torch.zeros(max_num_tokens, dtype=torch.int32),
        context_positions=torch.zeros(max_num_tokens, dtype=torch.int64),
        context_slot_mapping=torch.zeros(max_num_tokens, dtype=torch.int32),
        sample_indices=torch.zeros(sample_capacity, dtype=torch.int64),
        sample_pos=torch.zeros(sample_capacity, dtype=torch.int64),
        sample_idx_mapping=torch.zeros(sample_capacity, dtype=torch.int32),
        temperature=torch.zeros(max_num_reqs, dtype=torch.float32),
        seeds=torch.zeros(max_num_reqs, dtype=torch.int64),
    )
    state = SimpleNamespace(
        num_sampled=torch.ones(num_reqs, dtype=torch.int32),
        num_rejected=torch.zeros(num_reqs, dtype=torch.int32),
        last_sampled=torch.zeros(max_num_reqs, dtype=torch.int64),
        next_prefill_tokens=torch.zeros(max_num_reqs, dtype=torch.int32),
        input_temperature=torch.ones(max_num_reqs, dtype=torch.float32),
        input_seeds=torch.zeros(max_num_reqs, dtype=torch.int64),
    )
    block_table = torch.ones((max_num_reqs, 8), dtype=torch.int32)

    recording = _RecordingKernel()
    with patch(f"{MODULE}._prepare_dflash_inputs_kernel", recording), \
         patch(f"{MODULE}.init_device_properties_triton"), \
         patch(f"{MODULE}.get_vectorcore_num", return_value=vectorcores):
        prepare_dflash_inputs_triton(
            input_buffers,
            outputs.query_slot_mapping,
            outputs.context_positions,
            outputs.context_slot_mapping,
            outputs.sample_indices,
            outputs.sample_pos,
            outputs.sample_idx_mapping,
            outputs.temperature,
            outputs.seeds,
            input_batch,
            state.num_sampled,
            state.num_rejected,
            state.last_sampled,
            state.next_prefill_tokens,
            state.input_temperature,
            state.input_seeds,
            block_table,
            block_size=128,
            cp_rank=0,
            cp_size=1,
            cp_interleave=1,
            parallel_drafting_token_id=151669,
            num_query_per_req=num_query_per_req,
            num_speculative_steps=num_speculative_steps,
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
            max_model_len=max(2 * max_num_tokens, 65536),
            sample_from_anchor=False,
            kv_cache_block_size=128,
        )
    return recording


def _widths(recording):
    return {call.kwargs["BLOCK_SIZE"] for call in recording.calls}


@pytest.mark.parametrize("scheduled_tokens", [
    [1], [2], [3], [129], [193], [257], [385], [769], [1025],
    [1537], [2049], [3073], [6145], [8192],
    [3, 17, 29, 61],  # ragged mixed batch
    [8192, 1, 2],  # wide spread across requests
])
def test_tile_width_is_a_single_constant_specialization(scheduled_tokens):
    """Every launch must use exactly one width: _MAX_CONTEXT_BLOCK_SIZE."""
    recording = _run_launcher(scheduled_tokens)
    assert recording.calls, "launcher did not launch the kernel"
    assert _widths(recording) == {_MAX_CONTEXT_BLOCK_SIZE}


@pytest.mark.parametrize("vectorcores", [4, 8, 16, 48, 64])
@pytest.mark.parametrize("scheduled_tokens, nqpr, steps", [
    ([1], 9, 8),
    ([8192], 9, 8),
    ([17, 33, 129], 4, 3),
    ([2049], 18, 17),
])
def test_tile_width_independent_of_vectorcores_and_draft_shapes(
        vectorcores, scheduled_tokens, nqpr, steps):
    """The width must not depend on device shape or speculative geometry."""
    recording = _run_launcher(scheduled_tokens, vectorcores=vectorcores,
                              num_query_per_req=nqpr,
                              num_speculative_steps=steps)
    assert _widths(recording) == {_MAX_CONTEXT_BLOCK_SIZE}


@pytest.mark.parametrize("scheduled_tokens", [
    [1], [9], [65], [129], [385], [1025], [3073], [8192],
    [3, 17, 29, 61],
])
@pytest.mark.parametrize("vectorcores", [4, 48, 64])
def test_context_partition_fits_in_single_tile(scheduled_tokens, vectorcores):
    """workers_per_req * BLOCK_SIZE must cover max_target_query_len so the
    single masked tile never drops the tail of a worker's partition."""
    recording = _run_launcher(scheduled_tokens, vectorcores=vectorcores)
    max_len = max(scheduled_tokens)
    for call in recording.calls:
        num_reqs, workers_per_req = call.grid
        assert num_reqs == len(scheduled_tokens)
        assert workers_per_req * call.kwargs["BLOCK_SIZE"] >= max_len, (
            f"partition {max_len} tokens over {workers_per_req} workers "
            f"with width {call.kwargs['BLOCK_SIZE']} would drop the tail")
