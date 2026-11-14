# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU contract tests for the DFlash input-preparation kernel launcher.

``BLOCK_SIZE`` is a ``tl.constexpr``, so any value derived from the per-step
batch shape becomes a separate Triton specialization that is compiled the first
time it is seen. Issue #20334 measured ~2.3-3.4 s of host stall for each such
first encounter during serving. The launcher therefore passes a constant tile
width; these tests pin that contract and the invariant that makes it correct.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_ascend.ops.triton.spec_decode import prepare_dflash_inputs as dflash_ops


class _CaptureKernel:
    def __init__(self) -> None:
        self.grid: Any = None
        self.args: Any = None
        self.kwargs: Any = None

    def __getitem__(self, grid):
        self.grid = grid
        return self.launch

    def launch(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


def _launch_recording_outputs(monkeypatch, vectorcore_count: int = 48) -> _CaptureKernel:
    capture = _CaptureKernel()
    monkeypatch.setattr(dflash_ops, "_prepare_dflash_inputs_kernel", capture)
    monkeypatch.setattr(dflash_ops, "init_device_properties_triton", lambda: None)
    monkeypatch.setattr(dflash_ops, "get_vectorcore_num", lambda: vectorcore_count)
    return capture


def _run_launcher(
    capture: _CaptureKernel, scheduled_tokens: list[int], num_query_per_req: int, num_speculative_steps: int
) -> None:
    """Run the launcher on CPU mocks and capture the launched constexprs."""
    num_reqs = len(scheduled_tokens)
    tokens = torch.zeros(1, dtype=torch.int64)
    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_scheduled_tokens=torch.tensor(scheduled_tokens, dtype=torch.int64),
        positions=tokens,
        query_start_loc=tokens,
        idx_mapping=tokens,
    )
    input_buffers = SimpleNamespace(
        input_ids=tokens,
        positions=tokens,
        query_start_loc=tokens,
        seq_lens=tokens,
    )

    dflash_ops.prepare_dflash_inputs_triton(
        input_buffers=input_buffers,
        query_slot_mapping=tokens,
        context_positions=tokens,
        context_slot_mapping=tokens,
        sample_indices=tokens,
        sample_pos=tokens,
        sample_idx_mapping=tokens,
        temperature=tokens,
        seeds=tokens,
        input_batch=input_batch,
        num_sampled=tokens,
        num_rejected=tokens,
        last_sampled=tokens,
        next_prefill_tokens=tokens,
        input_temperature=tokens,
        input_seeds=tokens,
        block_table=torch.zeros(num_reqs, 4, dtype=torch.int64),
        block_size=128,
        cp_rank=0,
        cp_size=1,
        cp_interleave=1,
        parallel_drafting_token_id=-1,
        num_query_per_req=num_query_per_req,
        num_speculative_steps=num_speculative_steps,
        max_num_reqs=num_reqs,
        max_num_tokens=8192,
        max_model_len=131072,
        kv_cache_block_size=128,
    )


@pytest.mark.parametrize(
    "scheduled_tokens",
    [
        [1],
        [2],
        [128],
        [129],
        [256],
        [257],
        [1024],
        [4096],
        [8191],
        [8192],
        [8192, 1, 129, 6145],
    ],
)
def test_block_size_is_a_single_constant_specialization(monkeypatch, scheduled_tokens):
    capture = _launch_recording_outputs(monkeypatch)

    _run_launcher(capture, scheduled_tokens, num_query_per_req=1, num_speculative_steps=3)

    assert capture.kwargs["BLOCK_SIZE"] == dflash_ops._MAX_CONTEXT_BLOCK_SIZE


def test_block_size_does_not_depend_on_batch_shape_or_vectorcores(monkeypatch):
    observed = set()
    for vectorcore_count in (24, 40, 48, 64):
        for scheduled_tokens in ([1, 1], [129, 1], [8192, 4096], [8192]):
            for num_query_per_req, num_speculative_steps in ((1, 1), (4, 3), (16, 7)):
                capture = _launch_recording_outputs(monkeypatch, vectorcore_count)
                _run_launcher(capture, scheduled_tokens, num_query_per_req, num_speculative_steps)
                observed.add(capture.kwargs["BLOCK_SIZE"])

    assert observed == {dflash_ops._MAX_CONTEXT_BLOCK_SIZE}


@pytest.mark.parametrize("max_target_query_len", [1, 2, 128, 129, 256, 257, 1024, 4096, 8192])
@pytest.mark.parametrize("vectorcore_count", [24, 48, 64])
def test_context_partition_fits_in_one_tile(monkeypatch, max_target_query_len, vectorcore_count):
    """A fixed tile is only correct while every worker partition fits in it.

    ``workers_per_req`` (grid dim 1) must keep the per-worker context partition
    at or below ``BLOCK_SIZE``; the kernel covers that partition with a single
    masked tile and does not loop over tiles.
    """
    capture = _launch_recording_outputs(monkeypatch, vectorcore_count)

    _run_launcher(capture, [max_target_query_len], num_query_per_req=1, num_speculative_steps=3)

    workers_per_req = capture.grid[1]
    assert workers_per_req * capture.kwargs["BLOCK_SIZE"] >= max_target_query_len
