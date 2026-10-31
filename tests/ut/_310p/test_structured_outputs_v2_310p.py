# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import DEFAULT, MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.output import GrammarOutput

from vllm_ascend._310p.worker.v2.rejection_sampler import RejectionSampler310V2
from vllm_ascend._310p.worker.v2.structured_outputs import apply_grammar_bitmask


@pytest.mark.parametrize(("limit", "invalid_rows"), [(None, [3, 4]), (0, [3, 4]), (1, [4]), (2, [])])
def test_bitmask_reorders_fixed_stride_for_mixed_variable_drafts(limit, invalid_rows):
    # Runner order differs from scheduler order; the plain request has no mask.
    batch = SimpleNamespace(
        req_ids=["plain", "short", "long"],
        cu_num_logits_np=np.array([0, 1, 2, 5], dtype=np.int32),
        num_draft_tokens=2,
    )
    masks: np.ndarray = np.zeros((6, 2), dtype=np.int32)
    # Exercise the signed high bit (token 31) and a partial final word (token 34).
    for row, token in enumerate([31, 34, 0, 5, 6, 7]):
        masks.view(np.uint32)[row, token // 32] = np.uint32(1 << (token % 32))
    grammar = GrammarOutput(["long", "short"], masks, None if limit is None else [limit, 0])
    logits = torch.ones(5, 35, dtype=torch.float16)

    invalid = apply_grammar_bitmask(logits, batch, grammar)

    assert logits[0].eq(1).all()
    assert logits[1:].argmax(dim=-1).tolist() == [5, 31, 34, 0]
    assert logits[1:].isfinite().sum(dim=-1).tolist() == [1, 1, 1, 1]
    assert ([] if invalid is None else np.flatnonzero(invalid).tolist()) == invalid_rows


@pytest.mark.parametrize("grammar_req_ids", [[], ["grammar"]])
def test_bitmask_without_mtp(grammar_req_ids):
    batch = SimpleNamespace(req_ids=["grammar"], cu_num_logits_np=np.array([0, 1]), num_draft_tokens=0)
    grammar = GrammarOutput(grammar_req_ids, np.full((len(grammar_req_ids), 1), 2, dtype=np.int32))
    logits = torch.ones(1, 3)

    assert apply_grammar_bitmask(logits, batch, grammar) is None
    assert logits.tolist() == ([[-torch.inf, 1, -torch.inf]] if grammar_req_ids else [[1, 1, 1]])


@pytest.mark.parametrize("temperature", [0.0, 0.7])
@pytest.mark.parametrize(("num_drafts", "limit"), [(1, 0), (3, 1)])
def test_rejection_never_accepts_unconstrained_drafts(temperature, num_drafts, limit):
    sampler = SimpleNamespace(
        max_num_reqs=1,
        apply_sampling_params=lambda logits, *args: logits,
        _temperature_np=np.array([temperature]),
    )
    input_ids = torch.ones(num_drafts + 1, dtype=torch.int32)
    indices = torch.arange(num_drafts + 1)
    offsets = torch.tensor([0, num_drafts + 1], dtype=torch.int32)
    batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping_np=np.array([0], dtype=np.int32),
        expanded_idx_mapping=torch.zeros(num_drafts + 1, dtype=torch.int32),
        input_ids_cpu=input_ids,
        input_ids=input_ids,
        logits_indices_np=indices.numpy(),
        logits_indices=indices,
        cu_num_logits_np=offsets.numpy(),
        cu_num_logits=offsets,
        seq_lens_np=np.array([8]),
        prefill_len_np=np.array([4]),
    )
    logits = torch.full((num_drafts + 1, 3), -torch.inf)
    logits[:, 1] = 0
    invalid = np.arange(num_drafts + 1) > limit

    with (
        patch("vllm_ascend._310p.worker.v2.rejection_sampler.is_pin_memory_available", return_value=False),
        patch.multiple(torch.npu, Stream=DEFAULT, Event=DEFAULT, current_stream=DEFAULT, stream=DEFAULT),
    ):
        runner = RejectionSampler310V2(
            sampler,
            MagicMock(num_speculative_tokens=num_drafts, rejection_sample_method="standard"),
            torch.device("cpu"),
        )
        output = runner(logits, batch, None, invalid)

    assert output.num_sampled.tolist() == [limit + 1]
    assert output.num_rejected.tolist() == [num_drafts - limit]
    assert output.sampled_token_ids[0].tolist() == [1] * (limit + 1) + [-1] * (num_drafts - limit)
    assert input_ids.eq(1).all()
