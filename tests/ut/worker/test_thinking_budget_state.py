# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from vllm.v1.sample.logits_processor.interface import BatchUpdate

from vllm_ascend.worker.thinking_budget_state import (
    DSV4_EOS_TOKEN_ID,
    DSV4_THINK_END_TOKEN_ID,
    DSV4_THINK_START_TOKEN_ID,
    PREMATURE_EOS_POLICY_ALLOW,
    PREMATURE_EOS_POLICY_MASK_IN_REASONING,
    AscendThinkingBudgetStateHolder,
)

VOCAB_SIZE = DSV4_THINK_END_TOKEN_ID + 8


def _reasoning_config(start_token_ids=None, end_token_ids=None):
    if start_token_ids is None:
        start_token_ids = [DSV4_THINK_START_TOKEN_ID]
    if end_token_ids is None:
        end_token_ids = [DSV4_THINK_END_TOKEN_ID]
    return SimpleNamespace(
        reasoning_start_token_ids=start_token_ids,
        reasoning_end_token_ids=end_token_ids,
    )


def _sampling_params(ignore_eos=False, eos_token_id=DSV4_EOS_TOKEN_ID, thinking_token_budget=None):
    return SimpleNamespace(
        ignore_eos=ignore_eos,
        eos_token_id=eos_token_id,
        thinking_token_budget=thinking_token_budget,
    )


def _holder(policy=PREMATURE_EOS_POLICY_MASK_IN_REASONING, num_spec_tokens=0, reasoning_config=None):
    return AscendThinkingBudgetStateHolder(
        reasoning_config or _reasoning_config(),
        max_num_seqs=4,
        num_spec_tokens=num_spec_tokens,
        device=torch.device("cpu"),
        is_pin_memory=False,
        premature_eos_policy=policy,
    )


def _sync_request(holder, prompt_token_ids, output_token_ids, params=None):
    holder.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=[],
            added=[(0, params or _sampling_params(), prompt_token_ids, output_token_ids)],
            moved=[],
        )
    )


def test_masks_eos_incrementally_until_think_end():
    output_token_ids: list[int] = []
    holder = _holder()
    _sync_request(holder, [DSV4_THINK_START_TOKEN_ID], output_token_ids)

    logits = torch.zeros((1, VOCAB_SIZE))
    holder.apply_to_logits(logits, predict_bonus_token=False, spec_token_ids=None)
    assert torch.isneginf(logits[0, DSV4_EOS_TOKEN_ID])

    output_token_ids.extend([10, 11])
    holder.update_state([output_token_ids], spec_token_ids=None)
    assert holder._state[0]["in_reasoning"] is True
    assert holder._state[0]["consumed_output_len"] == 2

    output_token_ids.append(DSV4_THINK_END_TOKEN_ID)
    logits = torch.zeros((1, VOCAB_SIZE))
    holder.apply_to_logits(logits, predict_bonus_token=False, spec_token_ids=None)
    assert logits[0, DSV4_EOS_TOKEN_ID] == 0


def test_masks_spec_target_and_bonus_positions():
    holder = _holder(num_spec_tokens=3)
    _sync_request(holder, [DSV4_THINK_START_TOKEN_ID], [])

    spec_token_ids = [[10, DSV4_THINK_END_TOKEN_ID, 11]]
    logits = torch.zeros((3, VOCAB_SIZE))
    holder.apply_to_logits(logits, predict_bonus_token=False, spec_token_ids=spec_token_ids)

    assert torch.isneginf(logits[0, DSV4_EOS_TOKEN_ID])
    assert torch.isneginf(logits[1, DSV4_EOS_TOKEN_ID])
    assert logits[2, DSV4_EOS_TOKEN_ID] == 0

    holder = _holder(num_spec_tokens=2)
    _sync_request(holder, [DSV4_THINK_START_TOKEN_ID], [])

    logits = torch.zeros((1, VOCAB_SIZE))
    holder.apply_to_logits(
        logits,
        predict_bonus_token=True,
        spec_token_ids=[[10, DSV4_THINK_END_TOKEN_ID]],
    )
    assert logits[0, DSV4_EOS_TOKEN_ID] == 0

    holder = _holder(num_spec_tokens=2)
    _sync_request(holder, [DSV4_THINK_START_TOKEN_ID], [])
    logits = torch.zeros((1, VOCAB_SIZE))
    holder.apply_to_logits(logits, predict_bonus_token=True, spec_token_ids=[[10, 11]])
    assert torch.isneginf(logits[0, DSV4_EOS_TOKEN_ID])


def test_eos_mask_requires_explicit_dsv4_policy_and_sampling_eos():
    holder = _holder(policy=PREMATURE_EOS_POLICY_ALLOW)
    _sync_request(holder, [DSV4_THINK_START_TOKEN_ID], [])
    assert not holder.has_tracked_requests()

    holder = _holder()
    _sync_request(holder, [DSV4_THINK_START_TOKEN_ID], [], params=_sampling_params(ignore_eos=True))
    assert not holder.has_tracked_requests()

    holder = _holder()
    _sync_request(holder, [DSV4_THINK_START_TOKEN_ID], [], params=_sampling_params(eos_token_id=2))
    assert not holder.has_tracked_requests()

    holder = _holder(reasoning_config=_reasoning_config(start_token_ids=[7], end_token_ids=[8]))
    _sync_request(holder, [7], [])
    assert not holder.has_tracked_requests()
