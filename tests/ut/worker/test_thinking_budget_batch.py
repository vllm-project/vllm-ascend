# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm import SamplingParams
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_ascend.sample.sampler import AscendSampler
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

START = 90
END = 91
VOCAB_SIZE = 128


@pytest.fixture
def batch():
    with patch("vllm_ascend.worker.npu_input_batch.MultiGroupBlockTable"):
        yield NPUInputBatch(
            max_num_reqs=4,
            max_model_len=64,
            max_num_batched_tokens=16,
            device=torch.device("cpu"),
            pin_memory=False,
            vocab_size=VOCAB_SIZE,
            block_sizes=[16],
            kernel_block_sizes=[16],
            reasoning_config=SimpleNamespace(
                reasoning_start_token_ids=[START],
                reasoning_end_token_ids=[END],
            ),
        )


def request(name, budget, output=()):
    return CachedRequestState(
        req_id=name,
        prompt_token_ids=[1, START],
        mm_features=[],
        sampling_params=SamplingParams(temperature=0, thinking_token_budget=budget),
        generator=None,
        block_ids=([0],),
        num_computed_tokens=2,
        output_token_ids=list(output),
    )


def forced_requests(batch):
    batch.refresh_metadata()
    holder = batch.sampling_metadata.thinking_budget_state_holder
    holder.update_state(batch.req_output_token_ids, None)
    logits = torch.zeros(batch.num_reqs, VOCAB_SIZE)
    holder.apply_to_logits(logits, predict_bonus_token=False, spec_token_ids=None)
    return {name for name, row in batch.req_id_to_index.items() if logits[row, END] > 0}


def test_mixed_budgets_follow_swap_and_condense(batch):
    for req in [request("zero", 0), request("long", 4), request("plain", None)]:
        batch.add_request(req)
    assert forced_requests(batch) == {"zero"}
    batch.swap_states(0, 2)
    assert forced_requests(batch) == {"zero"}
    batch.remove_request("long")
    batch.condense()
    assert forced_requests(batch) == {"zero"}
    batch.remove_request("zero")
    batch.add_request(request("replacement", 8))
    batch.condense()
    assert forced_requests(batch) == set()


@pytest.mark.parametrize("budget", [0, 1, 3])
def test_resume_restores_committed_budget(batch, budget):
    req = request("resumed", budget, [10] * budget)
    batch.add_request(req)
    batch.add_request(request("other", 16))
    assert forced_requests(batch) == {"resumed"}
    batch.remove_request("resumed")
    batch.condense()
    assert forced_requests(batch) == set()
    batch.add_request(req)
    assert forced_requests(batch) == {"resumed"}


@pytest.mark.parametrize("budget", [None, -1, 1, 3])
def test_natural_end_and_slot_reuse_disable_forcing(batch, budget):
    batch.add_request(request("finished", budget, [10, END, 11]))
    assert forced_requests(batch) == set()
    batch.remove_request("finished")
    batch.add_request(request("new", 0))
    batch.condense()
    assert forced_requests(batch) == {"new"}


def test_all_requests_removed_then_slots_reused(batch):
    batch.add_request(request("finished", 0))
    batch.add_request(request("other", 1))
    assert forced_requests(batch) == {"finished"}
    for req_id in ["finished", "other"]:
        batch.remove_request(req_id)
    batch.condense()
    batch.refresh_metadata()
    assert batch.num_reqs == 0
    batch.add_request(request("new-unlimited", None))
    batch.add_request(request("new-zero", 0))
    assert forced_requests(batch) == {"new-zero"}


def test_input_batch_budget_reaches_normal_sampler(batch):
    batch.add_request(request("zero", 0))
    batch.add_request(request("reached", 3, [10, 11, 12]))
    batch.add_request(request("plain", None))
    batch.refresh_metadata()
    logits = torch.zeros(3, VOCAB_SIZE)
    logits[:, 42] = 1
    output = AscendSampler()(logits, batch.sampling_metadata)
    assert output.sampled_token_ids.flatten().tolist() == [END, END, 42]


def test_eos_removal_preserves_other_budget_and_clears_reused_slot(batch):
    eos = 92
    ordinary_token = 10
    batch.add_request(request("ends-with-eos", 16))
    batch.add_request(request("remaining", 1))
    sampler = AscendSampler()

    def sample_logits(eos_row):
        batch.refresh_metadata()
        logits = torch.zeros(batch.num_reqs, VOCAB_SIZE)
        logits[:, ordinary_token] = 1
        logits[eos_row, eos] = 2
        return sampler(logits, batch.sampling_metadata).sampled_token_ids.flatten().tolist()

    assert sample_logits(eos_row=0) == [eos, ordinary_token]
    batch.req_output_token_ids[batch.req_id_to_index["remaining"]].append(ordinary_token)
    # The scheduler removes a request when its sampled token is EOS.
    batch.remove_request("ends-with-eos")
    # GPUModelRunner fills removed slots before condensing and refreshing.
    batch.add_request(request("new-unlimited", None, [ordinary_token] * 16))
    batch.condense()
    assert sample_logits(eos_row=0) == [eos, END]
