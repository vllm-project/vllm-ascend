# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v1._310p.model_runner_310p import NPUModelRunner310
from vllm_ascend.worker.v1.model_runner import NPUModelRunner


def _buffer(size, dtype=torch.int32):
    tensor = torch.zeros(size, dtype=dtype)
    return SimpleNamespace(cpu=tensor, gpu=tensor, np=tensor.numpy(), copy_to_gpu=MagicMock())


def _prepare_draft_counts(computed, prompts, scheduled, drafts, consumer=False, runner_cls=NPUModelRunner):
    # Exercise the real input preparation with CPU buffers; no model or NPU
    # kernels are needed to check per-request speculative metadata.
    count = len(scheduled)
    capacity = sum(scheduled) + 8
    req_ids = [f"req{i}" for i in range(count)]
    computed = np.array(computed, dtype=np.int32)
    prompt_lengths = np.array(prompts, dtype=np.int32)
    runner = runner_cls.__new__(runner_cls)
    runner.device = torch.device("cpu")
    runner.arange_np = np.arange(capacity, dtype=np.int32)
    runner._positions_cpu_buf = torch.empty(capacity, dtype=torch.int64)
    runner._positions_np_buf = runner._positions_cpu_buf.numpy()
    for name in (
        "query_pos",
        "query_start_loc",
        "input_ids",
        "req_indices",
        "num_scheduled_tokens",
        "num_accepted_tokens",
        "num_decode_draft_tokens",
        "discard_request_indices",
    ):
        setattr(runner, name, _buffer(capacity))
    runner.discard_request_mask = _buffer(capacity, torch.bool)
    runner.positions = torch.zeros(capacity, dtype=torch.int64)
    runner.num_computed_tokens = torch.zeros(capacity, dtype=torch.int32)
    runner.seq_lens = torch.zeros(capacity, dtype=torch.int32)
    runner.optimistic_seq_lens_cpu = torch.zeros(capacity, dtype=torch.int32)
    token_ids = torch.zeros((count, max(prompts) + capacity), dtype=torch.int32)
    runner.input_batch = SimpleNamespace(
        num_reqs=count,
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        num_computed_tokens_cpu=computed,
        num_computed_tokens_cpu_tensor=torch.from_numpy(computed),
        num_prompt_tokens=prompt_lengths,
        token_ids_cpu=token_ids.numpy(),
        token_ids_cpu_tensor=token_ids,
        block_table=MagicMock(),
        req_prompt_embeds={},
        prev_req_id_to_index=None,
    )
    runner.requests = {r: SimpleNamespace(num_tokens=prompts[i]) for i, r in enumerate(req_ids)}
    runner.dcp_size = 1
    runner.use_async_scheduling = False
    runner.use_async_spec_decode = False
    runner.enable_prompt_embeds = False
    runner.uses_mrope = False
    runner._has_gdn = False
    runner._needs_seq_lens_cpu_sync = False
    runner.is_kv_consumer = consumer
    runner.lora_config = None
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=5)
    runner.num_accepted_tokens_event = None
    runner.valid_sampled_token_count_gpu = None
    runner._compute_prev_positions = MagicMock()
    runner._prepare_input_ids = MagicMock()
    runner._fill_replay_start = MagicMock(return_value=None)
    runner._build_attn_state = MagicMock()
    runner._calc_spec_decode_metadata = MagicMock(return_value=SimpleNamespace(logits_indices=torch.tensor([0])))

    def cumsum_and_arange(lengths, out):
        out[: sum(lengths)] = np.concatenate([np.arange(n) for n in lengths])
        return np.cumsum(lengths, dtype=np.int32)

    runner._get_cumsum_and_arange = cumsum_and_arange
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=sum(scheduled),
        num_scheduled_tokens=dict(zip(req_ids, scheduled)),
        scheduled_spec_decode_tokens={req_ids[i]: ids for i, ids in drafts.items()},
        scheduled_new_reqs=[SimpleNamespace(req_id=r) for r in req_ids],
    )
    with (
        patch("vllm_ascend.worker.v1.model_runner.lmhead_tp_enable", return_value=False),
        patch("vllm_ascend.worker.v1._310p.model_runner_310p.lmhead_tp_enable", return_value=False),
        patch("vllm_ascend.worker.v1._310p.model_runner_310p.is_rc_device", return_value=False),
    ):
        runner._prepare_inputs(scheduler_output, np.array(scheduled, dtype=np.int32))
    return runner.num_decode_draft_tokens.np[:count].tolist()


@pytest.mark.parametrize("consumer", [False, True])
@pytest.mark.parametrize("draft_count", [1, 3, 5])
def test_cached_prompt_tail_uses_verifier_metadata(consumer, draft_count):
    assert _prepare_draft_counts([128], [129], [draft_count + 1], {0: [-1] * draft_count}, consumer) == [draft_count]


@pytest.mark.parametrize("consumer", [False, True])
def test_chunked_prefill_is_not_a_verifier_window(consumer):
    assert _prepare_draft_counts([128], [256], [8], {0: [-1] * 5}, consumer) == [-1]


def test_mixed_batch_classifies_each_request_by_scheduled_width():
    # Cached prompt tail, ordinary decode, chunked prefill, and plain decode.
    assert _prepare_draft_counts(
        [128, 130, 0, 64],
        [129, 129, 32, 64],
        [6, 4, 8, 1],
        {0: [-1] * 5, 1: [7, 8, 9], 2: [-1] * 3},
    ) == [5, 3, -1, -1]


@pytest.mark.parametrize("consumer", [False, True])
def test_310p_uses_the_same_verifier_classification(consumer):
    assert _prepare_draft_counts(
        [128, 130, 0],
        [129, 129, 32],
        [6, 4, 8],
        {0: [-1] * 5, 1: [7, 8, 9], 2: [-1] * 3},
        consumer=consumer,
        runner_cls=NPUModelRunner310,
    ) == [5, 3, -1]
