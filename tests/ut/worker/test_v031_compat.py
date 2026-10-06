# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.outputs import ModelRunnerOutput

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.v2 import aclgraph_utils


@pytest.mark.parametrize("randomize_inputs", [False, True])
def test_dummy_run_accepts_profile_randomization_keyword(randomize_inputs):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(mm_encoder_only=True),
        ),
    )
    hidden, last = runner._dummy_run(1, is_profile=True, randomize_inputs=randomize_inputs)
    assert hidden.numel() == last.numel() == 0


def test_randomized_dummy_token_ids_are_in_vocabulary_and_restored():
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1))
    runner.model_config = SimpleNamespace(get_vocab_size=lambda: 128)
    runner.input_ids = SimpleNamespace(gpu=torch.zeros(32, dtype=torch.int32))
    inputs = runner.input_ids.gpu[:16]
    with runner.maybe_randomize_inputs(inputs, None, randomize_inputs=True):
        assert torch.all((inputs >= 0) & (inputs < 128))
        assert torch.count_nonzero(runner.input_ids.gpu[16:]) == 0
    assert torch.count_nonzero(runner.input_ids.gpu) == 0


@pytest.mark.parametrize("cp_size", [1, 2])
def test_capture_only_prepares_dcp_lengths_for_context_parallel(cp_size):
    batch = SimpleNamespace(
        seq_lens=torch.tensor([3]),
        num_reqs=1,
        num_reqs_after_padding=2,
        dcp_local_seq_lens=None,
    )
    buffers = SimpleNamespace(dcp_local_seq_lens=torch.empty(2, dtype=torch.int32))
    blocks = SimpleNamespace(cp_size=cp_size, cp_rank=0, cp_interleave=1)
    prepared = torch.tensor([2, 0], dtype=torch.int32)
    state = MagicMock()
    with (
        patch.object(aclgraph_utils.AscendInputBatch, "make_dummy", return_value=batch),
        patch.object(aclgraph_utils.cudagraph_utils, "build_slot_mappings_by_layer", return_value={}),
        patch.object(aclgraph_utils, "prepare_dcp_local_seq_lens", return_value=prepared) as prepare,
    ):
        aclgraph_utils._prepare_pcp_inputs_to_capture(
            1, 2, state, buffers, blocks, [], MagicMock(), True, pcp_manager=MagicMock()
        )
    if cp_size == 1:
        prepare.assert_not_called()
        assert batch.dcp_local_seq_lens is None
    else:
        prepare.assert_called_once_with(buffers.dcp_local_seq_lens, batch.seq_lens, 1, 2, 0, 1, num_reqs_padded=2)
        assert batch.dcp_local_seq_lens is prepared
    assert state.prepare_attn.call_args.args[0] is batch


def test_ascend_pp_mtp_output_preserves_draft_tokens():
    # Platform initialization installs Ascend's deliberate extension of the
    # upstream output type. A plain upstream dataclass inspection misses it.
    output = ModelRunnerOutput(
        req_ids=["request"],
        req_id_to_index={"request": 0},
        sampled_token_ids=[[42]],
        spec_token_ids=[[43, 44]],
    )
    assert output.spec_token_ids == [[43, 44]]
