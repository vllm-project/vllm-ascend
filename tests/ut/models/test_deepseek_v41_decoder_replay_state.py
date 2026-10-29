# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode, get_current_vllm_config

from vllm_ascend.models.deepseek_v41 import decoder_replay_state as replay_mod
from vllm_ascend.models.deepseek_v41.decoder_replay_layers import AscendDecoderReplayLayers
from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState
from vllm_ascend.worker.v2.model_states.default import AscendModelState


@pytest.fixture
def state(monkeypatch):
    monkeypatch.setattr(replay_mod, "enable_dsa_cp", lambda: False)
    monkeypatch.setattr(replay_mod, "get_mrv2_in_profile_run", lambda: False)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=32),
        cache_config=SimpleNamespace(block_size=2),
        parallel_config=SimpleNamespace(data_parallel_size=1, tensor_parallel_size=1),
    )
    layers = AscendDecoderReplayLayers(2, lambda *states: states, [])
    state = replay_mod.DecoderReplayState(config, layers, torch.device("cpu"))
    state.slots = torch.empty((1, 32), dtype=torch.int64)
    monkeypatch.setattr(state, "_gather", Mock())
    monkeypatch.setattr(state, "_build_metadata", Mock(return_value={}))
    return state


def inputs(*, dummy=False):
    batch = SimpleNamespace(
        num_reqs=3,
        query_start_loc_np=np.array([0, 1, 7, 13], dtype=np.int32),
        is_prefilling_np=np.array([False, True, True]),
        idx_mapping_np=np.array([2, 0, 1]),
        seq_lens_np=np.array([1, 6, 6], dtype=np.int32),
        is_dummy=dummy,
    )
    return replay_mod.DecoderReplayInputs(batch, (), torch.empty(1, 13), [], torch.zeros(3, dtype=torch.int32))


def test_gather_uses_current_kernel_argument_order(state, monkeypatch):
    gather = Mock()
    monkeypatch.setattr(replay_mod, "gather_replay_batch", gather)
    batch = SimpleNamespace(
        num_reqs=1,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        seq_lens=torch.tensor([3], dtype=torch.int32),
        positions=torch.arange(3),
    )
    slots = torch.empty((1, 3), dtype=torch.int64)
    encoder_start = torch.zeros(1, dtype=torch.int32)
    replay_mod.DecoderReplayState._gather(
        state, replay_mod.DecoderReplayInputs(batch, (), slots, [], encoder_start), warmup=True
    )

    args = gather.call_args.args
    assert len(args) == 13
    assert args[2] is batch.seq_lens
    assert args[3] is encoder_start
    assert args[4] is batch.positions
    assert args[5] is slots
    assert gather.call_args.kwargs == {"warmup": True}


def test_prefill_trims_but_decode_and_prompt_logprobs_keep_rows(state):
    state.keeps_rows[0] = True
    state.prepare(inputs(), CUDAGraphMode.NONE)
    assert state.num_actual_tokens == 9
    assert state.dropped_before.np[:4].tolist() == [0, 0, 0, 4]
    assert state._build_metadata.call_args.args[1].tolist() == [0, 1, 7, 9]


@pytest.mark.parametrize(
    ("local_end", "expected_kept"),
    [(4, 0), (8, 2)],
    ids=["owner-before-global-tail", "owner-of-global-tail"],
)
def test_pcp_decoder_replay_keeps_only_original_tail_owners(state, local_end, expected_kept):
    state.config.parallel_config.prefill_context_parallel_size = 2
    local_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens_after_padding=4,
        query_start_loc_np=np.array([0, 4], dtype=np.int32),
        seq_lens_np=np.array([local_end], dtype=np.int32),
        idx_mapping_np=np.array([0], dtype=np.int32),
    )
    global_batch = SimpleNamespace(
        num_reqs=1,
        query_start_loc_np=np.array([0, 8], dtype=np.int32),
        seq_lens_np=np.array([8], dtype=np.int32),
        is_prefilling_np=np.array([True]),
        idx_mapping_np=np.array([0], dtype=np.int32),
    )
    replay_inputs = replay_mod.DecoderReplayInputs(
        local_batch,
        (),
        torch.empty(1, 8),
        [],
        torch.zeros(1, dtype=torch.int32),
        SimpleNamespace(global_batch=global_batch),
    )
    kept_lens, padded, replay_start = state._pcp_kept_lengths(replay_inputs)
    assert kept_lens.tolist() == [expected_kept]
    assert padded == 2
    assert replay_start.tolist() == [6]


def test_pcp_decoder_replay_keeps_prompt_logprob_rows(state):
    state.keeps_rows[0] = True
    local_batch = SimpleNamespace(num_reqs=1, num_tokens_after_padding=4)
    global_batch = SimpleNamespace(
        num_reqs=1,
        query_start_loc_np=np.array([0, 8], dtype=np.int32),
        is_prefilling_np=np.array([True]),
        idx_mapping_np=np.array([0], dtype=np.int32),
    )
    replay_inputs = replay_mod.DecoderReplayInputs(
        local_batch,
        (),
        torch.empty(1, 8),
        [],
        torch.zeros(1, dtype=torch.int32),
        SimpleNamespace(global_batch=global_batch),
    )
    assert state._pcp_kept_lengths(replay_inputs) is None


def test_lazy_decoder_replay_builder_uses_current_config(state, monkeypatch):
    class Backend:
        def get_builder_cls(self):
            assert get_current_vllm_config() is state.config
            return object

    monkeypatch.setattr(replay_mod, "is_circular_kv_cache_spec", lambda spec: False)
    state.slots = None
    group = SimpleNamespace(kv_cache_spec=object(), backend=Backend())
    replay_inputs = inputs()
    replay_inputs = replay_mod.DecoderReplayInputs(
        replay_inputs.batch,
        replay_inputs.block_tables,
        replay_inputs.slot_mappings,
        [[group]],
        replay_inputs.encoder_replay_start,
    )
    state.prepare(replay_inputs, CUDAGraphMode.FULL)
    state._gather.assert_called_once()


def test_decoder_replay_builder_prepares_pcp_rope(state, monkeypatch):
    prepared = Mock()

    class Builder:
        def __init__(self, *args):
            pass

        def prepare_source_rope(self):
            prepared()

    monkeypatch.setattr(replay_mod, "AscendDSAV41MetadataBuilder", Builder)
    monkeypatch.setattr(replay_mod, "is_circular_kv_cache_spec", lambda spec: False)
    state.slots = None
    group = SimpleNamespace(
        kv_cache_spec=object(),
        layer_names=["decoder.swa_cache"],
        backend=SimpleNamespace(get_builder_cls=lambda: Builder),
    )
    replay_inputs = replay_mod.DecoderReplayInputs(inputs().batch, (), torch.empty(1, 13), [[group]], torch.zeros(3))
    state._initialize(replay_inputs)
    prepared.assert_called_once_with()


@pytest.mark.parametrize("mode", [CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE])
def test_graph_step_clears_previous_replay(state, mode):
    state.layers.rows = torch.arange(2)
    state.prepare(inputs(), mode)
    assert state.layers.rows is None
    state._gather.assert_not_called()


@pytest.mark.parametrize("capture,profile", [(True, False), (False, True)])
def test_capture_and_profile_do_not_prepare_replay(state, monkeypatch, capture, profile):
    monkeypatch.setattr(replay_mod, "get_mrv2_in_profile_run", lambda: profile)
    state.prepare(inputs(), CUDAGraphMode.NONE, for_capture=capture)
    assert state.layers.rows is None
    state._gather.assert_not_called()


def test_idle_dp_rank_participates_when_peer_trims(state, monkeypatch):
    state.config.parallel_config.data_parallel_size = 2
    monkeypatch.setattr(replay_mod, "should_skip_allreduce_across_dp_group", lambda config: False)
    monkeypatch.setattr(replay_mod, "should_skip_dp_coordination", lambda: False)
    monkeypatch.setattr(replay_mod, "get_dp_group", lambda: SimpleNamespace(rank_in_group=0, cpu_group=None))

    def agree(values, group):
        values[:, 1] = torch.tensor([1, 2])

    monkeypatch.setattr(replay_mod.dist, "all_reduce", agree)
    state.prepare(inputs(dummy=True), CUDAGraphMode.NONE)
    assert state.layers.rows.numel() == 13
    assert state.dp_tokens.tolist() == [13, 2]
    assert state.dropped_before.np[:4].tolist() == [0, 0, 0, 0]


def test_unsynchronized_dp_does_not_trim(state, monkeypatch):
    state.config.parallel_config.data_parallel_size = 2
    monkeypatch.setattr(replay_mod, "should_skip_allreduce_across_dp_group", lambda config: True)
    state.prepare(inputs(), CUDAGraphMode.NONE)
    assert state.layers.rows is None
    state._gather.assert_not_called()


def test_dsa_cp_pads_replay_without_adding_actual_rows(state, monkeypatch):
    monkeypatch.setattr(replay_mod, "enable_dsa_cp", lambda: True)
    state.config.parallel_config.tensor_parallel_size = 4
    state.prepare(inputs(), CUDAGraphMode.NONE)
    assert state.layers.num_actual_rows == 5
    assert state.layers.rows.numel() == 8


@pytest.mark.parametrize("prompt_logprobs", [None, 0, 3])
def test_request_reuse_refreshes_prompt_logprob_retention(state, monkeypatch, prompt_logprobs):
    monkeypatch.setattr(AscendModelState, "add_request", lambda *args: None)
    model_state = EngramModelState.__new__(EngramModelState)
    model_state.decoder_replay_state = state
    model_state._replay_start_np = np.zeros(4, dtype=np.int32)
    state.keeps_rows[1] = True
    request = SimpleNamespace(replay_start=7, sampling_params=SimpleNamespace(prompt_logprobs=prompt_logprobs))
    model_state.add_request(1, request)
    assert state.keeps_rows[1] == (prompt_logprobs is not None)
    assert model_state._replay_start_np[1] == 7
