# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, call

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
        parallel_config=SimpleNamespace(data_parallel_size=1, tensor_parallel_size=1, prefill_context_parallel_size=1),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
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
        has_prefill=True,
        num_computed_prefill_tokens_np=np.zeros(3, dtype=np.int32),
        prefill_len_np=np.array([1, 6, 6], dtype=np.int32),
        is_dummy=dummy,
    )
    return replay_mod.DecoderReplayInputs(batch, (), torch.empty(1, 13), [], torch.zeros(3, dtype=torch.int32))


@pytest.mark.parametrize("has_kv_floor", [False, True])
def test_gather_uses_current_kernel_argument_order(state, monkeypatch, has_kv_floor):
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
    kv_floor = torch.ones(1, dtype=torch.int32) if has_kv_floor else None
    replay_mod.DecoderReplayState._gather(
        state, replay_mod.DecoderReplayInputs(batch, (), slots, [], encoder_start), kv_floor=kv_floor, warmup=True
    )

    args = gather.call_args.args
    assert len(args) == 13
    assert args[2] is batch.seq_lens
    assert args[3] is encoder_start
    assert args[4] is batch.positions
    assert args[5] is slots
    assert gather.call_args.kwargs["kv_floor"] is kv_floor
    assert gather.call_args.kwargs["warmup"] is True


def test_prefill_trims_but_decode_and_prompt_logprobs_keep_rows(state):
    state.keeps_rows[0] = True
    state.prepare(inputs(), CUDAGraphMode.NONE)
    assert state.num_actual_tokens == 9
    assert state.dropped_before.np[:4].tolist() == [0, 0, 0, 4]
    assert state._build_metadata.call_args.args[1].tolist() == [0, 1, 7, 9]


@pytest.mark.parametrize(
    ("remaining_offset", "keep_rows", "one_row_enabled", "expected_kept"),
    [(-1, False, True, 2), (0, False, True, 1), (1, False, True, 1), (0, True, True, 6), (0, False, False, 2)],
)
def test_one_row_replay_respects_prompt_boundary_and_full_outputs(
    state, remaining_offset, keep_rows, one_row_enabled, expected_kept
):
    replay_inputs = inputs()
    threshold = state.config.scheduler_config.max_num_batched_tokens + state.layers.window
    replay_inputs.batch.prefill_len_np[1] += threshold + remaining_offset
    state.keeps_rows[0] = keep_rows
    state._one_row_enabled = one_row_enabled
    state.prepare(replay_inputs, CUDAGraphMode.NONE)
    offsets = state._build_metadata.call_args.args[1]
    assert np.diff(offsets).tolist() == [1, expected_kept, 2]
    assert state.skipped_tail_np.tolist() == [expected_kept == 1, False, False, False]


def test_short_chunk_after_one_row_replay_limits_reads_and_clears_history(state):
    replay_inputs = inputs()
    batch = replay_inputs.batch
    batch.prefill_len_np[1] += state.config.scheduler_config.max_num_batched_tokens + state.layers.window
    state.prepare(replay_inputs, CUDAGraphMode.NONE)
    assert state.needs_eager_forward(batch.idx_mapping_np, batch.is_prefilling_np)

    batch.query_start_loc_np = np.array([0, 1, 2, 8], dtype=np.int32)
    batch.num_computed_prefill_tokens_np[1] = 6
    batch.seq_lens_np[1] = batch.prefill_len_np[1] = 7
    state.prepare(replay_inputs, CUDAGraphMode.NONE)
    assert state.layers.rows is not None
    assert state._gather.call_args.kwargs["kv_floor"].tolist() == [0, 6, 0]
    assert not state.skipped_tail_np.any()
    assert not state.needs_eager_forward(batch.idx_mapping_np, batch.is_prefilling_np)


@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("one_row", [False, True])
@pytest.mark.parametrize(
    ("local_end", "owns_tail"),
    [(8, False), (16, True)],
    ids=["owner-before-global-tail", "owner-of-global-tail"],
)
def test_pcp_decoder_replay_keeps_only_original_tail_owners(
    state, monkeypatch, local_end, owns_tail, one_row, sequence_parallel
):
    state.config.parallel_config.prefill_context_parallel_size = 2
    state.config.parallel_config.tensor_parallel_size = 4 if sequence_parallel else 1
    state.layers.sequence_parallel = sequence_parallel
    local_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens_after_padding=8,
        query_start_loc_np=np.array([0, 8], dtype=np.int32),
        seq_lens_np=np.array([local_end], dtype=np.int32),
        idx_mapping_np=np.array([0], dtype=np.int32),
    )
    remaining = state.config.scheduler_config.max_num_batched_tokens + state.layers.window if one_row else 0
    global_batch = SimpleNamespace(
        num_reqs=1,
        query_start_loc_np=np.array([0, 16], dtype=np.int32),
        seq_lens_np=np.array([16], dtype=np.int32),
        is_prefilling_np=np.array([True]),
        idx_mapping_np=np.array([0], dtype=np.int32),
        has_prefill=True,
        is_dummy=False,
        num_computed_prefill_tokens_np=np.array([0], dtype=np.int32),
        prefill_len_np=np.array([16 + remaining], dtype=np.int32),
    )
    replay_inputs = replay_mod.DecoderReplayInputs(
        local_batch,
        (),
        torch.empty(1, 16),
        [],
        torch.zeros(1, dtype=torch.int32),
        SimpleNamespace(global_batch=global_batch),
    )
    build_context = Mock(return_value=object())
    monkeypatch.setattr(state, "_build_replay_pcp_context", build_context)
    state.prepare(replay_inputs, CUDAGraphMode.NONE)
    kept = 1 if one_row else state.layers.window
    assert build_context.call_args.args[1].tolist() == [kept]
    assert state.num_actual_tokens == (kept if owns_tail else 0)
    assert state.layers.rows.numel() == state.num_actual_tokens
    assert state.num_tokens == (4 if sequence_parallel else kept)
    assert state._gather.call_args.kwargs["kv_floor"].tolist() == [16 - kept]


def test_pcp_decoder_replay_keeps_prompt_logprob_rows(state):
    state.keeps_rows[0] = True
    local_batch = SimpleNamespace(num_reqs=1, num_tokens_after_padding=4)
    global_batch = SimpleNamespace(
        num_reqs=1,
        query_start_loc_np=np.array([0, 8], dtype=np.int32),
        is_prefilling_np=np.array([True]),
        idx_mapping_np=np.array([0], dtype=np.int32),
        has_prefill=True,
        is_dummy=False,
        num_computed_prefill_tokens_np=np.array([0], dtype=np.int32),
        prefill_len_np=np.array([8 + state.config.scheduler_config.max_num_batched_tokens + state.layers.window]),
    )
    replay_inputs = replay_mod.DecoderReplayInputs(
        local_batch,
        (),
        torch.empty(1, 8),
        [],
        torch.zeros(1, dtype=torch.int32),
        SimpleNamespace(global_batch=global_batch),
    )
    state.prepare(replay_inputs, CUDAGraphMode.NONE)
    assert state.layers.rows is None
    state._gather.assert_not_called()


def test_pcp_following_short_chunk_preserves_read_floor_without_saving_rows(state):
    batch = SimpleNamespace(
        num_reqs=1,
        num_tokens_after_padding=1,
        query_start_loc_np=np.array([0, 1], dtype=np.int32),
        seq_lens_np=np.array([8], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([7], dtype=np.int32),
        idx_mapping_np=np.array([0], dtype=np.int32),
    )
    replay_inputs = replay_mod.DecoderReplayInputs(
        batch, (), torch.empty(1, 1), [], torch.zeros(1, dtype=torch.int32), SimpleNamespace(global_batch=batch)
    )
    kept = np.array([1], dtype=np.int32)
    assert state._pcp_kept_lengths(replay_inputs, kept, None) is None
    kept_lens, padded, floor = state._pcp_kept_lengths(replay_inputs, kept, np.array([True]))
    assert kept_lens.tolist() == [1]
    assert padded == 1
    assert floor.tolist() == [7]


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
    assert state._gather.call_args_list == [
        call(replay_inputs, warmup=True),
        call(replay_inputs, kv_floor=state._kv_floor.gpu, warmup=True),
    ]


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


@pytest.mark.parametrize(
    ("prompt_logprobs", "prompt_token_ids", "return_routed_experts", "keep_rows"),
    [
        (None, None, False, False),
        (0, None, False, True),
        (3, None, False, True),
        (None, [1], False, True),
        (None, None, True, True),
    ],
)
def test_request_reuse_refreshes_prompt_logprob_retention(
    state, monkeypatch, prompt_logprobs, prompt_token_ids, return_routed_experts, keep_rows
):
    monkeypatch.setattr(AscendModelState, "add_request", lambda *args: None)
    model_state = EngramModelState.__new__(EngramModelState)
    model_state.decoder_replay_state = state
    model_state._replay_start_np = np.zeros(4, dtype=np.int32)
    model_state.vllm_config = SimpleNamespace(
        aux_output_config=SimpleNamespace(enable_return_routed_experts=return_routed_experts)
    )
    state.keeps_rows[1] = True
    state.skipped_tail_np[1] = True
    request = SimpleNamespace(
        replay_start=7,
        sampling_params=SimpleNamespace(prompt_logprobs=prompt_logprobs, prompt_logprob_token_ids=prompt_token_ids),
    )
    model_state.add_request(1, request)
    assert state.keeps_rows[1] == keep_rows
    assert not state.skipped_tail_np[1]
    assert model_state._replay_start_np[1] == 7
