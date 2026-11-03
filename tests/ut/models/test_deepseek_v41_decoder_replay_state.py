# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode, get_current_vllm_config

from vllm_ascend.attention.utils import uses_pcp_sharded_layout
from vllm_ascend.models.deepseek_v41 import decoder_replay_state as replay_mod
from vllm_ascend.models.deepseek_v41.decoder_replay_layers import AscendDecoderReplayLayers
from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState
from vllm_ascend.worker.v2.model_states.default import AscendModelState


@pytest.fixture
def state(monkeypatch):
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
        num_tokens_after_padding=16,
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
        is_dummy=False,
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
    local_batch = SimpleNamespace(num_reqs=1, num_tokens_after_padding=4, is_dummy=False)
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


@pytest.mark.parametrize("actual_tokens", [0, 2])
def test_replay_metadata_preserves_pcp_layout_contract(state, monkeypatch, actual_tokens):
    state.num_tokens = 2
    state.num_actual_tokens = actual_tokens
    state.config.model_config = SimpleNamespace(max_model_len=16)
    state.slots = torch.arange(64).reshape(2, 32)
    state.replay_pcp_context = SimpleNamespace(global_batch=object())
    offsets = np.array([0, actual_tokens], dtype=np.int32)
    state.offsets[:2] = torch.from_numpy(offsets)
    monkeypatch.setattr(replay_mod, "get_cos_and_sin_dsa", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        replay_mod,
        "get_pcp_group",
        lambda: SimpleNamespace(all_gather=lambda tensor, dim: torch.cat((tensor, tensor), dim=dim)),
    )
    metadata = object()

    def build(prefix, common, **kwargs):
        # Exercise the same layout guard as the real PCP attention builder.
        # Gathered slots are invalid unless the common metadata owns a context,
        # including on ranks with no retained suffix queries.
        assert uses_pcp_sharded_layout(common, True)
        assert common.pcp_context is state.replay_pcp_context
        assert common.pcp_cache_group_idx == 1
        assert common.slot_mapping.numel() == 4
        assert common.num_actual_tokens == actual_tokens
        assert common.num_input_tokens == 2
        return metadata

    state.builders = [(1, ["decoder.swa_cache"], SimpleNamespace(build=build, take_device_metadata_tasks=lambda: []))]
    batch = SimpleNamespace(
        num_reqs=1,
        seq_lens=torch.tensor([8], dtype=torch.int32),
        seq_lens_np=np.array([8], dtype=np.int32),
        max_query_len=4,
        attn_state=None,
        is_prefilling_np=np.array([True]),
    )
    replay_inputs = replay_mod.DecoderReplayInputs(
        batch, (torch.zeros(1, 1), torch.zeros(1, 1)), state.slots, [], torch.zeros(1)
    )
    result = replay_mod.DecoderReplayState._build_metadata(state, replay_inputs, offsets)
    assert result["decoder.swa_cache"] is metadata


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
    assert state.layers.rows is None
    assert state.layers.context_factory is not None
    assert state.dp_tokens.tolist() == [16, 2]
    assert state.num_actual_tokens == 0
    state._gather.assert_not_called()
    state._build_metadata.assert_not_called()


def test_unsynchronized_dp_does_not_trim(state, monkeypatch):
    state.config.parallel_config.data_parallel_size = 2
    monkeypatch.setattr(replay_mod, "should_skip_allreduce_across_dp_group", lambda config: True)
    state.prepare(inputs(), CUDAGraphMode.NONE)
    assert state.layers.rows is None
    state._gather.assert_not_called()


@pytest.fixture
def synchronized_dp(state, monkeypatch):
    state.config.parallel_config.data_parallel_size = 2
    monkeypatch.setattr(replay_mod, "should_skip_allreduce_across_dp_group", lambda config: False)
    monkeypatch.setattr(replay_mod, "should_skip_dp_coordination", lambda: False)
    monkeypatch.setattr(replay_mod, "get_dp_group", lambda: SimpleNamespace(rank_in_group=0, cpu_group=None))

    def agree(values, group):
        values[:, 1] = torch.tensor([1, 2])

    monkeypatch.setattr(replay_mod.dist, "all_reduce", agree)
    return state


@pytest.mark.parametrize("local_tokens", [0, 1, 3])
def test_untrimmed_pcp_rank_keeps_collective_width(synchronized_dp, monkeypatch, local_tokens):
    state = synchronized_dp
    batch = SimpleNamespace(
        num_reqs=1,
        num_tokens_after_padding=4,
        query_start_loc_np=np.array([0, local_tokens], dtype=np.int32),
        is_prefilling_np=np.array([False]),
        idx_mapping_np=np.array([0], dtype=np.int32),
        is_dummy=False,
    )
    context = SimpleNamespace(
        global_batch=SimpleNamespace(
            num_reqs=1,
            query_start_loc_np=np.array([0, 3], dtype=np.int32),
            is_prefilling_np=np.array([False]),
            idx_mapping_np=np.array([0], dtype=np.int32),
        )
    )
    state.prepare(
        replay_mod.DecoderReplayInputs(batch, (), torch.empty(1, 8), [], torch.zeros(1), context),
        CUDAGraphMode.NONE,
    )
    assert state.dp_tokens.tolist() == [4, 2]
    assert state.num_actual_tokens == local_tokens
    assert state.layers.rows is None
    assert state.layers.context_factory is not None
    state._gather.assert_not_called()
    state._build_metadata.assert_not_called()


def test_dummy_pcp_does_not_need_ownership_indices(synchronized_dp):
    source = inputs(dummy=True)
    source = replay_mod.DecoderReplayInputs(
        source.batch, (), source.slot_mappings, [], source.encoder_replay_start, SimpleNamespace()
    )
    synchronized_dp.prepare(source, CUDAGraphMode.NONE)
    assert synchronized_dp.dp_tokens.tolist() == [16, 2]
    assert synchronized_dp.layers.rows is None
    synchronized_dp._gather.assert_not_called()


def test_skipped_dp_coordination_never_publishes_partial_counts(synchronized_dp, monkeypatch):
    monkeypatch.setattr(replay_mod, "should_skip_dp_coordination", lambda: True)
    reduce = Mock()
    monkeypatch.setattr(replay_mod.dist, "all_reduce", reduce)
    synchronized_dp.prepare(inputs(), CUDAGraphMode.NONE)
    reduce.assert_not_called()
    assert synchronized_dp.dp_tokens is None
    assert synchronized_dp.layers.context_factory is None


@pytest.mark.parametrize("actual_tokens", [0, 2])
def test_compacted_pcp_rank_uses_padded_dp_count(synchronized_dp, monkeypatch, actual_tokens):
    state = synchronized_dp
    source = inputs()
    source = replay_mod.DecoderReplayInputs(
        source.batch,
        (),
        source.slot_mappings,
        [],
        source.encoder_replay_start,
        SimpleNamespace(global_batch=source.batch),
    )
    monkeypatch.setattr(
        state,
        "_pcp_kept_lengths",
        lambda _: (np.array([0, actual_tokens, 0], dtype=np.int32), 2, torch.zeros(3)),
    )
    builder = Mock(return_value=object())
    monkeypatch.setattr(state, "_build_replay_pcp_context", builder)
    state.prepare(source, CUDAGraphMode.NONE)
    assert state.dp_tokens.tolist() == [2, 2]
    assert state.layers.rows.numel() == actual_tokens
    assert state.layers.num_padded_rows == 2
    builder.assert_called_once()


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_replay_context_preserves_layout_masks_and_restores_mc2(state, monkeypatch, compact, fail, sequence_parallel):
    original_padding = torch.tensor([False, True, False, True])
    state.layers.sequence_parallel = sequence_parallel
    gather_padding = Mock(side_effect=lambda *args, **kwargs: original_padding.clone())
    monkeypatch.setattr(replay_mod, "tensor_model_parallel_all_gather", gather_padding)
    monkeypatch.setattr(replay_mod, "get_tp_group", lambda: SimpleNamespace(world_size=2, rank_in_group=0))
    outer = SimpleNamespace(
        attn_metadata={"original": object()},
        slot_mapping={"original": torch.arange(4)},
        is_padding=original_padding[:2] if sequence_parallel else original_padding,
        additional_kwargs={"eplb_heat_collection_status": True},
    )
    state.num_tokens = 4
    state.num_actual_tokens = 2 if compact else 3
    state.dp_tokens = torch.tensor([4, 2])
    state.rows[:4] = torch.tensor([0, 2, 99, 99])
    state.layers.rows = state.rows[:2] if compact else None
    state.metadata = {"compacted": object()}
    mask = torch.tensor([False, True, False, True, True, False])
    saved_mask = mask.clone()
    inner = SimpleNamespace(additional_kwargs={"mc2_mask": mask})
    current = [outer]
    monkeypatch.setattr(replay_mod, "get_forward_context", lambda: current[0])
    monkeypatch.setattr(replay_mod, "get_mc2_mask", lambda: mask)

    @contextmanager
    def forward(metadata, config, **kwargs):
        assert metadata is (state.metadata if compact else outer.attn_metadata)
        assert kwargs["slot_mapping"] is (None if compact else outer.slot_mapping)
        assert kwargs["num_tokens_across_dp"].tolist() == [4, 2]
        expected = [False, False, True, True] if compact else original_padding.tolist()
        assert kwargs["is_padding"].tolist() == (expected[:2] if sequence_parallel else expected)
        mask.fill_(True)  # Platform hook rewrites the shared MC2 buffer.
        current[0] = inner
        try:
            yield
        finally:
            current[0] = outer

    monkeypatch.setattr(replay_mod, "set_forward_context", forward)
    try:
        with state.forward_context():
            assert mask.tolist() == (
                [True, True, False, False, False, False] if compact else [True, False, True, False, False, False]
            )
            assert inner.additional_kwargs["eplb_heat_collection_status"] is True
            if fail:
                raise RuntimeError("decoder failure")
    except RuntimeError:
        assert fail
    assert torch.equal(mask, saved_mask)
    assert torch.equal(original_padding, torch.tensor([False, True, False, True]))
    assert gather_padding.call_count == int(sequence_parallel)


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
