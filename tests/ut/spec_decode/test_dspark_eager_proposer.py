# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import vllm_ascend.spec_decode as spec_decode_module
import vllm_ascend.spec_decode.deepseek_v4_dspark_proposer as dspark_proposer_module
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.deepseek_v4_dspark_proposer import (
    AscendDeepSeekV4DSparkProposer,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def _group(gid: int, block_size: int = 8):
    return SimpleNamespace(
        kv_cache_group_id=gid,
        kv_cache_spec=SimpleNamespace(block_size=block_size),
        layer_names=[f"draft.layer.{gid}"],
    )


@pytest.mark.parametrize("method", ["mtp", "dspark"])
def test_dspark_registration_precedes_generic_mtp(monkeypatch, method):
    marker = object()
    monkeypatch.setattr(
        spec_decode_module,
        "AscendDeepSeekV4DSparkProposer",
        lambda *args: marker,
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(dspark_block_size=5))
        )
    )

    assert spec_decode_module.get_spec_decode_method(method, config, "npu", "runner") is marker


def test_anchor_first_inputs_use_each_cache_group_and_rejected_anchor():
    proposer = AscendDeepSeekV4DSparkProposer.__new__(AscendDeepSeekV4DSparkProposer)
    proposer.device = torch.device("cpu")
    proposer.block_size = proposer.num_speculative_tokens = 3
    proposer.max_model_len = 64
    proposer.max_num_tokens = 16
    proposer.max_query_tokens = 6
    proposer.draft_attn_groups = [_group(0), _group(1)]
    proposer._draft_attn_layer_names_ordered = ["draft.layer.0", "draft.layer.1"]
    proposer._per_group_block_tables = {
        0: torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]], dtype=torch.int32),
        1: torch.tensor([[30, 31, 32, 33, 34], [40, 41, 42, 43, 44]], dtype=torch.int32),
    }
    proposer._query_slot_buffers = {}
    proposer._context_slot_buffers = {}
    proposer.runner = None
    proposer._seed_buffer = torch.zeros(2, dtype=torch.int64)
    proposer._draft_buffer = torch.zeros((2, 3), dtype=torch.int64)
    proposer._token_to_req_buffer = torch.zeros(6, dtype=torch.int32)
    proposer._dflash_hidden_states = torch.zeros((8, 2))
    proposer._context_positions_buffer = torch.zeros(8, dtype=torch.int32)
    proposer.positions = torch.zeros(6, dtype=torch.int32)
    proposer.input_ids = torch.zeros(6, dtype=torch.int64)
    proposer.parallel_drafting_token_id = 99
    proposer.arange_dspark = torch.arange(32, dtype=torch.int32)

    target_positions = torch.tensor([10, 11, 12, 13, 30, 31, 32, 33], dtype=torch.int32)
    target_hidden = torch.arange(16, dtype=torch.float32).view(8, 2)
    cad = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
        seq_lens=torch.tensor([14, 34], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([14, 34], dtype=torch.int32),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=2,
        num_actual_tokens=8,
        num_input_tokens=8,
        max_query_len=4,
        actual_seq_lengths_q=[4, 4],
        block_table_tensor=None,
        slot_mapping=torch.arange(8, dtype=torch.int32),
        positions=target_positions,
        attn_state=AscendAttentionState.SpecDecoding,
        decode_token_per_req=1,
        max_seq_len=34,
    )

    num_tokens, indices, returned, _ = proposer.set_inputs_first_pass(
        torch.arange(8),
        torch.tensor([101, 202]),
        target_positions,
        target_hidden,
        torch.tensor([2, 5], dtype=torch.int32),
        cad,
        torch.full((2,), torch.iinfo(torch.int32).max, dtype=torch.int32),
    )

    assert num_tokens == 6
    torch.testing.assert_close(indices, torch.arange(6, dtype=torch.int32))
    torch.testing.assert_close(proposer.input_ids, torch.tensor([101, 99, 99, 202, 99, 99]))
    torch.testing.assert_close(proposer.positions, torch.tensor([13, 14, 15, 32, 33, 34], dtype=torch.int32))
    torch.testing.assert_close(returned.seq_lens, torch.tensor([16, 35], dtype=torch.int32))
    torch.testing.assert_close(returned.query_start_loc, torch.tensor([0, 3, 6], dtype=torch.int32))
    assert proposer._dflash_num_context == 8
    assert proposer._query_slots_by_gid[0].data_ptr() != proposer._query_slots_by_gid[1].data_ptr()
    torch.testing.assert_close(proposer._dflash_hidden_states[:8], target_hidden)


def test_dspark_masks_draft_tokens_past_max_model_len():
    proposer = AscendDeepSeekV4DSparkProposer.__new__(AscendDeepSeekV4DSparkProposer)
    proposer.device = torch.device("cpu")
    proposer.block_size = proposer.num_speculative_tokens = 3
    proposer.max_model_len = 64
    proposer.max_num_tokens = 4
    proposer.max_query_tokens = 3
    proposer.draft_attn_groups = [_group(0)]
    proposer._draft_attn_layer_names_ordered = ["draft.layer.0"]
    proposer._per_group_block_tables = {0: torch.arange(8, dtype=torch.int32).view(1, 8)}
    proposer._query_slot_buffers = {}
    proposer._context_slot_buffers = {}
    proposer.runner = None
    proposer._seed_buffer = torch.zeros(1, dtype=torch.int64)
    proposer._token_to_req_buffer = torch.zeros(3, dtype=torch.int32)
    proposer._dflash_hidden_states = torch.zeros((4, 2))
    proposer._context_positions_buffer = torch.zeros(4, dtype=torch.int32)
    proposer.positions = torch.zeros(3, dtype=torch.int32)
    proposer.input_ids = torch.zeros(3, dtype=torch.int64)
    proposer.parallel_drafting_token_id = 99
    proposer.arange_dspark = torch.arange(8, dtype=torch.int32)
    target_positions = torch.tensor([61, 62], dtype=torch.int32)
    cad = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([63], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([63], dtype=torch.int32),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=1,
        num_actual_tokens=2,
        num_input_tokens=2,
        max_query_len=2,
        actual_seq_lengths_q=[2],
        block_table_tensor=None,
        slot_mapping=torch.arange(2, dtype=torch.int32),
        positions=target_positions,
        attn_state=AscendAttentionState.SpecDecoding,
        decode_token_per_req=1,
        max_seq_len=63,
    )

    proposer.set_inputs_first_pass(
        torch.arange(2),
        torch.tensor([101]),
        target_positions,
        torch.zeros((2, 2)),
        None,
        cad,
        None,
    )

    torch.testing.assert_close(proposer.positions, torch.tensor([63, 0, 0], dtype=torch.int32))
    torch.testing.assert_close(proposer._query_slots_by_gid[0], torch.tensor([63, -1, -1], dtype=torch.int32))
    torch.testing.assert_close(cad.seq_lens, torch.tensor([64], dtype=torch.int32))


def test_dspark_rejects_input_without_valid_anchor():
    proposer = AscendDeepSeekV4DSparkProposer.__new__(AscendDeepSeekV4DSparkProposer)
    proposer.device = torch.device("cpu")
    proposer.block_size = proposer.num_speculative_tokens = 2
    proposer.max_model_len = 64
    proposer.max_num_tokens = 2
    proposer.max_query_tokens = 2
    proposer.draft_attn_groups = [_group(0)]
    proposer._per_group_block_tables = {0: torch.arange(8, dtype=torch.int32).view(1, 8)}
    proposer.runner = None
    proposer._seed_buffer = torch.zeros(1, dtype=torch.int64)
    proposer.arange_dspark = torch.arange(8, dtype=torch.int32)
    positions = torch.tensor([4, 5], dtype=torch.int32)
    cad = SimpleNamespace(
        num_reqs=1,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
    )

    try:
        proposer.set_inputs_first_pass(
            torch.arange(2),
            torch.tensor([101]),
            positions,
            torch.zeros((2, 2)),
            None,
            cad,
            torch.tensor([2], dtype=torch.int32),
        )
    except ValueError as error:
        assert "valid anchor token" in str(error)
    else:
        raise AssertionError("expected invalid DSpark anchor metadata to fail")


def test_dspark_ignores_padded_rows_on_idle_dp_rank():
    proposer = AscendDeepSeekV4DSparkProposer.__new__(AscendDeepSeekV4DSparkProposer)
    proposer.device = torch.device("cpu")
    proposer.dp_rank = 1
    proposer.block_size = proposer.num_speculative_tokens = 2
    proposer.max_model_len = 64
    proposer.max_num_tokens = 2
    proposer.max_query_tokens = 2
    proposer.draft_attn_groups = [_group(0)]
    proposer._draft_attn_layer_names_ordered = ["draft.layer.0"]
    proposer._per_group_block_tables = {0: torch.arange(8, dtype=torch.int32).view(1, 8)}
    proposer._query_slot_buffers = {}
    proposer._context_slot_buffers = {}
    proposer.runner = SimpleNamespace(_num_reqs_across_dp=torch.tensor([1, 0]), dp_rank=1)
    proposer._seed_buffer = torch.zeros(1, dtype=torch.int64)
    proposer._token_to_req_buffer = torch.zeros(2, dtype=torch.int32)
    proposer._dflash_hidden_states = torch.zeros((2, 2))
    proposer._context_positions_buffer = torch.zeros(2, dtype=torch.int32)
    proposer.positions = torch.zeros(2, dtype=torch.int32)
    proposer.input_ids = torch.zeros(2, dtype=torch.int64)
    proposer.parallel_drafting_token_id = 99
    proposer.arange_dspark = torch.arange(8, dtype=torch.int32)
    positions = torch.tensor([4], dtype=torch.int32)
    cad = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([5], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([5], dtype=torch.int32),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=1,
        num_actual_tokens=1,
        num_input_tokens=1,
        max_query_len=1,
        actual_seq_lengths_q=[1],
        block_table_tensor=None,
        slot_mapping=torch.zeros(1, dtype=torch.int32),
        positions=positions,
        attn_state=AscendAttentionState.SpecDecoding,
        decode_token_per_req=1,
        max_seq_len=5,
    )

    num_tokens, indices, returned, _ = proposer.set_inputs_first_pass(
        torch.zeros(1, dtype=torch.int64),
        torch.tensor([101]),
        positions,
        torch.zeros((1, 2)),
        torch.tensor([-1057308040], dtype=torch.int32),
        cad,
        torch.tensor([1057308040], dtype=torch.int32),
    )

    assert num_tokens == 0
    assert indices.numel() == 0
    assert returned.num_reqs == 0
    assert returned.query_start_loc.tolist() == [0]


def test_sequential_markov_sampling_uses_previous_draft_token():
    class FakeModel:
        def compute_logits(self, hidden_states):
            return torch.zeros((hidden_states.shape[0], 5))

        def markov_embed(self, token_ids):
            return token_ids

        def markov_bias(self, token_ids):
            logits = torch.zeros((token_ids.shape[0], 5))
            logits.scatter_(1, ((token_ids + 1) % 5).unsqueeze(1), 10)
            return logits

    proposer = AscendDeepSeekV4DSparkProposer.__new__(AscendDeepSeekV4DSparkProposer)
    proposer.model = FakeModel()
    proposer.runner = SimpleNamespace(input_batch=SimpleNamespace(idx_mapping=None))
    proposer.block_size = 2
    proposer._probabilistic = False
    proposer._seed_buffer = torch.tensor([1, 3])
    proposer._draft_buffer = torch.zeros((2, 2), dtype=torch.int64)
    proposer.positions = torch.arange(4, dtype=torch.int32)

    tokens = proposer._sample_sequential(
        torch.zeros((4, 3)),
        2,
        SimpleNamespace(all_greedy=True),
    )

    torch.testing.assert_close(tokens, torch.tensor([[2, 3], [4, 0]]))
    assert proposer.take_last_draft_logits() is None


def test_probabilistic_sampling_caches_processed_draft_logits(monkeypatch):
    class FakeModel:
        def compute_logits(self, hidden_states):
            return torch.arange(hidden_states.shape[0] * 5, dtype=torch.float32).view(-1, 5)

        def markov_embed(self, token_ids):
            return token_ids

        def markov_bias(self, token_ids):
            return torch.zeros((token_ids.shape[0], 5))

    calls = []

    def fake_gumbel_sample(
        logits,
        idx_mapping,
        temperature,
        seeds,
        positions,
        *,
        output_processed_logits,
        output_processed_logits_col,
        **kwargs,
    ):
        del kwargs
        step = int(output_processed_logits_col)
        output_processed_logits[:, step].copy_(logits)
        calls.append((idx_mapping.clone(), temperature.clone(), seeds.clone(), positions.clone()))
        return logits.argmax(dim=-1)

    monkeypatch.setattr(dspark_proposer_module, "gumbel_sample", fake_gumbel_sample)
    proposer = AscendDeepSeekV4DSparkProposer.__new__(AscendDeepSeekV4DSparkProposer)
    proposer.model = FakeModel()
    proposer.runner = SimpleNamespace(input_batch=SimpleNamespace(idx_mapping=torch.tensor([1, 0])), sampler=None)
    proposer.vllm_config = SimpleNamespace(model_config=SimpleNamespace(seed=7))
    proposer.block_size = 2
    proposer._probabilistic = True
    proposer._seed_buffer = torch.tensor([1, 3])
    proposer._draft_buffer = torch.zeros((2, 2), dtype=torch.int64)
    proposer._sampling_seed_buffer = torch.zeros(2, dtype=torch.int64)
    proposer._idx_mapping_buffer = torch.arange(2, dtype=torch.int32)
    proposer.positions = torch.arange(4, dtype=torch.int32)
    proposer.arange_dspark = torch.arange(8, dtype=torch.int32)

    tokens = proposer._sample_sequential(
        torch.zeros((4, 3)),
        2,
        SimpleNamespace(
            all_greedy=False,
            temperature=torch.tensor([0.5, 0.75]),
            generators={},
        ),
    )
    draft_logits = proposer.take_last_draft_logits()

    torch.testing.assert_close(tokens, torch.tensor([[4, 4], [4, 4]]))
    assert draft_logits is not None
    torch.testing.assert_close(
        draft_logits,
        torch.arange(20, dtype=torch.float32).view(2, 2, 5),
    )
    assert len(calls) == 2
    torch.testing.assert_close(calls[0][0], torch.tensor([1, 0], dtype=torch.int32))
    torch.testing.assert_close(calls[0][1], torch.tensor([0.5, 0.75]))
    torch.testing.assert_close(calls[0][2], torch.tensor([7, 9980], dtype=torch.int64))


def test_runner_flattens_cached_logits_by_active_request():
    runner = SimpleNamespace(
        _draft_logits=torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4),
        _draft_logits_req_ids=["req-b", "req-a"],
        input_batch=SimpleNamespace(req_ids=["req-a", "req-c", "req-b"]),
    )
    metadata = SimpleNamespace(num_draft_tokens=[2, 0, 1])

    logits = NPUModelRunner._get_spec_decode_draft_logits(cast(Any, runner), metadata)

    expected = torch.cat((runner._draft_logits[1, :2], runner._draft_logits[0, :1]))
    torch.testing.assert_close(logits, expected)
