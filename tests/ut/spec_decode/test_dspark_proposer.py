# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from vllm.v1.worker.utils import AttentionGroup

import vllm_ascend.models.deepseek_v4_dspark as dspark_model_module
import vllm_ascend.spec_decode.dspark_proposer as dspark_proposer_module
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def test_dspark_perf_record_writes_jsonl(monkeypatch, tmp_path):
    path = tmp_path / "dspark_perf.jsonl"
    monkeypatch.setenv("VLLM_ASCEND_DSPARK_PERF_TRACE_PATH", str(path))
    monkeypatch.setenv("VLLM_ASCEND_DSPARK_PERF_TRACE_MAX_RECORDS", "1")
    monkeypatch.setenv("VLLM_ASCEND_DSPARK_PERF_TRACE_SYNC", "0")

    proposer = SimpleNamespace(device=torch.device("cpu"), _dspark_perf_trace_records=0)
    start_ns = dspark_proposer_module.time.perf_counter_ns()
    AscendDSparkProposer._write_dspark_perf_record(
        proposer,
        "stage_a",
        start_ns,
        num_tokens=2,
    )
    AscendDSparkProposer._write_dspark_perf_record(
        proposer,
        "stage_b",
        start_ns,
        num_tokens=3,
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["stage"] == "stage_a"
    assert record["num_tokens"] == 2
    assert record["elapsed_ms"] >= 0
    assert proposer._dspark_perf_trace_records == 1


class _FakeDSparkModel:
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids.to(torch.long)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        vocab_size = 5
        bias = torch.zeros(
            (markov_embed.numel(), vocab_size),
            dtype=torch.float32,
            device=markov_embed.device,
        )
        next_ids = (markov_embed.to(torch.long) + 1) % vocab_size
        bias.scatter_(1, next_ids.view(-1, 1), 10.0)
        return bias


def test_dspark_sample_sequential_uses_previous_draft_token(monkeypatch):
    monkeypatch.setattr(
        dspark_proposer_module,
        "greedy_sample",
        lambda logits: logits.argmax(dim=-1),
    )
    proposer = SimpleNamespace(
        num_speculative_tokens=3,
        model=_FakeDSparkModel(),
        _dspark_seed_buffer=torch.tensor([1, 3], dtype=torch.int64),
        _dspark_draft_buffer=torch.zeros((2, 3), dtype=torch.int64),
    )
    head_hidden = torch.zeros((6, 5), dtype=torch.float32)
    token_indices = torch.arange(6, dtype=torch.int32)

    draft_tokens = AscendDSparkProposer._sample_sequential(
        proposer,
        num_reqs=2,
        head_hidden=head_hidden,
        token_indices_to_sample=token_indices,
    )

    assert draft_tokens.data_ptr() == proposer._dspark_draft_buffer.data_ptr()
    torch.testing.assert_close(
        draft_tokens,
        torch.tensor([[2, 3, 4], [4, 0, 1]], dtype=torch.int64),
    )
    assert proposer._dspark_last_draft_logits is None
    assert proposer._dspark_last_draft_logit_components is None


def test_dspark_sample_sequential_full_vocab_greedy_uses_direct_argmax(monkeypatch):
    monkeypatch.setattr(
        dspark_proposer_module,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_reduce_sample=False),
    )

    def unexpected_tp_greedy(logits):
        raise AssertionError("full-vocab DSpark logits should use direct argmax")

    monkeypatch.setattr(dspark_proposer_module, "greedy_sample", unexpected_tp_greedy)
    model = SimpleNamespace(
        compute_logits=lambda hidden_states: hidden_states,
        markov_embed=lambda token_ids: token_ids,
        markov_bias=lambda markov_embed: torch.zeros((markov_embed.numel(), 4), dtype=torch.float32),
    )
    proposer = SimpleNamespace(
        num_speculative_tokens=1,
        model=model,
        _dspark_seed_buffer=torch.tensor([0], dtype=torch.int64),
        _dspark_draft_buffer=torch.zeros((1, 1), dtype=torch.int64),
    )
    head_hidden = torch.tensor([[0.0, 1.0, 8.0, 3.0]], dtype=torch.float32)

    draft_tokens = AscendDSparkProposer._sample_sequential(
        proposer,
        num_reqs=1,
        head_hidden=head_hidden,
        token_indices_to_sample=torch.tensor([0], dtype=torch.int32),
    )

    torch.testing.assert_close(draft_tokens, torch.tensor([[2]], dtype=torch.int64))


def test_dspark_sample_sequential_reduce_sample_uses_tp_greedy(monkeypatch):
    monkeypatch.setattr(
        dspark_proposer_module,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_reduce_sample=True),
    )
    calls = []

    def fake_tp_greedy(logits):
        calls.append(logits.clone())
        return torch.tensor([3], dtype=torch.int64)

    monkeypatch.setattr(dspark_proposer_module, "greedy_sample", fake_tp_greedy)
    model = SimpleNamespace(
        compute_logits=lambda hidden_states: hidden_states,
        markov_embed=lambda token_ids: token_ids,
        markov_bias=lambda markov_embed: torch.zeros((markov_embed.numel(), 4), dtype=torch.float32),
    )
    proposer = SimpleNamespace(
        num_speculative_tokens=1,
        model=model,
        _dspark_seed_buffer=torch.tensor([0], dtype=torch.int64),
        _dspark_draft_buffer=torch.zeros((1, 1), dtype=torch.int64),
    )

    draft_tokens = AscendDSparkProposer._sample_sequential(
        proposer,
        num_reqs=1,
        head_hidden=torch.tensor([[0.0, 8.0, 1.0, 3.0]], dtype=torch.float32),
        token_indices_to_sample=torch.tensor([0], dtype=torch.int32),
    )

    assert len(calls) == 1
    torch.testing.assert_close(draft_tokens, torch.tensor([[3]], dtype=torch.int64))


def test_dspark_sample_sequential_caches_greedy_logits_for_accept_debug(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DSPARK_ACCEPT_DEBUG_PATH", "/tmp/dspark_accept_debug.jsonl")
    monkeypatch.setenv("VLLM_ASCEND_DSPARK_ACCEPT_DEBUG_TOPK", "2")
    monkeypatch.setattr(
        dspark_proposer_module,
        "greedy_sample",
        lambda logits: logits.argmax(dim=-1),
    )
    proposer = SimpleNamespace(
        num_speculative_tokens=2,
        model=_FakeDSparkModel(),
        _dspark_seed_buffer=torch.tensor([1], dtype=torch.int64),
        _dspark_draft_buffer=torch.zeros((1, 2), dtype=torch.int64),
        _dspark_last_draft_logits=None,
        _dspark_last_draft_probs=None,
        _dspark_last_draft_logit_components=None,
    )
    head_hidden = torch.zeros((2, 5), dtype=torch.float32)

    AscendDSparkProposer._sample_sequential(
        proposer,
        num_reqs=1,
        head_hidden=head_hidden,
        token_indices_to_sample=torch.arange(2, dtype=torch.int32),
    )

    assert proposer._dspark_last_draft_logits.shape == (1, 2, 5)
    assert proposer._dspark_last_draft_logits.is_contiguous()
    components = proposer._dspark_last_draft_logit_components
    assert components is not None
    torch.testing.assert_close(components["prev_token_ids"], torch.tensor([[1, 2]], dtype=torch.int64))
    torch.testing.assert_close(components["base_logit_at_draft"], torch.zeros((1, 2), dtype=torch.float32))
    torch.testing.assert_close(components["markov_bias_at_draft"], torch.full((1, 2), 10.0))
    torch.testing.assert_close(components["final_logit_at_draft"], torch.full((1, 2), 10.0))
    torch.testing.assert_close(components["base_rank_of_draft"], torch.ones((1, 2), dtype=torch.int32))
    torch.testing.assert_close(components["markov_bias_rank_of_draft"], torch.ones((1, 2), dtype=torch.int32))
    torch.testing.assert_close(components["final_rank_of_draft"], torch.ones((1, 2), dtype=torch.int32))
    assert components["final_top_ids"].shape == (1, 2, 2)
    torch.testing.assert_close(components["final_top_ids"][0, :, 0], torch.tensor([2, 3], dtype=torch.int64))
    torch.testing.assert_close(components["final_top_values"][0, :, 0], torch.full((2,), 10.0))
    take_components = AscendDSparkProposer.take_last_draft_logit_components.__get__(proposer)
    assert take_components() is components
    assert proposer._dspark_last_draft_logit_components is None


def test_dspark_sample_sequential_caches_probabilistic_draft_distribution(monkeypatch):
    calls = []

    def fake_gumbel_sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature,
        output_processed_logits=None,
        output_processed_logits_col=None,
        use_fp64=False,
    ):
        del use_fp64
        calls.append(
            {
                "expanded_idx_mapping": expanded_idx_mapping.clone(),
                "temperature": temperature.clone(),
                "seed": seed.clone(),
                "pos": pos.clone(),
                "apply_temperature": apply_temperature,
                "col": output_processed_logits_col.clone(),
            }
        )
        processed = logits.float()
        random_mask = temperature[expanded_idx_mapping.long()] != 0
        if random_mask.any():
            processed = processed.clone()
            processed[random_mask] = processed[random_mask] / temperature[expanded_idx_mapping.long()][
                random_mask
            ].unsqueeze(-1)
        if output_processed_logits is not None:
            col = int(output_processed_logits_col.item())
            output_processed_logits[:, col, :].copy_(processed)
        return processed.argmax(dim=-1)

    monkeypatch.setattr(dspark_proposer_module, "gumbel_sample", fake_gumbel_sample)

    proposer = SimpleNamespace(
        num_speculative_tokens=2,
        model=_FakeDSparkModel(),
        _dspark_seed_buffer=torch.tensor([1, 3], dtype=torch.int64),
        _dspark_sampling_seed_buffer=torch.zeros(2, dtype=torch.int64),
        _dspark_idx_mapping_buffer=torch.arange(2, dtype=torch.int32),
        _dspark_draft_buffer=torch.zeros((2, 2), dtype=torch.int64),
        _dspark_probabilistic=True,
        _dspark_last_draft_logits=None,
        _dspark_last_draft_probs=None,
        use_fp64_gumbel=False,
        positions=torch.tensor([10, 11, 20, 21], dtype=torch.int32),
    )
    proposer._get_draft_sampling_temperature = AscendDSparkProposer._get_draft_sampling_temperature.__get__(proposer)
    proposer._get_runner_sampling_state_seeds = AscendDSparkProposer._get_runner_sampling_state_seeds.__get__(proposer)
    proposer._get_draft_sampling_seeds = AscendDSparkProposer._get_draft_sampling_seeds.__get__(proposer)
    proposer._get_draft_idx_mapping = AscendDSparkProposer._get_draft_idx_mapping.__get__(proposer)
    proposer._sample_draft_ids = AscendDSparkProposer._sample_draft_ids.__get__(proposer)
    head_hidden = torch.zeros((4, 5), dtype=torch.float32)
    token_indices = torch.arange(4, dtype=torch.int32)
    gen0 = torch.Generator(device="cpu").manual_seed(123)
    gen1 = torch.Generator(device="cpu").manual_seed(456)
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        temperature=torch.tensor([1.0, 2.0], dtype=torch.float32),
        generators={0: gen0, 1: gen1},
    )

    draft_tokens = AscendDSparkProposer._sample_sequential(
        proposer,
        num_reqs=2,
        head_hidden=head_hidden,
        token_indices_to_sample=token_indices,
        sampling_metadata=sampling_metadata,
    )

    torch.testing.assert_close(
        draft_tokens,
        torch.tensor([[2, 3], [4, 0]], dtype=torch.int64),
    )
    assert proposer._dspark_last_draft_logits.shape == (2, 2, 5)
    assert proposer._dspark_last_draft_logits.is_contiguous()
    assert proposer._dspark_last_draft_probs is None
    assert proposer._dspark_last_draft_logit_components is None
    assert len(calls) == 2
    torch.testing.assert_close(
        calls[0]["expanded_idx_mapping"],
        torch.tensor([0, 1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        calls[0]["temperature"],
        torch.tensor([1.0, 2.0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        calls[0]["seed"],
        torch.tensor([123, 456], dtype=torch.int64),
    )
    torch.testing.assert_close(
        calls[0]["pos"],
        torch.tensor([9, 19], dtype=torch.int32),
    )
    torch.testing.assert_close(
        calls[1]["pos"],
        torch.tensor([10, 20], dtype=torch.int32),
    )
    assert calls[0]["apply_temperature"] is True
    assert int(calls[0]["col"].item()) == 0
    assert int(calls[1]["col"].item()) == 1
    torch.testing.assert_close(
        proposer._dspark_last_draft_logits[0],
        torch.tensor(
            [
                [0.0, 0.0, 10.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 10.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        proposer._dspark_last_draft_logits[1],
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 5.0],
                [5.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_dspark_runner_flattens_draft_logit_components_by_req_id():
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["req-b", "req-a"]),
        _draft_logit_components={
            "prev_token_ids": torch.tensor([[10, 11], [20, 21]], dtype=torch.int64),
            "final_top_ids": torch.tensor(
                [
                    [[100, 101], [110, 111]],
                    [[200, 201], [210, 211]],
                ],
                dtype=torch.int64,
            ),
        },
        _draft_logit_components_req_ids=["req-a", "req-b"],
    )
    runner._get_spec_decode_draft_distribution = NPUModelRunner._get_spec_decode_draft_distribution.__get__(runner)
    runner._get_spec_decode_draft_logit_components = NPUModelRunner._get_spec_decode_draft_logit_components.__get__(
        runner
    )
    metadata = SimpleNamespace(num_draft_tokens=[2, 1])

    components = runner._get_spec_decode_draft_logit_components(metadata)

    assert components is not None
    torch.testing.assert_close(components["prev_token_ids"], torch.tensor([20, 21, 10], dtype=torch.int64))
    torch.testing.assert_close(
        components["final_top_ids"],
        torch.tensor([[200, 201], [210, 211], [100, 101]], dtype=torch.int64),
    )


def test_dspark_draft_idx_mapping_uses_runner_input_batch_mapping():
    proposer = SimpleNamespace(
        runner=SimpleNamespace(
            input_batch=SimpleNamespace(
                idx_mapping=torch.tensor([3, 1, 2], dtype=torch.int64),
            )
        ),
        _dspark_idx_mapping_buffer=torch.arange(3, dtype=torch.int32),
    )

    idx_mapping = AscendDSparkProposer._get_draft_idx_mapping(
        proposer,
        2,
        torch.device("cpu"),
    )

    torch.testing.assert_close(idx_mapping, torch.tensor([3, 1], dtype=torch.int32))


def test_dspark_set_inputs_first_pass_uses_anchor_first_block(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", "1")
    device = torch.device("cpu")
    block_size = 3
    batch_size = 2
    proposer = SimpleNamespace(
        device=device,
        num_speculative_tokens=block_size,
        parallel_drafting_token_id=99,
        kernel_block_size=8,
        token_arange_np=np.arange(16, dtype=np.int32),
        arange_dspark=torch.arange(32, dtype=torch.int32),
        input_ids=torch.zeros(batch_size * block_size, dtype=torch.int64),
        positions=torch.zeros(batch_size * block_size, dtype=torch.int32),
        _slot_mapping_buffer=torch.zeros(batch_size * block_size, dtype=torch.int32),
        _request_slots_buffer=torch.zeros(batch_size * block_size, dtype=torch.int32),
        _dspark_seed_buffer=torch.full((4,), -1, dtype=torch.int64),
        _dflash_hidden_states=torch.zeros(8, 2, dtype=torch.float32),
        _context_positions_buffer=torch.zeros(8, dtype=torch.int32),
        _context_request_slots_buffer=torch.zeros(8, dtype=torch.int32),
        _context_slot_mapping_buffer=torch.zeros(8, dtype=torch.int32),
    )
    proposer._assign_request_slots = lambda size: [4, 5]
    proposer._get_draft_block_table = AscendDSparkProposer._get_draft_block_table.__get__(proposer)
    proposer._slot_mapping_from_block_table = AscendDSparkProposer._slot_mapping_from_block_table.__get__(proposer)

    target_positions = torch.tensor([5, 6, 7, 15, 16, 17], dtype=torch.int32)
    target_hidden_states = torch.arange(12, dtype=torch.float32).view(6, 2)
    next_token_ids = torch.tensor([101, 202], dtype=torch.int64)
    # The CPU mirror is the authoritative host-side range source for DSpark
    # input preparation; the device mirror is intentionally stale here.
    cad = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2, 6], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 3, 6], dtype=torch.int32),
        seq_lens=torch.tensor([8, 18], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([8, 18], dtype=torch.int32),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=batch_size,
        num_actual_tokens=6,
        num_input_tokens=6,
        max_query_len=3,
        actual_seq_lengths_q=[3, 3],
        block_table_tensor=torch.tensor(
            [
                [10, 20, 30],
                [11, 21, 31],
            ],
            dtype=torch.int32,
        ),
        slot_mapping=torch.tensor([100, 101, 102, 103, 104, 105], dtype=torch.int32),
        positions=target_positions,
        attn_state=AscendAttentionState.SpecDecoding,
        decode_token_per_req=1,
        max_seq_len=18,
    )

    num_query_total, token_indices, returned_cad, maybe_graph_input = AscendDSparkProposer.set_inputs_first_pass(
        proposer,
        target_token_ids=torch.arange(6, dtype=torch.int64),
        next_token_ids=next_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        token_indices_to_sample=None,
        cad=cad,
        num_rejected_tokens_gpu=None,
    )

    assert returned_cad is cad
    assert maybe_graph_input is None
    assert num_query_total == batch_size * block_size
    torch.testing.assert_close(token_indices, torch.arange(6, dtype=torch.int32))
    torch.testing.assert_close(proposer._dspark_seed_buffer[:4], torch.tensor([101, 202, 0, 0]))
    torch.testing.assert_close(
        proposer.input_ids,
        torch.tensor([101, 99, 99, 202, 99, 99], dtype=torch.int64),
    )
    torch.testing.assert_close(
        proposer.positions,
        torch.tensor([8, 9, 10, 18, 19, 20], dtype=torch.int32),
    )
    torch.testing.assert_close(
        proposer._request_slots_buffer,
        torch.tensor([4, 4, 4, 5, 5, 5], dtype=torch.int32),
    )
    torch.testing.assert_close(
        cad.slot_mapping,
        torch.tensor([160, 161, 162, 250, 251, 252], dtype=torch.int32),
    )
    torch.testing.assert_close(cad.query_start_loc, torch.tensor([0, 3, 6], dtype=torch.int32))
    torch.testing.assert_close(cad.query_start_loc_cpu, torch.tensor([0, 3, 6], dtype=torch.int32))
    torch.testing.assert_close(proposer._dspark_query_start_loc, cad.query_start_loc)
    torch.testing.assert_close(proposer._dspark_query_start_loc_cpu, cad.query_start_loc_cpu)
    torch.testing.assert_close(cad.seq_lens, torch.tensor([11, 21], dtype=torch.int32))
    assert cad.max_query_len == block_size
    assert cad.max_seq_len == 21
    assert cad.num_actual_tokens == 6
    assert cad.causal is False
    assert cad.attn_mask is None
    assert cad.attn_state == AscendAttentionState.ChunkedPrefill
    assert cad.decode_token_per_req == block_size
    assert cad.actual_seq_lengths_q == [block_size, block_size]
    torch.testing.assert_close(proposer._dflash_hidden_states[:6], target_hidden_states)
    torch.testing.assert_close(proposer._context_positions_buffer[:6], target_positions)
    torch.testing.assert_close(
        proposer._context_request_slots_buffer[:6],
        torch.tensor([4, 4, 4, 5, 5, 5], dtype=torch.int32),
    )
    torch.testing.assert_close(
        proposer._context_slot_mapping_buffer[:6],
        torch.tensor([100, 101, 102, 103, 104, 105], dtype=torch.int32),
    )
    assert proposer._dflash_num_context == 6


def test_dspark_slot_mapping_from_block_table_keeps_explicit_slots_for_empty_request():
    block_size = 4
    proposer = SimpleNamespace(kernel_block_size=block_size)
    slot_mapping_from_block_table = AscendDSparkProposer._slot_mapping_from_block_table.__get__(proposer)
    block_table = torch.tensor(
        [
            [12, 3, 14],
            [99, 88, 77],
            [42, 7, 64],
        ],
        dtype=torch.int32,
    )

    first_req_slots = slot_mapping_from_block_table(
        torch.tensor([1, 4, 7, 9], dtype=torch.int32),
        0,
        block_table,
    )
    empty_req_slots = slot_mapping_from_block_table(
        torch.empty(0, dtype=torch.int32),
        1,
        block_table,
    )
    third_req_slots = slot_mapping_from_block_table(
        torch.tensor([0, 5, 10], dtype=torch.int32),
        2,
        block_table,
    )

    torch.testing.assert_close(first_req_slots, torch.tensor([49, 12, 15, 57], dtype=torch.int32))
    assert empty_req_slots.dtype == torch.int32
    assert empty_req_slots.numel() == 0
    torch.testing.assert_close(third_req_slots, torch.tensor([168, 29, 258], dtype=torch.int32))


def test_dspark_set_inputs_first_pass_masks_tokens_past_max_model_len(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", "1")
    device = torch.device("cpu")
    block_size = 3
    proposer = SimpleNamespace(
        device=device,
        num_speculative_tokens=block_size,
        parallel_drafting_token_id=99,
        kernel_block_size=8,
        max_model_len=10,
        token_arange_np=np.arange(16, dtype=np.int32),
        arange_dspark=torch.arange(32, dtype=torch.int32),
        input_ids=torch.zeros(block_size, dtype=torch.int64),
        positions=torch.zeros(block_size, dtype=torch.int32),
        _slot_mapping_buffer=torch.zeros(block_size, dtype=torch.int32),
        _request_slots_buffer=torch.zeros(block_size, dtype=torch.int32),
        _dspark_seed_buffer=torch.full((2,), -1, dtype=torch.int64),
        _dflash_hidden_states=torch.zeros(4, 2, dtype=torch.float32),
        _context_positions_buffer=torch.zeros(4, dtype=torch.int32),
        _context_request_slots_buffer=torch.zeros(4, dtype=torch.int32),
        _context_slot_mapping_buffer=torch.zeros(4, dtype=torch.int32),
    )
    proposer._assign_request_slots = lambda size: [4]
    proposer._get_draft_block_table = AscendDSparkProposer._get_draft_block_table.__get__(proposer)
    proposer._slot_mapping_from_block_table = AscendDSparkProposer._slot_mapping_from_block_table.__get__(proposer)

    target_positions = torch.tensor([6, 7, 8], dtype=torch.int32)
    cad = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 3], dtype=torch.int32),
        seq_lens=torch.tensor([9], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([9], dtype=torch.int32),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=1,
        num_actual_tokens=3,
        num_input_tokens=3,
        max_query_len=3,
        actual_seq_lengths_q=[3],
        block_table_tensor=torch.tensor([[10, 20]], dtype=torch.int32),
        slot_mapping=torch.tensor([100, 101, 102], dtype=torch.int32),
        positions=target_positions,
        attn_state=AscendAttentionState.SpecDecoding,
        decode_token_per_req=1,
        max_seq_len=9,
    )

    AscendDSparkProposer.set_inputs_first_pass(
        proposer,
        target_token_ids=torch.arange(3, dtype=torch.int64),
        next_token_ids=torch.tensor([101], dtype=torch.int64),
        target_positions=target_positions,
        target_hidden_states=torch.arange(6, dtype=torch.float32).view(3, 2),
        token_indices_to_sample=None,
        cad=cad,
        num_rejected_tokens_gpu=None,
    )

    torch.testing.assert_close(proposer.positions, torch.tensor([9, 0, 0], dtype=torch.int32))
    torch.testing.assert_close(cad.slot_mapping, torch.tensor([161, -1, -1], dtype=torch.int32))
    torch.testing.assert_close(cad.seq_lens, torch.tensor([10], dtype=torch.int32))
    assert cad.max_seq_len == 10


def test_dspark_standard_dsa_uses_draft_group_block_table(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)

    class FakeBlockTable:
        def __init__(self, table):
            self._table = table

        def get_device_tensor(self):
            return self._table

    class FakeMultiGroupBlockTable:
        def __init__(self, tables):
            self._tables = tables

        def __getitem__(self, idx):
            return self._tables[idx]

    device = torch.device("cpu")
    block_size = 3
    batch_size = 2
    draft_block_table = torch.tensor(
        [
            [30, 40, 50],
            [31, 41, 51],
        ],
        dtype=torch.int32,
    )
    proposer = SimpleNamespace(
        device=device,
        num_speculative_tokens=block_size,
        parallel_drafting_token_id=99,
        kernel_block_size=8,
        kv_cache_gid=1,
        runner=SimpleNamespace(
            input_batch=SimpleNamespace(
                block_table=FakeMultiGroupBlockTable(
                    [
                        FakeBlockTable(torch.full((batch_size, 3), 9, dtype=torch.int32)),
                        FakeBlockTable(draft_block_table),
                    ]
                )
            )
        ),
        token_arange_np=np.arange(16, dtype=np.int32),
        arange_dspark=torch.arange(32, dtype=torch.int32),
        input_ids=torch.zeros(batch_size * block_size, dtype=torch.int64),
        positions=torch.zeros(batch_size * block_size, dtype=torch.int32),
        _slot_mapping_buffer=torch.zeros(batch_size * block_size, dtype=torch.int32),
        _request_slots_buffer=torch.zeros(batch_size * block_size, dtype=torch.int32),
        _dspark_seed_buffer=torch.full((4,), -1, dtype=torch.int64),
        _dflash_hidden_states=torch.zeros(8, 2, dtype=torch.float32),
        _context_positions_buffer=torch.zeros(8, dtype=torch.int32),
        _context_request_slots_buffer=torch.zeros(8, dtype=torch.int32),
        _context_slot_mapping_buffer=torch.zeros(8, dtype=torch.int32),
        draft_attn_groups=[
            AttentionGroup(
                _FakeBackend,
                ["draft.swa"],
                SimpleNamespace(block_size=8),
                1,
                [_FakeMetadataBuilder(SimpleNamespace(block_size=8), ["draft.swa"], SimpleNamespace(), device)],
            )
        ],
    )
    proposer._assign_request_slots = lambda size: [4, 5]
    proposer._get_draft_block_table = AscendDSparkProposer._get_draft_block_table.__get__(proposer)
    proposer._get_draft_block_table_for_gid = AscendDSparkProposer._get_draft_block_table_for_gid.__get__(proposer)
    proposer._get_draft_block_tables = AscendDSparkProposer._get_draft_block_tables.__get__(proposer)
    proposer._layer_map_from_gid_map = AscendDSparkProposer._layer_map_from_gid_map.__get__(proposer)
    proposer._slot_mapping_buffer_for_gid = AscendDSparkProposer._slot_mapping_buffer_for_gid.__get__(proposer)
    proposer._slot_mapping_from_block_table = AscendDSparkProposer._slot_mapping_from_block_table.__get__(proposer)

    target_positions = torch.tensor([5, 6, 7, 15, 16, 17], dtype=torch.int32)
    cad = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 3, 6], dtype=torch.int32),
        seq_lens=torch.tensor([8, 18], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([8, 18], dtype=torch.int32),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=batch_size,
        num_actual_tokens=6,
        num_input_tokens=6,
        max_query_len=3,
        actual_seq_lengths_q=[3, 3],
        block_table_tensor=torch.full((batch_size, 3), 99, dtype=torch.int32),
        slot_mapping=torch.arange(100, 106, dtype=torch.int32),
        positions=target_positions,
        attn_state=AscendAttentionState.SpecDecoding,
        decode_token_per_req=1,
        max_seq_len=18,
    )

    AscendDSparkProposer.set_inputs_first_pass(
        proposer,
        target_token_ids=torch.arange(6, dtype=torch.int64),
        next_token_ids=torch.tensor([101, 202], dtype=torch.int64),
        target_positions=target_positions,
        target_hidden_states=torch.arange(12, dtype=torch.float32).view(6, 2),
        token_indices_to_sample=None,
        cad=cad,
        num_rejected_tokens_gpu=None,
    )

    torch.testing.assert_close(proposer._dspark_block_table, draft_block_table)
    torch.testing.assert_close(
        proposer._context_slot_mapping_buffer[:6],
        torch.tensor([245, 246, 247, 335, 408, 409], dtype=torch.int32),
    )
    torch.testing.assert_close(
        cad.slot_mapping,
        torch.tensor([320, 321, 322, 410, 411, 412], dtype=torch.int32),
    )
    assert proposer._dspark_block_tables_by_layer["draft.swa"] is proposer._dspark_block_table


def test_dspark_standard_dsa_keeps_compact_block_table_order(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)

    class FakeBlockTable:
        def __init__(self, table):
            self._table = table

        def get_device_tensor(self):
            return self._table

    class FakeMultiGroupBlockTable:
        def __init__(self, tables):
            self._tables = tables

        def __getitem__(self, idx):
            return self._tables[idx]

    device = torch.device("cpu")
    block_size = 3
    draft_block_table = torch.tensor(
        [
            [1, 2, 3],
            [10, 11, 12],
            [30, 40, 50],
        ],
        dtype=torch.int32,
    )
    proposer = SimpleNamespace(
        device=device,
        num_speculative_tokens=block_size,
        parallel_drafting_token_id=99,
        kernel_block_size=8,
        kv_cache_gid=1,
        runner=SimpleNamespace(
            input_batch=SimpleNamespace(
                req_ids=["live"],
                req_id_to_index={"live": 2},
                block_table=FakeMultiGroupBlockTable(
                    [
                        FakeBlockTable(torch.full((3, 3), 9, dtype=torch.int32)),
                        FakeBlockTable(draft_block_table),
                    ]
                ),
            )
        ),
        token_arange_np=np.arange(16, dtype=np.int32),
        arange_dspark=torch.arange(32, dtype=torch.int32),
        input_ids=torch.zeros(block_size, dtype=torch.int64),
        positions=torch.zeros(block_size, dtype=torch.int32),
        _slot_mapping_buffer=torch.zeros(block_size, dtype=torch.int32),
        _request_slots_buffer=torch.zeros(block_size, dtype=torch.int32),
        _dspark_seed_buffer=torch.full((2,), -1, dtype=torch.int64),
        _dflash_hidden_states=torch.zeros(4, 2, dtype=torch.float32),
        _context_positions_buffer=torch.zeros(4, dtype=torch.int32),
        _context_request_slots_buffer=torch.zeros(4, dtype=torch.int32),
        _context_slot_mapping_buffer=torch.zeros(4, dtype=torch.int32),
        draft_attn_groups=[
            AttentionGroup(
                _FakeBackend,
                ["draft.swa"],
                SimpleNamespace(block_size=8),
                1,
                [_FakeMetadataBuilder(SimpleNamespace(block_size=8), ["draft.swa"], SimpleNamespace(), device)],
            )
        ],
    )
    proposer._assign_request_slots = lambda size: [4]
    proposer._get_draft_block_table = AscendDSparkProposer._get_draft_block_table.__get__(proposer)
    proposer._get_draft_block_table_for_gid = AscendDSparkProposer._get_draft_block_table_for_gid.__get__(proposer)
    proposer._get_draft_block_tables = AscendDSparkProposer._get_draft_block_tables.__get__(proposer)
    proposer._layer_map_from_gid_map = AscendDSparkProposer._layer_map_from_gid_map.__get__(proposer)
    proposer._slot_mapping_buffer_for_gid = AscendDSparkProposer._slot_mapping_buffer_for_gid.__get__(proposer)
    proposer._slot_mapping_from_block_table = AscendDSparkProposer._slot_mapping_from_block_table.__get__(proposer)

    target_positions = torch.tensor([5, 6], dtype=torch.int32)
    cad = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([7], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([7], dtype=torch.int32),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=1,
        num_actual_tokens=2,
        num_input_tokens=2,
        max_query_len=2,
        actual_seq_lengths_q=[2],
        block_table_tensor=torch.full((1, 3), 99, dtype=torch.int32),
        slot_mapping=torch.arange(100, 102, dtype=torch.int32),
        positions=target_positions,
        attn_state=AscendAttentionState.SpecDecoding,
        decode_token_per_req=1,
        max_seq_len=7,
    )

    AscendDSparkProposer.set_inputs_first_pass(
        proposer,
        target_token_ids=torch.arange(2, dtype=torch.int64),
        next_token_ids=torch.tensor([101], dtype=torch.int64),
        target_positions=target_positions,
        target_hidden_states=torch.arange(4, dtype=torch.float32).view(2, 2),
        token_indices_to_sample=None,
        cad=cad,
        num_rejected_tokens_gpu=None,
    )

    torch.testing.assert_close(proposer._dspark_block_table, draft_block_table[:1])
    torch.testing.assert_close(
        proposer._context_slot_mapping_buffer[:2],
        torch.tensor([13, 14], dtype=torch.int32),
    )
    torch.testing.assert_close(cad.slot_mapping, torch.tensor([15, 16, 17], dtype=torch.int32))


def test_dspark_standard_dsa_prefers_runner_per_group_metadata(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)

    class FakeBlockTable:
        def __init__(self, table):
            self._table = table

        def get_device_tensor(self):
            return self._table

    class FakeMultiGroupBlockTable:
        def __init__(self, tables):
            self._tables = tables

        def __getitem__(self, idx):
            return self._tables[idx]

    device = torch.device("cpu")
    proposer = SimpleNamespace(
        device=device,
        num_speculative_tokens=2,
        parallel_drafting_token_id=99,
        kernel_block_size=4,
        kv_cache_gid=1,
        runner=SimpleNamespace(
            input_batch=SimpleNamespace(
                block_table=FakeMultiGroupBlockTable(
                    [
                        FakeBlockTable(torch.full((1, 3), 7, dtype=torch.int32)),
                        FakeBlockTable(torch.zeros((1, 3), dtype=torch.int32)),
                    ]
                )
            )
        ),
        token_arange_np=np.arange(16, dtype=np.int32),
        arange_dspark=torch.arange(32, dtype=torch.int32),
        input_ids=torch.zeros(2, dtype=torch.int64),
        positions=torch.zeros(2, dtype=torch.int32),
        _slot_mapping_buffer=torch.zeros(2, dtype=torch.int32),
        _request_slots_buffer=torch.zeros(2, dtype=torch.int32),
        _dspark_seed_buffer=torch.full((2,), -1, dtype=torch.int64),
        _dflash_hidden_states=torch.zeros(4, 2, dtype=torch.float32),
        _context_positions_buffer=torch.zeros(4, dtype=torch.int32),
        _context_request_slots_buffer=torch.zeros(4, dtype=torch.int32),
        _context_slot_mapping_buffer=torch.zeros(4, dtype=torch.int32),
        _dspark_per_group_block_tables={},
        _dspark_per_group_slot_mappings={},
        _dspark_query_slot_mapping_buffers={},
        _dspark_context_slot_mapping_buffers={},
        draft_attn_groups=[
            AttentionGroup(
                _FakeBackend,
                ["draft.swa"],
                SimpleNamespace(block_size=4),
                1,
                [_FakeMetadataBuilder(SimpleNamespace(block_size=4), ["draft.swa"], SimpleNamespace(), device)],
            )
        ],
    )
    proposer._assign_request_slots = lambda size: [3]
    proposer.set_per_group_attn_metadata = AscendDSparkProposer.set_per_group_attn_metadata.__get__(proposer)
    proposer._get_draft_block_table = AscendDSparkProposer._get_draft_block_table.__get__(proposer)
    proposer._get_draft_block_table_for_gid = AscendDSparkProposer._get_draft_block_table_for_gid.__get__(proposer)
    proposer._get_draft_block_tables = AscendDSparkProposer._get_draft_block_tables.__get__(proposer)
    proposer._layer_map_from_gid_map = AscendDSparkProposer._layer_map_from_gid_map.__get__(proposer)
    proposer._slot_mapping_buffer_for_gid = AscendDSparkProposer._slot_mapping_buffer_for_gid.__get__(proposer)
    proposer._slot_mapping_from_block_table = AscendDSparkProposer._slot_mapping_from_block_table.__get__(proposer)

    runner_block_table = torch.tensor([[10, 11, 12]], dtype=torch.int32)
    runner_slot_mapping = torch.tensor([500, 501], dtype=torch.int32)
    proposer.set_per_group_attn_metadata(1, runner_block_table, runner_slot_mapping)

    target_positions = torch.tensor([5, 6], dtype=torch.int32)
    cad = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([7], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([7], dtype=torch.int32),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=1,
        num_actual_tokens=2,
        num_input_tokens=2,
        max_query_len=2,
        actual_seq_lengths_q=[2],
        block_table_tensor=torch.full((1, 3), 99, dtype=torch.int32),
        slot_mapping=torch.arange(2, dtype=torch.int32),
        positions=target_positions,
        attn_state=AscendAttentionState.SpecDecoding,
        decode_token_per_req=1,
        max_seq_len=7,
    )

    AscendDSparkProposer.set_inputs_first_pass(
        proposer,
        target_token_ids=torch.arange(2, dtype=torch.int64),
        next_token_ids=torch.tensor([101], dtype=torch.int64),
        target_positions=target_positions,
        target_hidden_states=torch.arange(4, dtype=torch.float32).view(2, 2),
        token_indices_to_sample=None,
        cad=cad,
        num_rejected_tokens_gpu=None,
    )

    torch.testing.assert_close(proposer._dspark_block_table, runner_block_table)
    assert proposer._dspark_block_table.data_ptr() != runner_block_table.data_ptr()
    torch.testing.assert_close(proposer._context_slot_mapping_buffer[:2], torch.tensor([45, 46], dtype=torch.int32))
    torch.testing.assert_close(cad.slot_mapping, torch.tensor([47, 48], dtype=torch.int32))
    assert proposer._dspark_block_tables_by_layer["draft.swa"] is proposer._dspark_block_table


def test_dspark_standard_dsa_block_table_uses_stable_buffer(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)

    class FakeBlockTable:
        def __init__(self, table):
            self.table = table

        def get_device_tensor(self, batch_size=None):
            del batch_size
            return self.table

    class FakeMultiGroupBlockTable:
        def __init__(self, block_table):
            self.block_table = block_table

        def __getitem__(self, idx):
            assert idx == 1
            return self.block_table

    fake_block_table = FakeBlockTable(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))
    proposer = SimpleNamespace(
        device=torch.device("cpu"),
        kv_cache_gid=1,
        runner=SimpleNamespace(input_batch=SimpleNamespace(block_table=FakeMultiGroupBlockTable(fake_block_table))),
        _dspark_max_request_slots=4,
        _dspark_block_table_buffers_by_gid={},
    )

    first = AscendDSparkProposer._get_draft_block_table_for_gid(
        proposer,
        SimpleNamespace(block_table_tensor=None),
        2,
        1,
    )
    assert first is not None
    first_ptr = first.data_ptr()
    torch.testing.assert_close(first, torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))

    fake_block_table.table = torch.tensor([[9, 8], [7, 6]], dtype=torch.int32)
    second = AscendDSparkProposer._get_draft_block_table_for_gid(
        proposer,
        SimpleNamespace(block_table_tensor=None),
        2,
        1,
    )

    assert second is not None
    assert second.data_ptr() == first_ptr
    torch.testing.assert_close(second, torch.tensor([[9, 8], [7, 6]], dtype=torch.int32))
    torch.testing.assert_close(first, torch.tensor([[9, 8], [7, 6]], dtype=torch.int32))


def test_dspark_standard_dsa_block_table_uses_max_model_len_capacity(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)

    proposer = SimpleNamespace(
        max_model_len=256,
        kernel_block_size=16,
        draft_attn_groups=[
            SimpleNamespace(
                kv_cache_group_id=1,
                kv_cache_spec=SimpleNamespace(block_size=64),
            )
        ],
        _dspark_max_request_slots=2,
        _dspark_block_table_buffers_by_gid={},
    )

    first = AscendDSparkProposer._copy_dspark_block_table_for_gid(
        proposer,
        1,
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        2,
    )
    first_ptr = first.data_ptr()

    assert first.shape == (2, 4)
    torch.testing.assert_close(first[:, :2], torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))
    torch.testing.assert_close(first[:, 2:], torch.zeros((2, 2), dtype=torch.int32))

    second = AscendDSparkProposer._copy_dspark_block_table_for_gid(
        proposer,
        1,
        torch.tensor([[9, 8, 7], [6, 5, 4]], dtype=torch.int32),
        2,
    )

    assert second.data_ptr() == first_ptr
    assert second.shape == (2, 4)
    torch.testing.assert_close(second[:, :3], torch.tensor([[9, 8, 7], [6, 5, 4]], dtype=torch.int32))
    torch.testing.assert_close(second[:, 3:], torch.zeros((2, 1), dtype=torch.int32))


def test_dspark_standard_dsa_keeps_per_layer_block_tables(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)

    class FakeBlockTable:
        def __init__(self, table):
            self._table = table

        def get_device_tensor(self):
            return self._table

    class FakeMultiGroupBlockTable:
        def __init__(self, tables):
            self._tables = tables

        def __getitem__(self, idx):
            return self._tables[idx]

    device = torch.device("cpu")
    proposer = object.__new__(AscendDSparkProposer)
    proposer.device = device
    proposer.num_speculative_tokens = 2
    proposer.parallel_drafting_token_id = 99
    proposer.kernel_block_size = 4
    proposer.kv_cache_gid = 1
    proposer.max_num_tokens = 8
    proposer.max_query_tokens = 4
    proposer.token_arange_np = np.arange(16, dtype=np.int32)
    proposer.arange_dspark = torch.arange(32, dtype=torch.int32)
    proposer.input_ids = torch.zeros(4, dtype=torch.int64)
    proposer.positions = torch.zeros(4, dtype=torch.int32)
    proposer._slot_mapping_buffer = torch.zeros(4, dtype=torch.int32)
    proposer._request_slots_buffer = torch.zeros(4, dtype=torch.int32)
    proposer._dspark_seed_buffer = torch.full((2,), -1, dtype=torch.int64)
    proposer._dflash_hidden_states = torch.zeros(8, 2, dtype=torch.float32)
    proposer._context_positions_buffer = torch.zeros(8, dtype=torch.int32)
    proposer._context_request_slots_buffer = torch.zeros(8, dtype=torch.int32)
    proposer._context_slot_mapping_buffer = torch.zeros(8, dtype=torch.int32)
    proposer._dspark_query_slot_mapping_buffers = {}
    proposer._dspark_context_slot_mapping_buffers = {}
    proposer._assign_request_slots = lambda size: [0]

    group1_table = torch.tensor([[10, 11, 12]], dtype=torch.int32)
    group2_table = torch.tensor([[30, 31, 32, 33, 34]], dtype=torch.int32)
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            block_table=FakeMultiGroupBlockTable(
                [
                    FakeBlockTable(torch.full((1, 3), 7, dtype=torch.int32)),
                    FakeBlockTable(group1_table),
                    FakeBlockTable(group2_table),
                ]
            )
        )
    )
    proposer.draft_attn_groups = [
        AttentionGroup(
            _FakeBackend,
            ["layer.a"],
            SimpleNamespace(block_size=4),
            1,
            [_FakeMetadataBuilder(SimpleNamespace(block_size=4), ["layer.a"], SimpleNamespace(), device)],
        ),
        AttentionGroup(
            _FakeBackend,
            ["layer.b"],
            SimpleNamespace(block_size=2),
            2,
            [_FakeMetadataBuilder(SimpleNamespace(block_size=2), ["layer.b"], SimpleNamespace(), device)],
        ),
    ]

    target_positions = torch.tensor([5, 6], dtype=torch.int32)
    cad = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([7], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([7], dtype=torch.int32),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=1,
        num_actual_tokens=2,
        num_input_tokens=2,
        max_query_len=2,
        actual_seq_lengths_q=[2],
        block_table_tensor=torch.full((1, 3), 99, dtype=torch.int32),
        slot_mapping=torch.arange(2, dtype=torch.int32),
        positions=target_positions,
        attn_state=AscendAttentionState.SpecDecoding,
        decode_token_per_req=1,
        max_seq_len=7,
    )

    AscendDSparkProposer.set_inputs_first_pass(
        proposer,
        target_token_ids=torch.arange(2, dtype=torch.int64),
        next_token_ids=torch.tensor([101], dtype=torch.int64),
        target_positions=target_positions,
        target_hidden_states=torch.arange(4, dtype=torch.float32).view(2, 2),
        token_indices_to_sample=None,
        cad=cad,
        num_rejected_tokens_gpu=None,
    )

    torch.testing.assert_close(proposer._dspark_block_tables_by_layer["layer.a"], group1_table)
    torch.testing.assert_close(proposer._dspark_block_tables_by_layer["layer.b"], group2_table)
    torch.testing.assert_close(
        proposer._dspark_context_slot_mappings_by_layer["layer.a"],
        torch.tensor([45, 46], dtype=torch.int32),
    )
    torch.testing.assert_close(
        proposer._dspark_context_slot_mappings_by_layer["layer.b"],
        torch.tensor([65, 66], dtype=torch.int32),
    )
    torch.testing.assert_close(
        proposer._dspark_query_slot_mappings_by_layer["layer.a"][:2],
        torch.tensor([47, 48], dtype=torch.int32),
    )
    torch.testing.assert_close(
        proposer._dspark_query_slot_mappings_by_layer["layer.b"][:2],
        torch.tensor([67, 68], dtype=torch.int32),
    )
    torch.testing.assert_close(cad.slot_mapping, torch.tensor([47, 48], dtype=torch.int32))

    AscendDSparkProposer._pad_draft_query_buffers(proposer, num_actual_tokens=2, num_input_tokens=6)
    assert proposer._dspark_query_slot_mappings_by_layer["layer.b"].shape[0] == proposer.max_num_tokens
    torch.testing.assert_close(
        proposer._dspark_query_slot_mappings_by_layer["layer.b"][:6],
        torch.tensor([67, 68, -1, -1, -1, -1], dtype=torch.int32),
    )


def test_dspark_set_inputs_first_pass_stores_rejected_context_tokens(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", "1")
    device = torch.device("cpu")
    block_size = 3
    batch_size = 2
    proposer = SimpleNamespace(
        device=device,
        num_speculative_tokens=block_size,
        parallel_drafting_token_id=99,
        kernel_block_size=8,
        token_arange_np=np.arange(16, dtype=np.int32),
        arange_dspark=torch.arange(64, dtype=torch.int32),
        input_ids=torch.zeros(batch_size * block_size, dtype=torch.int64),
        positions=torch.zeros(batch_size * block_size, dtype=torch.int32),
        _slot_mapping_buffer=torch.zeros(batch_size * block_size, dtype=torch.int32),
        _request_slots_buffer=torch.zeros(batch_size * block_size, dtype=torch.int32),
        _dspark_seed_buffer=torch.full((4,), -1, dtype=torch.int64),
        _dflash_hidden_states=torch.zeros(8, 2, dtype=torch.float32),
        _context_positions_buffer=torch.zeros(8, dtype=torch.int32),
        _context_request_slots_buffer=torch.zeros(8, dtype=torch.int32),
        _context_slot_mapping_buffer=torch.zeros(8, dtype=torch.int32),
    )
    proposer._assign_request_slots = lambda size: [2, 7]
    proposer._get_draft_block_table = AscendDSparkProposer._get_draft_block_table.__get__(proposer)
    proposer._slot_mapping_from_block_table = AscendDSparkProposer._slot_mapping_from_block_table.__get__(proposer)

    target_positions = torch.tensor([10, 11, 12, 13, 30, 31, 32, 33], dtype=torch.int32)
    target_hidden_states = torch.arange(16, dtype=torch.float32).view(8, 2)
    next_token_ids = torch.tensor([101, 202], dtype=torch.int64)
    num_rejected_tokens = torch.tensor([1, 2], dtype=torch.int32)
    cad = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 3, 8], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
        seq_lens=torch.tensor([14, 34], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([14, 34], dtype=torch.int32),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=batch_size,
        num_actual_tokens=8,
        num_input_tokens=8,
        max_query_len=4,
        actual_seq_lengths_q=[4, 4],
        block_table_tensor=torch.tensor(
            [
                [10, 20, 30, 40, 50],
                [11, 21, 31, 41, 51],
            ],
            dtype=torch.int32,
        ),
        slot_mapping=torch.arange(200, 208, dtype=torch.int32),
        positions=target_positions,
        attn_state=AscendAttentionState.SpecDecoding,
        decode_token_per_req=1,
        max_seq_len=34,
    )

    num_query_total, token_indices, returned_cad, maybe_graph_input = AscendDSparkProposer.set_inputs_first_pass(
        proposer,
        target_token_ids=torch.arange(8, dtype=torch.int64),
        next_token_ids=next_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        token_indices_to_sample=None,
        cad=cad,
        num_rejected_tokens_gpu=num_rejected_tokens,
    )

    assert returned_cad is cad
    assert maybe_graph_input is None
    assert num_query_total == batch_size * block_size
    torch.testing.assert_close(token_indices, torch.arange(6, dtype=torch.int32))
    torch.testing.assert_close(
        proposer.input_ids,
        torch.tensor([101, 99, 99, 202, 99, 99], dtype=torch.int64),
    )
    torch.testing.assert_close(
        proposer.positions,
        torch.tensor([13, 14, 15, 32, 33, 34], dtype=torch.int32),
    )
    torch.testing.assert_close(
        cad.slot_mapping,
        torch.tensor([165, 166, 167, 408, 409, 410], dtype=torch.int32),
    )
    torch.testing.assert_close(cad.query_start_loc, torch.tensor([0, 3, 6], dtype=torch.int32))
    torch.testing.assert_close(cad.query_start_loc_cpu, torch.tensor([0, 3, 6], dtype=torch.int32))
    torch.testing.assert_close(proposer._dspark_query_start_loc, cad.query_start_loc)
    torch.testing.assert_close(proposer._dspark_query_start_loc_cpu, cad.query_start_loc_cpu)
    torch.testing.assert_close(cad.seq_lens, torch.tensor([16, 35], dtype=torch.int32))
    assert cad.max_query_len == block_size
    assert cad.max_seq_len == 37
    assert cad.num_actual_tokens == 6
    assert cad.causal is False
    assert cad.attn_state == AscendAttentionState.ChunkedPrefill
    torch.testing.assert_close(
        proposer._dflash_hidden_states[:8],
        target_hidden_states,
    )
    torch.testing.assert_close(
        proposer._context_positions_buffer[:8],
        target_positions,
    )
    torch.testing.assert_close(
        proposer._context_request_slots_buffer[:8],
        torch.tensor([2, 2, 2, 2, 7, 7, 7, 7], dtype=torch.int32),
    )
    torch.testing.assert_close(
        proposer._context_slot_mapping_buffer[:8],
        torch.arange(200, 208, dtype=torch.int32),
    )
    assert proposer._dflash_num_context == 8


def test_dspark_build_model_inputs_first_pass_returns_query_slot_mapping():
    calls = []

    class FakeModel:
        def precompute_and_store_context_kv(
            self,
            context_states,
            context_positions,
            context_slot_mapping,
            context_request_slots,
        ):
            calls.append(
                (
                    context_states,
                    context_positions,
                    context_slot_mapping,
                    context_request_slots,
                )
            )

    proposer = SimpleNamespace(
        _dflash_num_context=2,
        _dspark_slots_to_reset=[],
        model=FakeModel(),
        _dflash_hidden_states=torch.arange(6, dtype=torch.float32).view(3, 2),
        _context_positions_buffer=torch.tensor([5, 6, 7], dtype=torch.int32),
        _context_slot_mapping_buffer=torch.tensor([50, 51, 52], dtype=torch.int32),
        _context_request_slots_buffer=torch.tensor([2, 2, 3], dtype=torch.int32),
        input_ids=torch.tensor([101, 99, 99], dtype=torch.int64),
        positions=torch.tensor([8, 9, 10], dtype=torch.int32),
        _request_slots_buffer=torch.tensor([4, 4, 4], dtype=torch.int32),
        _slot_mapping_buffer=torch.tensor([160, 161, 162], dtype=torch.int32),
    )

    model_inputs = AscendDSparkProposer.build_model_inputs_first_pass(proposer, 3)

    assert len(calls) == 1
    context_states, context_positions, context_slot_mapping, context_request_slots = calls[0]
    torch.testing.assert_close(context_states, proposer._dflash_hidden_states[:2])
    torch.testing.assert_close(context_positions, proposer._context_positions_buffer[:2])
    torch.testing.assert_close(context_slot_mapping, proposer._context_slot_mapping_buffer[:2])
    torch.testing.assert_close(context_request_slots, proposer._context_request_slots_buffer[:2])
    assert model_inputs["input_ids"].data_ptr() == proposer.input_ids.data_ptr()
    torch.testing.assert_close(model_inputs["positions"], proposer.positions)
    torch.testing.assert_close(model_inputs["request_slots"], proposer._request_slots_buffer)
    torch.testing.assert_close(model_inputs["slot_mapping"], proposer._slot_mapping_buffer)


class _FakeKVSpec:
    block_size = 64


class _FakeMetadataBuilder:
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device
        self.calls = []

    def build_for_drafting(self, common_attn_metadata, draft_index, **kwargs):
        self.calls.append(
            {
                "positions": common_attn_metadata.positions.clone(),
                "slot_mapping": common_attn_metadata.slot_mapping.clone(),
                "block_table": getattr(common_attn_metadata, "block_table_tensor", None),
                "num_input_tokens": common_attn_metadata.num_input_tokens,
                "num_actual_tokens": common_attn_metadata.num_actual_tokens,
                "causal": common_attn_metadata.causal,
                "attn_state": common_attn_metadata.attn_state,
                "draft_index": draft_index,
                "block_size": kwargs.get("block_size"),
            }
        )
        return SimpleNamespace(tag="metadata")


class _FakeBackend:
    @classmethod
    def full_cls_name(cls):
        return "fake.Backend"

    @staticmethod
    def get_builder_cls():
        return _FakeMetadataBuilder


class _FakeLayer:
    def get_attn_backend(self):
        return _FakeBackend


def test_dspark_initialize_attn_backend_standard_dsa(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)
    monkeypatch.setattr(
        dspark_proposer_module,
        "get_layers_from_vllm_config",
        lambda *args, **kwargs: {
            "model.layers.61.self_attn.swa_cache": _FakeLayer(),
            "model.layers.62.self_attn.swa_cache": _FakeLayer(),
        },
    )

    kv_spec = _FakeKVSpec()
    proposer = SimpleNamespace(
        model=SimpleNamespace(
            get_draft_kv_cache_layer_names=lambda: [
                "model.layers.61.self_attn.swa_cache",
                "model.layers.62.self_attn.swa_cache",
            ]
        ),
        vllm_config=SimpleNamespace(),
        device=torch.device("cpu"),
        num_speculative_tokens=5,
        block_size=5,
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=[
                    "model.layers.0.self_attn.swa_cache",
                    "model.layers.61.self_attn.swa_cache",
                    "model.layers.62.self_attn.swa_cache",
                ],
                kv_cache_spec=kv_spec,
            )
        ]
    )

    AscendDSparkProposer.initialize_attn_backend(proposer, kv_cache_config)

    assert proposer.attn_layer_names == [
        "model.layers.61.self_attn.swa_cache",
        "model.layers.62.self_attn.swa_cache",
    ]
    assert len(proposer.draft_attn_groups) == 1
    group = proposer.draft_attn_groups[0]
    assert group.backend is _FakeBackend
    assert group.kv_cache_spec is kv_spec
    assert group.kv_cache_group_id == 0
    assert proposer.kernel_block_size == 64
    assert proposer.block_size == 5


def test_dspark_build_standard_dsa_metadata_sets_query_buffers():
    builder = _FakeMetadataBuilder(_FakeKVSpec(), ["draft.swa"], SimpleNamespace(), torch.device("cpu"))
    proposer = SimpleNamespace(
        positions=torch.tensor([10, 11, 12, 99], dtype=torch.int32),
        _slot_mapping_buffer=torch.tensor([100, 101, 102, 999], dtype=torch.int32),
        draft_attn_groups=[
            AttentionGroup(
                _FakeBackend,
                ["draft.swa"],
                builder.kv_cache_spec,
                0,
                [builder],
            )
        ],
    )
    common_metadata = SimpleNamespace(
        positions=torch.empty(0, dtype=torch.int32),
        slot_mapping=torch.empty(0, dtype=torch.int32),
        num_input_tokens=0,
        num_actual_tokens=0,
        causal=True,
        attn_state=None,
    )

    result = AscendDSparkProposer._build_standard_dsa_attn_metadata(
        proposer,
        common_metadata,
        num_input_tokens=4,
        num_actual_tokens=3,
    )

    assert len(result) == 1
    assert result[0]["draft.swa"].tag == "metadata"
    call = builder.calls[0]
    torch.testing.assert_close(call["positions"], torch.tensor([10, 11, 12, 0], dtype=torch.int32))
    torch.testing.assert_close(call["slot_mapping"], torch.tensor([100, 101, 102, -1], dtype=torch.int32))
    assert call["num_input_tokens"] == 4
    assert call["num_actual_tokens"] == 3
    assert call["causal"] is False
    assert call["attn_state"] == AscendAttentionState.ChunkedPrefill
    assert call["draft_index"] == 1
    assert call["block_size"] == 64


def test_dspark_standard_dsa_metadata_uses_stable_graph_buffers():
    class StableMetadataBuilder(_FakeMetadataBuilder):
        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            self.step = 0

        def build_for_drafting(self, common_attn_metadata, draft_index, **kwargs):
            del common_attn_metadata, draft_index, kwargs
            self.step += 1
            return SimpleNamespace(
                prefill=SimpleNamespace(
                    dspark_swa_indices=torch.full((4, 1, 8), self.step, dtype=torch.int32),
                    dspark_swa_lens=torch.full((4,), self.step, dtype=torch.int32),
                    sas_metadata=torch.full((3,), self.step, dtype=torch.int32),
                )
            )

    builder = StableMetadataBuilder(_FakeKVSpec(), ["draft.swa"], SimpleNamespace(), torch.device("cpu"))
    proposer = SimpleNamespace(
        positions=torch.tensor([10, 11, 12, 0], dtype=torch.int32),
        _slot_mapping_buffer=torch.tensor([100, 101, 102, -1], dtype=torch.int32),
        draft_attn_groups=[
            AttentionGroup(
                _FakeBackend,
                ["draft.swa"],
                builder.kv_cache_spec,
                0,
                [builder],
            )
        ],
        _dspark_standard_dsa_graph_buffers={},
    )
    common_metadata = SimpleNamespace(
        positions=torch.empty(0, dtype=torch.int32),
        slot_mapping=torch.empty(0, dtype=torch.int32),
        num_input_tokens=0,
        num_actual_tokens=0,
        causal=True,
        attn_state=None,
    )

    first = AscendDSparkProposer._build_standard_dsa_attn_metadata(
        proposer,
        common_metadata,
        num_input_tokens=4,
        num_actual_tokens=3,
    )[0]["draft.swa"].prefill
    first_ptrs = (
        first.dspark_swa_indices.data_ptr(),
        first.dspark_swa_lens.data_ptr(),
        first.sas_metadata.data_ptr(),
    )

    second = AscendDSparkProposer._build_standard_dsa_attn_metadata(
        proposer,
        common_metadata,
        num_input_tokens=4,
        num_actual_tokens=3,
    )[0]["draft.swa"].prefill

    assert first_ptrs == (
        second.dspark_swa_indices.data_ptr(),
        second.dspark_swa_lens.data_ptr(),
        second.sas_metadata.data_ptr(),
    )
    torch.testing.assert_close(second.dspark_swa_indices, torch.full((4, 1, 8), 2, dtype=torch.int32))
    torch.testing.assert_close(second.dspark_swa_lens, torch.full((4,), 2, dtype=torch.int32))
    torch.testing.assert_close(second.sas_metadata, torch.full((3,), 2, dtype=torch.int32))


def test_dspark_standard_dsa_propose_pads_model_inputs(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)

    context_calls = []

    class FakeForwardContextManager:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_set_ascend_forward_context(attn_metadata, vllm_config, **kwargs):
        context_calls.append((attn_metadata, vllm_config, kwargs))
        return FakeForwardContextManager()

    monkeypatch.setattr(
        dspark_proposer_module,
        "set_ascend_forward_context",
        fake_set_ascend_forward_context,
    )
    monkeypatch.setattr(
        dspark_proposer_module,
        "get_forward_context",
        lambda: SimpleNamespace(moe_layer_index=None),
    )

    model_calls = []
    precompute_calls = []

    def clone_value(value):
        if isinstance(value, dict):
            return {key: val.clone() if isinstance(val, torch.Tensor) else val for key, val in value.items()}
        if isinstance(value, torch.Tensor):
            return value.clone()
        return value

    class FakeModel:
        def precompute_and_store_context_kv(
            self,
            context_states,
            context_positions,
            context_slot_mapping,
            context_request_slots,
        ):
            precompute_calls.append(
                (
                    context_states.clone(),
                    context_positions.clone(),
                    clone_value(context_slot_mapping),
                    context_request_slots.clone(),
                )
            )

        def __call__(
            self,
            *,
            input_ids,
            positions,
            inputs_embeds,
            request_slots,
            slot_mapping,
            block_table,
            dspark_query_start_loc=None,
            dspark_seq_lens=None,
            dspark_token_to_req_indices=None,
        ):
            del inputs_embeds
            model_calls.append(
                {
                    "input_ids": input_ids.clone(),
                    "positions": positions.clone(),
                    "request_slots": request_slots.clone(),
                    "slot_mapping": clone_value(slot_mapping),
                    "block_table": clone_value(block_table),
                    "dspark_query_start_loc": clone_value(dspark_query_start_loc),
                    "dspark_seq_lens": clone_value(dspark_seq_lens),
                    "dspark_token_to_req_indices": clone_value(dspark_token_to_req_indices),
                }
            )
            return torch.arange(input_ids.numel() * 4, dtype=torch.float32).view(input_ids.numel(), 4)

    builder = _FakeMetadataBuilder(_FakeKVSpec(), ["draft.swa"], SimpleNamespace(), torch.device("cpu"))

    def sync_metadata_across_dp(
        num_tokens,
        is_draft_model,
        cudagraph_mode=dspark_proposer_module.CUDAGraphMode.NONE,
        allow_dp_padding=False,
    ):
        del num_tokens, is_draft_model, allow_dp_padding
        return (
            6,
            torch.tensor([6], dtype=torch.int32),
            cudagraph_mode,
        )

    proposer = SimpleNamespace(
        device=torch.device("cpu"),
        vllm_config=SimpleNamespace(),
        runner=SimpleNamespace(_sync_metadata_across_dp=sync_metadata_across_dp),
        model=FakeModel(),
        num_speculative_tokens=5,
        parallel_drafting_token_id=99,
        kernel_block_size=64,
        token_arange_np=np.arange(16, dtype=np.int32),
        arange_dspark=torch.arange(32, dtype=torch.int32),
        input_ids=torch.zeros(6, dtype=torch.int64),
        positions=torch.zeros(6, dtype=torch.int32),
        _slot_mapping_buffer=torch.zeros(6, dtype=torch.int32),
        _request_slots_buffer=torch.zeros(6, dtype=torch.int32),
        _dspark_token_to_req_indices_buffer=torch.zeros(6, dtype=torch.int32),
        _dspark_token_to_req_indices=None,
        _dspark_query_start_loc=None,
        _dspark_seq_lens=None,
        _dspark_seed_buffer=torch.full((2,), -1, dtype=torch.int64),
        _dflash_hidden_states=torch.zeros(4, 4, dtype=torch.float32),
        _context_positions_buffer=torch.zeros(4, dtype=torch.int32),
        _context_request_slots_buffer=torch.zeros(4, dtype=torch.int32),
        _context_slot_mapping_buffer=torch.zeros(4, dtype=torch.int32),
        _dspark_slots_to_reset=[],
        draft_attn_groups=[
            AttentionGroup(
                _FakeBackend,
                ["draft.swa"],
                builder.kv_cache_spec,
                0,
                [builder],
            )
        ],
    )
    proposer._assign_request_slots = lambda size: [3]

    sample_calls = []

    def fake_sample_sequential(num_reqs, head_hidden, token_indices_to_sample, sampling_metadata):
        sample_calls.append((num_reqs, head_hidden.clone(), token_indices_to_sample.clone(), sampling_metadata))
        return torch.tensor([[7, 8, 9, 10, 11]], dtype=torch.int64)

    proposer._sample_sequential = fake_sample_sequential
    proposer._pad_draft_query_buffers = AscendDSparkProposer._pad_draft_query_buffers.__get__(proposer)
    proposer._get_draft_block_table = AscendDSparkProposer._get_draft_block_table.__get__(proposer)
    proposer._get_draft_block_table_for_gid = AscendDSparkProposer._get_draft_block_table_for_gid.__get__(proposer)
    proposer._get_draft_block_tables = AscendDSparkProposer._get_draft_block_tables.__get__(proposer)
    proposer._layer_map_from_gid_map = AscendDSparkProposer._layer_map_from_gid_map.__get__(proposer)
    proposer._slot_mapping_buffer_for_gid = AscendDSparkProposer._slot_mapping_buffer_for_gid.__get__(proposer)
    proposer._slot_mapping_from_block_table = AscendDSparkProposer._slot_mapping_from_block_table.__get__(proposer)
    proposer._build_standard_dsa_attn_metadata = AscendDSparkProposer._build_standard_dsa_attn_metadata.__get__(
        proposer
    )
    proposer.set_inputs_first_pass = AscendDSparkProposer.set_inputs_first_pass.__get__(proposer)

    target_positions = torch.tensor([3, 4], dtype=torch.int32)
    target_hidden_states = torch.arange(8, dtype=torch.float32).view(2, 4)
    cad = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([5], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([5], dtype=torch.int32),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=1,
        num_actual_tokens=2,
        num_input_tokens=2,
        max_query_len=2,
        actual_seq_lengths_q=[2],
        block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([30, 31], dtype=torch.int32),
        positions=target_positions,
        attn_state=AscendAttentionState.SpecDecoding,
        decode_token_per_req=1,
        max_seq_len=5,
    )

    draft_tokens = AscendDSparkProposer._propose(
        proposer,
        target_token_ids=torch.tensor([1, 2], dtype=torch.int64),
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=torch.tensor([111], dtype=torch.int64),
        token_indices_to_sample=None,
        common_attn_metadata=cad,
        target_model_batch_desc=SimpleNamespace(),
        sampling_metadata=SimpleNamespace(),
    )

    torch.testing.assert_close(draft_tokens, torch.tensor([[7, 8, 9, 10, 11]], dtype=torch.int64))
    assert len(context_calls) == 1
    _, _, context_kwargs = context_calls[0]
    assert context_kwargs["num_tokens"] == 6
    assert context_kwargs["num_actual_tokens"] == 5
    assert len(precompute_calls) == 1
    assert len(model_calls) == 1
    torch.testing.assert_close(
        model_calls[0]["input_ids"],
        torch.tensor([111, 99, 99, 99, 99, 99], dtype=torch.int64),
    )
    torch.testing.assert_close(
        model_calls[0]["positions"],
        torch.tensor([5, 6, 7, 8, 9, 0], dtype=torch.int32),
    )
    torch.testing.assert_close(
        model_calls[0]["request_slots"],
        torch.tensor([3, 3, 3, 3, 3, 0], dtype=torch.int32),
    )
    assert set(precompute_calls[0][2]) == {"draft.swa"}
    torch.testing.assert_close(precompute_calls[0][2]["draft.swa"], torch.tensor([3, 4], dtype=torch.int32))
    assert set(model_calls[0]["slot_mapping"]) == {"draft.swa"}
    torch.testing.assert_close(
        model_calls[0]["slot_mapping"]["draft.swa"],
        torch.tensor([5, 6, 7, 8, 9, -1], dtype=torch.int32),
    )
    assert set(model_calls[0]["block_table"]) == {"draft.swa"}
    torch.testing.assert_close(model_calls[0]["block_table"]["draft.swa"], torch.tensor([[0]], dtype=torch.int32))
    torch.testing.assert_close(
        model_calls[0]["dspark_query_start_loc"],
        torch.tensor([0, 5], dtype=torch.int32),
    )
    torch.testing.assert_close(model_calls[0]["dspark_seq_lens"], torch.tensor([10], dtype=torch.int32))
    torch.testing.assert_close(
        model_calls[0]["dspark_token_to_req_indices"],
        torch.tensor([0, 0, 0, 0, 0, -1], dtype=torch.int32),
    )
    assert len(sample_calls) == 1
    assert sample_calls[0][1].shape[0] == 6
    torch.testing.assert_close(sample_calls[0][2], torch.arange(5, dtype=torch.int32))
    metadata_call = builder.calls[0]
    assert metadata_call["num_input_tokens"] == 6
    assert metadata_call["num_actual_tokens"] == 5
    torch.testing.assert_close(metadata_call["positions"], model_calls[0]["positions"])
    torch.testing.assert_close(metadata_call["slot_mapping"], model_calls[0]["slot_mapping"]["draft.swa"])


def test_dspark_dummy_run_keeps_drafter_eager_when_graph_disabled(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)

    context_calls = []

    class FakeForwardContextManager:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_set_ascend_forward_context(attn_metadata, vllm_config, **kwargs):
        context_calls.append((attn_metadata, vllm_config, kwargs))
        return FakeForwardContextManager()

    def fake_get_forward_context():
        runtime_mode = context_calls[-1][2]["aclgraph_runtime_mode"]
        return SimpleNamespace(cudagraph_runtime_mode=runtime_mode)

    monkeypatch.setattr(
        dspark_proposer_module,
        "set_ascend_forward_context",
        fake_set_ascend_forward_context,
    )
    monkeypatch.setattr(
        dspark_proposer_module,
        "get_forward_context",
        fake_get_forward_context,
    )

    class FakeModel:
        def precompute_and_store_context_kv(
            self,
            context_states,
            context_positions,
            context_slot_mapping,
            context_request_slots,
        ):
            del context_states, context_positions, context_slot_mapping, context_request_slots

        def __call__(
            self,
            *,
            input_ids,
            positions,
            inputs_embeds,
            request_slots,
            slot_mapping,
            block_table,
            dspark_query_start_loc=None,
            dspark_seq_lens=None,
            dspark_token_to_req_indices=None,
        ):
            del (
                positions,
                inputs_embeds,
                request_slots,
                slot_mapping,
                block_table,
                dspark_query_start_loc,
                dspark_seq_lens,
                dspark_token_to_req_indices,
            )
            return torch.zeros((input_ids.numel(), 4), dtype=torch.float32)

    def sync_metadata_across_dp(
        num_tokens,
        is_draft_model,
        cudagraph_mode=dspark_proposer_module.CUDAGraphMode.NONE,
        allow_dp_padding=False,
    ):
        del is_draft_model, allow_dp_padding
        return (
            num_tokens,
            torch.tensor([num_tokens], dtype=torch.int32),
            cudagraph_mode,
        )

    proposer = SimpleNamespace(
        use_cuda_graph=False,
        runner=SimpleNamespace(_sync_metadata_across_dp=sync_metadata_across_dp),
        vllm_config=SimpleNamespace(),
        model=FakeModel(),
        num_speculative_tokens=5,
        max_query_tokens=8,
        input_ids=torch.zeros(8, dtype=torch.int64),
        positions=torch.zeros(8, dtype=torch.int32),
        hidden_states=torch.zeros(8, 4, dtype=torch.float32),
        _context_positions_buffer=torch.zeros(8, dtype=torch.int32),
        _context_request_slots_buffer=torch.zeros(8, dtype=torch.int32),
        _slot_mapping_buffer=torch.zeros(8, dtype=torch.int32),
        _request_slots_buffer=torch.zeros(8, dtype=torch.int32),
        parallel_drafting_token_id=99,
        draft_attn_groups=[object()],
    )
    proposer._pad_draft_query_buffers = AscendDSparkProposer._pad_draft_query_buffers.__get__(proposer)
    proposer._sample_sequential = lambda num_reqs, head_hidden, token_indices_to_sample, sampling_metadata: torch.zeros(
        (num_reqs, proposer.num_speculative_tokens),
        dtype=torch.int64,
    )
    proposer._update_full_graph_params = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("draft dummy_run must not update full-graph params when use_cuda_graph is false")
    )

    AscendDSparkProposer.dummy_run(
        proposer,
        num_tokens=5,
        num_reqs=1,
        aclgraph_runtime_mode=dspark_proposer_module.CUDAGraphMode.FULL,
    )

    assert len(context_calls) == 1
    assert context_calls[0][2]["aclgraph_runtime_mode"] == dspark_proposer_module.CUDAGraphMode.NONE


def test_dspark_load_model_disables_dspark_drafter_graph(monkeypatch):
    base_seen_use_cuda_graph = []
    warnings = []

    def fake_base_load_model(self, model):
        del model
        base_seen_use_cuda_graph.append(self.use_cuda_graph)

    monkeypatch.setattr(dspark_proposer_module.AscendDflashProposer, "load_model", fake_base_load_model)
    monkeypatch.setattr(
        dspark_proposer_module.logger,
        "warning_once",
        lambda *args, **kwargs: warnings.append(args),
    )

    proposer = object.__new__(AscendDSparkProposer)
    proposer.use_cuda_graph = True

    AscendDSparkProposer.load_model(proposer, model=object())

    assert base_seen_use_cuda_graph == [False]
    assert proposer.use_cuda_graph is False
    assert proposer._runnable.__func__ is AscendDSparkProposer._run_dspark_model
    assert len(warnings) == 1
    assert "DSpark drafter ACLGraph is disabled" in warnings[0][0]


def test_dspark_draft_vllm_config_disables_inner_compile(monkeypatch):
    @dataclass
    class FakeModelConfig:
        enforce_eager: bool

    @dataclass
    class FakeCompilationConfig:
        mode: object

    @dataclass
    class FakeVllmConfig:
        model_config: FakeModelConfig
        compilation_config: FakeCompilationConfig

    base_config = FakeVllmConfig(
        model_config=FakeModelConfig(enforce_eager=False),
        compilation_config=FakeCompilationConfig(mode=dspark_proposer_module.CompilationMode.VLLM_COMPILE),
    )
    monkeypatch.setattr(
        dspark_proposer_module.AscendDflashProposer,
        "_create_draft_vllm_config",
        lambda self: base_config,
    )

    proposer = object.__new__(AscendDSparkProposer)
    draft_config = AscendDSparkProposer._create_draft_vllm_config(proposer)

    assert draft_config.model_config.enforce_eager is True
    assert draft_config.compilation_config.mode == dspark_proposer_module.CompilationMode.NONE
    assert base_config.model_config.enforce_eager is False
    assert base_config.compilation_config.mode == dspark_proposer_module.CompilationMode.VLLM_COMPILE


def test_dspark_propose_uses_aclgraph_dispatch_when_enabled(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)

    context_calls = []

    class FakeForwardContextManager:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_set_ascend_forward_context(attn_metadata, vllm_config, **kwargs):
        context_calls.append((attn_metadata, vllm_config, kwargs))
        return FakeForwardContextManager()

    def fake_get_forward_context():
        return SimpleNamespace(
            cudagraph_runtime_mode=context_calls[-1][2]["aclgraph_runtime_mode"],
            moe_layer_index=None,
        )

    monkeypatch.setattr(dspark_proposer_module, "set_ascend_forward_context", fake_set_ascend_forward_context)
    monkeypatch.setattr(dspark_proposer_module, "get_forward_context", fake_get_forward_context)
    monkeypatch.setattr(dspark_proposer_module, "_EXTRA_CTX", SimpleNamespace(capturing=False))

    first_batch_descriptor = dspark_proposer_module.BatchDescriptor(num_tokens=6, num_reqs=1, uniform=True)
    batch_descriptor = dspark_proposer_module.BatchDescriptor(num_tokens=12, num_reqs=2, uniform=True)

    class FakeDispatcher:
        def __init__(self):
            self.calls = []

        def dispatch(self, *, num_tokens, uniform_decode, has_lora, valid_modes=None):
            self.calls.append(
                {
                    "num_tokens": num_tokens,
                    "uniform_decode": uniform_decode,
                    "has_lora": has_lora,
                    "valid_modes": valid_modes,
                }
            )
            mode = dspark_proposer_module.CUDAGraphMode.FULL
            descriptor = first_batch_descriptor if len(self.calls) == 1 else batch_descriptor
            return mode, descriptor

    dispatcher = FakeDispatcher()
    sync_calls = []

    def fake_sync_metadata(num_tokens, is_draft_model, cudagraph_mode, allow_dp_padding):
        sync_calls.append((num_tokens, is_draft_model, cudagraph_mode, allow_dp_padding))
        return (
            12,
            torch.tensor([12], dtype=torch.int32),
            cudagraph_mode,
        )

    class FakeModel:
        def precompute_and_store_context_kv(self, *args):
            pass

    proposer = SimpleNamespace(
        runner=SimpleNamespace(
            input_batch=SimpleNamespace(lora_id_to_lora_request={}),
            cudagraph_dispatcher=dispatcher,
            _sync_metadata_across_dp=fake_sync_metadata,
        ),
        vllm_config=SimpleNamespace(),
        device=torch.device("cpu"),
        use_cuda_graph=True,
        dp_rank=0,
        num_speculative_tokens=5,
        draft_attn_groups=[],
        model=FakeModel(),
        _dflash_num_context=0,
        _dspark_slots_to_reset=[],
        _dflash_hidden_states=torch.empty((0, 4)),
        _context_positions_buffer=torch.empty((0,), dtype=torch.int32),
        _context_slot_mapping_buffer=torch.empty((0,), dtype=torch.int32),
        _context_request_slots_buffer=torch.empty((0,), dtype=torch.int32),
        _slot_mapping_buffer=torch.empty((0,), dtype=torch.int32),
        _dspark_token_to_req_indices_buffer=torch.zeros(8, dtype=torch.int32),
        token_indices_to_sample=torch.full((16,), -1, dtype=torch.int32),
    )

    common_attn_metadata = SimpleNamespace(
        num_reqs=1,
        seq_lens=torch.tensor([10], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 5], dtype=torch.int32),
    )
    token_indices = torch.arange(5, dtype=torch.int32)
    proposer.set_inputs_first_pass = lambda **kwargs: (5, token_indices, common_attn_metadata, None)

    build_calls = []
    pad_calls = []
    runnable_calls = []
    sample_calls = []

    def fake_build_standard_dsa_attn_metadata(common, num_input_tokens, num_actual_tokens):
        build_calls.append((common, num_input_tokens, num_actual_tokens))
        return [{"draft.swa": object()}]

    proposer._build_standard_dsa_attn_metadata = fake_build_standard_dsa_attn_metadata
    proposer._pad_draft_query_buffers = lambda num_actual_tokens, num_input_tokens: pad_calls.append(
        (num_actual_tokens, num_input_tokens)
    )

    def fake_runnable(**kwargs):
        runnable_calls.append(kwargs)
        return torch.zeros((12, 4), dtype=torch.float32)

    def fake_sample(num_reqs, head_hidden, token_indices_to_sample, sampling_metadata):
        sample_calls.append(
            {
                "num_reqs": num_reqs,
                "head_hidden": head_hidden,
                "token_indices_to_sample": token_indices_to_sample,
                "sampling_metadata": sampling_metadata,
            }
        )
        return torch.tensor([[7, 8, 9, 10, 11], [70, 80, 90, 100, 110]], dtype=torch.int64)

    proposer._runnable = fake_runnable
    proposer._sample_sequential = fake_sample

    draft_tokens = AscendDSparkProposer._propose(
        proposer,
        target_token_ids=torch.tensor([1], dtype=torch.int64),
        target_positions=torch.tensor([4], dtype=torch.int32),
        target_hidden_states=torch.zeros(1, 4),
        next_token_ids=torch.tensor([2], dtype=torch.int64),
        token_indices_to_sample=None,
        common_attn_metadata=common_attn_metadata,
        target_model_batch_desc=SimpleNamespace(uniform=True),
        sampling_metadata=SimpleNamespace(),
    )

    torch.testing.assert_close(draft_tokens, torch.tensor([[7, 8, 9, 10, 11]], dtype=torch.int64))
    assert dispatcher.calls == [
        {"num_tokens": 5, "uniform_decode": True, "has_lora": False, "valid_modes": None},
        {
            "num_tokens": 12,
            "uniform_decode": True,
            "has_lora": False,
            "valid_modes": {dspark_proposer_module.CUDAGraphMode.FULL},
        },
    ]
    assert sync_calls == [(6, True, dspark_proposer_module.CUDAGraphMode.FULL, True)]
    assert len(context_calls) == 1
    _, _, context_kwargs = context_calls[0]
    assert context_kwargs["aclgraph_runtime_mode"] == dspark_proposer_module.CUDAGraphMode.FULL
    assert context_kwargs["batch_descriptor"] == batch_descriptor
    assert context_kwargs["num_tokens"] == 12
    torch.testing.assert_close(context_kwargs["num_tokens_across_dp"], torch.tensor([12], dtype=torch.int32))
    assert context_kwargs["num_actual_tokens"] == 5
    assert build_calls == [(common_attn_metadata, 12, 5)]
    assert pad_calls == [(5, 12)]
    assert len(runnable_calls) == 1
    assert runnable_calls[0]["num_input_tokens"] == 12
    assert len(sample_calls) == 1
    assert sample_calls[0]["num_reqs"] == 2
    torch.testing.assert_close(
        proposer.token_indices_to_sample[:10],
        torch.arange(10, dtype=torch.int32),
    )
    assert sample_calls[0]["token_indices_to_sample"].data_ptr() == proposer.token_indices_to_sample[:10].data_ptr()


def test_dspark_propose_runs_padded_graph_reqs_and_slices_result(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)

    context_calls = []

    class FakeForwardContextManager:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_set_ascend_forward_context(attn_metadata, vllm_config, **kwargs):
        context_calls.append((attn_metadata, vllm_config, kwargs))
        return FakeForwardContextManager()

    monkeypatch.setattr(dspark_proposer_module, "set_ascend_forward_context", fake_set_ascend_forward_context)
    monkeypatch.setattr(
        dspark_proposer_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            cudagraph_runtime_mode=context_calls[-1][2]["aclgraph_runtime_mode"],
            moe_layer_index=None,
        ),
    )
    monkeypatch.setattr(dspark_proposer_module, "_EXTRA_CTX", SimpleNamespace(capturing=False))

    class FakeDispatcher:
        def __init__(self):
            self.calls = 0

        def dispatch(self, *, num_tokens, uniform_decode, has_lora, valid_modes=None):
            del uniform_decode, has_lora, valid_modes
            self.calls += 1
            if self.calls == 1:
                return dspark_proposer_module.CUDAGraphMode.NONE, dspark_proposer_module.BatchDescriptor(
                    num_tokens=6,
                    num_reqs=1,
                    uniform=True,
                )
            return dspark_proposer_module.CUDAGraphMode.FULL, dspark_proposer_module.BatchDescriptor(
                num_tokens=num_tokens,
                num_reqs=2,
                uniform=True,
            )

    proposer = SimpleNamespace(
        runner=SimpleNamespace(
            input_batch=SimpleNamespace(lora_id_to_lora_request={}),
            cudagraph_dispatcher=FakeDispatcher(),
            _sync_metadata_across_dp=lambda num_tokens, is_draft_model, cudagraph_mode, allow_dp_padding: (
                12,
                torch.tensor([12], dtype=torch.int32),
                dspark_proposer_module.CUDAGraphMode.FULL,
            ),
        ),
        vllm_config=SimpleNamespace(),
        device=torch.device("cpu"),
        use_cuda_graph=True,
        dp_rank=0,
        num_speculative_tokens=5,
        draft_attn_groups=[],
        model=type("FakeModel", (), {"precompute_and_store_context_kv": lambda self, *args: None})(),
        _dflash_num_context=0,
        _dspark_slots_to_reset=[],
        _dflash_hidden_states=torch.empty((0, 4)),
        _context_positions_buffer=torch.empty((0,), dtype=torch.int32),
        _context_slot_mapping_buffer=torch.empty((0,), dtype=torch.int32),
        _context_request_slots_buffer=torch.empty((0,), dtype=torch.int32),
        _slot_mapping_buffer=torch.empty((0,), dtype=torch.int32),
        token_indices_to_sample=torch.full((16,), -1, dtype=torch.int32),
        _dspark_token_to_req_indices_buffer=torch.zeros(10, dtype=torch.int32),
        _dspark_last_draft_logits=None,
        _dspark_last_draft_probs=None,
        _dspark_last_draft_logit_components=None,
    )
    common_attn_metadata = SimpleNamespace(
        num_reqs=1,
        seq_lens=torch.tensor([10], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 5], dtype=torch.int32),
        actual_seq_lengths_q=[5],
    )
    token_indices = torch.arange(5, dtype=torch.int32)
    proposer.set_inputs_first_pass = lambda **kwargs: (5, token_indices, common_attn_metadata, None)
    build_calls = []
    pad_calls = []
    runnable_calls = []
    sample_calls = []

    def fake_build_standard_dsa_attn_metadata(common, num_input_tokens, num_actual_tokens):
        build_calls.append((common, num_input_tokens, num_actual_tokens))
        return []

    proposer._build_standard_dsa_attn_metadata = fake_build_standard_dsa_attn_metadata
    proposer._pad_draft_query_buffers = lambda num_actual_tokens, num_input_tokens: pad_calls.append(
        (num_actual_tokens, num_input_tokens)
    )

    def fake_runnable(**kwargs):
        runnable_calls.append(kwargs)
        return torch.zeros((12, 4), dtype=torch.float32)

    def fake_sample(num_reqs, head_hidden, token_indices_to_sample, sampling_metadata):
        del head_hidden, token_indices_to_sample, sampling_metadata
        sample_calls.append(num_reqs)
        proposer._dspark_last_draft_logits = torch.arange(30, dtype=torch.float32).view(2, 5, 3)
        proposer._dspark_last_draft_logit_components = {
            "final_top_ids": torch.arange(20, dtype=torch.int64).view(2, 5, 2),
        }
        return torch.tensor([[7, 8, 9, 10, 11], [70, 80, 90, 100, 110]], dtype=torch.int64)

    proposer._runnable = fake_runnable
    proposer._sample_sequential = fake_sample

    draft_tokens = AscendDSparkProposer._propose(
        proposer,
        target_token_ids=torch.tensor([1], dtype=torch.int64),
        target_positions=torch.tensor([4], dtype=torch.int32),
        target_hidden_states=torch.zeros(1, 4),
        next_token_ids=torch.tensor([2], dtype=torch.int64),
        token_indices_to_sample=None,
        common_attn_metadata=common_attn_metadata,
        target_model_batch_desc=SimpleNamespace(uniform=True),
        sampling_metadata=SimpleNamespace(),
    )

    torch.testing.assert_close(draft_tokens, torch.tensor([[7, 8, 9, 10, 11]], dtype=torch.int64))
    assert runnable_calls[0]["num_input_tokens"] == 12
    assert sample_calls == [2]
    assert common_attn_metadata.num_reqs == 2
    torch.testing.assert_close(common_attn_metadata.query_start_loc, torch.tensor([0, 5, 5], dtype=torch.int32))
    torch.testing.assert_close(common_attn_metadata.query_start_loc_cpu, torch.tensor([0, 5, 5], dtype=torch.int32))
    torch.testing.assert_close(common_attn_metadata.seq_lens, torch.tensor([10, 0], dtype=torch.int32))
    torch.testing.assert_close(common_attn_metadata._seq_lens_cpu, torch.tensor([10, 0], dtype=torch.int32))
    assert common_attn_metadata.actual_seq_lengths_q == [5, 0]
    assert pad_calls == [(5, 12)]
    assert build_calls == [(common_attn_metadata, 12, 5)]
    assert context_calls[0][2]["num_tokens"] == 12
    assert context_calls[0][2]["num_actual_tokens"] == 5
    assert proposer._dspark_last_draft_logits.shape == (1, 5, 3)
    assert proposer._dspark_last_draft_logit_components["final_top_ids"].shape == (1, 5, 2)


def test_dspark_dsa_impl_exposes_graph_param_update_hook():
    from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPImpl
    from vllm_ascend.attention.dsa_v1 import AscendDSAImpl

    assert AscendDSAImpl.update_graph_params(None, None, 1) is None
    assert AscendDSACPImpl.update_graph_params(None, None, 1) is None


def test_dspark_attention_uses_standard_cache_pta_when_enabled(monkeypatch):
    calls = []

    def fake_standard_cache_attention(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.full_like(args[0], 3.0)

    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)
    monkeypatch.setattr(
        dspark_model_module,
        "dspark_attention_from_standard_cache",
        fake_standard_cache_attention,
    )
    cache = torch.zeros(4, 8, 1, 4)
    attention = SimpleNamespace(
        dsa_attn=SimpleNamespace(
            swa_cache_layer=SimpleNamespace(kv_cache=cache, block_size=8),
        ),
        attn_sink=torch.tensor([0.5], dtype=torch.float32),
        n_local_heads=1,
        block_size=2,
        window_size=4,
        scale=0.25,
    )
    q = torch.zeros(2, 1, 4)
    positions = torch.tensor([10, 11], dtype=torch.int32)
    slot_mapping = torch.tensor([10, 11], dtype=torch.int32)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)

    output = dspark_model_module.DeepseekV4DSparkAttention._run_standard_dspark_attention(
        attention,
        q,
        positions,
        slot_mapping,
        block_table,
    )

    assert len(calls) == 1
    assert calls[0][0][1] is cache
    assert calls[0][0][2] is block_table
    assert calls[0][0][3] is positions
    assert calls[0][0][4] is slot_mapping
    assert calls[0][0][5] is None
    assert calls[0][0][8] == 4
    assert calls[0][0][9] == 8
    torch.testing.assert_close(output, torch.full_like(q, 3.0))


def test_dspark_standard_swa_store_can_be_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", "1")

    class FailIfTouched:
        @property
        def swa_cache_layer(self):
            raise AssertionError("standard SWA cache must not be touched when disabled")

    attention = SimpleNamespace(dsa_attn=FailIfTouched())

    dspark_model_module.DeepseekV4DSparkAttention._store_standard_swa_kv(
        attention,
        torch.zeros(1, 1, 4),
        torch.tensor([0], dtype=torch.int32),
    )


def test_dspark_standard_swa_store_unwraps_singleton_cache(monkeypatch):
    from vllm_ascend.device import device_op as device_op_module

    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)
    calls = []

    def fake_scatter(cache, shared_kv, slot_mapping):
        calls.append((cache, shared_kv, slot_mapping))

    monkeypatch.setattr(
        device_op_module.DeviceOperator,
        "dsa_kv_compress_scatter",
        staticmethod(fake_scatter),
    )

    cache = torch.zeros(1, 64, 1, 4)
    shared_kv = torch.ones(1, 1, 4)
    slot_mapping = torch.tensor([[0, 0]], dtype=torch.int32)
    attention = SimpleNamespace(
        dsa_attn=SimpleNamespace(
            swa_cache_layer=SimpleNamespace(kv_cache=[[cache]], block_size=64),
        ),
    )

    dspark_model_module.DeepseekV4DSparkAttention._store_standard_swa_kv(
        attention,
        shared_kv,
        slot_mapping,
    )

    assert len(calls) == 1
    assert calls[0][0] is cache
    assert calls[0][1] is shared_kv
    torch.testing.assert_close(calls[0][2], slot_mapping)
