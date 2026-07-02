# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import torch

import vllm_ascend.spec_decode.dspark_proposer as dspark_proposer_module
import vllm_ascend.models.deepseek_v4_dspark as dspark_model_module
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer
from vllm.v1.worker.utils import AttentionGroup


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
            processed[random_mask] = processed[random_mask] / temperature[
                expanded_idx_mapping.long()
            ][random_mask].unsqueeze(-1)
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
    proposer._get_draft_sampling_temperature = AscendDSparkProposer._get_draft_sampling_temperature.__get__(
        proposer
    )
    proposer._get_runner_sampling_state_seeds = AscendDSparkProposer._get_runner_sampling_state_seeds.__get__(
        proposer
    )
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


def test_dspark_set_inputs_first_pass_uses_anchor_first_block():
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

    target_positions = torch.tensor([5, 6, 7, 15, 16, 17], dtype=torch.int32)
    target_hidden_states = torch.arange(12, dtype=torch.float32).view(6, 2)
    next_token_ids = torch.tensor([101, 202], dtype=torch.int64)
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


def test_dspark_set_inputs_first_pass_skips_rejected_context_tokens():
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

    target_positions = torch.tensor([10, 11, 12, 13, 30, 31, 32, 33], dtype=torch.int32)
    target_hidden_states = torch.arange(16, dtype=torch.float32).view(8, 2)
    next_token_ids = torch.tensor([101, 202], dtype=torch.int64)
    num_rejected_tokens = torch.tensor([1, 2], dtype=torch.int32)
    cad = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
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
    torch.testing.assert_close(cad.seq_lens, torch.tensor([16, 35], dtype=torch.int32))
    assert cad.max_query_len == block_size
    assert cad.max_seq_len == 37
    assert cad.num_actual_tokens == 6
    assert cad.causal is False
    assert cad.attn_state == AscendAttentionState.ChunkedPrefill
    torch.testing.assert_close(
        proposer._dflash_hidden_states[:5],
        torch.cat([target_hidden_states[:3], target_hidden_states[4:6]], dim=0),
    )
    torch.testing.assert_close(
        proposer._context_positions_buffer[:5],
        torch.tensor([10, 11, 12, 30, 31], dtype=torch.int32),
    )
    torch.testing.assert_close(
        proposer._context_request_slots_buffer[:5],
        torch.tensor([2, 2, 2, 7, 7], dtype=torch.int32),
    )
    torch.testing.assert_close(
        proposer._context_slot_mapping_buffer[:5],
        torch.tensor([200, 201, 202, 204, 205], dtype=torch.int32),
    )
    assert proposer._dflash_num_context == 5


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
    monkeypatch.setenv("VLLM_ASCEND_DSPARK_USE_STANDARD_DSA", "1")
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
    assert proposer.block_size == 64


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


def test_dspark_standard_dsa_propose_pads_model_inputs(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DSPARK_USE_STANDARD_DSA", "1")

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
                    context_slot_mapping.clone(),
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
        ):
            del inputs_embeds
            model_calls.append(
                {
                    "input_ids": input_ids.clone(),
                    "positions": positions.clone(),
                    "request_slots": request_slots.clone(),
                    "slot_mapping": slot_mapping.clone(),
                }
            )
            return torch.arange(input_ids.numel() * 4, dtype=torch.float32).view(input_ids.numel(), 4)

    builder = _FakeMetadataBuilder(_FakeKVSpec(), ["draft.swa"], SimpleNamespace(), torch.device("cpu"))
    proposer = SimpleNamespace(
        device=torch.device("cpu"),
        vllm_config=SimpleNamespace(),
        runner=SimpleNamespace(
            _sync_metadata_across_dp=lambda num_tokens, is_draft_model: (
                6,
                torch.tensor([6], dtype=torch.int32),
                None,
            )
        ),
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
    torch.testing.assert_close(
        model_calls[0]["slot_mapping"],
        torch.tensor([5, 6, 7, 8, 9, -1], dtype=torch.int32),
    )
    assert len(sample_calls) == 1
    assert sample_calls[0][1].shape[0] == 6
    torch.testing.assert_close(sample_calls[0][2], torch.arange(5, dtype=torch.int32))
    metadata_call = builder.calls[0]
    assert metadata_call["num_input_tokens"] == 6
    assert metadata_call["num_actual_tokens"] == 5
    torch.testing.assert_close(metadata_call["positions"], model_calls[0]["positions"])
    torch.testing.assert_close(metadata_call["slot_mapping"], model_calls[0]["slot_mapping"])


def test_dspark_dummy_run_keeps_drafter_eager_when_graph_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DSPARK_USE_STANDARD_DSA", "1")

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
        ):
            del positions, inputs_embeds, request_slots, slot_mapping
            return torch.zeros((input_ids.numel(), 4), dtype=torch.float32)

    proposer = SimpleNamespace(
        use_cuda_graph=False,
        runner=SimpleNamespace(
            _sync_metadata_across_dp=lambda num_tokens, is_draft_model: (
                num_tokens,
                torch.tensor([num_tokens], dtype=torch.int32),
                None,
            )
        ),
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


def test_dspark_attention_uses_standard_dsa_when_enabled(monkeypatch):
    calls = []

    def fake_dsa_attn(positions, hidden_states, llama_4_scaling):
        calls.append((positions, hidden_states, llama_4_scaling))
        return hidden_states + 1

    monkeypatch.setenv("VLLM_ASCEND_DSPARK_USE_STANDARD_DSA", "1")
    attention = SimpleNamespace(dsa_attn=fake_dsa_attn)
    positions = torch.tensor([1, 2], dtype=torch.int32)
    hidden_states = torch.zeros(2, 3)

    output = dspark_model_module.DeepseekV4DSparkAttention.forward(
        attention,
        positions,
        hidden_states,
        llama_4_scaling=None,
        request_slots=torch.tensor([0, 0], dtype=torch.int32),
        slot_mapping=torch.tensor([10, 11], dtype=torch.int32),
    )

    assert len(calls) == 1
    assert calls[0][0] is positions
    assert calls[0][1] is hidden_states
    torch.testing.assert_close(output, torch.ones(2, 3))


def test_dspark_standard_swa_store_is_env_gated(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_STANDARD_DSA", raising=False)

    class FailIfTouched:
        @property
        def swa_cache_layer(self):
            raise AssertionError("standard SWA cache must not be touched by default")

    attention = SimpleNamespace(dsa_attn=FailIfTouched())

    dspark_model_module.DeepseekV4DSparkAttention._store_standard_swa_kv(
        attention,
        torch.zeros(1, 1, 4),
        torch.tensor([0], dtype=torch.int32),
    )


def test_dspark_standard_swa_store_unwraps_singleton_cache(monkeypatch):
    from vllm_ascend.device import device_op as device_op_module

    monkeypatch.setenv("VLLM_ASCEND_DSPARK_USE_STANDARD_DSA", "1")
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
