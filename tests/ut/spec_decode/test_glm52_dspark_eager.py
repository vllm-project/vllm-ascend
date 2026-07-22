from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_ascend.spec_decode.dspark_proposer as dspark_module
import vllm_ascend.spec_decode.llm_base_proposer as base_module
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer
from vllm_ascend.spec_decode.llm_base_proposer import (
    AscendSpecDecodeBaseProposer,
)


def _make_config(*, bonus_anchor=None, draft_sample_method="greedy"):
    hf_config = SimpleNamespace()
    if bonus_anchor is not None:
        hf_config.dspark_bonus_anchor = bonus_anchor
    draft_model_config = SimpleNamespace(
        hf_config=hf_config,
        get_hidden_size=lambda: 3,
    )
    speculative_config = SimpleNamespace(
        draft_model_config=draft_model_config,
        draft_sample_method=draft_sample_method,
    )
    return SimpleNamespace(speculative_config=speculative_config)


def _fake_dflash_init(self, vllm_config, device, runner=None):
    self.vllm_config = vllm_config
    self.speculative_config = vllm_config.speculative_config
    self.draft_model_config = self.speculative_config.draft_model_config
    self.num_speculative_tokens = 7
    self.max_batch_size = 4
    self.max_num_tokens = 64
    self.dtype = torch.float32
    self.device = device
    self.hidden_size = 2
    self.hidden_states = torch.zeros((64, 2), device=device)
    self._dflash_hidden_states = torch.zeros((64, 2), device=device)


@pytest.mark.parametrize(
    ("bonus_anchor", "sample_from_anchor", "num_query_per_req"),
    [
        (None, True, 7),
        (False, True, 7),
        (True, False, 8),
    ],
)
def test_dspark_initializes_anchor_contract_and_forces_eager(
    monkeypatch,
    bonus_anchor,
    sample_from_anchor,
    num_query_per_req,
):
    monkeypatch.setattr(
        AscendDflashProposer,
        "__init__",
        _fake_dflash_init,
    )

    proposer = AscendDSparkProposer(
        _make_config(bonus_anchor=bonus_anchor),
        torch.device("cpu"),
    )

    assert proposer.sample_from_anchor is sample_from_anchor
    assert proposer.num_query_per_req == num_query_per_req
    assert proposer.max_query_tokens == 4 * num_query_per_req
    assert proposer.positions.shape == (4 * num_query_per_req,)
    assert proposer._slot_mapping_buffer.shape == (4 * num_query_per_req,)
    assert proposer.use_cuda_graph is False


def test_dspark_rejects_probabilistic_draft_sampling(monkeypatch):
    monkeypatch.setattr(
        AscendDflashProposer,
        "__init__",
        _fake_dflash_init,
    )

    with pytest.raises(ValueError, match="probabilistic"):
        AscendDSparkProposer(
            _make_config(draft_sample_method="probabilistic"),
            torch.device("cpu"),
        )


class _KernelRecorder:
    def __init__(self):
        self.grid = None
        self.kwargs = None

    def __getitem__(self, grid):
        self.grid = grid
        return self

    def __call__(self, **kwargs):
        self.kwargs = kwargs


def _make_input_proposer(*, sample_from_anchor, num_query_per_req):
    kv_cache_spec = SimpleNamespace(block_size=128)
    attn_group = SimpleNamespace(
        kv_cache_group_id=0,
        kv_cache_spec=kv_cache_spec,
    )
    return SimpleNamespace(
        num_speculative_tokens=7,
        sample_from_anchor=sample_from_anchor,
        num_query_per_req=num_query_per_req,
        max_query_tokens=2 * num_query_per_req,
        device=torch.device("cpu"),
        parallel_drafting_token_id=999,
        kv_cache_gid=0,
        draft_attn_groups=[attn_group],
        _per_group_block_tables={0: torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)},
        _per_group_slot_mappings={0: torch.tensor([10, 11, 12, 13], dtype=torch.int32)},
        _per_group_query_slot_mapping_buffers={0: torch.zeros(2 * num_query_per_req, dtype=torch.int32)},
        _per_group_context_slot_mapping_buffers={0: torch.zeros(32, dtype=torch.int32)},
        _layer_group_idx=[0],
        _dspark_seed_buffer=torch.zeros(4, dtype=torch.int64),
        _dflash_hidden_states=torch.zeros((32, 3), dtype=torch.float32),
        _context_positions_buffer=torch.zeros(32, dtype=torch.int32),
        input_ids=torch.zeros(2 * num_query_per_req, dtype=torch.int64),
        positions=torch.zeros(2 * num_query_per_req, dtype=torch.int32),
        arange_dflash=torch.arange(3, dtype=torch.int32),
        token_arange_np=np.arange(3, dtype=np.int32),
    )


@pytest.mark.parametrize(
    ("sample_from_anchor", "num_query_per_req"),
    [
        (True, 7),
        (False, 8),
    ],
)
def test_dspark_first_pass_uses_anchor_width_and_updates_metadata(
    monkeypatch,
    sample_from_anchor,
    num_query_per_req,
):
    kernel = _KernelRecorder()
    monkeypatch.setattr(
        dspark_module,
        "copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
        kernel,
    )
    proposer = _make_input_proposer(
        sample_from_anchor=sample_from_anchor,
        num_query_per_req=num_query_per_req,
    )
    cad = SimpleNamespace(
        num_reqs=2,
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([10, 20], dtype=torch.int32),
        max_seq_len=20,
        actual_seq_lengths_q=[],
        decode_token_per_req=0,
    )
    target_hidden_states = torch.arange(12, dtype=torch.float32).reshape(
        4,
        3,
    )
    num_rejected_tokens = torch.tensor([1, 2], dtype=torch.int32)

    num_query_total, token_indices, updated_cad, long_seq = AscendDSparkProposer.set_inputs_first_pass(
        proposer,
        target_token_ids=torch.zeros(2, dtype=torch.int64),
        next_token_ids=torch.tensor([101, 102], dtype=torch.int64),
        target_positions=torch.tensor([9, 19], dtype=torch.int32),
        target_hidden_states=target_hidden_states,
        token_indices_to_sample=None,
        cad=cad,
        num_rejected_tokens_gpu=num_rejected_tokens,
    )

    assert kernel.grid == (1,)
    assert kernel.kwargs["num_query_per_req"] == num_query_per_req
    assert kernel.kwargs["num_speculative_tokens"] == 7
    assert kernel.kwargs["SAMPLE_FROM_ANCHOR"] is sample_from_anchor
    assert num_query_total == 2 * num_query_per_req
    assert token_indices.shape == (14,)
    assert long_seq is None
    torch.testing.assert_close(
        proposer._dspark_seed_buffer,
        torch.tensor([101, 102, 0, 0], dtype=torch.int64),
    )
    torch.testing.assert_close(
        proposer._dflash_hidden_states[:4],
        target_hidden_states,
    )
    torch.testing.assert_close(
        updated_cad.query_start_loc,
        torch.tensor(
            [0, num_query_per_req, 2 * num_query_per_req],
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        updated_cad.seq_lens,
        torch.tensor(
            [
                10 - 1 + num_query_per_req,
                20 - 2 + num_query_per_req,
            ],
            dtype=torch.int32,
        ),
    )
    assert updated_cad.actual_seq_lengths_q == [num_query_per_req] * 2
    assert updated_cad.decode_token_per_req == num_query_per_req
    assert updated_cad.num_actual_tokens == 2 * num_query_per_req
    assert updated_cad.num_input_tokens == 2 * num_query_per_req
    assert updated_cad.max_query_len == num_query_per_req
    assert updated_cad.max_seq_len == 20 + num_query_per_req
    assert updated_cad.slot_mapping.shape == (2 * num_query_per_req,)
    assert updated_cad.positions.shape == (2 * num_query_per_req,)
    assert updated_cad.causal is False
    assert updated_cad.attn_mask is None
    assert updated_cad.attn_state == AscendAttentionState.ChunkedPrefill


class _MarkovModel:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states
        self.compute_logits_input = None

    def __call__(self, **kwargs):
        return self.hidden_states

    def compute_logits(self, hidden_states):
        self.compute_logits_input = hidden_states.clone()
        return torch.zeros((hidden_states.shape[0], 2), dtype=torch.float32)

    def markov_embed(self, token_ids):
        return torch.zeros((token_ids.shape[0], 2), dtype=torch.float32)

    def markov_bias(self, hidden_states):
        return torch.zeros((hidden_states.shape[0], 2), dtype=torch.float32)


def test_dspark_markov_logits_use_sampled_hidden_states(monkeypatch):
    monkeypatch.setattr(base_module, "lmhead_tp_enable", lambda: False)
    monkeypatch.setattr(
        base_module,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_reduce_sample=False),
    )
    hidden_states = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )
    model = _MarkovModel(hidden_states)
    proposer = SimpleNamespace(
        input_ids=torch.zeros(4, dtype=torch.int64),
        _get_positions=lambda num_tokens: torch.arange(
            num_tokens,
            dtype=torch.int32,
        ),
        method="dspark",
        _context_slot_mapping_buffers=[],
        build_model_inputs_first_pass=lambda *args: None,
        pass_hidden_states_to_model=False,
        model=model,
        model_returns_tuple=lambda: False,
        _share_mtp_indices=False,
        maybe_all_gather_and_unpad=lambda last, positions, hidden: (
            last,
            positions,
            hidden,
        ),
        runner=SimpleNamespace(pcp_manager=None),
        pcp_size=1,
        num_speculative_tokens=1,
        parallel_drafting=True,
        _dspark_draft_buffer=torch.zeros((2, 2), dtype=torch.int64),
        _dspark_seed_buffer=torch.tensor([7, 8], dtype=torch.int64),
        device=torch.device("cpu"),
    )
    token_indices = torch.tensor([1, 3], dtype=torch.int64)

    output = AscendSpecDecodeBaseProposer._run_merged_draft(
        proposer,
        num_input_tokens=4,
        batch_size=2,
        token_indices_to_sample=token_indices,
        target_positions=torch.arange(4, dtype=torch.int32),
        inputs_embeds=None,
        multi_steps_attn_metadata=[],
        num_tokens=4,
    )

    torch.testing.assert_close(
        model.compute_logits_input,
        hidden_states[token_indices],
    )
    torch.testing.assert_close(
        output,
        torch.zeros((2, 1), dtype=torch.int64),
    )
