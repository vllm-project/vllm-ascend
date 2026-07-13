from types import SimpleNamespace

import torch
from vllm.config import CUDAGraphMode

import vllm_ascend.spec_decode.dflash_proposer as dflash_module
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer


class _ContextRecorder:
    def __init__(self):
        self.calls = 0

    def precompute_and_store_context_kv(self, hidden_states, positions, slot_mapping):
        self.calls += 1
        self.hidden_states = hidden_states
        self.positions = positions
        self.slot_mapping = slot_mapping


def _make_proposer():
    return SimpleNamespace(
        _dflash_num_context=3,
        _dflash_hidden_states=torch.zeros(16, 4),
        _context_positions_buffer=torch.zeros(16, dtype=torch.int32),
        _context_slot_mapping_buffer=torch.zeros(16, dtype=torch.int32),
        input_ids=torch.zeros(16, dtype=torch.int32),
        positions=torch.zeros(16, dtype=torch.int32),
        model=_ContextRecorder(),
        _use_dspark_block_contract=lambda: True,
    )


def test_dspark_eager_precomputes_actual_context(monkeypatch):
    proposer = _make_proposer()
    monkeypatch.setattr(
        dflash_module,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE),
    )

    model_inputs = AscendDflashProposer.build_model_inputs_first_pass(
        proposer,
        num_input_tokens=8,
    )

    assert proposer.model.calls == 1
    assert proposer.model.hidden_states.shape[0] == 3
    assert proposer.model.positions.shape[0] == 3
    assert proposer.model.slot_mapping.shape[0] == 3
    assert model_inputs["input_ids"].shape[0] == 8
    assert model_inputs["positions"].shape[0] == 8


def test_dspark_graph_precomputes_fixed_padded_context(monkeypatch):
    proposer = _make_proposer()
    monkeypatch.setattr(
        dflash_module,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL),
    )

    model_inputs = AscendDflashProposer.build_model_inputs_first_pass(
        proposer,
        num_input_tokens=8,
    )

    assert proposer.model.calls == 1
    assert proposer.model.hidden_states.shape[0] == 8
    assert proposer.model.positions.shape[0] == 8
    assert proposer.model.slot_mapping.shape[0] == 8
    assert model_inputs["input_ids"].shape[0] == 8
    assert model_inputs["positions"].shape[0] == 8
