from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from vllm.config import CompilationMode, CUDAGraphMode

import vllm_ascend.spec_decode as spec_decode_module
import vllm_ascend.spec_decode.dflash_proposer as dflash_module
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer


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


class _KernelRecorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        assert grid == (1,)

        def launch(**kwargs):
            self.calls.append(kwargs)

        return launch


def test_dspark_input_preparation_uses_device_kernel(monkeypatch):
    kernel = _KernelRecorder()
    monkeypatch.setattr(
        dflash_module,
        "copy_and_expand_dflash_inputs_kernel_single_grid",
        kernel,
    )
    proposer = SimpleNamespace(
        num_speculative_tokens=2,
        max_query_tokens=12,
        use_cuda_graph=True,
        _dflash_hidden_states=torch.zeros(12, 4),
        _context_positions_buffer=torch.zeros(12, dtype=torch.int32),
        _context_slot_mapping_buffer=torch.zeros(12, dtype=torch.int32),
        input_ids=torch.zeros(12, dtype=torch.int32),
        positions=torch.zeros(12, dtype=torch.int32),
        _slot_mapping_buffer=torch.zeros(12, dtype=torch.int32),
        parallel_drafting_token_id=99,
        kernel_block_size=4,
        device=torch.device("cpu"),
        arange_dflash=torch.arange(32, dtype=torch.int32),
        token_arange_np=np.arange(32, dtype=np.int32),
    )
    cad = SimpleNamespace(
        num_reqs=2,
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 5], dtype=torch.int32),
        seq_lens=torch.tensor([5, 13], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([5, 13], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([5, 13], dtype=torch.int32),
        block_table_tensor=torch.tensor(
            [[10, 11, 12, 13], [20, 21, 22, 23]],
            dtype=torch.int32,
        ),
        slot_mapping=torch.arange(5, dtype=torch.int32),
        actual_seq_lengths_q=[2, 3],
        decode_token_per_req=1,
        max_query_len=3,
        max_seq_len=13,
        positions=None,
        positions_cpu=torch.empty(0, dtype=torch.int32),
        causal=True,
        attn_mask=object(),
        attn_state=None,
    )

    result = AscendDflashProposer._set_dspark_inputs_first_pass(
        proposer,
        target_token_ids=torch.arange(5, dtype=torch.int32),
        next_token_ids=torch.tensor([7, 8], dtype=torch.int32),
        target_positions=torch.tensor([3, 4, 10, 11, 12], dtype=torch.int32),
        target_hidden_states=torch.ones(5, 4),
        cad=cad,
        num_rejected_tokens_gpu=torch.tensor([1, 0], dtype=torch.int32),
    )

    assert result[0] == 6
    assert len(kernel.calls) == 1
    assert kernel.calls[0]["RECOMPUTE_CONTEXT_SLOTS"] is True
    torch.testing.assert_close(
        cad.query_start_loc,
        torch.tensor([0, 3, 6], dtype=torch.int32),
    )
    torch.testing.assert_close(cad.seq_lens, torch.tensor([7, 16], dtype=torch.int32))
    torch.testing.assert_close(
        cad._seq_lens_cpu,
        torch.tensor([7, 16], dtype=torch.int32),
    )
    assert cad.actual_seq_lengths_q == [3, 3]
    assert cad.max_seq_len == 16


def test_dspark_padded_graph_metadata_uses_static_sentinels():
    def adjust_tensor(tensor, desired_size):
        if tensor.shape[0] >= desired_size:
            return tensor[:desired_size]
        padding = torch.zeros(
            desired_size - tensor.shape[0],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat((tensor, padding))

    proposer = SimpleNamespace(
        num_speculative_tokens=2,
        _adjust_tensor=adjust_tensor,
    )
    cad = SimpleNamespace(
        seq_lens=torch.tensor([7, 16, 0], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([7, 16], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([7, 16], dtype=torch.int32),
        actual_seq_lengths_q=[3, 3],
    )

    AscendDSparkProposer._stabilize_padded_graph_metadata(
        proposer,
        cad,
        actual_num_reqs=2,
        padded_num_reqs=3,
    )

    torch.testing.assert_close(
        cad.seq_lens,
        torch.tensor([7, 16, 1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        cad._seq_lens_cpu,
        torch.tensor([7, 16, 1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        cad.seq_lens_cpu,
        torch.tensor([7, 16, 1], dtype=torch.int32),
    )
    assert cad.actual_seq_lengths_q == [3, 3, 3]


def test_dspark_draft_config_disables_inner_compile(monkeypatch):
    @dataclass
    class FakeCompilationConfig:
        mode: CompilationMode

    @dataclass
    class FakeVllmConfig:
        compilation_config: FakeCompilationConfig

    base_config = FakeVllmConfig(
        compilation_config=FakeCompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
        )
    )
    monkeypatch.setattr(
        AscendDflashProposer,
        "_create_draft_vllm_config",
        lambda self: base_config,
    )
    proposer = object.__new__(AscendDSparkProposer)
    proposer.use_cuda_graph = True

    draft_config = AscendDSparkProposer._create_draft_vllm_config(proposer)

    assert draft_config.compilation_config.mode == CompilationMode.NONE
    assert base_config.compilation_config.mode == CompilationMode.VLLM_COMPILE


def test_dspark_checkpoint_selects_dedicated_proposer(monkeypatch):
    monkeypatch.setattr(
        spec_decode_module,
        "AscendDSparkProposer",
        lambda vllm_config, device, runner: "dspark",
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["DSparkDraftModel"],
                )
            )
        )
    )

    proposer = spec_decode_module.get_spec_decode_method(
        "dflash",
        vllm_config,
        torch.device("cpu"),
        runner=None,
    )

    assert proposer == "dspark"
