from types import SimpleNamespace

import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.dspark_proposer import AscendDsparkProposer


class _ContextRecorder:
    def precompute_and_store_context_kv(self, hidden_states, positions, slot_mapping):
        self.hidden_states = hidden_states.clone()
        self.positions = positions.clone()
        self.slot_mapping = slot_mapping.clone()


def _make_proposer():
    hidden_states = torch.arange(64, dtype=torch.float32).reshape(16, 4)
    positions = torch.zeros(16, dtype=torch.int32)
    positions[:3] = torch.tensor([10, 11, 12], dtype=torch.int32)
    slots = torch.zeros(16, dtype=torch.int32)
    slots[:3] = torch.tensor([110, -1, 112], dtype=torch.int32)
    return SimpleNamespace(
        _dflash_num_context=3,
        _dflash_hidden_states=hidden_states,
        _context_positions_buffer=positions,
        _context_slot_mapping_buffer=slots,
        input_ids=torch.zeros(16, dtype=torch.int32),
        positions=torch.zeros(16, dtype=torch.int32),
        model=_ContextRecorder(),
        use_cuda_graph=True,
        _dspark_context_kv_precomputed=False,
        _precompute_live_context_kv=lambda: None,
    )


def _bind_live_context_precompute(proposer):
    proposer._precompute_live_context_kv = lambda: AscendDsparkProposer._precompute_live_context_kv(proposer)


def test_dspark_precompute_filters_rejected_context_rows():
    proposer = _make_proposer()
    _bind_live_context_precompute(proposer)

    AscendDsparkProposer._precompute_live_context_kv(proposer)

    torch.testing.assert_close(
        proposer.model.hidden_states,
        proposer._dflash_hidden_states[torch.tensor([0, 2])],
    )
    torch.testing.assert_close(
        proposer.model.positions,
        torch.tensor([10, 12], dtype=torch.int32),
    )
    torch.testing.assert_close(
        proposer.model.slot_mapping,
        torch.tensor([110, 112], dtype=torch.int32),
    )


def test_dspark_eager_build_precomputes_live_context():
    proposer = _make_proposer()
    _bind_live_context_precompute(proposer)

    model_inputs = AscendDsparkProposer.build_model_inputs_first_pass(
        proposer,
        num_input_tokens=8,
    )

    assert proposer.model.hidden_states.shape[0] == 2
    assert model_inputs["input_ids"].shape[0] == 8
    assert model_inputs["positions"].shape[0] == 8


def test_dspark_query_graph_skips_precomputed_context():
    proposer = _make_proposer()
    _bind_live_context_precompute(proposer)
    proposer._dspark_context_kv_precomputed = True

    model_inputs = AscendDsparkProposer.build_model_inputs_first_pass(
        proposer,
        num_input_tokens=8,
    )

    assert not hasattr(proposer.model, "hidden_states")
    assert model_inputs["input_ids"].shape[0] == 8


def test_dspark_graph_precomputes_context_before_query_replay():
    proposer = _make_proposer()
    _bind_live_context_precompute(proposer)
    forward_context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL)

    assert AscendDsparkProposer.prepare_dspark_context_kv_for_graph(
        proposer,
        forward_context,
    )
    assert proposer._dspark_context_kv_precomputed
    assert proposer.model.hidden_states.shape[0] == 2


def test_dspark_context_precompute_is_not_split_from_eager_query():
    proposer = _make_proposer()
    _bind_live_context_precompute(proposer)
    forward_context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE)

    assert not AscendDsparkProposer.prepare_dspark_context_kv_for_graph(
        proposer,
        forward_context,
    )
    assert not proposer._dspark_context_kv_precomputed
    assert not hasattr(proposer.model, "hidden_states")


def test_dspark_stabilizes_graph_padding_metadata():
    proposer = SimpleNamespace(
        _dspark_query_tokens_per_req=8,
        _adjust_tensor=lambda tensor, size: torch.nn.functional.pad(
            tensor,
            (0, size - tensor.shape[0]),
        ),
    )
    metadata = SimpleNamespace(
        seq_lens=torch.tensor([1010, 1012, 0, 0], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([1010, 1012], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([1010, 1012], dtype=torch.int32),
        actual_seq_lengths_q=[8, 8],
    )

    AscendDsparkProposer._stabilize_padded_graph_metadata(
        proposer,
        metadata,
        actual_num_reqs=2,
        padded_num_reqs=4,
    )

    torch.testing.assert_close(
        metadata.seq_lens,
        torch.tensor([1010, 1012, 1, 1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata._seq_lens_cpu,
        torch.tensor([1010, 1012, 1, 1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.seq_lens_cpu,
        torch.tensor([1010, 1012, 1, 1], dtype=torch.int32),
    )
    assert metadata.actual_seq_lengths_q == [8, 8, 8, 8]


def test_dspark_uses_dflash_graph_capture_metadata_path():
    assert AscendDsparkProposer.dummy_run is AscendDflashProposer.dummy_run
