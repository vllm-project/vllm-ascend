from types import SimpleNamespace

import torch
from vllm.compilation import monitor as cudagraph_monitor
from vllm.config import CUDAGraphMode

from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer


class _ContextRecorder:
    def precompute_and_store_context_kv(self, hidden_states, positions, slot_mapping):
        if isinstance(slot_mapping, list):
            slot_mapping = slot_mapping[0]
        self.hidden_states = hidden_states.clone()
        self.positions = positions.clone()
        self.slot_mapping = slot_mapping.clone()


class _GraphRecorder:
    def __init__(self):
        self.calls = 0

    def __call__(self, hidden_states, positions, slot_mapping):
        self.calls += 1
        self.hidden_states = hidden_states
        self.positions = positions
        self.slot_mapping = slot_mapping
        return hidden_states


def _make_proposer():
    hidden_states = torch.arange(64, dtype=torch.float32).reshape(16, 4)
    positions = torch.zeros(16, dtype=torch.int32)
    positions[:3] = torch.tensor([10, 11, 12], dtype=torch.int32)
    slots = torch.zeros(16, dtype=torch.int32)
    slots[:3] = torch.tensor([110, -1, 112], dtype=torch.int32)
    proposer = SimpleNamespace(
        _dflash_num_context=3,
        _dflash_hidden_states=hidden_states,
        _context_positions_buffer=positions,
        _context_slot_mapping_buffers=[slots],
        _per_group_context_slot_mapping_buffers={0: slots},
        _per_group_query_slot_mapping_buffers={
            0: torch.ones(16, dtype=torch.int32)
        },
        input_ids=torch.zeros(16, dtype=torch.int32),
        positions=torch.zeros(16, dtype=torch.int32),
        model=_ContextRecorder(),
        use_cuda_graph=True,
        max_query_tokens=16,
        _dspark_query_tokens_per_req=8,
        _dspark_context_kv_graph=None,
        _dspark_context_kv_precomputed=False,
        _precompute_live_context_kv=lambda: None,
    )
    proposer._primary_context_slot_mapping = lambda: (
        AscendDSparkProposer._primary_context_slot_mapping(proposer)
    )
    proposer._use_dspark_context_kv_bucket_graph = lambda _: False
    proposer._get_dspark_context_kv_bucket = lambda num_context: (
        AscendDSparkProposer._get_dspark_context_kv_bucket(
            proposer,
            num_context,
        )
    )
    return proposer


def _bind_live_context_precompute(proposer):
    proposer._precompute_live_context_kv = lambda: (
        AscendDSparkProposer._precompute_live_context_kv(proposer)
    )


def test_dspark_precompute_filters_rejected_context_rows():
    proposer = _make_proposer()
    _bind_live_context_precompute(proposer)

    AscendDSparkProposer._precompute_live_context_kv(proposer)

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

    AscendDSparkProposer.build_model_inputs_first_pass(
        proposer,
        num_input_tokens=8,
        context_slots=proposer._context_slot_mapping_buffers,
    )

    assert proposer.model.hidden_states.shape[0] == 2


def test_dspark_query_graph_skips_precomputed_context():
    proposer = _make_proposer()
    _bind_live_context_precompute(proposer)
    proposer._dspark_context_kv_precomputed = True

    AscendDSparkProposer.build_model_inputs_first_pass(
        proposer,
        num_input_tokens=8,
        context_slots=proposer._context_slot_mapping_buffers,
    )

    assert not hasattr(proposer.model, "hidden_states")


def test_dspark_graph_precomputes_context_before_query_replay():
    proposer = _make_proposer()
    _bind_live_context_precompute(proposer)
    forward_context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL)

    assert AscendDSparkProposer.prepare_dspark_context_kv_for_graph(
        proposer,
        forward_context,
    )
    assert proposer._dspark_context_kv_precomputed
    assert proposer.model.hidden_states.shape[0] == 2


def test_dspark_context_kv_bucket_uses_query_block_width():
    proposer = _make_proposer()

    assert AscendDSparkProposer._get_dspark_context_kv_bucket(proposer, 1) == 8
    assert AscendDSparkProposer._get_dspark_context_kv_bucket(proposer, 9) == 16
    assert AscendDSparkProposer._get_dspark_context_kv_bucket(proposer, 16) == 16


def test_dspark_context_kv_bucket_rejects_unsupported_shape():
    proposer = _make_proposer()

    assert AscendDSparkProposer._get_dspark_context_kv_bucket(proposer, 0) is None
    assert AscendDSparkProposer._get_dspark_context_kv_bucket(proposer, 17) is None


def test_dspark_bucket_graph_restores_descriptor_and_capture_monitor(
    monkeypatch,
):
    monkeypatch.setattr(
        cudagraph_monitor,
        "cudagraph_capturing_enabled",
        False,
    )
    proposer = _make_proposer()
    graph = _GraphRecorder()
    query_descriptor = object()
    proposer._dspark_context_kv_graph = graph
    forward_context = SimpleNamespace(batch_descriptor=query_descriptor)

    assert AscendDSparkProposer._precompute_context_kv_bucket_graph(
        proposer,
        forward_context,
    )
    assert graph.calls == 1
    assert graph.hidden_states.shape[0] == 8
    assert graph.positions.shape[0] == 8
    assert graph.slot_mapping.shape[0] == 8
    assert forward_context.batch_descriptor is query_descriptor
    assert not cudagraph_monitor.cudagraph_capturing_enabled
    assert proposer._dspark_context_kv_precomputed


def test_dspark_graph_initializes_only_graph_visible_padding(monkeypatch):
    monkeypatch.setenv(
        "VLLM_ASCEND_ENABLE_GLM_DSPARK_CONTEXT_KV_BUCKET_GRAPH",
        "1",
    )
    proposer = _make_proposer()
    proposer.parallel_drafting_token_id = 99
    proposer._dflash_hidden_states.fill_(1)
    proposer._context_positions_buffer.fill_(1)
    proposer._context_slot_mapping_buffers[0].fill_(1)
    proposer.input_ids.fill_(1)
    proposer.positions.fill_(1)
    proposer._slot_mapping_buffer = torch.ones(16, dtype=torch.int32)
    proposer._per_group_query_slot_mapping_buffers = {
        0: proposer._slot_mapping_buffer
    }

    AscendDSparkProposer._initialize_graph_padding(
        proposer,
        num_context=3,
        num_query_total=8,
    )

    assert torch.all(proposer._dflash_hidden_states[:3] == 1)
    assert torch.all(proposer._dflash_hidden_states[3:8] == 0)
    assert torch.all(proposer._dflash_hidden_states[8:] == 1)
    assert torch.all(proposer._context_positions_buffer[3:8] == 0)
    assert torch.all(proposer._context_slot_mapping_buffers[0][3:8] == -1)
    assert torch.all(proposer.input_ids[:8] == 1)
    assert torch.all(proposer.input_ids[8:] == 99)
    assert torch.all(proposer.positions[:8] == 1)
    assert torch.all(proposer.positions[8:] == 0)
    assert torch.all(proposer._slot_mapping_buffer[:8] == 1)
    assert torch.all(proposer._slot_mapping_buffer[8:] == -1)


def test_dspark_context_precompute_is_not_split_from_eager_query():
    proposer = _make_proposer()
    _bind_live_context_precompute(proposer)
    forward_context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE)

    assert not AscendDSparkProposer.prepare_dspark_context_kv_for_graph(
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

    AscendDSparkProposer._stabilize_padded_graph_metadata(
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


def test_glm_dspark_uses_dflash_graph_capture_metadata_path(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        AscendDflashProposer,
        "dummy_run",
        lambda *args, **kwargs: sentinel,
    )
    proposer = SimpleNamespace(_dspark_sample_from_anchor=False)

    result = AscendDSparkProposer.dummy_run(proposer, num_tokens=8)

    assert result is sentinel
