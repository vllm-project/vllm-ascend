# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.models.deepseek_v41 import decoder_replay_layers as replay_mod
from vllm_ascend.models.deepseek_v41 import model as model_mod
from vllm_ascend.models.deepseek_v41.decoder_replay_sp import SPReplayPlan
from vllm_ascend.models.deepseek_v41.model import AscendDeepseekV41LLMForCausalLM, DeepseekV41Model


@pytest.mark.parametrize(("pcp_size", "prefix_caching"), [(1, True), (2, True), (2, False)])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_decoder_replay_enabled_with_or_without_prefix_caching(pcp_size, prefix_caching, dp_size):
    model = SimpleNamespace(
        config=SimpleNamespace(kv_source_layer_ids=[0], sliding_window=2, engram_layer_ids=[]),
        end_layer=3,
        decoder_replay_start=3,
        use_sequence_parallel=False,
        _run_replay_layers=lambda *states: states,
        topk_indices_buffer=torch.zeros(8, 2),
        candidate_indices_buffer=torch.zeros(8, 2),
        candidate_lengths_buffer=torch.zeros(8),
        topk_lengths_buffer=torch.zeros(8),
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(swa_bounded_replay=True, enable_prefix_caching=prefix_caching),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=pcp_size,
            data_parallel_size=dp_size,
            decode_context_parallel_size=1,
            use_ubatching=False,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
        speculative_config=None,
        lora_config=None,
    )
    replay = replay_mod.make_decoder_replay(model, config)
    assert replay is not None
    assert model.decoder_replay_start == 1


@pytest.mark.parametrize(
    ("runner_v2", "pcp_size", "adaptive", "enabled"),
    [(True, 1, False, True), (False, 1, False, False), (True, 2, False, False), (True, 1, True, False)],
)
def test_sp_decoder_replay_requires_host_known_mrv2_layout(runner_v2, pcp_size, adaptive, enabled):
    model = SimpleNamespace(
        config=SimpleNamespace(kv_source_layer_ids=[0], sliding_window=128, engram_layer_ids=[]),
        end_layer=3,
        use_sequence_parallel=True,
        _run_replay_layers=lambda *states: states,
        topk_indices_buffer=torch.empty(8, 2),
        candidate_indices_buffer=torch.empty(8, 2),
        candidate_lengths_buffer=torch.empty(8),
        topk_lengths_buffer=torch.empty(8),
    )
    config = SimpleNamespace(
        use_v2_model_runner=runner_v2,
        cache_config=SimpleNamespace(swa_bounded_replay=True),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=1,
            use_ubatching=False,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
        speculative_config=SimpleNamespace(
            draft_model_config=None, enable_adaptive_verification=adaptive, use_dspark=lambda: True
        ),
        lora_config=None,
    )
    replay = replay_mod.make_decoder_replay(model, config)
    assert (replay is not None) == enabled
    if enabled:
        assert replay.sequence_parallel


def test_packed_replay_compacts_sparse_lengths_with_indices(monkeypatch):
    buffers = [torch.arange(6) + offset for offset in (0, 10, 20, 30)]
    model = SimpleNamespace(
        config=SimpleNamespace(kv_source_layer_ids=[2], engram_layer_ids=[1], sliding_window=2),
        end_layer=5,
        use_sequence_parallel=False,
        topk_indices_buffer=buffers[0],
        candidate_indices_buffer=buffers[1],
        candidate_lengths_buffer=buffers[2],
        topk_lengths_buffer=buffers[3],
        _run_replay_layers=lambda hidden, optional, positions: (hidden,),
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(swa_bounded_replay=True),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_ubatching=False,
        ),
        speculative_config=None,
        lora_config=None,
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
    )
    replay = replay_mod.make_decoder_replay(model, config)
    assert replay is not None
    replay.rows = torch.tensor([4, 5])
    replay.num_actual_rows = 2
    replay.context_factory = nullcontext
    replay(torch.arange(6), None, torch.arange(6))
    for buffer, offset in zip(buffers, (0, 10, 20, 30)):
        assert buffer[:2].tolist() == [offset + 4, offset + 5]


def test_replay_scatter_preserves_row_zero_with_padding(monkeypatch):
    buffer = torch.arange(6).reshape(6, 1)
    hidden = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    replay = replay_mod.AscendDecoderReplayLayers(
        2, lambda hidden, mix, positions, ids: (hidden + 10, mix, hidden + 20), [buffer]
    )
    replay.rows = torch.tensor([0, 4, 5, 0])
    replay.num_actual_rows = 3
    replay.context_factory = nullcontext
    output, mix, aux = replay(hidden, hidden, torch.arange(6), None)
    assert output[:, 0].tolist() == [10, 0, 0, 0, 14, 15]
    assert aux[:, 0].tolist() == [20, 0, 0, 0, 24, 25]
    assert mix[:, 0].tolist() == [0, 0, 0, 0, 4, 5]
    assert buffer[:4, 0].tolist() == [0, 4, 5, 0]


def test_pcp_replay_empty_owner_runs_padded_rows_without_scattering(monkeypatch):
    buffer = torch.arange(4).reshape(4, 1)

    def run_layers(hidden, mix, positions):
        assert hidden.shape[0] == 2
        assert not hidden.count_nonzero()
        assert not mix.count_nonzero()
        assert not positions.count_nonzero()
        return hidden + 1, mix

    replay = replay_mod.AscendDecoderReplayLayers(2, run_layers, [buffer])
    replay.rows = torch.empty(0, dtype=torch.int64)
    replay.num_actual_rows = 0
    replay.num_padded_rows = 2
    replay.context_factory = nullcontext
    hidden, mix = replay(torch.arange(4).float().reshape(4, 1), torch.ones(4, 1), torch.arange(4))
    assert not hidden.count_nonzero()
    assert not mix.count_nonzero()
    assert buffer[:2, 0].tolist() == [0, 0]


def test_replay_without_rows_keeps_inputs():
    hidden = torch.arange(6)
    replay = replay_mod.AscendDecoderReplayLayers(2, lambda *states: states, [])
    assert replay(hidden, None) == (hidden, None)


def test_untrimmed_replica_enters_decoder_context_without_gathering():
    entered = []

    @contextmanager
    def decoder_context():
        entered.append(True)
        yield
        entered.append(False)

    hidden = torch.arange(6)

    def run_layers(*states):
        assert entered == [True]
        assert states[0] is hidden
        return states

    buffer = torch.arange(6)
    replay = replay_mod.AscendDecoderReplayLayers(2, run_layers, [buffer])
    replay.context_factory = decoder_context
    assert replay(hidden, None) == (hidden, None)
    assert entered == [True, False]
    assert torch.equal(buffer, torch.arange(6))


@pytest.fixture(params=[False, True], ids=["non_sp", "sp"])
def compiled_replay_model(monkeypatch, request):
    class Layer(torch.nn.Module):
        engram = None

        def __init__(self, index):
            super().__init__()
            self.layer_idx = index
            self.weight = torch.nn.Parameter(torch.eye(3) * (index + 2), requires_grad=False)

        def forward(self, positions, hidden, mix, unused, input_ids):
            return torch.nn.functional.linear(hidden, self.weight), mix + 1

        @staticmethod
        def hc_collapse(hidden, mix):
            return hidden.mean(dim=1)

    model = DeepseekV41Model.__new__(DeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.hc_mult = 1
    model.use_sequence_parallel = request.param
    model._mtp_hidden_buffer = None
    model.shared_attention_state = SimpleNamespace(reset=lambda: None)
    model.layers = torch.nn.ModuleList(Layer(i) for i in range(3))
    model.norm = torch.nn.Identity()
    model.end_layer = 3
    model.start_layer = 0
    model.decoder_replay_start = 1
    model.aux_hidden_state_layers = (1, 3)
    replay = replay_mod.AscendDecoderReplayLayers(2, model._run_replay_layers, [])
    replay.sequence_parallel = model.use_sequence_parallel
    model.decoder_replay_layers = replay
    # One-rank SP isolates the compile dispatch. Multi-rank data movement is
    # covered separately by the real Gloo tests.
    monkeypatch.setattr(model_mod, "is_forward_context_available", lambda: False)
    monkeypatch.setattr(model_mod.envs, "VLLM_MOE_SKIP_PADDING", False)
    monkeypatch.setattr(model_mod, "sp_shard", lambda tensor: tensor.clone())
    monkeypatch.setattr(model_mod, "sp_all_gather", lambda tensor: tensor.clone())
    monkeypatch.setattr(replay_mod, "get_tp_group", lambda: SimpleNamespace(device_group=None))
    monkeypatch.setattr(torch.distributed, "all_to_all_single", lambda output, source, **kwargs: output.copy_(source))
    graphs, compiled_calls = [], []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    # Match vLLM's unguarded dispatch: runtime state changes must be handled
    # outside the compiled entry, without relying on Dynamo recompilation.
    compiled = torch.compile(
        model.forward,
        backend=backend,
        fullgraph=True,
        options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
    )

    def compiled_entry(self, *args, **kwargs):
        compiled_calls.append(True)
        return compiled(*args, **kwargs)

    # Model the decorated backbone's __call__ with a real compiled callable;
    # the outer production forward chooses __call__ versus raw forward.
    monkeypatch.setattr(DeepseekV41Model, "__call__", compiled_entry)
    causal = SimpleNamespace(model=model)
    embeddings = torch.arange(18, dtype=torch.float32).reshape(6, 3)

    def forward(**kwargs):
        return AscendDeepseekV41LLMForCausalLM.forward(
            causal,
            torch.arange(6),
            torch.arange(6),
            inputs_embeds=embeddings,
            engram_lookups={},
            engram_mask=torch.empty(0, dtype=torch.bool),
            **kwargs,
        )

    try:
        yield SimpleNamespace(
            model=model, replay=replay, forward=forward, graphs=graphs, calls=compiled_calls, embeddings=embeddings
        )
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("replay_enabled", [False, True])
def test_decode_compiles_every_layer_including_decoder_tail(compiled_replay_model, replay_enabled):
    case = compiled_replay_model
    if not replay_enabled:
        case.model.decoder_replay_layers = None
    output, aux = case.forward()
    torch.testing.assert_close(output, case.embeddings * 24)
    torch.testing.assert_close(aux[0], case.embeddings)
    torch.testing.assert_close(aux[1], case.embeddings * 6)
    assert len(case.graphs) == 1
    # Check the graph contents, not just that compile() returned the right
    # values: an opaque replay op must not swallow the two late linear layers.
    linears = [node for node in case.graphs[0].graph.nodes if node.target == torch.nn.functional.linear]
    assert len(linears) == len(case.model.layers)
    assert all("dsv41_sp_decoder_replay" not in str(node.target) for node in case.graphs[0].graph.nodes)


def test_capture_engram_inputs_keep_compiled_entry(compiled_replay_model):
    case = compiled_replay_model
    entered = []

    @contextmanager
    def captured_engram_inputs(ids, positions, lookback, offsets, valid_tokens):
        entered.append((offsets.tolist(), valid_tokens.item()))
        yield {"engram_lookups": {}, "engram_mask": torch.empty(0, dtype=torch.bool)}

    case.model.captured_engram_inputs = captured_engram_inputs
    output, _ = case.forward(engram_query_start_loc=torch.tensor([0, 6]), engram_valid_token_count=torch.tensor(6))
    torch.testing.assert_close(output, case.embeddings * 24)
    assert entered == [([0, 6], 6)]
    assert len(case.calls) == 1
    assert sum(node.target == torch.nn.functional.linear for node in case.graphs[0].graph.nodes) == 3


def test_replay_plans_bypass_compiled_entry_then_return_to_decode(compiled_replay_model):
    case = compiled_replay_model
    case.forward()  # compile ordinary warmup/decode with no replay plan
    contexts = []

    @contextmanager
    def decoder_context():
        contexts.append("enter")
        try:
            yield
        finally:
            contexts.append("exit")

    for offsets, kept, rows in (([0, 6], [2], [4, 5]), ([0, 1, 6], [1, 2], [0, 4, 5])):
        case.replay.rows = torch.tensor(rows)
        case.replay.num_actual_rows = len(rows)
        case.replay.context_factory = decoder_context
        if case.model.use_sequence_parallel:
            case.replay.sp_plan = SPReplayPlan.build(
                np.array(offsets), np.array(kept), 6, len(rows), 1, 0, torch.device("cpu")
            )
        output, aux = case.forward()
        expected = torch.zeros_like(case.embeddings)
        expected[rows] = case.embeddings[rows] * 24
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(aux[0], case.embeddings)
        torch.testing.assert_close(aux[1], expected / 4)
        assert len(case.calls) == 1
    assert contexts == ["enter", "exit", "enter", "exit"]

    # An untrimmed DP peer must enter its decoder context too, even without
    # local rows to compact. It must not reuse the original compiled metadata.
    case.replay.rows = None
    case.replay.sp_plan = None
    output, _ = case.forward()
    torch.testing.assert_close(output, case.embeddings * 24)
    assert len(case.calls) == 1
    assert contexts[-2:] == ["enter", "exit"]

    # prepare() clears this context for the next ordinary decode/capture.
    case.replay.context_factory = None
    output, _ = case.forward()
    torch.testing.assert_close(output, case.embeddings * 24)
    assert len(case.calls) == 2
    assert len(case.graphs) == 1


def test_model_replays_only_layers_after_kv_sources(monkeypatch):
    seen = []

    class Layer:
        engram = None

        def __init__(self, index):
            self.layer_idx = index

        def __call__(self, positions, hidden, mix, unused, input_ids):
            seen.append((self.layer_idx, positions.tolist()))
            return hidden + 1, mix

        @staticmethod
        def hc_collapse(hidden, mix):
            return hidden.mean(dim=1)

    model = DeepseekV41Model.__new__(DeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.hc_mult = 1
    model.use_sequence_parallel = False
    model._mtp_hidden_buffer = None
    model.shared_attention_state = SimpleNamespace(reset=lambda: None)
    model.layers = [Layer(i) for i in range(3)]
    model.norm = lambda hidden: hidden
    model.end_layer = 3
    model.start_layer = 0
    model.needs_moe_input_ids = False
    model.decoder_replay_start = 1
    model.aux_hidden_state_layers = (1, 3)
    replay = replay_mod.AscendDecoderReplayLayers(2, model._run_replay_layers, [])
    replay.rows = torch.tensor([2, 3])
    replay.num_actual_rows = 2
    replay.context_factory = nullcontext
    model.decoder_replay_layers = replay
    output, aux = DeepseekV41Model.forward(
        model,
        torch.arange(4),
        torch.arange(4),
        None,
        inputs_embeds=torch.ones(4, 1),
        engram_lookups={},
        engram_mask=torch.empty(0, dtype=torch.bool),
    )
    assert seen == [(0, [0, 1, 2, 3]), (1, [2, 3]), (2, [2, 3])]
    assert output[:, 0].tolist() == [0, 0, 4, 4]
    assert aux[0][:, 0].tolist() == [1, 1, 1, 1]
    assert aux[1][:, 0].tolist() == [0, 0, 3, 3]
