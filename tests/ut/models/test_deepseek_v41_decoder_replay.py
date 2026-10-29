# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.models.deepseek_v41 import decoder_replay_layers as replay_mod
from vllm_ascend.models.deepseek_v41.model import DeepseekV41Model


@pytest.mark.parametrize(("pcp_size", "prefix_caching"), [(1, True), (2, True), (2, False)])
def test_decoder_replay_enabled_with_or_without_prefix_caching(pcp_size, prefix_caching):
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
            data_parallel_size=1,
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
    monkeypatch.setattr(replay_mod, "enable_dsa_cp", lambda: False)
    replay(torch.arange(6), None, torch.arange(6))
    for buffer, offset in zip(buffers, (0, 10, 20, 30)):
        assert buffer[:2].tolist() == [offset + 4, offset + 5]


def test_replay_scatter_preserves_row_zero_with_padding(monkeypatch):
    monkeypatch.setattr(replay_mod, "enable_dsa_cp", lambda: False)
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
    monkeypatch.setattr(replay_mod, "enable_dsa_cp", lambda: False)
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


def test_model_replays_only_layers_after_kv_sources(monkeypatch):
    monkeypatch.setattr(replay_mod, "enable_dsa_cp", lambda: False)
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
