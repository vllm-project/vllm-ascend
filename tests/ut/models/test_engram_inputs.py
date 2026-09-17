# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.models.deepseek_v41 import model as implementation


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(implementation, "get_ascend_config", lambda: SimpleNamespace(enable_engram=True))
    cls = implementation.DeepseekV41Model
    shell = SimpleNamespace(
        config=SimpleNamespace(engram_layer_ids=[1, 14], engram_max_ngram_size=4, engram_n_heads=8),
        layers=[SimpleNamespace(engram=SimpleNamespace(embed=SimpleNamespace(width=32))) for _ in range(15)],
        engram_rotation=torch.eye(32),
        _engram_max_tokens=16,
        _engram_input_buffers=None,
    )
    shell.prepare_engram_graph_inputs = cls.prepare_engram_graph_inputs.__get__(shell)
    shell.prepare_engram_inputs = cls.prepare_engram_inputs.__get__(shell)
    return shell


def test_capture_first_reuses_storage_and_refreshes_only_runtime_rows(model):
    captured = model.prepare_engram_graph_inputs(4)
    pointers = {layer: value.data_ptr() for layer, value in captured["engram_lookups"].items()}
    for count, padded in [(9, 12), (1, 4), (0, 4), (6, 8)]:
        values = {layer: torch.full((count, 768), float(count), dtype=torch.bfloat16) for layer in (1, 14)}
        model.prepare_engram = lambda *args, values=values, count=count: (values, torch.ones(count, dtype=torch.bool))
        # Detect accidental clearing of the entire capacity on a small decode.
        for buffer in captured["engram_lookups"].values():
            buffer[padded:].fill_(123)
        actual = model.prepare_engram_inputs(torch.arange(count), torch.arange(count), padded)
        assert actual["engram_mask"].data_ptr() == captured["engram_mask"].data_ptr()
        assert actual["engram_mask"][:count].all()
        assert not actual["engram_mask"][count:padded].any()
        for layer, buffer in actual["engram_lookups"].items():
            assert buffer.shape == (16, 768) and buffer.data_ptr() == pointers[layer]
            assert torch.equal(buffer[:count], values[layer])
            assert not buffer[count:padded].any()
            assert (buffer[padded:] == 123).all()


def test_disabled_engram_capture_and_replay_do_not_access_layers(model, monkeypatch):
    monkeypatch.setattr(implementation, "get_ascend_config", lambda: SimpleNamespace(enable_engram=False))
    model.layers = [SimpleNamespace(engram=None) for _ in range(15)]
    for result in (model.prepare_engram_graph_inputs(4), model.prepare_engram_inputs(None, torch.arange(4), 4)):
        assert result["engram_lookups"] == {}
        assert result["engram_mask"].numel() == 0


def test_sequence_parallel_slices_capacity_before_sharding(monkeypatch):
    seen = []

    class Layer:
        layer_idx = 1

        def __init__(self):
            self.engram = SimpleNamespace(wkv=self.project, q_weight=torch.ones(1, 32), k_weight=torch.ones(1, 32))

        def project(self, value):
            seen.append(value.clone())
            return torch.zeros(value.shape[0], 64, dtype=torch.bfloat16)

        def __call__(self, positions, hidden, pre_mix, *args, **kwargs):
            return hidden, pre_mix

        def hc_collapse(self, hidden, pre_mix):
            return hidden[:, 0]

    monkeypatch.setattr(implementation, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True))
    monkeypatch.setattr(implementation.envs, "VLLM_MOE_SKIP_PADDING", False)
    monkeypatch.setattr(implementation, "sp_shard", lambda value: value.chunk(2)[1])
    monkeypatch.setattr(implementation, "sp_all_gather", lambda value: torch.cat([value, value]))
    monkeypatch.setattr(implementation, "engram_gate", lambda hidden, *args: hidden)
    shell = SimpleNamespace(
        use_sequence_parallel=True,
        hc_mult=1,
        config=SimpleNamespace(hidden_size=32, rms_norm_eps=1e-6),
        shared_attention_state=SimpleNamespace(reset=lambda: None),
        aux_hidden_state_layers=[],
        needs_moe_input_ids=False,
        layers=[Layer()],
        engram_rotation=torch.eye(32),
        norm=lambda value: value,
    )
    lookup = torch.arange(16 * 96).reshape(16, 96).bfloat16()
    implementation.DeepseekV41Model.forward(
        shell,
        torch.arange(4),
        torch.arange(4),
        None,
        inputs_embeds=torch.zeros(4, 32, dtype=torch.bfloat16),
        engram_lookups={1: lookup},
        engram_mask=torch.ones(16, dtype=torch.bool),
    )
    assert torch.equal(seen[0], lookup[:4].chunk(2)[1])


@pytest.mark.parametrize("mode,capture", [("NONE", False), ("FULL", False), ("FULL", True)])
def test_runner_sync_preparation_and_capture_through_vl(monkeypatch, mode, capture):
    from vllm_ascend.models.deepseek_v41.vl_model import AscendDeepseekV41ForCausalLM
    from vllm_ascend.worker import model_runner_v1 as runner_module

    calls = []
    history_inputs = object()

    class LanguageModel(torch.nn.Module):
        def prepare_engram_inputs(self, *args):
            assert args[-1] is history_inputs
            calls.append("sync")
            return {}

        def prepare_engram_graph_inputs(self, *args):
            calls.append("capture")
            return {}

        def forward(self, *args, **kwargs):
            return 42

    wrapper = AscendDeepseekV41ForCausalLM.__new__(AscendDeepseekV41ForCausalLM)
    torch.nn.Module.__init__(wrapper)
    wrapper.language_model = LanguageModel()
    context = SimpleNamespace(cudagraph_runtime_mode=getattr(runner_module.CUDAGraphMode, mode))
    monkeypatch.setattr(runner_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    runner = SimpleNamespace(
        model=wrapper,
        enable_enpu=False,
        _engram_capture_active=capture,
        _get_engram_history_inputs=lambda: history_inputs,
        _update_full_graph_params_if_needed=lambda *args: None,
    )
    assert runner_module.NPUModelRunner._model_forward(runner, 4) == 42
    assert calls == ["capture" if capture else "sync"]


@pytest.mark.parametrize("num_reqs", [0, 2])
def test_runner_engram_history_selects_swa_group_and_full_requests(monkeypatch, num_reqs):
    from vllm_ascend.worker import model_runner_v1 as runner_module

    pages = torch.tensor([[7, 8, 9], [12, 13, 14], [99, 99, 99]], dtype=torch.int32)
    boundaries = torch.tensor([0, 3, 9, 99], dtype=torch.int32)
    groups = [
        SimpleNamespace(layer_names=["long_kv"], kv_cache_spec=object()),
        SimpleNamespace(layer_names=["swa"], kv_cache_spec=object()),
    ]
    monkeypatch.setattr(
        runner_module, "get_storage_block_size", lambda spec: 4 if spec is groups[1].kv_cache_spec else 8
    )
    runner = SimpleNamespace(
        model=SimpleNamespace(engram_cache_layer_name="swa"),
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        query_start_loc=SimpleNamespace(cpu=boundaries),
        input_batch=SimpleNamespace(
            num_reqs=num_reqs,
            block_table=[None, SimpleNamespace(get_cpu_tensor=lambda: pages)],
        ),
    )
    actual_boundaries, actual_pages, block_size = runner_module.NPUModelRunner._get_engram_history_inputs(runner)
    torch.testing.assert_close(actual_boundaries, boundaries[: num_reqs + 1])
    torch.testing.assert_close(actual_pages, pages[:num_reqs])
    assert actual_boundaries.data_ptr() == boundaries.data_ptr()
    if num_reqs:
        assert actual_pages.data_ptr() == pages.data_ptr()
    assert block_size == 4


def test_runner_disabled_engram_does_not_read_batch():
    from vllm_ascend.worker import model_runner_v1 as runner_module

    runner = SimpleNamespace(model=SimpleNamespace(engram_cache_layer_name=None))
    assert runner_module.NPUModelRunner._get_engram_history_inputs(runner) is None


def test_runner_dummy_engram_does_not_read_live_batch(monkeypatch):
    from unittest.mock import Mock

    from vllm_ascend.worker import model_runner_v1 as runner_module

    model = Mock(return_value=42)
    model.prepare_engram_inputs.return_value = {}
    runner = SimpleNamespace(model=model, enable_enpu=False, _update_full_graph_params_if_needed=lambda *args: None)
    monkeypatch.setattr(
        runner_module,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=runner_module.CUDAGraphMode.NONE),
    )
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    assert runner_module.NPUModelRunner._model_forward(runner, 4, is_dummy_run=True) == 42
    model.prepare_engram_inputs.assert_called_once_with(None, None, 4, None)


def test_engram_history_consumes_explicit_full_request_inputs(model):
    from unittest.mock import Mock

    pages = torch.tensor([[7, 8, 9], [12, 13, 14]], dtype=torch.int32)
    boundaries = torch.tensor([0, 3, 9], dtype=torch.int32)
    hashes = torch.zeros((9, 2, 24), dtype=torch.int64)
    mask = torch.ones(9, dtype=torch.bool)
    model.engram_history = SimpleNamespace(update=Mock(return_value=(hashes, mask)))
    table = model.layers[1].engram.embed
    table.route_many = Mock(return_value=[torch.zeros(9, 24, 32), torch.zeros(9, 24, 32)])
    inputs, positions = torch.arange(12), torch.arange(12)
    implementation.DeepseekV41Model.prepare_engram(model, inputs, positions, (boundaries, pages, 4))
    args = model.engram_history.update.call_args.args
    torch.testing.assert_close(args[0], inputs[:9])
    torch.testing.assert_close(args[1], positions[:9])
    torch.testing.assert_close(args[2], torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1]))
    assert args[3] is pages and args[4] == 4


def test_engram_dummy_routes_empty_hashes(model):
    from unittest.mock import Mock

    model.engram_history = SimpleNamespace(update=Mock())
    table = model.layers[1].engram.embed
    table.route_many = Mock(return_value=[torch.zeros(0, 24, 32), torch.zeros(0, 24, 32)])
    implementation.DeepseekV41Model.prepare_engram(model, torch.arange(4), torch.arange(4), None)
    model.engram_history.update.assert_not_called()
    ids = table.route_many.call_args.args[1]
    assert len(ids) == 2 and all(value.shape == (0, 24) for value in ids)
