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


def test_vmm_refreshes_captured_outputs_without_routing_or_copy(model, monkeypatch):
    captured = model.prepare_engram_graph_inputs(4)
    monkeypatch.setattr(implementation, "get_forward_context", lambda: SimpleNamespace(flash_comm_v1_enabled=False))
    model._engram_vmm_run = "unit-job"
    calls = []

    def prepare(hashes, mask):
        calls.append(hashes.shape[0])
        # Deliberately fill the entire capacity. The outer model must not apply
        # the old routed zero/copy path after direct final-buffer stores.
        for buffer in captured["engram_lookups"].values():
            buffer.fill_(7)
        captured["engram_mask"].fill_(True)
        return captured

    model._engram_vmm_inputs = SimpleNamespace(prepare=prepare)
    model._prepare_engram_hashes = lambda *args: (
        torch.zeros(1, 2, 24, dtype=torch.int64),
        torch.ones(1, dtype=torch.bool),
    )
    model.prepare_engram = lambda *args: pytest.fail("VMM must bypass owner routing")
    actual = model.prepare_engram_inputs(torch.arange(1), torch.arange(1), 4)
    assert actual is captured and calls == [1]
    assert all((buffer == 7).all() for buffer in actual["engram_lookups"].values())
    assert actual["engram_mask"].all()


def test_vmm_rejects_flashcomm_before_history(model, monkeypatch):
    model._engram_vmm_run = "unit-job"
    monkeypatch.setattr(implementation, "get_forward_context", lambda: SimpleNamespace(flash_comm_v1_enabled=True))
    model._prepare_engram_hashes = lambda *args: pytest.fail("must fail before mutating history")
    with pytest.raises(ValueError, match="FlashComm1"):
        model.prepare_engram_inputs(torch.arange(1), torch.arange(1), 4)


@pytest.mark.parametrize("tokens,padded", [(4, 3), (1, 17), (17, None), (1, -1)])
def test_invalid_capacity_fails_before_history_or_routing(model, tokens, padded):
    def unexpected(*args):
        raise AssertionError("must validate before routing")

    model.prepare_engram = unexpected
    with pytest.raises(ValueError, match="Engram padded token count"):
        model.prepare_engram_inputs(torch.arange(tokens), torch.arange(tokens), padded)


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

    class LanguageModel(torch.nn.Module):
        def prepare_engram_inputs(self, *args):
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
        _update_full_graph_params_if_needed=lambda *args: None,
    )
    assert runner_module.NPUModelRunner._model_forward(runner, 4) == 42
    assert calls == ["capture" if capture else "sync"]
