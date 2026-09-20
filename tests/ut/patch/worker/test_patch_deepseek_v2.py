# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model

from vllm_ascend.patch.worker import patch_deepseek_v2
from vllm_ascend.patch.worker.patch_deepseek_v2 import _should_skip_indexer_init


def _config(**overrides) -> SimpleNamespace:
    values = {"num_hidden_layers": 80}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_glm51_skip_topk_keeps_per_layer_indexer():
    assert not _should_skip_indexer_init(
        _config(),
        "model.layers.2.self_attn",
        skip_topk=True,
    )


def test_glm52_shared_layer_skips_indexer_init():
    assert _should_skip_indexer_init(
        _config(indexer_types=["full", "full", "shared"]),
        "model.layers.2.self_attn",
        skip_topk=True,
    )


def test_mtp_layer_keeps_indexer():
    indexer_types = ["full"] * 80 + ["shared"]
    assert not _should_skip_indexer_init(
        _config(indexer_types=indexer_types),
        "model.layers.80.self_attn",
        skip_topk=True,
    )


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("boundaries", [(0, 78), (0, 42, 78), (0, 20, 40, 59, 78)])
def test_aux_relay_matches_unpartitioned_forward(monkeypatch, native, boundaries):
    if native and not hasattr(DeepseekV2Model, "pack_local_aux_hidden_states"):
        pytest.skip("The installed vLLM release has no native aux relay")
    aux_layers = (0, 2, 20, 39, 58, 75, 78)
    ids = torch.zeros(4, dtype=torch.long)

    def layer(positions, hidden, residual, scaling):
        residual = torch.zeros_like(hidden) if residual is None else residual
        return hidden + 1, residual + 1

    def run(split):
        incoming = None
        for start, end in zip(split, split[1:]):
            first, last = start == 0, end == 78
            group = SimpleNamespace(is_first_rank=first, is_last_rank=last, world_size=len(split) - 1)
            monkeypatch.setattr(patch_deepseek_v2, "get_pp_group", lambda group=group: group)
            model = DeepseekV2Model.__new__(DeepseekV2Model)
            torch.nn.Module.__init__(model)
            model.config = SimpleNamespace(hidden_size=2)
            model.hidden_size = 2
            model.start_layer, model.end_layer = start, end
            model.layers = [layer] * 78
            model.aux_hidden_state_layers = aux_layers
            model._use_upstream_aux_relay = native
            model.embed_input_ids = lambda ids: torch.zeros(len(ids), 2)
            model.norm = lambda hidden, residual: (hidden + residual, None)
            if native:
                import vllm.distributed.parallel_state as parallel_state
                from vllm.v1.worker.gpu.pp_utils import PPHandler

                monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
                monkeypatch.setattr(parallel_state, "get_pp_group", lambda group=group: group)
                model._set_aux_hidden_state_layers(aux_layers)
            output = patch_deepseek_v2._patched_forward(model, ids if first else None, ids, incoming)
            if not last:
                if native:
                    # Use the actual upstream relay, without creating streams or process groups.
                    handler = SimpleNamespace(
                        aux_hidden_state_relay_keys=[
                            f"aux_hidden_states_{i}" for i in range(model._aux_slot_base_cached)
                        ]
                    )
                    output = PPHandler.relay_aux_hidden_states(handler, incoming, output)
                prefix = "aux_hidden_states_" if native else "pp_transport_aux_hidden_states_"
                expected_count = sum(idx <= end for idx in aux_layers)
                assert set(output.tensors) == {"hidden_states", "residual"} | {
                    f"{prefix}{idx}" for idx in range(expected_count)
                }
            incoming = output
        return output

    expected_hidden, expected_aux = run((0, 78))
    actual_hidden, actual_aux = run(boundaries)
    torch.testing.assert_close(actual_hidden, expected_hidden)
    assert len(actual_aux) == len(expected_aux) == len(aux_layers)
    for actual, expected in zip(actual_aux, expected_aux):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("legacy,v2", [(True, True), (False, True), (False, False)])
def test_aux_buffer_factory_uses_one_protocol(monkeypatch, legacy, v2):
    factory = object()
    model = SimpleNamespace(make_empty_intermediate_tensors=factory)
    monkeypatch.setattr(patch_deepseek_v2, "_original_deepseek_v2_model_init", lambda *args, **kwargs: None)
    monkeypatch.setattr(patch_deepseek_v2.pp_utils, "use_legacy_spec_pp", lambda: legacy)
    wrapped = object()
    monkeypatch.setattr(patch_deepseek_v2.pp_utils, "make_empty_intermediate_tensors", lambda *args: wrapped)
    patch_deepseek_v2._patched_deepseek_v2_model_init(model, vllm_config=SimpleNamespace(use_v2_model_runner=v2))
    assert model._use_upstream_aux_relay is (v2 and not legacy)
    assert model.make_empty_intermediate_tensors is (factory if v2 and not legacy else wrapped)


@pytest.mark.parametrize("native", [False, True])
def test_aux_setter_preserves_v1_behavior(native):
    if not hasattr(patch_deepseek_v2, "_set_aux_hidden_state_layers"):
        pytest.skip("The installed vLLM release uses its original setter")
    model = SimpleNamespace(_use_upstream_aux_relay=native, _set_aux_hidden_state_layers=Mock())
    layers = (2, 20, 39, 58, 75)
    patch_deepseek_v2._set_aux_hidden_state_layers(SimpleNamespace(model=model), layers)
    if native:
        model._set_aux_hidden_state_layers.assert_called_once_with(layers)
    else:
        model._set_aux_hidden_state_layers.assert_not_called()
        assert model.aux_hidden_state_layers == layers
