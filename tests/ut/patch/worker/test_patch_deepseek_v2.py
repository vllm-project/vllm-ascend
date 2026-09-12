# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model
from vllm.sequence import IntermediateTensors

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


class _FakePPGroup:
    def __init__(self, is_first_rank: bool, is_last_rank: bool):
        self.is_first_rank = is_first_rank
        self.is_last_rank = is_last_rank
        self.world_size = 2
        self.rank_in_group = 0 if is_first_rank else 1


class _AddLayer(torch.nn.Module):
    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta

    def forward(self, positions, hidden_states, residual, llama_4_scaling):
        del positions, llama_4_scaling
        residual = torch.zeros_like(hidden_states) if residual is None else residual + self.delta
        return hidden_states + self.delta, residual


def _model(start_layer: int, end_layer: int, aux_layers: tuple[int, ...], hidden_size: int = 2):
    model = DeepseekV2Model.__new__(DeepseekV2Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace()
    model.hidden_size = hidden_size
    model.start_layer = start_layer
    model.end_layer = end_layer
    # Decoders index their layers by absolute layer id, with placeholders for
    # the layers other PP stages own.
    model.layers = torch.nn.ModuleList([_AddLayer(1.0) for _ in range(end_layer)])
    model.aux_hidden_state_layers = tuple(aux_layers)
    model._aux_slot_base_cached = 0
    model._aux_upstream_total_cached = 0
    model.embed_input_ids = lambda input_ids: torch.zeros(input_ids.shape[0], hidden_size)
    model.norm = lambda hidden_states, residual: (hidden_states + residual, None)
    return model


def _run_forward(monkeypatch, model, pp_group, intermediate_tensors=None):
    monkeypatch.setattr(patch_deepseek_v2, "get_pp_group", lambda: pp_group)
    input_ids = torch.zeros(4, dtype=torch.long) if intermediate_tensors is None else None
    return patch_deepseek_v2._patched_forward(
        model,
        input_ids,
        torch.zeros(4, dtype=torch.long),
        intermediate_tensors,
    )


def test_deepseek_v2_decoder_opts_into_the_upstream_aux_relay():
    # Upstream refuses PP with a target-driven drafter unless the decoder
    # declares support, and relays the states through its own slots.
    assert DeepseekV2Model.supports_aux_hidden_states_over_pp is True
    assert DeepseekV2Model.AUX_HIDDEN_STATE_KEY == "aux_hidden_states_"
    for member in ("pack_local_aux_hidden_states", "collect_remote_aux_hidden_states"):
        assert callable(getattr(DeepseekV2Model, member))


def test_aux_capture_follows_the_upstream_layer_ids(monkeypatch):
    # Upstream ids are one-based layer outputs, plus the first stage's entry
    # state, so four layers with ids {0, 4} capture exactly two states.
    model = _model(0, 4, aux_layers=(0, 4))

    hidden_states, aux_hidden_states = _run_forward(monkeypatch, model, _FakePPGroup(True, True))

    assert hidden_states.shape == (4, 2)
    assert len(aux_hidden_states) == 2


def test_middle_stage_packs_only_its_own_states_after_the_upstream_slots(monkeypatch):
    # Split 42/36 with GLM-5.2's aux ids: the first stage owns ids 2/20/39, so
    # this stage publishes ids 58/75 in the slots that follow them.
    model = _model(42, 78, aux_layers=(2, 20, 39, 58, 75))
    model._aux_slot_base_cached = 3
    intermediate_tensors = IntermediateTensors(
        {
            "hidden_states": torch.zeros(4, 2),
            "residual": torch.zeros(4, 2),
        }
    )

    out = _run_forward(monkeypatch, model, _FakePPGroup(False, False), intermediate_tensors)

    assert isinstance(out, IntermediateTensors)
    aux_keys = {key for key in out.tensors if key.startswith("aux_hidden_states_")}
    # The relay forwards slots 0..2, so only 3 and 4 are produced here.
    assert aux_keys == {"aux_hidden_states_3", "aux_hidden_states_4"}


def test_last_stage_appends_local_states_after_the_relayed_ones(monkeypatch):
    model = _model(42, 78, aux_layers=(2, 20, 39, 58, 75))
    model._aux_slot_base_cached = 3
    model._aux_upstream_total_cached = 3
    intermediate_tensors = IntermediateTensors(
        {
            "hidden_states": torch.zeros(4, 2),
            "residual": torch.zeros(4, 2),
            "aux_hidden_states_0": torch.full((4, 2), 10.0),
            "aux_hidden_states_1": torch.full((4, 2), 11.0),
            "aux_hidden_states_2": torch.full((4, 2), 12.0),
        }
    )

    _, aux_hidden_states = _run_forward(monkeypatch, model, _FakePPGroup(False, True), intermediate_tensors)

    assert len(aux_hidden_states) == 5
    # Earlier stages' states first: they hold the lower layer indices.
    assert [float(state[0, 0]) for state in aux_hidden_states[:3]] == [10.0, 11.0, 12.0]
