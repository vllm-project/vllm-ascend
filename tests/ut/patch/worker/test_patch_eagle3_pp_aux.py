#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from types import SimpleNamespace

import torch
import torch.nn as nn
from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.sequence import IntermediateTensors

from vllm_ascend.patch.worker import patch_eagle3_pp_aux as eagle3_pp_aux


class _FakePPGroup:
    def __init__(self, is_first_rank: bool, is_last_rank: bool):
        self.is_first_rank = is_first_rank
        self.is_last_rank = is_last_rank


class _FakeLayer(nn.Module):
    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta

    def forward(self, positions, hidden_states, residual, kv_cache, attn_metadata, llama_4_scaling):
        del positions, kv_cache, attn_metadata, llama_4_scaling
        next_hidden_states = hidden_states + self.delta
        next_residual = torch.zeros_like(hidden_states) if residual is None else residual + self.delta
        return next_hidden_states, next_residual


class _FakeEagleMixinLayer(nn.Module):
    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta

    def forward(self, positions, hidden_states, residual):
        del positions
        next_hidden_states = hidden_states + self.delta
        next_residual = torch.zeros_like(hidden_states) if residual is None else residual + self.delta
        return next_hidden_states, next_residual


class _FakeDeepseekV2Model(nn.Module):
    def __init__(self, start_layer: int, end_layer: int, aux_hidden_state_layers: tuple[int, ...]):
        super().__init__()
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.aux_hidden_state_layers = aux_hidden_state_layers
        self.config = SimpleNamespace(hidden_size=2)
        self.layers = nn.ModuleList([_FakeLayer(float(i + 1)) for i in range(4)])

    def embed_input_ids(self, input_ids):
        return input_ids.to(torch.float32).unsqueeze(-1).expand(-1, self.config.hidden_size)

    def norm(self, hidden_states, residual):
        return hidden_states + residual, None

    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros((batch_size, self.config.hidden_size), dtype=dtype, device=device),
                "residual": torch.zeros((batch_size, self.config.hidden_size), dtype=dtype, device=device),
            }
        )


class _FakeEagleMixinModel(nn.Module, EagleModelMixin):
    def __init__(self, start_layer: int, end_layer: int, aux_hidden_state_layers: tuple[int, ...]):
        super().__init__()
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.aux_hidden_state_layers = aux_hidden_state_layers
        self.config = SimpleNamespace(hidden_size=2)
        self.layers = nn.ModuleList([_FakeEagleMixinLayer(float(i + 1)) for i in range(4)])

    def embed_input_ids(self, input_ids):
        return input_ids.to(torch.float32).unsqueeze(-1).expand(-1, self.config.hidden_size)

    def norm(self, hidden_states, residual):
        return hidden_states + residual, None

    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros((batch_size, self.config.hidden_size), dtype=dtype, device=device),
                "residual": torch.zeros((batch_size, self.config.hidden_size), dtype=dtype, device=device),
            }
        )


def test_extract_aux_from_intermediate_sorts_by_aux_index():
    aux_2 = torch.full((1, 2), 2.0)
    aux_10 = torch.full((1, 2), 10.0)
    intermediate = IntermediateTensors(
        {
            "hidden_states": torch.zeros((1, 2)),
            "aux_layer_10": aux_10,
            "aux_layer_2": aux_2,
        }
    )

    aux_states = eagle3_pp_aux._extract_aux_from_intermediate(intermediate)

    assert len(aux_states) == 2
    torch.testing.assert_close(aux_states[0], aux_2)
    torch.testing.assert_close(aux_states[1], aux_10)


def test_non_last_pp_rank_carries_previous_and_local_aux_states(monkeypatch):
    monkeypatch.setattr(
        eagle3_pp_aux,
        "get_pp_group",
        lambda: _FakePPGroup(is_first_rank=False, is_last_rank=False),
    )
    forward = eagle3_pp_aux._make_deepseek_v2_forward()
    model = _FakeDeepseekV2Model(start_layer=2, end_layer=3, aux_hidden_state_layers=(1, 2))
    previous_aux = torch.full((2, 2), 11.0)
    hidden_states = torch.full((2, 2), 3.0)
    residual = torch.full((2, 2), 5.0)
    intermediate = IntermediateTensors(
        {
            "hidden_states": hidden_states,
            "residual": residual,
            "aux_layer_0": previous_aux,
        }
    )

    output = forward(
        model,
        None,
        torch.arange(2),
        kv_caches=[None] * 4,
        attn_metadata=None,
        intermediate_tensors=intermediate,
    )

    assert isinstance(output, IntermediateTensors)
    assert set(output.tensors) == {"hidden_states", "residual", "aux_layer_0", "aux_layer_1"}
    torch.testing.assert_close(output["aux_layer_0"], previous_aux)
    torch.testing.assert_close(output["aux_layer_1"], hidden_states + residual)
    torch.testing.assert_close(output["hidden_states"], hidden_states + 3.0)
    torch.testing.assert_close(output["residual"], residual + 3.0)


def test_last_pp_rank_returns_complete_aux_states(monkeypatch):
    monkeypatch.setattr(
        eagle3_pp_aux,
        "get_pp_group",
        lambda: _FakePPGroup(is_first_rank=False, is_last_rank=True),
    )
    forward = eagle3_pp_aux._make_deepseek_v2_forward()
    model = _FakeDeepseekV2Model(start_layer=3, end_layer=4, aux_hidden_state_layers=(1, 3))
    previous_aux = torch.full((2, 2), 13.0)
    hidden_states = torch.full((2, 2), 7.0)
    residual = torch.full((2, 2), 2.0)
    intermediate = IntermediateTensors(
        {
            "hidden_states": hidden_states,
            "residual": residual,
            "aux_layer_0": previous_aux,
        }
    )

    output_hidden_states, aux_states = forward(
        model,
        None,
        torch.arange(2),
        kv_caches=[None] * 4,
        attn_metadata=None,
        intermediate_tensors=intermediate,
    )

    assert len(aux_states) == 2
    torch.testing.assert_close(aux_states[0], previous_aux)
    torch.testing.assert_close(aux_states[1], hidden_states + residual)
    torch.testing.assert_close(output_hidden_states, hidden_states + 4.0 + residual + 4.0)


def test_eagle_mixin_non_last_pp_rank_carries_previous_and_local_aux_states(monkeypatch):
    monkeypatch.setattr(
        eagle3_pp_aux,
        "get_pp_group",
        lambda: _FakePPGroup(is_first_rank=False, is_last_rank=False),
    )
    forward = eagle3_pp_aux._make_eagle_mixin_forward()
    model = _FakeEagleMixinModel(start_layer=2, end_layer=3, aux_hidden_state_layers=(1, 3))
    previous_aux = torch.full((2, 2), 11.0)
    hidden_states = torch.full((2, 2), 3.0)
    residual = torch.full((2, 2), 5.0)
    intermediate = IntermediateTensors(
        {
            "hidden_states": hidden_states,
            "residual": residual,
            "aux_layer_0": previous_aux,
        }
    )

    output = forward(
        model,
        None,
        torch.arange(2),
        intermediate_tensors=intermediate,
    )

    assert isinstance(output, IntermediateTensors)
    assert set(output.tensors) == {"hidden_states", "residual", "aux_layer_0", "aux_layer_1"}
    torch.testing.assert_close(output["aux_layer_0"], previous_aux)
    torch.testing.assert_close(output["aux_layer_1"], hidden_states + 3.0 + residual + 3.0)
    torch.testing.assert_close(output["hidden_states"], hidden_states + 3.0)
    torch.testing.assert_close(output["residual"], residual + 3.0)


def test_eagle_mixin_last_pp_rank_returns_complete_aux_states(monkeypatch):
    monkeypatch.setattr(
        eagle3_pp_aux,
        "get_pp_group",
        lambda: _FakePPGroup(is_first_rank=False, is_last_rank=True),
    )
    forward = eagle3_pp_aux._make_eagle_mixin_forward()
    model = _FakeEagleMixinModel(start_layer=3, end_layer=4, aux_hidden_state_layers=(1, 4))
    previous_aux = torch.full((2, 2), 13.0)
    hidden_states = torch.full((2, 2), 7.0)
    residual = torch.full((2, 2), 2.0)
    intermediate = IntermediateTensors(
        {
            "hidden_states": hidden_states,
            "residual": residual,
            "aux_layer_0": previous_aux,
        }
    )

    output_hidden_states, aux_states = forward(
        model,
        None,
        torch.arange(2),
        intermediate_tensors=intermediate,
    )

    assert len(aux_states) == 2
    torch.testing.assert_close(aux_states[0], previous_aux)
    torch.testing.assert_close(aux_states[1], hidden_states + 4.0 + residual + 4.0)
    torch.testing.assert_close(output_hidden_states, hidden_states + 4.0 + residual + 4.0)


def test_make_empty_intermediate_tensors_allocates_only_incoming_aux_layers():
    model = _FakeDeepseekV2Model(start_layer=2, end_layer=4, aux_hidden_state_layers=(0, 2, 3))

    eagle3_pp_aux._patch_make_empty_intermediate_tensors(model)
    result = model.make_empty_intermediate_tensors(
        batch_size=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert set(result.tensors) == {"hidden_states", "residual", "aux_layer_0"}
    assert result["aux_layer_0"].shape == (3, 2)
    assert result["aux_layer_0"].dtype == torch.float32


def test_patch_accepts_eagle_mixin_model():
    model = _FakeEagleMixinModel(start_layer=2, end_layer=4, aux_hidden_state_layers=(0, 2, 3))

    assert eagle3_pp_aux.patch_eagle3_pp_aux_propagation(model) is True
    assert model._eagle3_pp_aux_forward_patched is True
    assert model._eagle3_pp_aux_make_empty_patched is True

    result = model.make_empty_intermediate_tensors(
        batch_size=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert set(result.tensors) == {"hidden_states", "residual", "aux_layer_0"}


def test_patch_rejects_unsupported_model():
    unsupported_model = nn.Linear(2, 2)

    assert eagle3_pp_aux.patch_eagle3_pp_aux_propagation(unsupported_model) is False
