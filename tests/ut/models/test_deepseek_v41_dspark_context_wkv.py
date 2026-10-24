# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm_ascend.models.deepseek_v41 import dspark as dspark_module
from vllm_ascend.models.deepseek_v41.dspark import (
    DeepseekV41DSparkModel,
    DSparkDeepseekV41ForCausalLM,
)


def test_context_wkv_weights_are_duplicated_by_layer():
    weights = [
        ("mtp.0.attn.wkv.weight", torch.arange(4)),
        ("mtp.1.attn.wq_a.weight", torch.arange(3)),
        ("mtp.2.attn.wkv.scale", torch.tensor(0.5)),
        ("mtp.3.attn.wkv.weight", torch.arange(2)),
    ]

    duplicated = list(dspark_module._duplicate_context_wkv_weights(weights, num_layers=3))

    assert [name for name, _ in duplicated] == [
        "mtp.0.attn.wkv.weight",
        "context_wkv_proj.weight",
        "mtp.1.attn.wq_a.weight",
        "mtp.2.attn.wkv.scale",
        "context_wkv_proj.scale",
        "mtp.3.attn.wkv.weight",
    ]
    assert duplicated[1][1].shard_id == 0
    assert duplicated[4][1].shard_id == 2
    assert duplicated[0][1].data_ptr() == duplicated[1][1].data_ptr()
    assert duplicated[3][1].data_ptr() == duplicated[4][1].data_ptr()


def test_context_wkv_weights_load_into_the_matching_merged_shard(monkeypatch):
    load_calls = []

    def weight_loader(param, loaded_weight, shard_id=None):
        load_calls.append((param, loaded_weight, shard_id))

    original_param = torch.nn.Parameter(torch.empty(4))
    original_param.weight_loader = weight_loader
    merged_param = torch.nn.Parameter(torch.empty(4))
    merged_param.weight_loader = weight_loader

    params = {
        "model.layers.40.self_attn.wkv.weight": original_param,
        "model.context_wkv_proj.weight": merged_param,
    }

    model = SimpleNamespace(
        model=SimpleNamespace(
            num_dspark_layers=1,
            get_expert_mapping=lambda: [],
        ),
        config=SimpleNamespace(num_attention_heads=1),
        rotation_path=None,
        named_parameters=lambda: params.items(),
    )

    def remap(name):
        if name.startswith("context_wkv_proj."):
            return f"model.{name}"
        return "model.layers.40.self_attn.wkv.weight"

    model._remap_dspark_name = remap
    monkeypatch.setattr(dspark_module, "process_eagle_weight", lambda *_: None)
    monkeypatch.setattr(dspark_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_module, "get_tensor_model_parallel_rank", lambda: 0)

    loaded_weight = torch.arange(4)
    loaded_params = DSparkDeepseekV41ForCausalLM.load_weights(
        model,
        [("mtp.0.attn.wkv.weight", loaded_weight)],
    )

    assert loaded_params == set(params)
    assert len(load_calls) == 2
    assert load_calls[0][0] is original_param
    assert load_calls[0][1] is loaded_weight
    assert load_calls[0][2] is None
    assert load_calls[1][0] is merged_param
    assert load_calls[1][1].data_ptr() == loaded_weight.data_ptr()
    assert load_calls[1][2] == 0


def test_context_wkv_synthetic_name_is_remapped_to_model():
    model = DSparkDeepseekV41ForCausalLM.__new__(DSparkDeepseekV41ForCausalLM)

    assert model._remap_dspark_name("context_wkv_proj.weight") == "model.context_wkv_proj.weight"


def test_context_kv_uses_one_stacked_wkv_projection(monkeypatch):
    projection_calls = []
    rope_calls = []
    rope = object()
    stacked_output = torch.arange(24, dtype=torch.float32).view(2, 12)

    class StackedProjection:
        def __call__(self, context_states):
            projection_calls.append(context_states)
            return stacked_output

    attns = [
        SimpleNamespace(
            name=f"attn-{layer_idx}",
            rotary_emb=SimpleNamespace(layername=f"layer-{layer_idx}"),
        )
        for layer_idx in range(3)
    ]
    layers = {
        "40": SimpleNamespace(self_attn=attns[0]),
        "41": SimpleNamespace(self_attn=attns[1]),
        "42": SimpleNamespace(self_attn=attns[2]),
    }

    def get_rope(positions, layer_names):
        rope_calls.append((positions, layer_names))
        return rope

    monkeypatch.setattr(dspark_module, "get_cos_and_sin_dsa", get_rope)

    projected = []
    stored = []
    model = SimpleNamespace(
        context_wkv_proj=StackedProjection(),
        num_dspark_layers=3,
        config=SimpleNamespace(head_dim=4),
        layers=layers,
    )

    def project_shared_kv(kv, positions, attn, rope=None):
        projected.append((kv.clone(), positions, attn, rope))
        return kv.unsqueeze(1)

    def store_standard_swa_kv(shared_kv, slot_mapping, attn):
        stored.append((shared_kv.clone(), slot_mapping, attn))

    model._project_shared_kv = project_shared_kv
    model._store_standard_swa_kv = store_standard_swa_kv

    context_states = torch.zeros(2, 5)
    positions = torch.tensor([7, 8])
    slot_mappings = [
        torch.tensor([0, 1]),
        None,
        torch.tensor([4, 5]),
    ]

    DeepseekV41DSparkModel.precompute_and_store_context_kv(
        model,
        context_states,
        positions,
        slot_mappings,
    )

    assert len(projection_calls) == 1
    assert projection_calls[0] is context_states
    assert len(rope_calls) == 1
    assert rope_calls[0][0] is positions
    assert rope_calls[0][1] == ["layer-0", "layer-1", "layer-2"]
    assert len(projected) == 2
    assert len(stored) == 2

    expected = stacked_output.view(2, 3, 4)
    torch.testing.assert_close(projected[0][0], expected[:, 0])
    torch.testing.assert_close(projected[1][0], expected[:, 2])
    assert projected[0][3] is rope
    assert projected[1][3] is rope
    assert stored[0][1] is slot_mappings[0]
    assert stored[1][1] is slot_mappings[2]


def test_context_kv_profiles_stacked_projection_without_slot_mappings():
    projection_calls = []

    class StackedProjection:
        def __call__(self, context_states):
            projection_calls.append(context_states)
            return torch.zeros(2, 12)

    model = SimpleNamespace(
        context_wkv_proj=StackedProjection(),
        num_dspark_layers=3,
        config=SimpleNamespace(head_dim=4),
        layers={str(layer_idx): SimpleNamespace(self_attn=SimpleNamespace()) for layer_idx in range(3)},
    )

    def fail_project_shared_kv(*_args, **_kwargs):
        raise AssertionError("KV preparation must be skipped without slot mappings")

    def fail_store_standard_swa_kv(*_args, **_kwargs):
        raise AssertionError("KV cache writes must be skipped without slot mappings")

    model._project_shared_kv = fail_project_shared_kv
    model._store_standard_swa_kv = fail_store_standard_swa_kv

    context_states = torch.zeros(2, 5)
    DeepseekV41DSparkModel.precompute_and_store_context_kv(
        model,
        context_states,
        torch.tensor([7, 8]),
    )

    assert len(projection_calls) == 1
    assert projection_calls[0] is context_states


def test_context_kv_skips_projection_for_empty_positions():
    class FailProjection:
        def __call__(self, _context_states):
            raise AssertionError("empty positions must skip the context projection")

    model = SimpleNamespace(context_wkv_proj=FailProjection())
    DeepseekV41DSparkModel.precompute_and_store_context_kv(
        model,
        torch.zeros(2, 5),
        torch.empty(0, dtype=torch.long),
    )
