# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm_ascend.models.deepseek_v4 import dspark as dspark_module
from vllm_ascend.models.deepseek_v4.dspark import DeepseekV4DSparkModel


def test_context_wkv_weights_are_duplicated_by_layer():
    weights = [
        ("mtp.0.attn.wkv.weight", torch.arange(4)),
        ("mtp.1.attn.wq_a.weight", torch.arange(3)),
        ("mtp.2.attn.wkv.scale", torch.tensor(0.5)),
        ("mtp.3.attn.wkv.weight", torch.arange(2)),
    ]

    duplicated = list(
        dspark_module._duplicate_context_wkv_weights(
            weights,
            num_layers=3,
        )
    )

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

    # detach() does not copy storage.
    assert duplicated[0][1].data_ptr() == duplicated[1][1].data_ptr()
    assert duplicated[3][1].data_ptr() == duplicated[4][1].data_ptr()


def test_context_kv_uses_one_stacked_wkv_projection():
    projection_calls = []

    stacked_output = torch.arange(
        24,
        dtype=torch.float32,
    ).view(2, 12)

    class StackedProjection:
        def __call__(self, context_states):
            projection_calls.append(context_states)
            return stacked_output

    attns = [SimpleNamespace(name=f"attn-{i}") for i in range(3)]

    layers = {
        "40": SimpleNamespace(self_attn=attns[0]),
        "41": SimpleNamespace(self_attn=attns[1]),
        "42": SimpleNamespace(self_attn=attns[2]),
    }

    projected = []
    stored = []

    model = SimpleNamespace(
        context_wkv_proj=StackedProjection(),
        num_dspark_layers=3,
        config=SimpleNamespace(head_dim=4),
        layers=layers,
    )

    def project_shared_kv(kv, positions, attn):
        projected.append((kv.clone(), positions, attn))
        return kv.unsqueeze(1)

    def store_standard_swa_kv(shared_kv, slot_mapping, attn):
        stored.append(
            (
                shared_kv.clone(),
                slot_mapping,
                attn,
            )
        )

    model._project_shared_kv = project_shared_kv
    model._store_standard_swa_kv = store_standard_swa_kv

    context_states = torch.zeros(2, 5)
    positions = torch.tensor([7, 8])

    slot_mappings = [
        torch.tensor([0, 1]),
        None,
        torch.tensor([4, 5]),
    ]

    DeepseekV4DSparkModel.precompute_and_store_context_kv(
        model,
        context_states,
        positions,
        slot_mappings,
    )

    assert len(projection_calls) == 1
    assert projection_calls[0] is context_states

    assert len(projected) == 2
    assert len(stored) == 2

    expected = stacked_output.view(2, 3, 4)

    torch.testing.assert_close(
        projected[0][0],
        expected[:, 0],
    )
    torch.testing.assert_close(
        projected[1][0],
        expected[:, 2],
    )

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

    def fail_project_shared_kv(*_):
        raise AssertionError("KV preparation must be skipped without slot mappings")

    def fail_store_standard_swa_kv(*_):
        raise AssertionError("KV cache writes must be skipped without slot mappings")

    model._project_shared_kv = fail_project_shared_kv
    model._store_standard_swa_kv = fail_store_standard_swa_kv

    context_states = torch.zeros(2, 5)
    DeepseekV4DSparkModel.precompute_and_store_context_kv(
        model,
        context_states,
        torch.tensor([7, 8]),
    )

    assert len(projection_calls) == 1
    assert projection_calls[0] is context_states
