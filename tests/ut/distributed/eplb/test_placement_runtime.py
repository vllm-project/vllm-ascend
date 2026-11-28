# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from vllm_ascend.distributed.eplb.placement_runtime import (
    RuntimeRoleSpec,
    bind_role_placement_to_model,
    build_model_role_spec,
    collective_call,
    load_placement_snapshot,
    local_expert_ids_for_role,
    parse_placement_snapshot,
    read_and_slice_target_placement,
    slice_pp_target_role,
    validate_role_against_spec,
)
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts


def _sample_payload(*, with_draft: bool = False) -> dict:
    payload: dict = {
        "version": 1,
        "ep_size": 2,
        "target": {
            "num_logical_experts": 4,
            "num_redundant_experts": 2,
            "physical_to_logical_map": [[3, 0, 1, 0, 2, 1], [1, 2, 0, 3, 1, 0]],
        },
    }
    if with_draft:
        payload["draft"] = {
            "num_logical_experts": 4,
            "num_redundant_experts": 2,
            "physical_to_logical_map": [[2, 1, 0, 0, 3, 1]],
        }
    return payload


class _FakeGroup:
    def __init__(self, rank: int, world_size: int) -> None:
        self._rank = rank
        self._size = world_size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def test_parse_and_load_placement_snapshot(tmp_path):
    path = tmp_path / "placement.json"
    payload = _sample_payload(with_draft=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

    snapshot = load_placement_snapshot(path)
    assert snapshot.version == 1
    assert snapshot.ep_size == 2
    assert snapshot.target.num_physical_experts == 6
    assert snapshot.target.num_layers == 2
    assert snapshot.draft is not None and snapshot.draft.num_layers == 1
    assert len(snapshot.digest()) == 64


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda d: d.update(version=3), "Unsupported EPLB placement version"),
        (lambda d: d.update(extra=1), "must contain"),
        (lambda d: d.update(ep_size=5), "divisible by ep_size"),
        (lambda d: d["target"].update(physical_to_logical_map=[[0, 1, 2, 0, 1, 2]]), "does not cover every logical"),
        (lambda d: d["target"].update(physical_to_logical_map=[[0, 0, 1, 2, 3, 1]]), "duplicate logical experts"),
        (lambda d: d["target"].update(physical_to_logical_map=[[0, 1, 2, 3, 4, 0]]), "invalid logical expert ID"),
    ],
)
def test_parse_placement_snapshot_rejects_invalid_inputs(mutator, match):
    payload = _sample_payload()
    mutator(payload)
    with pytest.raises(ValueError, match=match):
        parse_placement_snapshot(payload)


def test_build_model_role_spec_and_pp_slicing():
    assert build_model_role_spec(None) is None
    assert build_model_role_spec(SimpleNamespace(moe_layers=[])) is None
    moe_model = SimpleNamespace(
        moe_layers=[object(), object()], num_logical_experts=4, num_redundant_experts=2, num_physical_experts=6
    )
    spec = build_model_role_spec(moe_model)
    assert spec == RuntimeRoleSpec(num_logical_experts=4, num_redundant_experts=2, num_layers=2)

    snapshot = parse_placement_snapshot(_sample_payload())
    validate_role_against_spec(snapshot.target, "target", spec)
    stage0 = RuntimeRoleSpec(num_logical_experts=4, num_redundant_experts=2, num_layers=1)
    stage1 = RuntimeRoleSpec(num_logical_experts=4, num_redundant_experts=2, num_layers=1)
    assert slice_pp_target_role(snapshot, 2, 0, [None, spec]) is None
    sliced = slice_pp_target_role(snapshot, 2, 1, [stage0, stage1])
    assert sliced is not None and sliced.physical_to_logical_map == ((1, 2, 0, 3, 1, 0),)

    with pytest.raises(ValueError, match="layer count mismatch"):
        slice_pp_target_role(snapshot, 2, 0, [stage0])


def test_read_and_slice_target_placement_single_and_pp(tmp_path):
    path = tmp_path / "placement.json"
    path.write_text(json.dumps(_sample_payload()), encoding="utf-8")
    spec_full = RuntimeRoleSpec(num_logical_experts=4, num_redundant_experts=2, num_layers=2)

    with (
        patch("torch.distributed.get_global_rank", return_value=0),
        patch("torch.distributed.broadcast_object_list"),
        patch(
            "torch.distributed.all_gather_object",
            side_effect=lambda out, obj, group: out.__setitem__(slice(None), [obj] * group.size()),
        ),
    ):
        snap, target = read_and_slice_target_placement(str(path), lambda: spec_full, _FakeGroup(0, 2))
        assert snap.ep_size == 2 and target == snap.target

        stage_spec = RuntimeRoleSpec(num_logical_experts=4, num_redundant_experts=2, num_layers=1)
        raw_text = path.read_text(encoding="utf-8")
        with patch(
            "torch.distributed.broadcast_object_list",
            side_effect=lambda buf, **_: buf.__setitem__(0, raw_text) if len(buf) == 1 else None,
        ):
            snap_pp, pp_role = read_and_slice_target_placement(
                str(path), lambda: stage_spec, _FakeGroup(0, 2), _FakeGroup(0, 2), pp_rank=1
            )
        assert pp_role is not None and pp_role.physical_to_logical_map == snap_pp.target.physical_to_logical_map[1:]

        with pytest.raises(FileNotFoundError, match="could not be read"):
            read_and_slice_target_placement(str(tmp_path / "missing.json"), lambda: spec_full, _FakeGroup(0, 2))


def test_bind_role_placement_and_routed_experts_slot_mapping():
    snapshot = parse_placement_snapshot(_sample_payload())
    layers = []
    for _ in range(2):
        layer = AscendRoutedExperts.__new__(AscendRoutedExperts)
        nn.Module.__init__(layer)
        layer.moe_config = SimpleNamespace(num_logical_experts=4, num_experts=6, ep_rank=1, ep_size=2)
        layer._expert_map = torch.tensor([-1, -1, -1, 0, 1, 2, 3], dtype=torch.int32)
        layer.expert_map_manager = SimpleNamespace(
            _expert_map=layer._expert_map,
            map_global_to_local=lambda gid, em=layer._expert_map: int(em[gid].item()),
            num_fused_shared_experts=1,
        )
        layers.append(layer)
    moe_model = SimpleNamespace(moe_layers=layers)

    bind_role_placement_to_model(moe_model, snapshot.target, ep_size=2, ep_rank=1)
    assert local_expert_ids_for_role(snapshot.target, ep_size=2, ep_rank=1) == {0, 1, 2, 3}
    assert local_expert_ids_for_role(snapshot.target, ep_size=1, ep_rank=0) is None
    assert [layers[0]._map_global_expert_id_to_local_expert_id(i) for i in range(4)] == [0, 2, 1, -1]
    assert layers[0]._map_global_expert_id_to_local_expert_id(4) == 3

    with pytest.raises(ValueError, match="collective_test failed"):
        collective_call(lambda: (_ for _ in ()).throw(RuntimeError("boom")), _FakeGroup(0, 1), "collective_test")
