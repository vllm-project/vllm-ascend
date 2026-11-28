# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from vllm_ascend.ascend_config import EplbConfig
from vllm_ascend.distributed.eplb.eplb_state import AscendEplbLayerState
from vllm_ascend.distributed.eplb.placement import (
    ACTIVE_STATIC_CONTROLLER,
    ACTIVE_STATIC_LOAD,
    apply_static_rows_to_model,
    read_static_placement_snapshot,
    slice_static_role_rows,
)
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.ops.fused_moe.router.grouped_topk_router import AscendGroupedTopKRouter
from vllm_ascend.worker.v2.eplb import AscendEPLBController


class _FakeCpuGroup:
    def __init__(self, rank: int = 0, world_size: int = 2) -> None:
        self._rank = rank
        self._size = world_size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


class _FakeMoELayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eplb_state = AscendEplbLayerState()
        self.router = AscendGroupedTopKRouter(
            top_k=2,
            global_num_experts=6,
            num_expert_group=None,
            topk_group=None,
            eplb_state=self.eplb_state,
        )

    def set_eplb_state(self, moe_layer_idx, expert_load_view, logical_to_physical_map, logical_replica_count) -> None:
        self.eplb_state.set_layer_state(moe_layer_idx, expert_load_view, logical_to_physical_map, logical_replica_count)


class _FakeMoEModel(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.moe_layers = nn.ModuleList([_FakeMoELayer() for _ in range(num_layers)])
        self.num_routed_experts = 4
        self.num_logical_experts = 4
        self.num_redundant_experts = 2
        self.num_physical_experts = 6


def _payload(*, with_draft: bool = False) -> dict:
    data: dict = {
        "version": 1,
        "ep_size": 2,
        "target": {
            "num_logical_experts": 4,
            "num_redundant_experts": 2,
            "physical_to_logical_map": [[3, 0, 1, 0, 2, 1], [1, 2, 0, 3, 1, 0]],
        },
    }
    if with_draft:
        data["draft"] = {
            "num_logical_experts": 4,
            "num_redundant_experts": 2,
            "physical_to_logical_map": [[2, 1, 0, 0, 3, 1]],
        }
    return data


def _make_controller(path: str) -> AscendEPLBController:
    parallel_cfg = SimpleNamespace(
        enable_eplb=True,
        enable_elastic_ep=False,
        eplb_config=SimpleNamespace(policy="stair", log_balancedness=False, step_interval=10),
    )
    return AscendEPLBController(
        parallel_cfg, torch.device("cpu"), EplbConfig(mode="static", expert_placement_path=path)
    )


def test_static_snapshot_read_slice_and_layer_binding(tmp_path):
    path = tmp_path / "placement.json"
    path.write_text(json.dumps(_payload(with_draft=True)), encoding="utf-8")
    ep_group = SimpleNamespace(world_size=2, rank_in_group=1, cpu_group=_FakeCpuGroup(0, 2))

    with (
        patch("vllm_ascend.distributed.eplb.eplb_state.get_ep_group", return_value=ep_group),
        patch("torch.distributed.get_global_rank", return_value=0),
        patch("torch.distributed.broadcast_object_list"),
    ):
        snap = read_static_placement_snapshot(str(path), ep_group.cpu_group)
        model = _FakeMoEModel(2)
        rows = slice_static_role_rows(snap, "target", model, ep_group.cpu_group)
        local_ids = apply_static_rows_to_model(model, rows, ep_size=2, ep_rank=1, device=torch.device("cpu"))
        assert local_ids == {0, 1, 2, 3}
        for layer in model.moe_layers:
            layer.router._validate_eplb_state()
            assert layer.eplb_state.expert_replica_routing_table is not None
            assert layer.eplb_state.load_recording_enabled is False
            layer.eplb_state.record_load(None, 0, topk_ids=torch.zeros((1, 2), dtype=torch.int64), num_source_tokens=1)

        with pytest.raises(FileNotFoundError):
            read_static_placement_snapshot(str(tmp_path / "missing.json"), ep_group.cpu_group)
        with pytest.raises(ValueError, match="layer count mismatch"):
            slice_static_role_rows(snap, "target", _FakeMoEModel(1), ep_group.cpu_group)

    routed = AscendRoutedExperts.__new__(AscendRoutedExperts)
    nn.Module.__init__(routed)
    routed.moe_config = SimpleNamespace(num_logical_experts=4, num_experts=6, ep_rank=1, ep_size=2)
    emap = torch.tensor([-1, -1, -1, 0, 1, 2, 3], dtype=torch.int32)
    routed.expert_map_manager = SimpleNamespace(map_global_to_local=lambda gid: int(emap[gid].item()))
    routed._static_logical_to_local_slot = {0: 0, 2: 1, 1: 2}
    assert [routed._map_global_expert_id_to_local_expert_id(i) for i in range(5)] == [0, 2, 1, -1, 3]


def test_static_controller_target_draft_pp_and_dummy(tmp_path):
    path = tmp_path / "placement.json"
    path.write_text(json.dumps(_payload(with_draft=True)), encoding="utf-8")
    ep_group = SimpleNamespace(world_size=2, rank_in_group=0, cpu_group=_FakeCpuGroup(0, 2))
    pp_group = SimpleNamespace(rank_in_group=0, world_size=2, cpu_group=_FakeCpuGroup(0, 2))
    target_cfg, draft_cfg = SimpleNamespace(model="t"), SimpleNamespace(model="d")
    spec_cfg = SimpleNamespace(draft_model_config=draft_cfg, method="mtp", prefill_context_kv_only=False)
    vllm_cfg = SimpleNamespace(
        load_config=SimpleNamespace(load_format="safetensors"),
        parallel_config=SimpleNamespace(enable_eplb=True),
        speculative_config=spec_cfg,
    )

    with (
        patch("vllm_ascend.worker.v2.eplb.get_ep_group", return_value=ep_group),
        patch("vllm_ascend.distributed.eplb.eplb_state.get_ep_group", return_value=ep_group),
        patch("vllm_ascend.worker.v2.eplb.get_mixture_of_experts_model", side_effect=lambda m: m),
        patch("torch.distributed.get_global_rank", return_value=0),
        patch("torch.distributed.broadcast_object_list"),
    ):
        ctrl = _make_controller(str(path))
        ctrl.prepare_load()
        assert ctrl.state is None and ACTIVE_STATIC_CONTROLLER.get() is ctrl
        target_m, draft_m = _FakeMoEModel(2), _FakeMoEModel(1)
        ctrl.bind_model_for_static_load(target_m, vllm_cfg, target_cfg)
        assert target_m.moe_layers[0]._static_logical_to_local_slot == {3: 0, 0: 1, 1: 2}
        ctrl.bind_model_for_static_load(draft_m, vllm_cfg, draft_cfg)
        assert draft_m.moe_layers[0]._static_logical_to_local_slot == {2: 0, 1: 1, 0: 2}
        assert ctrl.maybe_register_speculator(MagicMock(), spec_cfg, False) is False
        assert ctrl.maybe_register_model(target_m, target_cfg) is False
        ctrl.maybe_start_async_loop(False, load_dummy_weights=False)
        assert ACTIVE_STATIC_CONTROLLER.get() is None and ACTIVE_STATIC_LOAD.get() is None
        ctrl.prepare_forward(target_cfg, 8)
        ctrl.step()

        def _fake_all_gather(out: list, obj: object, group: object = None) -> None:
            if isinstance(obj, bool):
                out[:] = [False, False]
            else:
                out[:] = [0, 2]

        # PP=2 dense stage 0 + missing registered draft error + dummy load
        with (
            patch("vllm_ascend.distributed.eplb.eplb_state.get_pp_group", return_value=pp_group),
            patch("torch.distributed.all_gather_object", side_effect=_fake_all_gather),
        ):
            ctrl.prepare_load()
            ctrl.bind_model_for_static_load(_FakeMoEModel(0), vllm_cfg, target_cfg)
            with pytest.raises(ValueError, match="no EPLB draft model is registered"):
                ctrl.maybe_start_async_loop(False, load_dummy_weights=False)

        dummy_cfg = SimpleNamespace(
            load_config=SimpleNamespace(load_format="dummy"),
            parallel_config=SimpleNamespace(enable_eplb=True),
            speculative_config=None,
        )
        ctrl.prepare_load()
        dummy_m = _FakeMoEModel(1)
        ctrl.bind_model_for_static_load(dummy_m, dummy_cfg, target_cfg)
        dummy_m.moe_layers[0].router._validate_eplb_state()
        ctrl.maybe_start_async_loop(False, load_dummy_weights=True)
        assert ACTIVE_STATIC_CONTROLLER.get() is None and ACTIVE_STATIC_LOAD.get() is None
