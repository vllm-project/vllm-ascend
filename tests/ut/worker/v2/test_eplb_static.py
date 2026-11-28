# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import vllm.distributed.eplb.eplb_state as upstream_eplb_state

from vllm_ascend.ascend_config import EplbConfig
from vllm_ascend.distributed.eplb.eplb_state import AscendEplbLayerState
from vllm_ascend.distributed.eplb.placement import (
    ACTIVE_STATIC_CONTROLLER,
    ACTIVE_STATIC_LOAD,
    apply_static_rows_to_model,
    read_static_placement_snapshot,
    slice_static_role_rows,
)
from vllm_ascend.ops.fused_moe.eplb import record_expert_tokens
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.ops.fused_moe.router.grouped_topk_router import AscendGroupedTopKRouter
from vllm_ascend.worker.v2.eplb import AscendEPLBController


class _FakeCpuGroup:
    def __init__(self, rank: int = 0, world_size: int = 2) -> None:
        self._rank, self._size = rank, world_size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


class _FakeMoELayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eplb_state = AscendEplbLayerState()
        self.router = AscendGroupedTopKRouter(
            top_k=2, global_num_experts=6, num_expert_group=None, topk_group=None, eplb_state=self.eplb_state
        )

    def set_eplb_state(self, moe_layer_idx, expert_load_view, logical_to_physical_map, logical_replica_count) -> None:
        self.eplb_state.set_layer_state(moe_layer_idx, expert_load_view, logical_to_physical_map, logical_replica_count)


class _FakeMoEModel(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.moe_layers = nn.ModuleList([_FakeMoELayer() for _ in range(num_layers)])
        self.num_routed_experts = self.num_logical_experts = 4
        self.num_redundant_experts, self.num_physical_experts = 2, 6


def _payload(*, with_draft: bool = True) -> dict:
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


def _make_controller(path: str, *, log_balancedness: bool = False) -> AscendEPLBController:
    parallel_cfg = SimpleNamespace(
        enable_eplb=True,
        enable_elastic_ep=False,
        num_ubatches=1,
        eplb_config=SimpleNamespace(
            policy="stair", log_balancedness=log_balancedness, log_balancedness_interval=2, step_interval=10
        ),
    )
    return AscendEPLBController(
        parallel_cfg,
        torch.device("cpu"),
        EplbConfig(mode="static", expert_placement_path=path, load_collection_phase="decode"),
    )


def test_static_snapshot_and_routed_experts(tmp_path):
    path = tmp_path / "placement.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    ep_group = SimpleNamespace(world_size=2, rank_in_group=1, cpu_group=_FakeCpuGroup(0, 2))
    with (
        patch("vllm_ascend.distributed.eplb.eplb_state.get_ep_group", return_value=ep_group),
        patch("torch.distributed.get_global_rank", return_value=0),
        patch("torch.distributed.broadcast_object_list"),
    ):
        snap = read_static_placement_snapshot(str(path), ep_group.cpu_group)
        model = _FakeMoEModel(2)
        rows = slice_static_role_rows(snap, "target", model, ep_group.cpu_group)
        assert apply_static_rows_to_model(model, rows, 2, 1, torch.device("cpu")) == {0, 1, 2, 3}
        for layer in model.moe_layers:
            layer.router._validate_eplb_state()
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


def test_static_controller_modes_pp_and_log_balancedness(tmp_path):
    path = tmp_path / "placement.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    dev_group = SimpleNamespace(rank=lambda: 0, size=lambda: 2)
    ep_group = SimpleNamespace(world_size=2, rank_in_group=0, cpu_group=_FakeCpuGroup(0, 2), device_group=dev_group)
    pp_group = SimpleNamespace(rank_in_group=0, world_size=2, cpu_group=_FakeCpuGroup(0, 2))
    target_cfg = SimpleNamespace(model="t", compile_factors=lambda: ["t"])
    draft_cfg = SimpleNamespace(model="d", compile_factors=lambda: ["d"])
    spec_cfg = SimpleNamespace(draft_model_config=draft_cfg, method="mtp", prefill_context_kv_only=False)
    vllm_cfg = SimpleNamespace(
        load_config=SimpleNamespace(load_format="safetensors"),
        parallel_config=SimpleNamespace(enable_eplb=True),
        speculative_config=spec_cfg,
    )

    def _fake_all_gather(out: list, obj: object, group: object = None) -> None:
        if isinstance(obj, bool):
            out[:] = [False, False]
        else:
            out[:] = [0, 2]

    with (
        patch("vllm_ascend.worker.v2.eplb.get_ep_group", return_value=ep_group),
        patch("vllm_ascend.distributed.eplb.eplb_state.get_ep_group", return_value=ep_group),
        patch.object(upstream_eplb_state, "get_ep_group", return_value=ep_group),
        patch.object(upstream_eplb_state, "CpuGpuEvent", MagicMock),
        patch.object(upstream_eplb_state, "all_reduce") as mock_all_reduce,
        patch.object(upstream_eplb_state.logger, "info") as mock_log_info,
        patch("torch.Event", side_effect=lambda *_, **__: SimpleNamespace(record=lambda *_: None, query=lambda: True)),
        patch("torch.ops.vllm.ascend_eplb_record_expert_tokens", side_effect=record_expert_tokens, create=True),
        patch("vllm_ascend.worker.v2.eplb.get_mixture_of_experts_model", side_effect=lambda m: m),
        patch("torch.distributed.get_global_rank", return_value=0),
        patch("torch.distributed.broadcast_object_list"),
    ):
        # 1. Default log_balancedness=False skips load recording completely
        ctrl = _make_controller(str(path), log_balancedness=False)
        ctrl.prepare_load()
        assert ctrl.state is None and ACTIVE_STATIC_CONTROLLER.get() is ctrl
        target_m, draft_m = _FakeMoEModel(2), _FakeMoEModel(1)
        ctrl.bind_model_for_static_load(target_m, vllm_cfg, target_cfg)
        ctrl.bind_model_for_static_load(draft_m, vllm_cfg, draft_cfg)
        assert target_m.moe_layers[0]._static_logical_to_local_slot == {3: 0, 0: 1, 1: 2}
        assert draft_m.moe_layers[0]._static_logical_to_local_slot == {2: 0, 1: 1, 0: 2}
        assert ctrl.maybe_register_speculator(MagicMock(), spec_cfg, False) is False
        assert ctrl.maybe_register_model(target_m, target_cfg) is False
        ctrl.maybe_start_async_loop(False, load_dummy_weights=False)
        assert ACTIVE_STATIC_CONTROLLER.get() is None and ACTIVE_STATIC_LOAD.get() is None

        # 2. PP=2 dense stage 0 + missing registered draft error + dummy load
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

        # 3. log_balancedness=True records only on log_balancedness_interval=2 steps
        log_ctrl = _make_controller(str(path), log_balancedness=True)
        log_ctrl.prepare_load()
        assert log_ctrl.state is not None and log_ctrl.state.static_mode is True
        t_log, d_log = _FakeMoEModel(2), _FakeMoEModel(1)
        log_ctrl.bind_model_for_static_load(t_log, vllm_cfg, target_cfg)
        log_ctrl.bind_model_for_static_load(d_log, vllm_cfg, draft_cfg)
        speculator = MagicMock()
        assert log_ctrl.maybe_register_speculator(speculator, spec_cfg, False) is True
        speculator.set_eplb_state.assert_called_once_with(log_ctrl.state)
        assert log_ctrl.maybe_register_model(t_log, target_cfg) is True
        log_ctrl.maybe_start_async_loop(True, load_dummy_weights=False)

        zeros_topk = torch.zeros((2, 2), dtype=torch.int64)
        log_ctrl.set_batch_phase(batch_has_prefill=True)
        log_ctrl.prepare_forward(target_cfg, 8)
        assert bool(log_ctrl.state.should_record_tensor.item()) is True
        t_log.moe_layers[0].eplb_state.record_load(
            torch.tensor([6, 2, 4], dtype=torch.int32), 1, topk_ids=zeros_topk, num_source_tokens=2
        )
        d_log.moe_layers[0].eplb_state.record_load(
            torch.tensor([3, 1, 2], dtype=torch.int32), 1, topk_ids=zeros_topk, num_source_tokens=2
        )
        log_ctrl.step()
        assert mock_all_reduce.call_count == 1 and bool(log_ctrl.state.should_record_tensor.item()) is False

        # Step 1 (1 % 2 != 0): skips recording, flushes step 0 balancedness log
        log_ctrl.prepare_forward(target_cfg, 8)
        assert bool(log_ctrl.state.should_record_tensor.item()) is False
        t_log.moe_layers[0].eplb_state.record_load(
            torch.tensor([9, 9, 9], dtype=torch.int32), 1, topk_ids=zeros_topk, num_source_tokens=2
        )
        assert all(int(ms.expert_load_pass.sum().item()) == 0 for ms in log_ctrl.state.model_states.values())
        log_ctrl.step()
        assert mock_all_reduce.call_count == 1
        logged = [call.args[0] % call.args[1:] for call in mock_log_info.call_args_list]
        assert any("for model t:" in m for m in logged) and any("for model d (draft):" in m for m in logged)

        with log_ctrl.suppress():
            assert bool(log_ctrl.state.should_record_tensor.item()) is False
            log_ctrl.step()
        log_ctrl.step(is_dummy=True, is_profile=True)
        assert log_ctrl.state.expert_rearrangement_step == 2
