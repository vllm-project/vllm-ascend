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
from vllm_ascend.distributed.eplb.placement_runtime import ACTIVE_STATIC_CONTROLLER, ACTIVE_STATIC_ROLE_PLACEMENT
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
        self.moe_config = SimpleNamespace(ep_size=2, ep_rank=0, num_logical_experts=4, num_experts=6)

    def set_eplb_state(self, moe_layer_idx, expert_load_view, logical_to_physical_map, logical_replica_count) -> None:
        self.eplb_state.set_layer_state(
            moe_layer_idx,
            expert_load_view,
            logical_to_physical_map,
            logical_replica_count,
        )


class _FakeMoEModel(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.moe_layers = nn.ModuleList([_FakeMoELayer() for _ in range(num_layers)])
        self.num_moe_layers = num_layers
        self.num_routed_experts = 4
        self.num_logical_experts = 4
        self.num_redundant_experts = 2
        self.num_physical_experts = 6
        self.num_expert_groups = 1

    def set_eplb_state(self, load_buffer, l2p, replicas) -> None:
        for idx, layer in enumerate(self.moe_layers):
            layer.set_eplb_state(idx, load_buffer[idx], l2p[idx], replicas[idx])


def _make_controller(path: str) -> AscendEPLBController:
    parallel_config = SimpleNamespace(
        enable_eplb=True,
        enable_elastic_ep=False,
        eplb_config=SimpleNamespace(policy="stair", log_balancedness=False, step_interval=10),
    )
    ascend_cfg = EplbConfig(mode="static", expert_placement_path=path)
    return AscendEPLBController(parallel_config, torch.device("cpu"), ascend_cfg)


def test_static_controller_loads_target_and_draft_without_communicator(tmp_path):
    path = tmp_path / "placement.json"
    payload = {
        "version": 1,
        "ep_size": 2,
        "target": {
            "num_logical_experts": 4,
            "num_redundant_experts": 2,
            "physical_to_logical_map": [[3, 0, 1, 0, 2, 1], [1, 2, 0, 3, 1, 0]],
        },
        "draft": {
            "num_logical_experts": 4,
            "num_redundant_experts": 2,
            "physical_to_logical_map": [[2, 1, 0, 0, 3, 1]],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    ep_group = SimpleNamespace(world_size=2, rank_in_group=0, cpu_group=_FakeCpuGroup(0, 2))
    target_model = _FakeMoEModel(2)
    draft_model = _FakeMoEModel(1)
    target_cfg = SimpleNamespace(model="target-m", compile_factors=lambda: ("target-m",))
    draft_cfg = SimpleNamespace(model="draft-m", compile_factors=lambda: ("draft-m",))
    spec_cfg = SimpleNamespace(draft_model_config=draft_cfg, method="mtp", prefill_context_kv_only=False)
    vllm_cfg = SimpleNamespace(
        load_config=SimpleNamespace(load_format="safetensors"),
        parallel_config=SimpleNamespace(enable_eplb=True),
        speculative_config=spec_cfg,
    )

    with (
        patch("vllm_ascend.worker.v2.eplb.get_ep_group", return_value=ep_group),
        patch("vllm_ascend.distributed.eplb.eplb_state.get_ep_group", return_value=ep_group),
        patch("vllm.distributed.eplb.eplb_state.CpuGpuEvent", MagicMock),
        patch("vllm_ascend.distributed.eplb.eplb_state.torch.accelerator.current_device_index", return_value=0),
        patch("vllm_ascend.worker.v2.eplb.get_mixture_of_experts_model", side_effect=lambda m: m),
        patch("vllm.v1.worker.gpu.eplb_utils.get_mixture_of_experts_model", side_effect=lambda m: m),
        patch("torch.distributed.get_global_rank", return_value=0),
        patch("torch.distributed.broadcast_object_list"),
        patch(
            "torch.distributed.all_gather_object",
            side_effect=lambda out, obj, group: out.__setitem__(slice(None), [obj] * group.size()),
        ),
    ):
        controller = _make_controller(str(path))
        controller.prepare_load()
        assert ACTIVE_STATIC_CONTROLLER.get() is controller

        controller.bind_model_for_static_load(target_model, vllm_cfg, target_cfg)
        assert ACTIVE_STATIC_ROLE_PLACEMENT.get() == controller._snapshot.target  # type: ignore[union-attr]
        assert target_model.moe_layers[0]._static_logical_to_local_slot == {3: 0, 0: 1, 1: 2}

        controller.bind_model_for_static_load(draft_model, vllm_cfg, draft_cfg)
        assert draft_model.moe_layers[0]._static_logical_to_local_slot == {2: 0, 1: 1, 0: 2}

        speculator = SimpleNamespace(
            vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(enable_eplb=True)),
            model=draft_model,
            set_eplb_state=MagicMock(),
        )
        assert controller.maybe_register_speculator(speculator, spec_cfg, False)
        assert controller.maybe_register_model(target_model, target_cfg)
        controller.maybe_start_async_loop(True, load_dummy_weights=False)

        assert ACTIVE_STATIC_CONTROLLER.get() is None
        assert ACTIVE_STATIC_ROLE_PLACEMENT.get() is None
        state = controller.state
        assert state is not None and len(state.model_states) == 2
        for ms in state.model_states.values():
            assert ms.communicator is None
            assert ms.expert_buffer == []
            for layer in ms.model.moe_layers:
                assert layer.eplb_state.load_recording_enabled is False
                assert layer.eplb_state.expert_replica_routing_table is not None

        controller.prepare_forward(target_cfg, 16)
        controller.step()
        state.prepare_forward(target_cfg, 16)
        state.step()
        assert state.rearrange() is None


def test_static_controller_pp_zero_moe_stage_dummy_and_unexpected_draft(tmp_path):
    path = tmp_path / "placement.json"
    payload = {
        "version": 1,
        "ep_size": 2,
        "target": {
            "num_logical_experts": 4,
            "num_redundant_experts": 2,
            "physical_to_logical_map": [[3, 0, 1, 0, 2, 1]],
        },
        "draft": {
            "num_logical_experts": 4,
            "num_redundant_experts": 2,
            "physical_to_logical_map": [[2, 1, 0, 0, 3, 1]],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    ep_group = SimpleNamespace(world_size=2, rank_in_group=0, cpu_group=_FakeCpuGroup(0, 2))
    pp_group = SimpleNamespace(rank_in_group=0, world_size=2, cpu_group=_FakeCpuGroup(0, 2))
    dense_stage_model = _FakeMoEModel(0)
    full_stage_model = _FakeMoEModel(1)
    target_cfg = SimpleNamespace(model="target-m", compile_factors=lambda: ("target-m",))
    vllm_cfg = SimpleNamespace(
        load_config=SimpleNamespace(load_format="safetensors"),
        parallel_config=SimpleNamespace(enable_eplb=True),
        speculative_config=None,
    )

    def _pp_gather(out, obj, group):
        if group is pp_group.cpu_group and not isinstance(obj, bool):
            out[:] = [None, SimpleNamespace(num_logical_experts=4, num_redundant_experts=2, num_layers=1)]
        else:
            out[:] = [obj] * group.size()

    with (
        patch("vllm_ascend.worker.v2.eplb.get_ep_group", return_value=ep_group),
        patch("vllm_ascend.worker.v2.eplb.get_pp_group", return_value=pp_group),
        patch("vllm_ascend.distributed.eplb.eplb_state.get_ep_group", return_value=ep_group),
        patch("vllm.distributed.eplb.eplb_state.CpuGpuEvent", MagicMock),
        patch("vllm_ascend.distributed.eplb.eplb_state.torch.accelerator.current_device_index", return_value=0),
        patch("vllm_ascend.worker.v2.eplb.get_mixture_of_experts_model", side_effect=lambda m: m),
        patch("vllm.v1.worker.gpu.eplb_utils.get_mixture_of_experts_model", side_effect=lambda m: m),
        patch("torch.distributed.get_global_rank", return_value=0),
        patch("torch.distributed.broadcast_object_list"),
        patch("torch.distributed.all_gather_object", side_effect=_pp_gather),
    ):
        controller = _make_controller(str(path))
        controller.prepare_load()
        controller.bind_model_for_static_load(dense_stage_model, vllm_cfg, target_cfg)
        assert ACTIVE_STATIC_ROLE_PLACEMENT.get() is None
        assert controller.maybe_register_model(dense_stage_model, target_cfg) is False
        with pytest.raises(ValueError, match="no EPLB draft model is registered"):
            controller.maybe_start_async_loop(False, load_dummy_weights=False)

        # Dummy weight load builds default initial static state without reading snapshot
        controller.prepare_load()
        dummy_cfg = SimpleNamespace(
            load_config=SimpleNamespace(load_format="dummy"),
            parallel_config=SimpleNamespace(enable_eplb=True),
            speculative_config=None,
        )
        controller.bind_model_for_static_load(full_stage_model, dummy_cfg, target_cfg)
        assert controller.maybe_register_model(full_stage_model, target_cfg) is True
        controller.maybe_start_async_loop(True, load_dummy_weights=True)
        assert controller.state is not None and len(controller.state.model_states) == 1
