# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod
from vllm_ascend.worker.sentinel import npu_worker_sentinel as sentinel_module
from vllm_ascend.worker.sentinel.eplb_redistribute import _reload_local_slots, scale_from_float_to_int64
from vllm_ascend.worker.sentinel.npu_worker_sentinel import WorkerSentinel


def test_fault_barrier_returns_empty_output_and_quarantines_following_calls():
    sentinel = SimpleNamespace(worker_faulted=False, reset_device=Mock())
    worker = SimpleNamespace(rank=0, worker_sentinel=sentinel)
    forward = Mock(side_effect=RuntimeError("MegaMoe device fault"))
    wrapped = sentinel_module.fault_barrier_wrapper(forward)
    with patch.object(sentinel_module, "_sentinel", sentinel):
        assert wrapped(worker) is sentinel_module.EMPTY_MODEL_RUNNER_OUTPUT
        assert sentinel.worker_faulted
        sentinel.reset_device.assert_called_once()
        assert wrapped(worker) is sentinel_module.EMPTY_MODEL_RUNNER_OUTPUT
    forward.assert_called_once_with(worker)


def test_scale_down_checks_dummy_batch_before_restoring_steady_state_timeout():
    sentinel = object.__new__(WorkerSentinel)
    calls = Mock()
    sentinel.worker = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=3,
            tensor_parallel_size=1,
            fault_tolerance_config=SimpleNamespace(engine_recovery_timeout_sec=300),
        ),
        execute_dummy_batch=calls.dummy,
    )
    sentinel._validate_scale_down_preconditions = Mock()
    sentinel.activate_cpu_group_timeouts = calls.activate
    request = object()
    with (
        patch.object(sentinel_module.GPUWorkerSentinel, "scale_down", side_effect=calls.scale_down),
        patch.object(sentinel_module, "get_dp_group", return_value=SimpleNamespace(cpu_group=object())),
        patch.object(sentinel_module, "set_gloo_backend_timeout", side_effect=calls.timeout),
        patch.object(torch.npu, "synchronize", side_effect=calls.synchronize),
    ):
        sentinel.scale_down(request)
    assert [call[0] for call in calls.mock_calls] == ["scale_down", "timeout", "dummy", "activate", "synchronize"]
    calls.activate.assert_called_once_with(request)


@pytest.mark.parametrize("mega_moe", [False, True])
def test_retry_publishes_mega_moe_mask_writes_before_leaving_quarantine(mega_moe):
    sentinel = object.__new__(WorkerSentinel)
    sentinel.worker = SimpleNamespace(vllm_config=object())
    sentinel.worker_faulted = True
    sentinel.reset_device = Mock()
    request = object()
    with (
        patch.object(sentinel_module.GPUWorkerSentinel, "retry") as retry,
        patch.object(sentinel_module, "use_cann_megamoe", return_value=mega_moe),
        patch.object(torch.npu, "synchronize") as synchronize,
        patch.object(sentinel_module, "get_ep_all2all_manager", side_effect=AssertionError("unexpected EP lookup")),
    ):
        sentinel.retry(request)
    sentinel.reset_device.assert_called_once()
    retry.assert_called_once_with(request)
    assert synchronize.call_count == int(mega_moe)
    assert not sentinel.worker_faulted


@pytest.mark.parametrize("mega_moe", [False, True])
def test_scale_down_uses_backend_specific_physical_expert_ids(mega_moe):
    # Rank 1 is dead: the expert on rank 2 must keep id 8 for MegaMoe,
    # but become id 4 for MC2. A tail-rank-only test would miss this bug.
    table = torch.tensor([[0], [8], [12]], dtype=torch.int32)
    layer = SimpleNamespace(eplb_state=SimpleNamespace(expert_replica_routing_table=table))
    state = SimpleNamespace(
        physical_to_logical_map=torch.empty(1, 16),
        logical_replica_count=torch.empty(1, 3),
        model=SimpleNamespace(moe_layers=[layer]),
    )
    sentinel = object.__new__(WorkerSentinel)
    sentinel._eplb_model_state = lambda: state
    sentinel.worker = SimpleNamespace(model_runner=SimpleNamespace(model=state.model), vllm_config=object())
    manager = SimpleNamespace(
        uses_mega_moe=mega_moe,
        query_active_mask=lambda: torch.tensor([0, 1, 0, 0]),
        get_elastic_info=Mock(return_value=torch.zeros(12)),
    )
    with (
        patch.object(sentinel_module, "mark_dead_expert_slots_inplace"),
        patch.object(sentinel_module, "redistribute_expert_placement", return_value={(0, 1)}),
        patch.object(sentinel_module, "_commit_eplb_maps"),
        patch.object(sentinel_module, "rebuild_model_expert_maps"),
        patch.object(sentinel_module, "reload_experts_from_disk") as reload_experts,
        patch.object(sentinel_module, "refresh_model_routing_tables"),
        patch.object(sentinel_module, "get_ep_group", return_value=SimpleNamespace(world_size=4)),
        patch.object(sentinel_module, "get_ep_all2all_manager", return_value=manager),
    ):
        sentinel._redistribute_experts({1})
    reload_experts.assert_called_once_with(state.model, sentinel.worker.vllm_config, {(0, 1)})
    assert table.flatten().tolist() == ([0, 8, 12] if mega_moe else [0, 4, 8])


@pytest.mark.parametrize(
    "fused,mega_moe,bound,error",
    [
        (1, True, True, None),
        (1, True, False, "warmup"),
        (1, False, False, "enable_fused_mc2"),
        (0, False, False, None),
    ],
)
def test_scale_down_rejects_unsupported_fused_backend(fused, mega_moe, bound, error):
    sentinel = object.__new__(WorkerSentinel)
    sentinel.worker = SimpleNamespace(
        use_v2_model_runner=True,
        model_runner=SimpleNamespace(eplb_state=object()),
        parallel_config=SimpleNamespace(eplb_config=SimpleNamespace(num_redundant_experts=8)),
        vllm_config=object(),
    )
    with (
        patch.object(
            sentinel_module,
            "get_ascend_config",
            return_value=SimpleNamespace(enable_fused_mc2=fused, enable_mc2_hierarchy_comm=False),
        ),
        patch.object(sentinel_module, "use_cann_megamoe", return_value=mega_moe),
        patch.object(sentinel_module, "get_ep_all2all_manager", return_value=SimpleNamespace(uses_mega_moe=bound)),
        patch.object(sentinel_module.torch_npu, "npu_moe_distribute_dispatch_v2", create=True),
    ):
        if error:
            with pytest.raises(ValueError, match=error):
                sentinel._validate_scale_down_preconditions()
        else:
            sentinel._validate_scale_down_preconditions()


def test_expert_reload_updates_mega_moe_packed_scales_without_replacing_storage():
    layer = SimpleNamespace(
        quant_method=object.__new__(AscendW8A8DynamicFusedMoEMethod),
        moe_config=SimpleNamespace(moe_parallel_config=SimpleNamespace(tp_rank=0, tp_size=1)),
        w13_weight_list=[torch.zeros(4, 6, dtype=torch.int8)],
        w2_weight_list=[torch.zeros(3, 4, dtype=torch.int8)],
        w13_weight_scale_fp32_list=[torch.zeros(6)],
        w2_weight_scale_list=[torch.zeros(4)],
        fused_w1_scale_list=[torch.zeros(6, dtype=torch.int64)],
        fused_w2_scale_list=[torch.zeros(4, dtype=torch.int64)],
    )
    tensors = {
        "gate_proj.weight": torch.full((3, 4), 2, dtype=torch.int8),
        "up_proj.weight": torch.full((3, 4), 3, dtype=torch.int8),
        "down_proj.weight": torch.full((4, 3), 4, dtype=torch.int8),
        "gate_proj.weight_scale": torch.full((3,), 0.25),
        "up_proj.weight_scale": torch.full((3,), 0.5),
        "down_proj.weight_scale": torch.full((4,), 0.75),
    }
    weight_address = layer.w13_weight_list[0].data_ptr()
    scale_address = layer.fused_w1_scale_list[0].data_ptr()
    with patch("vllm_ascend.worker.sentinel.eplb_redistribute.torch_npu.npu_format_cast", side_effect=lambda x, _: x):
        assert _reload_local_slots([layer], {(0, 0): 0}, {(0, 0): tensors}) == 1
    assert layer.w13_weight_list[0].data_ptr() == weight_address
    assert layer.fused_w1_scale_list[0].data_ptr() == scale_address
    assert layer.w13_weight_list[0][0].tolist() == [2, 2, 2, 3, 3, 3]
    torch.testing.assert_close(
        layer.fused_w1_scale_list[0], scale_from_float_to_int64(torch.tensor([0.25] * 3 + [0.5] * 3))
    )
    torch.testing.assert_close(layer.fused_w2_scale_list[0], scale_from_float_to_int64(torch.full((4,), 0.75)))
