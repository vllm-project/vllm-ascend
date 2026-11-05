# SPDX-License-Identifier: Apache-2.0
"""State-copy key coverage and compile-only startup regression tests."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tests.ut.model_executor.warmup.helpers import make_registry
from vllm_ascend.ops import kda_state_copy_plan as plans
from vllm_ascend.ops.triton.batch_memcpy import _BATCH_MEMCPY_KERNEL
from vllm_ascend.ops.triton.kda_state_copy_kernel import _KDA_STATE_COPY_KERNEL, KDAStateCopyKernel


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("aligned", [False, True])
@pytest.mark.parametrize("rows,stride,payload", [(1, 1, 1), (7, 393232, 196608), (2, 2**31 + 32, 16)])
def test_keys_cover_directions_index_types_and_bound_layout(dtype, aligned, rows, stride, payload):
    keys = _KDA_STATE_COPY_KERNEL.get_warmup_keys(
        dtype=dtype,
        cache_rows=rows,
        cache_stride_elements=stride,
        payload_elements=payload,
        cache_aligned=aligned,
    )
    assert set(keys) == {
        KDAStateCopyKernel.CompileKey(dtype, index_dtype, rows, stride, payload, aligned, to_cache)
        for index_dtype in (torch.int32, torch.int64)
        for to_cache in (False, True)
    }


def test_preparation_uses_metadata_only_and_registration_requires_window():
    state = SimpleNamespace(
        device=SimpleNamespace(type="npu"),
        shape=(7, 2, 3, 4),
        dtype=torch.float32,
        stride=lambda: (80, 12, 4, 1),
        data_ptr=lambda: 4,
    )
    with (
        patch.object(torch, "empty", side_effect=AssertionError("device allocation")),
        patch.object(torch, "zeros", side_effect=AssertionError("device allocation")),
    ):
        plan = plans.KDAStateCopyPlan.prepare(state, 129)
        plan.seal()
    registry = make_registry()
    plan.register_warmup()
    assert len(registry) == 0
    with registry.activate():
        plan.register_warmup()
    assert len(registry) == 1


def test_compile_uses_only_warmup_and_runtime_preserves_two_dimensional_grid():
    owner = KDAStateCopyKernel()
    kernel = Mock()
    kernel.arg_names = _KDA_STATE_COPY_KERNEL.kernel.arg_names
    owner.kernel = kernel
    keys = owner.get_warmup_keys(
        dtype=torch.float32,
        cache_rows=7,
        cache_stride_elements=393232,
        payload_elements=196608,
        cache_aligned=False,
    )
    for key in keys:
        owner.compile(key)
    assert kernel.warmup.call_count == 4
    for call in kernel.warmup.call_args_list:
        args = call.kwargs
        assert args["grid"] == (1,)
        assert args["cache_ptr"].data_ptr() == 1
        assert args["packed_ptr"].data_ptr() == 0
        assert args["flags_ptr"].dtype == (args["indices_ptr"].dtype if args["TO_CACHE"] else torch.bool)
    kernel.__getitem__ = Mock(return_value=Mock())
    inputs = owner.warmup_inputs(keys[0])
    owner(**(inputs | {"selected": 129}))
    kernel.__getitem__.assert_called_once_with((129, 24))


def test_late_registration_deduplicates_layers_and_includes_fallback():
    plan = plans.KDAStateCopyPlan()
    plan._signature = ("npu", torch.float32, (7, 2, 3, 4), (80, 12, 4, 1), 0)
    plan._payload = 24
    plan._scalars = (7, 80, 24, 1)
    fallback = plans.StridedKDAFallbackPlan()
    fallback._signature = ("npu", torch.float16, (7, 2, 3, 4), (80, 12, 4, 1), 0)
    context = {name: SimpleNamespace(_ascend_kda_state_copy=p) for name, p in enumerate((plan, plan, fallback))}
    context["unrelated"] = SimpleNamespace()
    registry = make_registry()
    plans.register_kda_state_copy_warmup(context)
    assert len(registry) == 0
    with registry.activate():
        plans.register_kda_state_copy_warmup(context)
    assert set(registry._registrations) == {_KDA_STATE_COPY_KERNEL, _BATCH_MEMCPY_KERNEL}
    assert len(registry._registrations[_KDA_STATE_COPY_KERNEL]) == 1
    assert _BATCH_MEMCPY_KERNEL.get_warmup_keys(block_size=8192) == [_BATCH_MEMCPY_KERNEL.CompileKey(8192)]


def test_orchestrator_registers_bound_plans_before_warmup():
    import importlib

    from tests.ut.model_executor.warmup.helpers import make_mock_worker

    kw = importlib.import_module("vllm_ascend.model_executor.warmup.kernel_warmup")
    worker = make_mock_worker()
    fallback = plans.StridedKDAFallbackPlan()
    fallback._signature = ("npu", torch.float16, (7, 2, 3, 4), (80, 12, 4, 1), 0)
    worker.vllm_config.compilation_config.static_forward_context = {
        "layer": SimpleNamespace(_ascend_kda_state_copy=fallback)
    }
    registry = worker.model_runner.jit_warmup_registry
    registry.warmup = Mock(side_effect=lambda: _BATCH_MEMCPY_KERNEL in registry._registrations or pytest.fail())
    with patch.object(kw, "_register_dspark_swa_indices_warmup", return_value=0):
        kw.kernel_warmup(worker)
    registry.warmup.assert_called_once()


def test_disabled_warmup_does_not_register_or_compile():
    import importlib

    from tests.ut.model_executor.warmup.helpers import make_mock_worker

    kw = importlib.import_module("vllm_ascend.model_executor.warmup.kernel_warmup")
    worker = make_mock_worker(enable_jit_warmup=False)
    with patch.object(plans, "register_kda_state_copy_warmup") as register:
        kw.kernel_warmup(worker)
    register.assert_not_called()
