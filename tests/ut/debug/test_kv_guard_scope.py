# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Outputs without a guaranteed sole consumer retain synchronous checking."""

from types import SimpleNamespace as NS

import pytest

from .test_kv_guard_runtime import runtime_fixture as runtime_fixture
from .test_kv_guard_v2 import adapter_fixture as adapter_fixture
from .test_kv_guard_v2 import message, prepare


@pytest.mark.parametrize("pooling,tp,pp", [(False, 1, 1), (True, 1, 1), (False, 2, 1), (False, 1, 2)])
def test_only_single_worker_generation_can_defer_checks_to_sample_output(adapter, pooling, tp, pp):
    runtime = adapter.runtime
    runtime.config.scheduler_config = NS(async_scheduling=True)
    adapter.runner.sample_tokens = lambda grammar: None
    adapter.runner.is_pooling_model = pooling
    runtime.config.parallel_config.tensor_parallel_size = tp
    runtime.config.parallel_config.pipeline_parallel_size = pp
    runtime.configure(adapter.runner)
    assert runtime._defer_audit is (not pooling and tp == 1 and pp == 1)


@pytest.mark.parametrize("pooling,tp,pp", [(True, 1, 1), (False, 2, 1), (False, 1, 2)])
def test_failure_is_detected_before_pool_or_nonreply_output_can_bypass_the_getter(adapter, pooling, tp, pp):
    runtime = adapter.runtime
    runtime.config.scheduler_config = NS(async_scheduling=True)
    adapter.runner.sample_tokens = lambda grammar: None
    adapter.runner.is_pooling_model = pooling
    runtime.config.parallel_config.tensor_parallel_size = tp
    runtime.config.parallel_config.pipeline_parallel_size = pp
    runtime.action = "raise"
    runtime.state.emit = runtime.report
    runtime.configure(adapter.runner)
    with pytest.raises(runtime.subject.KVGuardError, match="write_target_mismatch"):
        adapter.wrap_execute(lambda output: prepare(adapter, ["r"], [129], [[1]], [1], "FULL"))(message())
    assert runtime._pending is None


@pytest.mark.parametrize("missing", ["pooling", "tp", "pp"])
def test_unknown_output_layout_keeps_synchronous_validation(adapter, missing):
    runtime = adapter.runtime
    runtime.config.scheduler_config = NS(async_scheduling=True)
    adapter.runner.sample_tokens = lambda grammar: None
    adapter.runner.is_pooling_model = False
    parallel = runtime.config.parallel_config
    parallel.tensor_parallel_size = parallel.pipeline_parallel_size = 1
    if missing == "pooling":
        del adapter.runner.is_pooling_model
    else:
        delattr(parallel, "tensor_parallel_size" if missing == "tp" else "pipeline_parallel_size")
    runtime.configure(adapter.runner)
    assert not runtime._defer_audit
