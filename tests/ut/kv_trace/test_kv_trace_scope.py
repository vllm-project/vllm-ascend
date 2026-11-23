# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output paths without a sole consumer keep their original synchronous audit."""

from types import SimpleNamespace as NS

import pytest

from .test_kv_trace_runtime import runtime_fixture as runtime_fixture
from .test_kv_trace_v2 import adapter_fixture as adapter_fixture


@pytest.mark.parametrize(
    "pooling,tp,pp,expected", [(False, 1, 1, True), (True, 1, 1, False), (False, 2, 1, False), (False, 1, 2, False)]
)
def test_only_single_worker_generation_defers(adapter, pooling, tp, pp, expected):
    runtime = adapter.runtime
    runtime.config.scheduler_config = NS(async_scheduling=True)
    runtime.config.parallel_config.tensor_parallel_size = tp
    runtime.config.parallel_config.pipeline_parallel_size = pp
    adapter.runner.is_pooling_model = pooling
    runtime.configure(adapter.runner)
    assert runtime._defer_audit is expected
