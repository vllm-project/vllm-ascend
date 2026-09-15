# SPDX-License-Identifier: Apache-2.0

import inspect

import vllm.v1.worker.gpu.sample.thinking_budget as thinking_budget

from vllm_ascend.patch.platform.patch_thinking_budget import (
    _thinking_budget_kernel,
    _update_committed_marker_cache_kernel,
)


def test_ascend_thinking_budget_kernels_are_registered():
    assert thinking_budget._thinking_budget_kernel is _thinking_budget_kernel
    assert (
        thinking_budget._update_committed_marker_cache_kernel
        is _update_committed_marker_cache_kernel
    )


def test_local_marker_scan_uses_static_slots():
    source = inspect.getsource(_thinking_budget_kernel.fn)
    assert "tl.static_range(0, 4)" in source
    assert "tl.range(start_lo" not in source
    assert "tl.range(end_lo" not in source
