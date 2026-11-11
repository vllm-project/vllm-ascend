# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Keep process-wide upstream kernel overrides local to 310P test modules."""

import importlib

import pytest
from vllm.triton_utils import dispatcher, triton

from vllm_ascend._310p.worker.v2.sample.dispatch import register_sampling_kernel_overrides


@pytest.fixture(scope="module")
def sampling_kernel_overrides():
    modules = [
        (importlib.import_module("vllm.v1.worker.gpu.sample.gumbel"), "_temperature_kernel"),
        (importlib.import_module("vllm.v1.worker.gpu.sample.min_p"), "_min_p_kernel"),
        (importlib.import_module("vllm.v1.worker.gpu.sample.logit_bias"), "_bias_kernel"),
    ]
    originals = [(module, name, getattr(module, name)) for module, name in modules]
    registry = dispatcher._registry.copy()
    previous_vars = dispatcher.__dict__.get("vars")
    had_next_power = hasattr(triton, "next_power_of_2")
    previous_next_power = getattr(triton, "next_power_of_2", None)
    register_sampling_kernel_overrides.cache_clear()
    register_sampling_kernel_overrides()
    try:
        yield
    finally:
        # Restore the original objects, including aliases, without wrapping them again.
        rebind = getattr(dispatcher, "_rebind_impls", None) or dispatcher._rebind_kernels
        rebind([(getattr(module, name), original) for module, name, original in originals])
        for module, name, original in originals:
            setattr(module, name, original)
        dispatcher._registry.clear()
        dispatcher._registry.update(registry)
        if previous_vars is None:
            del dispatcher.vars
        else:
            dispatcher.vars = previous_vars
        if had_next_power:
            triton.next_power_of_2 = previous_next_power
        else:
            del triton.next_power_of_2
        register_sampling_kernel_overrides.cache_clear()
