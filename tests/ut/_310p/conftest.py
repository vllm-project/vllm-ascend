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
    had_next_power = hasattr(triton, "next_power_of_2")
    register_sampling_kernel_overrides.cache_clear()
    register_sampling_kernel_overrides()
    try:
        yield
    finally:
        dispatcher._rebind_kernels([(getattr(module, name), original) for module, name, original in originals])
        for module, name, original in originals:
            setattr(module, name, original)
        dispatcher._registry.clear()
        dispatcher._registry.update(registry)
        if not had_next_power:
            del triton.next_power_of_2
        register_sampling_kernel_overrides.cache_clear()
