# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Resolve the target-dependent AR implementation before and after vLLM #60335."""

try:
    from vllm.v1.worker.gpu.spec_decode.target_dependent_ar import speculator as ar_speculator_module
    from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.cudagraph_utils import SpeculatorCudaGraphManager
    from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
        TargetDependentARSpeculator as AutoRegressiveSpeculator,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {
        "vllm.v1.worker.gpu.spec_decode.target_dependent_ar",
        "vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator",
    }:
        raise
    from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as ar_speculator_module
    from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import SpeculatorCudaGraphManager
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import AutoRegressiveSpeculator

__all__ = ["AutoRegressiveSpeculator", "SpeculatorCudaGraphManager", "ar_speculator_module"]
