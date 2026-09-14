#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from types import ModuleType

import vllm.triton_utils as _vllm_triton_utils
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

# main2main compat: vllm main decorates sampling/spec-decode Triton kernels with
# triton_kernel_dispatcher_with_warmup, whose _is_autotuned reads
# ``triton.runtime.autotuner.Autotuner`` at import time. When Triton is disabled,
# vllm.triton_utils.triton is a TritonPlaceholder without `runtime`, so importing
# vllm.v1.spec_decode.utils raises AttributeError. Give the placeholder the
# attributes it reads; a dummy Autotuner matches no kernel, so detection is inert.
if not HAS_TRITON and not hasattr(_vllm_triton_utils.triton, "runtime"):
    _autotuner_stub = ModuleType("triton.runtime.autotuner")
    for _attr_name, _attr_value in (
        ("Autotuner", type("Autotuner", (), {})),
        ("Heuristics", type("Heuristics", (), {})),
    ):
        setattr(_autotuner_stub, _attr_name, _attr_value)
    _runtime_stub = ModuleType("triton.runtime")
    _autotuner_attr = "autotuner"
    _runtime_attr = "runtime"
    setattr(_runtime_stub, _autotuner_attr, _autotuner_stub)
    setattr(_vllm_triton_utils.triton, _runtime_attr, _runtime_stub)

if HAS_TRITON:
    import vllm_ascend.patch.worker.patch_triton
    import vllm_ascend.patch.worker.patch_v2.patch_triton  # noqa


import vllm_ascend.patch.worker.patch_distributed  # noqa
import vllm_ascend.patch.worker.patch_minimax_m2  # noqa
import vllm_ascend.patch.worker.patch_mamba_utils  # noqa
import vllm_ascend.patch.worker.patch_bind_kv_cache  # noqa
import vllm_ascend.patch.worker.patch_step3p5  # noqa

if get_current_hardware_profile().supports(HardwareCapability.STANDARD_WORKER_PATCHES):
    import vllm_ascend.patch.worker.patch_qwen3_5  # noqa
    import vllm_ascend.patch.worker.patch_qwen3_dflash  # noqa
    import vllm_ascend.patch.worker.patch_qwen3vl  # noqa
else:
    import vllm_ascend.patch.worker.patch_idex_310  # noqa
    import vllm_ascend.patch.worker.patch_v2.patch_spec_decode_310  # noqa
import vllm_ascend.patch.worker.patch_rejection_sampler  # noqa

import vllm_ascend.patch.worker.patch_kimi_k25  # noqa
import vllm_ascend.patch.worker.patch_eagle3_init  # noqa
import vllm_ascend.patch.worker.patch_cudagraph  # noqa
import vllm_ascend.patch.worker.patch_deepseek_v2  # noqa

# vLLM's use_v2_model_runner may enable the v2 runner without the
# VLLM_USE_V2_MODEL_RUNNER env var (e.g. based on model architecture).
# We always patch it so that on Ascend the v2 runner is enabled only
# when the env var is explicitly set.
import vllm_ascend.patch.worker.patch_v2.patch_use_v2_model_runner  # noqa

import vllm_ascend.patch.worker.patch_fused_moe  # noqa

import vllm_ascend.patch.worker.patch_v2.patch_uva  # noqa
import vllm_ascend.patch.worker.patch_v2.patch_input_batch  # noqa
import vllm_ascend.patch.worker.patch_v2.patch_model_state  # noqa
import vllm_ascend.patch.worker.patch_v2.patch_block_table  # noqa
import vllm_ascend.patch.worker.patch_v2.patch_attn_utils  # noqa

import vllm_ascend.patch.worker.patch_v2.patch_eagle_speculator  # noqa
import vllm_ascend.patch.worker.patch_v2.patch_dflash_speculator  # noqa
import vllm_ascend.patch.worker.patch_v2.patch_dspark  # noqa

# 310P: draft FULL must use AutoRegressiveAclGraphManager310 (no FIA graph_task).
# patch_eagle_speculator above installs the 910 manager; re-override here.
if not get_current_hardware_profile().supports(HardwareCapability.STANDARD_WORKER_PATCHES):
    from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as _ar_spec

    from vllm_ascend._310p.worker.v2.spec_decode.aclgraph import (
        AutoRegressiveAclGraphManager310,
    )

    _ar_spec.SpeculatorCudaGraphManager = AutoRegressiveAclGraphManager310

# only patch routed experts capture in main2main.
import vllm_ascend.patch.worker.patch_routed_experts_capture  # noqa
