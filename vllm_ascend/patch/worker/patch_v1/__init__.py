#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""MRv1-only worker patches.

Every module in this package patches a symbol that is only reachable through
the V1 Model Runner (``vllm.v1.worker.gpu_model_runner`` and its
``vllm.v1.worker.*`` helper modules). The V2 Model Runner (``vllm.v1.worker.gpu.*``)
never touches them, so the whole package is dead code once MRv1 is removed:
delete this folder and the single ``patch_v1`` import in
``vllm_ascend/patch/worker/__init__.py`` to drop V1 patch support.
"""

import vllm_ascend.patch.worker.patch_v1.patch_cudagraph  # noqa
import vllm_ascend.patch.worker.patch_v1.patch_mamba_utils  # noqa
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

if not get_current_hardware_profile().supports(HardwareCapability.STANDARD_WORKER_PATCHES):
    import vllm_ascend.patch.worker.patch_v1.patch_idex_310  # noqa

# ``patch_eagle3_pp_aux.py`` lives here as well but is not imported eagerly: it
# only defines helpers and has no import-time side effect. Its sole consumer is
# ``vllm_ascend.worker.v1.model_runner``, which imports
# ``patch_eagle3_pp_aux_propagation`` lazily while loading an Eagle3 target model.
