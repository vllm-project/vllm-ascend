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

from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    import vllm_ascend.patch.worker.patch_triton

# isort: off
import vllm_ascend.patch.platform.patch_sched_yield  # noqa
import vllm_ascend.patch.worker.patch_unquantized_gemm  # noqa
import vllm_ascend.patch.worker.patch_bert  # noqa
import vllm_ascend.patch.worker.patch_distributed  # noqa
import vllm_ascend.patch.worker.patch_multimodal_merge  # noqa
import vllm_ascend.patch.worker.patch_qwen3_next  # noqa
import vllm_ascend.patch.worker.patch_qwen3_next_mtp  # noqa
import vllm_ascend.patch.worker.patch_rejection_sampler  # noqa
import vllm_ascend.patch.worker.patch_v2_eagle  # noqa
import vllm_ascend.patch.worker.patch_v2_uva  # noqa
import vllm_ascend.patch.worker.patch_huanyuan_vl  # noqa
import vllm_ascend.patch.worker.patch_routed_experts_capturer  # noqa
import vllm_ascend.patch.worker.patch_npugraph_ex_triton  # noqa
import vllm_ascend.patch.worker.patch_kimi_k25  # noqa
