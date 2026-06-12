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

from vllm_ascend.utils import is_310p, vllm_version_is

# v2 model runner patches depend on upstream main APIs beyond v0.21.0.
_V2_MODEL_RUNNER_SUPPORTED = not vllm_version_is("0.21.0")

if HAS_TRITON:
    import vllm_ascend.patch.worker.patch_triton

    if _V2_MODEL_RUNNER_SUPPORTED:
        import vllm_ascend.patch.worker.patch_v2.patch_triton  # noqa


import vllm_ascend.patch.worker.patch_weight_utils  # noqa
import vllm_ascend.patch.worker.patch_distributed  # noqa
import vllm_ascend.patch.worker.patch_minimax_m2  # noqa
import vllm_ascend.patch.worker.patch_minimax_m2_linear_attn  # noqa
import vllm_ascend.patch.worker.patch_mamba_utils  # noqa
import vllm_ascend.patch.worker.patch_qwen3_next_mtp  # noqa

if not is_310p():
    import vllm_ascend.patch.worker.patch_qwen3_5  # noqa
    import vllm_ascend.patch.worker.patch_gdn_attn  # noqa
    import vllm_ascend.patch.worker.patch_qwen3_dflash  # noqa
    import vllm_ascend.patch.worker.patch_qwen3vl  # noqa
else:
    import vllm_ascend.patch.worker.patch_idex_310  # noqa
import vllm_ascend.patch.worker.patch_rejection_sampler  # noqa

# torchair/npugraph_ex is only available on NPU; silently skip when missing
# so that CPU-only environments (e.g. UT runners without torch_npu) can still
# import this module without crashing.
try:  # noqa: SIM105
    import vllm_ascend.patch.worker.patch_npugraph_ex_triton  # noqa
except ImportError:
    pass
import vllm_ascend.patch.worker.patch_kimi_k25  # noqa
import vllm_ascend.patch.worker.patch_draft_quarot  # noqa
import vllm_ascend.patch.worker.patch_cudagraph  # noqa
import vllm_ascend.patch.worker.patch_deepseek_mtp  # noqa
import vllm_ascend.patch.worker.patch_gqa_c8  # noqa

if _V2_MODEL_RUNNER_SUPPORTED:
    import vllm_ascend.patch.worker.patch_v2.patch_uva  # noqa
    import vllm_ascend.patch.worker.patch_v2.patch_input_batch  # noqa
    import vllm_ascend.patch.worker.patch_v2.patch_model_state  # noqa
    import vllm_ascend.patch.worker.patch_v2.patch_block_table  # noqa
    import vllm_ascend.patch.worker.patch_v2.patch_attn_utils  # noqa

# only patch routed experts capture in main2main.
if _V2_MODEL_RUNNER_SUPPORTED:
    import vllm_ascend.patch.worker.patch_routed_experts_capture  # noqa
