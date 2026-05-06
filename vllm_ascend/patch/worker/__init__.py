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

import contextlib

from vllm.triton_utils import HAS_TRITON

from vllm_ascend.utils import is_310p, vllm_version_is

# v2 model runner is only supported on vllm > 0.20.2.
_V2_MODEL_RUNNER_SUPPORTED = not vllm_version_is("0.20.2")

if HAS_TRITON:
    import vllm_ascend.patch.worker.patch_mamba_ssd  # noqa
    import vllm_ascend.patch.worker.patch_triton

    if _V2_MODEL_RUNNER_SUPPORTED:
        import vllm_ascend.patch.worker.patch_v2.patch_triton  # noqa
else:
    import vllm.model_executor.layers.mamba.ops.causal_conv1d

    from vllm_ascend._310p.ops.causal_conv1d import causal_conv1d_fn as _ascend_causal_conv1d_fn
    from vllm_ascend._310p.ops.causal_conv1d import causal_conv1d_update as _ascend_causal_conv1d_update

    vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_fn = _ascend_causal_conv1d_fn
    vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_update = _ascend_causal_conv1d_update


import vllm_ascend.patch.worker.patch_weight_utils  # noqa
import vllm_ascend.patch.platform.patch_sched_yield  # noqa
import vllm_ascend.patch.worker.patch_bert  # noqa
import vllm_ascend.patch.worker.patch_distributed  # noqa
import vllm_ascend.patch.worker.patch_minimax_m2  # noqa
import vllm_ascend.patch.worker.patch_minimax_m2_linear_attn  # noqa
import vllm_ascend.patch.worker.patch_mamba_utils  # noqa
import vllm_ascend.patch.worker.patch_mamba_weights  # noqa
import vllm_ascend.patch.worker.patch_qwen3_next_mtp  # noqa

if not is_310p():
    import vllm_ascend.patch.worker.patch_qwen3_5  # noqa
    import vllm_ascend.patch.worker.patch_gdn_attn  # noqa
    import vllm_ascend.patch.worker.patch_qwen3vl  # noqa
    if not vllm_version_is("0.19.1"):
        with contextlib.suppress(ModuleNotFoundError):
            import vllm_ascend.patch.worker.patch_qwen3_dflash  # noqa
else:
    import vllm_ascend.patch.worker.patch_idex_310  # noqa
import vllm_ascend.patch.worker.patch_rejection_sampler  # noqa
import vllm_ascend.patch.worker.patch_huanyuan_vl  # noqa
import vllm_ascend.patch.worker.patch_npugraph_ex_triton  # noqa
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
