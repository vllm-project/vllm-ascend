#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

import vllm_ascend.ops.activation  # noqa
import vllm_ascend.ops.bailing_moe_linear_attn  # noqa
import vllm_ascend.ops.conv  # noqa
import vllm_ascend.ops.cv_linear  # noqa
import vllm_ascend.ops.dsa  # noqa
import vllm_ascend.ops.flashcomm2_oshard_manager  # noqa
import vllm_ascend.ops.fused_moe.comm_utils  # noqa
import vllm_ascend.ops.fused_moe.experts_selector  # noqa
import vllm_ascend.ops.fused_moe.fused_moe  # noqa
import vllm_ascend.ops.fused_moe.gate_linear  # noqa
import vllm_ascend.ops.fused_moe.moe_comm_method  # noqa
import vllm_ascend.ops.fused_moe.moe_mlp  # noqa
import vllm_ascend.ops.fused_moe.moe_runtime_args  # noqa
import vllm_ascend.ops.fused_moe.moe_stage_contracts  # noqa
import vllm_ascend.ops.fused_moe.moe_stage_params  # noqa
import vllm_ascend.ops.fused_moe.prepare_finalize  # noqa
import vllm_ascend.ops.fused_moe.token_dispatcher  # noqa
import vllm_ascend.ops.gdn  # noqa
import vllm_ascend.ops.gdn_attn_builder  # noqa
import vllm_ascend.ops.layer_shard_linear  # noqa
import vllm_ascend.ops.layernorm  # noqa
import vllm_ascend.ops.linear  # noqa
import vllm_ascend.ops.linear_op  # noqa
import vllm_ascend.ops.mhc  # noqa
import vllm_ascend.ops.mla  # noqa
import vllm_ascend.ops.mm_encoder_attention  # noqa
import vllm_ascend.ops.qwen2_decoder  # noqa
import vllm_ascend.ops.register_custom_ops  # noqa
import vllm_ascend.ops.rel_pos_attention  # noqa
import vllm_ascend.ops.rope_dsv4  # noqa
import vllm_ascend.ops.rotary_embedding  # noqa
import vllm_ascend.ops.vocab_parallel_embedding  # noqa
import vllm_ascend.ops.weight_prefetch  # noqa

from vllm_ascend.device.device_config import DeviceConfig

if DeviceConfig.supports_triton:
    import vllm_ascend.ops.triton.batch_invariant.matmul  # noqa
    import vllm_ascend.ops.triton.batch_invariant.mean  # noqa
    import vllm_ascend.ops.triton.batch_invariant.rmsnorm  # noqa
    import vllm_ascend.ops.triton.batch_invariant.softmax  # noqa
    import vllm_ascend.ops.triton.batch_memcpy  # noqa
    import vllm_ascend.ops.triton.bincount  # noqa
    import vllm_ascend.ops.triton.fla.chunk  # noqa
    import vllm_ascend.ops.triton.fla.chunk_delta_h  # noqa
    import vllm_ascend.ops.triton.fla.chunk_delta_hupdate  # noqa
    import vllm_ascend.ops.triton.fla.chunk_o  # noqa
    import vllm_ascend.ops.triton.fla.chunk_o_update  # noqa
    import vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt  # noqa
    import vllm_ascend.ops.triton.fla.cumsum  # noqa
    import vllm_ascend.ops.triton.fla.fused_qkvzba_split_reshape  # noqa
    import vllm_ascend.ops.triton.fla.l2norm  # noqa
    import vllm_ascend.ops.triton.fla.layernorm_guard  # noqa
    import vllm_ascend.ops.triton.fla.sigmoid_gating  # noqa
    import vllm_ascend.ops.triton.fla.solve_tril  # noqa
    import vllm_ascend.ops.triton.fla.utils  # noqa
    import vllm_ascend.ops.triton.fla.wy_fast  # noqa
    import vllm_ascend.ops.triton.fused_gdn_gating  # noqa
    import vllm_ascend.ops.triton.gdn_chunk_meta  # noqa
    import vllm_ascend.ops.triton.layernorm_gated  # noqa
    import vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_mrope  # noqa
    import vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_rope  # noqa
    import vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_rope_simt  # noqa
    import vllm_ascend.ops.triton.linearnorm.split_qkv_tp_rmsnorm_rope  # noqa
    import vllm_ascend.ops.triton.mamba.causal_conv1d  # noqa
    import vllm_ascend.ops.triton.mamba.lightning_attn  # noqa
    import vllm_ascend.ops.triton.mul_add  # noqa
    import vllm_ascend.ops.triton.muls_add  # noqa
    import vllm_ascend.ops.triton.penalty  # noqa
    import vllm_ascend.ops.triton.reject_sample  # noqa
    import vllm_ascend.ops.triton.rms_norm  # noqa
    import vllm_ascend.ops.triton.rope  # noqa
    import vllm_ascend.ops.triton.spec_decode.utils  # noqa
    import vllm_ascend.ops.triton.activation.swiglu_quant  # noqa
    import vllm_ascend.ops.triton.triton_utils  # noqa
