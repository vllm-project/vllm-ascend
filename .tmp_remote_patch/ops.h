/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <optional>
#include <torch/library.h>

#include <vector>
#include "kernels/types.h"
#include "torch_npu/csrc/aten/common/from_blob.h"

namespace vllm_ascend {
  extern void bgmv_shrink_impl(
        AscendType type,
        void *stream,
        void *x,
        void *weight,
        void *indices,
        uint32_t indicesSize,
        void *y, 
        uint32_t batch_size,
        uint32_t num_tokens_per_core,
        uint32_t input_hidden_dim,
        uint32_t lora_rank,
        float scale,
        uint32_t aiv_num);

    extern void bgmv_expand_impl(
        AscendType type,
        void *stream,
        void *x,
        void *weight,
        void *indices,
        uint32_t indicesSize,
        void *y,
        void *y_out,
        uint32_t batch_size,
        uint32_t num_tokens_per_core,
        uint32_t lora_rank,
        uint32_t output_hidden_dim,
        uint32_t slice_offset,
        uint32_t output_full_dim,
        uint32_t aiv_num);

    extern void bgmv_moe_w13_impl(
        AscendType type,
        void *stream,
        void *x,
        void *weight_a0,
        void *weight_a1,
        void *weight_b0,
        void *weight_b1,
        void *indices,
        uint32_t indices_size,
        void *workspace,
        void *y,
        uint32_t batch_size,
        uint32_t input_hidden_dim,
        uint32_t output_slice_dim,
        uint32_t slice_offset,
        uint32_t output_full_dim,
        float scale,
        uint32_t aiv_num);

    extern void moe_lora_routing_impl(
        void *stream,
        void *expanded_row_idx,
        void *topk_ids,
        void *token_lora_indices,
        void *adapter_enabled,
        void *combined_indices,
        uint32_t num_rows,
        uint32_t num_tokens,
        uint32_t num_adapters,
        uint32_t top_k,
        uint32_t num_experts,
        bool index64,
        uint32_t enabled_type);

    extern void moe_lora_prefill_route_allgather_impl(
        void *stream, void *expanded_row_idx, void *routed_topk_ids,
        void *token_lora_indices, void *adapter_enabled, void *local_count,
        void *error_per_core, uint32_t canonical_rows, uint32_t local_rows,
        uint32_t num_tokens, uint32_t num_adapters, uint32_t top_k,
        uint32_t num_experts, uint32_t group_pitch, int64_t first_expert_idx,
        uint32_t block_dim, uint32_t route_tile_rows, bool index64,
        uint32_t enabled_type);

    extern void moe_lora_prefill_prefix_b1_impl(
        void *stream, void *local_count, void *core_prefix, void *group_total,
        uint32_t num_groups, uint32_t group_pitch, uint32_t num_cores,
        uint32_t block_dim, uint32_t prefix_tile_groups);

    extern void moe_lora_prefill_prefix_b2_impl(
        void *stream, void *group_total, void *error_per_core,
        void *group_start, void *group_count_i64, void *route_error,
        uint32_t num_groups, uint32_t group_pitch, uint32_t num_cores,
        uint32_t num_rows);

    extern void moe_lora_prefill_scatter_allgather_impl(
        void *stream, void *x, void *expanded_row_idx, void *routed_topk_ids,
        void *token_lora_indices, void *adapter_enabled, void *core_prefix,
        void *group_start, void *group_total, void *grouped_x, void *perm_record,
        uint32_t canonical_rows, uint32_t num_rows, uint32_t num_tokens,
        uint32_t num_adapters, uint32_t top_k, uint32_t num_experts,
        uint32_t num_groups, uint32_t group_pitch, uint32_t input_width,
        uint32_t grouped_stride, int64_t first_expert_idx, uint32_t block_dim,
        uint32_t route_tile_rows, uint32_t column_tile_elements,
        bool is_bfloat16, bool index64, uint32_t enabled_type);

    extern void moe_lora_prefill_route_alltoall_impl(
        void *stream, void *expert_count, void *exchanged_lora_indices,
        void *adapter_enabled, void *local_count, void *error_per_core,
        uint32_t num_rows, uint32_t num_adapters, uint32_t num_experts,
        uint32_t group_pitch, uint32_t block_dim, uint32_t route_tile_rows,
        bool count64,
        uint32_t enabled_type);

    extern void moe_lora_prefill_scatter_alltoall_impl(
        void *stream, void *x, void *expert_count,
        void *exchanged_lora_indices, void *adapter_enabled,
        void *core_prefix, void *group_start, void *grouped_x,
        void *perm_record, uint32_t num_rows, uint32_t num_adapters,
        uint32_t num_experts, uint32_t num_groups, uint32_t group_pitch,
        uint32_t input_width, uint32_t grouped_stride, uint32_t block_dim,
        uint32_t route_tile_rows, uint32_t column_tile_elements,
        bool is_bfloat16, bool count64, uint32_t enabled_type);

    extern void moe_lora_prefill_gather_by_perm_impl(
        void *stream, void *source, void *perm_record, void *grouped_x,
        uint32_t num_rows, uint32_t input_width, uint32_t grouped_stride,
        uint32_t block_dim, uint32_t route_tile_rows,
        uint32_t column_tile_elements, bool is_bfloat16);

    extern void moe_lora_prefill_scatter_add_impl(
        void *stream, void *delta, void *perm_record, void *y,
        uint32_t num_rows, uint32_t delta_width, uint32_t output_width,
        uint32_t output_offset, uint32_t block_dim, uint32_t route_tile_rows,
        uint32_t scatter_add_tile_elements, bool is_bfloat16);

    extern void sgmv_shrink_impl(
        AscendType type,
        void *stream,
        void *x,
        void *weight,
        void *loraIndices,
        uint32_t loraIndicesSize,
        void *seqLen,
        uint32_t seqLenSize,
        void *y,
        uint32_t batch_size,
        uint32_t num_tokens_per_core,
        uint32_t input_hidden_dim,
        uint32_t lora_rank,
        float scale);

    extern void sgmv_expand_impl(
        AscendType type,
        void *stream,
        void *x,
        void *weight,
        void *loraIndices,
        uint32_t loraIndicesSize,
        void *seqLen,
        uint32_t seqLenSize,
        void *y,
        void *y_out,
        uint32_t batch_size,
        uint32_t num_tokens_per_core,
        uint32_t lora_rank,
        uint32_t output_hidden_dim,
        uint32_t slice_offset,
        uint32_t output_full_dim);

    extern void mla_preprocess_impl(
        void* stream,
        void* hidden_state,
        void* quant_scale1,
        void* quant_offset1,
        void* wdqkv,
        void* bias1,
        void* gamma2,
        void* beta2,
        void* quant_scale2,
        void* quant_offset2,
        void* gamma3,
        void* sin1,
        void* cos1,
        void* sin2,
        void* cos2,
        void* keycache,
        void* slot_mapping,
        void* wuq,
        void* bias2,
        void* wuk,
        void* descale1,
        void* descale2,
        void* ctkv_scale,
        void* qnope_scale,
        void* q,
        void* keycache_out,
        void* q2,
        void* keycache_out2,
        void* inner_out,
        void* workspace,
        void* tiling,
        const uint32_t block_dim
    );

    extern void batch_matmul_transpose_impl(
        void* stream,
        void* gm_a,
        void* gm_b,
        void* gm_c,
        void* gm_tiling_data,
        const uint32_t block_dim
    );
}
