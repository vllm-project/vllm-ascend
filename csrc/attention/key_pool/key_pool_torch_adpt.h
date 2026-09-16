/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
#ifndef KEY_POOL_TORCH_ADPT_H
#define KEY_POOL_TORCH_ADPT_H

#include <algorithm>

namespace vllm_ascend {

constexpr int64_t KEY_POOL_DIM_ONE = 1;
constexpr int64_t KEY_POOL_DIM_TWO = 2;
constexpr int64_t KEY_POOL_DIM_THREE = 3;
constexpr int64_t KEY_POOL_MAX_DIM_SIZE = 8;
constexpr int64_t KEY_POOL_VALUE_ZERO = 0;

at::Tensor ConstructKeyPoolOutputTensor(const at::Tensor& hidden_states,
                                        const at::Tensor& wk,
                                        const at::Tensor& state_cache,
                                        const at::Tensor& cache_block_table,
                                        int64_t cmp_ratio)
{
    auto hidden_states_dim = hidden_states.dim();
    at::SmallVector<int64_t, KEY_POOL_MAX_DIM_SIZE> pooled_key_size;
    at::Tensor pooled_key;

    TORCH_CHECK(wk.defined(), "Check wk != nullptr failed");
    auto wk_dim = wk.dim();
    TORCH_CHECK(wk_dim == KEY_POOL_DIM_TWO, "wk dim num[", wk_dim,
                "] should be 2");

    TORCH_CHECK(hidden_states_dim == KEY_POOL_DIM_TWO ||
                    hidden_states_dim == KEY_POOL_DIM_THREE,
                "hidden_states dim num[", hidden_states_dim,
                "] should be 2 or 3");
    TORCH_CHECK(state_cache.dim() == KEY_POOL_DIM_THREE,
                "state_cache dim num[", state_cache.dim(), "] should be 3");
    TORCH_CHECK(cache_block_table.dim() == KEY_POOL_DIM_TWO,
                "cache_block_table dim num[", cache_block_table.dim(),
                "] should be 2");
    // The block table covers addressable history. A call can complete at
    // most ceil(tokens / cmp_ratio) pools for any one sequence.
    int64_t tokens_per_batch = hidden_states_dim == KEY_POOL_DIM_TWO
                                   ? hidden_states.size(0)
                                   : hidden_states.size(1);
    if (hidden_states_dim == KEY_POOL_DIM_TWO) {
        int64_t pcap = std::min(tokens_per_batch,
                               tokens_per_batch / cmp_ratio + cache_block_table.size(0));
        pooled_key_size = {pcap, wk.size(0)};
    } else {
        int64_t pcap = (tokens_per_batch + cmp_ratio - 1) / cmp_ratio;
        pooled_key_size = {hidden_states.size(0), pcap, wk.size(0)};
    }

    pooled_key = at::empty(pooled_key_size,
                           hidden_states.options().dtype(hidden_states.dtype()));
    return pooled_key;
}

at::Tensor npu_key_pool(const at::Tensor& hidden_states, const at::Tensor& wk,
                        const at::Tensor& gate_weight, const at::Tensor& ape,
                        at::Tensor& state_cache,
                        const at::Tensor& cache_block_table,
                        const at::Tensor& start_pos,
                        const c10::optional<at::Tensor>& norm_weight,
                        const c10::optional<at::Tensor>& norm_bias,
                        const c10::optional<at::Tensor>& cos,
                        const c10::optional<at::Tensor>& sin,
                        const c10::optional<at::Tensor>& cu_seqlens,
                        const c10::optional<at::Tensor>& seqused,
                        int64_t cmp_ratio, double norm_eps, int64_t rotary_mode)
{
    TORCH_CHECK(hidden_states.defined(), "Check hidden_states != nullptr failed");
    auto hidden_states_dim = hidden_states.dim();
    TORCH_CHECK(hidden_states_dim == KEY_POOL_DIM_TWO ||
                    hidden_states_dim == KEY_POOL_DIM_THREE,
                "hidden_states dim num[", hidden_states_dim,
                "] should be 2 or 3");

    TORCH_CHECK(cmp_ratio > KEY_POOL_VALUE_ZERO,
                "cmp_ratio should be greater than 0");
    TORCH_CHECK(cmp_ratio >= 2 && cmp_ratio <= 128 && (cmp_ratio & (cmp_ratio - 1)) == 0,
                "cmp_ratio must be one of 2, 4, 8, 16, 32, 64, 128");

    auto check_same_device = [&hidden_states](const at::Tensor& tensor,
                                              const char* name) {
        TORCH_CHECK(tensor.device() == hidden_states.device(), name,
                    " must be on the same device as hidden_states (",
                    hidden_states.device(), "), got ", tensor.device());
    };
    check_same_device(wk, "wk");
    check_same_device(gate_weight, "gate_weight");
    check_same_device(ape, "ape");
    check_same_device(state_cache, "state_cache");
    check_same_device(cache_block_table, "cache_block_table");
    check_same_device(start_pos, "start_pos");

    TORCH_CHECK(hidden_states.scalar_type() == at::kBFloat16 ||
                    hidden_states.scalar_type() == at::kHalf,
                "hidden_states must be BF16 or FP16");
    TORCH_CHECK(wk.scalar_type() == hidden_states.scalar_type() &&
                    gate_weight.scalar_type() == hidden_states.scalar_type(),
                "wk and gate_weight dtype must match hidden_states");
    TORCH_CHECK(wk.dim() == KEY_POOL_DIM_TWO &&
                    gate_weight.sizes() == wk.sizes(),
                "wk and gate_weight must have the same rank-2 shape");
    TORCH_CHECK(wk.size(1) == hidden_states.size(hidden_states_dim - 1),
                "wk input dimension must match hidden_states hidden size");
    TORCH_CHECK(ape.scalar_type() == at::kFloat &&
                    ape.dim() == KEY_POOL_DIM_TWO &&
                    ape.size(0) == cmp_ratio && ape.size(1) == wk.size(0),
                "ape must be FP32 with shape [cmp_ratio, wk.size(0)]");
    TORCH_CHECK(state_cache.scalar_type() == at::kFloat &&
                    state_cache.dim() == KEY_POOL_DIM_THREE &&
                    state_cache.size(2) == KEY_POOL_DIM_TWO * wk.size(0),
                "state_cache must be FP32 with shape [blocks, block_size, 2*D]");
    TORCH_CHECK(state_cache.stride(2) == 1 &&
                    state_cache.stride(1) == state_cache.size(2) &&
                    state_cache.stride(0) >= state_cache.size(1) * state_cache.size(2),
                "state_cache only supports padding between physical blocks");
    TORCH_CHECK(cache_block_table.scalar_type() == at::kInt &&
                    cache_block_table.dim() == KEY_POOL_DIM_TWO,
                "cache_block_table must be rank-2 INT32");
    TORCH_CHECK(start_pos.scalar_type() == at::kInt &&
                    start_pos.dim() == KEY_POOL_DIM_ONE &&
                    start_pos.size(0) == cache_block_table.size(0),
                "start_pos must be INT32 with shape [B]");

    TORCH_CHECK(norm_weight.has_value() == norm_bias.has_value(),
                "norm_weight and norm_bias must be provided as a pair");
    if (norm_weight.has_value()) {
        check_same_device(*norm_weight, "norm_weight");
        check_same_device(*norm_bias, "norm_bias");
        TORCH_CHECK(norm_weight->scalar_type() == at::kFloat &&
                        norm_bias->scalar_type() == at::kFloat,
                    "norm_weight and norm_bias must be FP32");
        TORCH_CHECK(norm_weight->dim() == KEY_POOL_DIM_ONE &&
                        norm_bias->dim() == KEY_POOL_DIM_ONE,
                    "norm_weight and norm_bias must be rank-1");
        TORCH_CHECK(norm_weight->size(0) == wk.size(0) &&
                        norm_bias->size(0) == wk.size(0),
                    "norm_weight and norm_bias size must match wk.size(0)");
    }
    TORCH_CHECK(norm_eps > 0.0, "norm_eps should be greater than 0");
    TORCH_CHECK(cos.has_value() == sin.has_value(),
                "cos and sin must be provided as a pair");
    TORCH_CHECK(!cos.has_value(),
                "KeyPool RoPE is not implemented in this stage");
    TORCH_CHECK(hidden_states_dim != KEY_POOL_DIM_TWO || cu_seqlens.has_value(),
                "cu_seqlens is required for TH layout");
    TORCH_CHECK(hidden_states_dim != KEY_POOL_DIM_THREE ||
                    !cu_seqlens.has_value(),
                "cu_seqlens must be absent for BSH layout");
    TORCH_CHECK(!seqused.has_value(),
                "seqused is reserved and must be None in this stage");
    at::Tensor pooled_key = ConstructKeyPoolOutputTensor(
        hidden_states, wk, state_cache, cache_block_table, cmp_ratio);

    auto state_cache_dim = state_cache.dim();
    TORCH_CHECK(state_cache_dim == KEY_POOL_DIM_THREE,
                "state_cache dim num[", state_cache_dim, "] should be 3");

    int64_t state_cache_stride_dim0 = state_cache.stride(0);

    if (cu_seqlens.has_value()) {
        check_same_device(*cu_seqlens, "cu_seqlens");
        TORCH_CHECK(cu_seqlens->scalar_type() == at::kInt,
                    "cu_seqlens must be INT32");
        TORCH_CHECK(cu_seqlens->dim() == KEY_POOL_DIM_ONE,
                    "cu_seqlens dim num[", cu_seqlens->dim(), "] should be 1");
        TORCH_CHECK(cu_seqlens->size(0) == cache_block_table.size(0) +
                                               KEY_POOL_DIM_ONE,
                    "cu_seqlens shape must be [B+1]");
    }
    EXEC_NPU_CMD(aclnnKeyPool, hidden_states, wk, gate_weight, ape, state_cache,
                 cache_block_table, start_pos, norm_weight, norm_bias, cos, sin,
                 cu_seqlens, seqused, cmp_ratio, norm_eps, rotary_mode,
                 state_cache_stride_dim0, pooled_key);

    return pooled_key;
}
}  // namespace vllm_ascend

#endif  // KEY_POOL_TORCH_ADPT_H
