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

#ifndef BUILD_DSPARK_SWA_INDICES_TORCH_ADPT_H
#define BUILD_DSPARK_SWA_INDICES_TORCH_ADPT_H

namespace vllm_ascend {

// Fused CANN kernel — replaces the PyTorch op-chain path with a single
// kernel launch. Only valid when every request has the same
// num_speculative_tokens draft tokens (DSpark uniform-query case); the Python
// dispatch checks this and falls back to the PyTorch path otherwise. The
// TORCH_CHECKs here are a defensive second guard.
at::Tensor build_dspark_swa_indices(
    const at::Tensor &block_table,
    const at::Tensor &query_start_loc,
    const at::Tensor &seq_lens,
    int64_t num_speculative_tokens,
    int64_t window_size,
    int64_t block_size,
    int64_t index_width,
    int64_t num_decode_tokens)
{
    TORCH_CHECK(num_speculative_tokens > 0, "num_speculative_tokens should be > 0, but got ", num_speculative_tokens);
    TORCH_CHECK(block_size > 0, "block_size should be > 0, but got ", block_size);
    TORCH_CHECK(index_width > 0, "index_width should be > 0, but got ", index_width);
    TORCH_CHECK(num_decode_tokens > 0, "num_decode_tokens should be > 0, but got ", num_decode_tokens);
    TORCH_CHECK(block_table.dim() == 2, "block_table should be 2D, but got dim ", block_table.dim());
    TORCH_CHECK(query_start_loc.dim() == 1 && query_start_loc.size(0) >= 2,
                "query_start_loc should be 1D with >= 2 elements");
    TORCH_CHECK(block_table.size(0) >= query_start_loc.size(0) - 1,
                "block_table rows < num_reqs");
    TORCH_CHECK(block_table.scalar_type() == at::kInt, "block_table must be int32");
    TORCH_CHECK(query_start_loc.scalar_type() == at::kInt, "query_start_loc must be int32");
    TORCH_CHECK(seq_lens.scalar_type() == at::kInt, "seq_lens must be int32");

    int64_t num_reqs = query_start_loc.size(0) - 1;
    TORCH_CHECK(num_decode_tokens == num_reqs * num_speculative_tokens,
                "fused kernel requires uniform query_lens: num_decode_tokens (", num_decode_tokens,
                ") must equal num_reqs (", num_reqs, ") * num_speculative_tokens (",
                num_speculative_tokens, ")");

    at::Tensor output = at::empty({num_decode_tokens, 1, index_width},
                                  block_table.options().dtype(at::kInt));

    EXEC_NPU_CMD(aclnnBuildDsparkSwaIndices,
                 block_table, query_start_loc, seq_lens,
                 num_speculative_tokens, window_size, block_size, output);
    return output;
}

}  // namespace vllm_ascend

#endif
