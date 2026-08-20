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

#ifndef SPARSE_ATTENTION_SCORE_PREFILL_V1_TORCH_ADPT_H
#define SPARSE_ATTENTION_SCORE_PREFILL_V1_TORCH_ADPT_H

#include <ATen/ATen.h>
#include <acl/acl.h>
#include <torch/torch.h>

namespace vllm_ascend {

at::Tensor npu_sparse_attention_score_prefill_v1(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &block_table, const at::Tensor &k2q_row_ptr,
    const at::Tensor &k2q_q_indices, const at::Tensor &k2q_slot_indices,
    int64_t num_key_value_heads, double scale_value, int64_t block_size,
    int64_t top_k, int64_t inner_precise,
    const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv) {
  for (size_t i = 0; i < query.sizes().size(); i++) {
    TORCH_CHECK(query.size(i) > 0,
                "All values within query's shape should be greater than 0, "
                "but shape[",
                i, "] is ", query.size(i));
  }

  at::Tensor output =
      at::empty(query.sizes(), query.options().dtype(query.dtype()));
  at::Tensor softmax_lse =
      at::empty({0}, query.options().dtype(at::kFloat));
  bool softmax_lse_flag = false;

  EXEC_NPU_CMD(aclnnSparseAttentionScorePrefill, query, key, value,
               block_table, k2q_row_ptr, k2q_q_indices, k2q_slot_indices,
               actual_seq_lengths, actual_seq_lengths_kv,
               num_key_value_heads, scale_value, block_size, top_k,
               inner_precise, softmax_lse_flag, output, softmax_lse);

  return output;
}

}  // namespace vllm_ascend

#endif
