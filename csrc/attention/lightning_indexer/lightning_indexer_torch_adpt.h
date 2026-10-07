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
#ifndef LIGHTNING_INDEXER_TORCH_ADPT_H
#define LIGHTNING_INDEXER_TORCH_ADPT_H

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> construct_lightning_indexer_output_tensor(
    const at::Tensor& query, const at::Tensor& key, int64_t sparse_count,
    const std::string& query_layout_str, const std::string& key_layout_str,
    bool return_value)
{
    constexpr int64_t SIZE = 8;
    constexpr int64_t DIM_0 = 0;
    constexpr int64_t DIM_1 = 1;
    constexpr int64_t DIM_2 = 2;

    at::SmallVector<int64_t, SIZE> output_size;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0,
                    "All values within query's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", query.size(i));
    }
    for (size_t i = 0; i < key.sizes().size(); i++) {
        TORCH_CHECK(key.size(i) > 0,
                    "All values within key's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", key.size(i));
    }
    TORCH_CHECK(sparse_count > 0,
                "sparse count should be greater than 0, but now is ",
                sparse_count);

    if (query_layout_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1),
                       key.size(DIM_2), sparse_count};
    } else {
        int64_t n_dim_index = (key_layout_str == "TND") ? DIM_1 : DIM_2;
        output_size = {query.size(DIM_0), key.size(n_dim_index),
                       sparse_count};
    }

    at::Tensor sparse_indices_out =
        at::empty(output_size, query.options().dtype(at::kInt));
    at::Tensor sparse_values_out;
    if (return_value) {
        sparse_values_out =
            at::empty(output_size, query.options().dtype(query.dtype()));
    } else {
        sparse_values_out = at::empty({0}, query.options().dtype(query.dtype()));
    }

    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out,
                                              sparse_values_out);
}

std::tuple<at::Tensor, at::Tensor> npu_lightning_indexer(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& weights,
    const c10::optional<at::Tensor>& actual_seq_lengths_query,
    const c10::optional<at::Tensor>& actual_seq_lengths_key,
    const c10::optional<at::Tensor>& block_table, c10::string_view layout_query,
    c10::string_view layout_key, int64_t sparse_count, int64_t sparse_mode,
    int64_t pre_tokens, int64_t next_tokens, bool return_value)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.");
    TORCH_CHECK(weights.numel() > 0, "Tensor weights is empty.");

    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);

    auto lightning_indexer_output = construct_lightning_indexer_output_tensor(
        query, key, sparse_count, query_layout_str, key_layout_str,
        return_value);
    at::Tensor sparse_indices_out = std::get<0>(lightning_indexer_output);
    at::Tensor sparse_values_out = std::get<1>(lightning_indexer_output);

    char* query_layout_ptr = const_cast<char*>(query_layout_str.c_str());
    char* key_layout_ptr = const_cast<char*>(key_layout_str.c_str());

    EXEC_NPU_CMD(aclnnLightningIndexer, query, key, weights,
                 actual_seq_lengths_query, actual_seq_lengths_key, block_table,
                 query_layout_ptr, key_layout_ptr, sparse_count, sparse_mode,
                 pre_tokens, next_tokens, return_value, sparse_indices_out,
                 sparse_values_out);

    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out,
                                              sparse_values_out);
}
}  // namespace vllm_ascend

#endif  // LIGHTNING_INDEXER_TORCH_ADPT_H
