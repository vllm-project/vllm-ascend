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
#ifndef LIGHTING_INDEXER_VLLM_TORCH_ADPT_H
#define LIGHTING_INDEXER_VLLM_TORCH_ADPT_H
namespace vllm_ascend {

at::Tensor npu_lightning_indexer(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table, c10::string_view layout_query,
    c10::string_view layout_key, int64_t sparse_count, int64_t sparse_mode)
{
    // npu tensor max size
    constexpr int32_t SIZE = 8;
    constexpr int32_t DIM_0 = 0;
    constexpr int32_t DIM_1 = 1;
    constexpr int32_t DIM_2 = 2;
    constexpr int32_t DIM_3 = 3;

    TORCH_CHECK(query.numel() > 0, "Query is empty.");
    TORCH_CHECK(key.numel() > 0, "Key is empty.");
    TORCH_CHECK(weights.numel() > 0, "Weights is empty.");
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
                                       "than 0, but shape[", i, "] is ", query.size(i));
    }
    TORCH_CHECK(sparse_count > 0, "sparse count should be greater than 0, but now is ", sparse_count);

    at::SmallVector<int64_t, SIZE> output_size;
    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);
    if (query_layout_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), key.size(DIM_2), sparse_count};
    } else {
        int n_dim_index = 0;
        n_dim_index = (key_layout_str == "TND") ? DIM_1 : DIM_2;
        output_size = {query.size(DIM_0), key.size(n_dim_index), sparse_count};
    }
    at::Tensor lightning_indexer_output = at::empty(output_size, query.options().dtype(at::kInt));
    // convert str
    char *query_layout_ptr = const_cast<char *>(query_layout_str.c_str());
    char *key_layout_ptr = const_cast<char *>(key_layout_str.c_str());
    EXEC_NPU_CMD(
        aclnnLightningIndexerVllm,
        query,
        key,
        weights,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
        query_layout_ptr,
        key_layout_ptr,
        sparse_count,
        sparse_mode,
        lightning_indexer_output);
    return lightning_indexer_output;
}
}
#endif