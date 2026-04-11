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
#ifndef LIGHTING_INDEXER_QUANT_VLLM_TORCH_ADPT_H
#define LIGHTING_INDEXER_QUANT_VLLM_TORCH_ADPT_H
namespace vllm_ascend {

at::Tensor npu_lightning_indexer_quant(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &query_dequant_scale, const at::Tensor &key_dequant_scale,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table, int64_t query_quant_mode, int64_t key_quant_mode,
    c10::string_view layout_query, c10::string_view layout_key, int64_t sparse_count, int64_t sparse_mode)
{
    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);

    const int SIZE = 8;
    const int DIM_0 = 0;
    const int DIM_1 = 1;
    const int DIM_2 = 2;
    const int DIM_3 = 3;

    at::SmallVector<int64_t, SIZE> output_size;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
            "than 0, but shape[", i, "] is ", query.size(i));
    }
    for (size_t i = 0; i < key.sizes().size(); i++) {
        TORCH_CHECK(key.size(i) > 0, "All values within key's shape should be greater "
            "than 0, but shape[", i, "] is ", key.size(i));
    }
    TORCH_CHECK(sparse_count > 0, "sparse count should be greater than 0, but now is ", sparse_count);
    int64_t keyHeadNum = (key_layout_str == "TND")? key.size(DIM_1) : key.size(DIM_2);
    if (query_layout_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), keyHeadNum, sparse_count};
    } else {
        output_size = {query.size(DIM_0), keyHeadNum, sparse_count};
    }
    at::Tensor lightning_indexer_quant_output = at::empty(output_size, query.options().dtype(at::kInt));

    // convert str
    char *query_layout_ptr = const_cast<char *>(query_layout_str.c_str());
    char *key_layout_ptr = const_cast<char *>(key_layout_str.c_str());

    EXEC_NPU_CMD(aclnnLightningIndexerQuant, 
                    query,
                    key, 
                    weights, 
                    query_dequant_scale, 
                    key_dequant_scale, 
                    actual_seq_lengths_query, 
                    actual_seq_lengths_key,
                    block_table, 
                    query_quant_mode, 
                    key_quant_mode, 
                    query_layout_ptr, 
                    key_layout_ptr, 
                    sparse_count, 
                    sparse_mode,
                    lightning_indexer_quant_output
                );

    return lightning_indexer_quant_output;

}
}
#endif