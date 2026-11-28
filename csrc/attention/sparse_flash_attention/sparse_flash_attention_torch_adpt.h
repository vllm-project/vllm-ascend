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
#ifndef SPARSE_FLASH_ATTENTION_TORCH_ADPT_H
#define SPARSE_FLASH_ATTENTION_TORCH_ADPT_H

namespace vllm_ascend {

namespace {

// Describe precisely the dense span occupied by this layer's pages. A layer can
// start at a nonzero offset in a larger allocation shared with other caches.
struct SfaPairedPages {
    const at::Tensor &tensor;
};

aclTensor *ConvertType(const SfaPairedPages &pages)
{
    static const auto createTensor = GET_OP_API_FUNC(aclCreateTensor);
    TORCH_CHECK(createTensor, "aclCreateTensor is unavailable");
    const auto &tensor = pages.tensor;
    const auto shape = tensor.sizes();
    auto *result = createTensor(shape.data(), shape.size(), ACL_BF16, tensor.strides().data(),
                                0, ACL_FORMAT_ND, shape.data(), shape.size(), tensor.data_ptr());
    TORCH_CHECK(result, "Could not create the PA_KR_BSND page descriptor");
    return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_sparse_flash_attention_output_tensor(
    const at::Tensor &query, const at::Tensor &key,
    const std::string &layout_query_str, const std::string &layout_kv_str, bool return_softmax_lse)
{
    constexpr int64_t SIZE = 8;
    constexpr int64_t DIM_0 = 0;
    constexpr int64_t DIM_1 = 1;
    constexpr int64_t DIM_2 = 2;
    constexpr int64_t DIM_3 = 3;
    constexpr int64_t DIM_4 = 4;

    TORCH_CHECK(layout_query_str == "BSND" || layout_query_str == "TND",
                "The layout of query only support BSND and TND, but got ",
                layout_query_str);
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0,
                    "All values within query's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", query.size(i));
    }

    at::SmallVector<int64_t, SIZE> output_size;
    if (layout_query_str == "TND") {
        TORCH_CHECK(query.dim() == DIM_3,
                    "When the layout of query is TND, the query dimension must be 3, but got ",
                    query.dim());
        output_size = {query.size(DIM_0), query.size(DIM_1),
                       query.size(DIM_2)};
    } else {
        TORCH_CHECK(query.dim() == DIM_4,
                    "When the layout of query is BSND, the query dimension must be 4, but got ",
                    query.dim());
        output_size = {query.size(DIM_0), query.size(DIM_1),
                       query.size(DIM_2), query.size(DIM_3)};
    }

    at::Tensor attention_output =
        at::empty(output_size, query.options().dtype(query.dtype()));
    at::SmallVector<int64_t, SIZE> softmax_size;
    if (return_softmax_lse) {
        if (query.dim() == DIM_3) {
            const bool paged = layout_kv_str == "PA_BSND" || layout_kv_str == "PA_KR_BSND";
            const auto kv_head_num = key.size(paged ? DIM_2 : DIM_1);
            softmax_size = {
                kv_head_num,
                query.size(DIM_0),
                query.size(DIM_1) / kv_head_num,
            };
        } else {
            softmax_size = {
                query.size(DIM_0),
                key.size(DIM_2),
                query.size(DIM_1),
                query.size(DIM_2) / key.size(DIM_2),
            };
        }
    } else {
        softmax_size = {0};
    }

    at::Tensor softmax_max =
        at::empty(softmax_size, query.options().dtype(at::kFloat));
    at::Tensor softmax_sum =
        at::empty(softmax_size, query.options().dtype(at::kFloat));
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(
        attention_output, softmax_max, softmax_sum);
}

}  // namespace

bool is_sfa_paired_page_available()
{
    static const bool available = [] {
        auto workspace = GetOpApiFuncAddr("aclnnVllmSfaPairedPageGetWorkspaceSize");
        auto run = GetOpApiFuncAddr("aclnnVllmSfaPairedPage");
        Dl_info workspace_info{}, run_info{};
        return workspace && run && dladdr(workspace, &workspace_info) && dladdr(run, &run_info) &&
               workspace_info.dli_fbase == run_info.dli_fbase;
    }();
    return available;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_sparse_flash_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices, double scale_value,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope, int64_t sparse_block_size,
    c10::string_view layout_query, c10::string_view layout_kv,
    int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens,
    int64_t attention_mode, bool return_softmax_lse)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.");
    TORCH_CHECK(value.numel() > 0, "Tensor value is empty.");
    TORCH_CHECK(sparse_indices.numel() > 0, "Tensor sparse_indices is empty.");

    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_kv);
    const bool paired_pages = layout_kv_str == "PA_KR_BSND";
    if (paired_pages) {
        TORCH_CHECK(layout_query_str == "TND" && key.dim() == 4 && key.size(2) == 1 &&
                        key.size(3) == 576 && key.scalar_type() == at::kBFloat16,
                    "PA_KR_BSND requires TND queries and BF16 pages shaped [P,B,1,576]");
        TORCH_CHECK(key.is_same(value) && key_rope.has_value() && key.is_same(*key_rope),
                    "PA_KR_BSND requires the same page tensor for key, value and key_rope");
        TORCH_CHECK(key.device() == query.device() && key.is_contiguous() && IsOpInputBaseFormat(key) &&
                        reinterpret_cast<uintptr_t>(key.data_ptr()) % 32 == 0,
                    "PA_KR_BSND requires contiguous base-format pages on the query device with a 32-byte-aligned start");
        TORCH_CHECK(is_sfa_paired_page_available(), "The installed OPP lacks aclnnVllmSfaPairedPage");
    }

    auto sparse_flash_attention_output =
        construct_sparse_flash_attention_output_tensor(
            query, key, layout_query_str, layout_kv_str, return_softmax_lse);
    at::Tensor attention_output = std::get<0>(sparse_flash_attention_output);
    at::Tensor softmax_max = std::get<1>(sparse_flash_attention_output);
    at::Tensor softmax_sum = std::get<2>(sparse_flash_attention_output);

    if (paired_pages) {
        const c10_npu::NPUGuard guard(query.device());
        SfaPairedPages pages{key};
        EXEC_NPU_CMD(aclnnVllmSfaPairedPage, query, pages, sparse_indices, block_table,
                     actual_seq_lengths_query, actual_seq_lengths_kv, query_rope, scale_value,
                     sparse_block_size, sparse_mode, pre_tokens, next_tokens, attention_mode,
                     return_softmax_lse, attention_output, softmax_max, softmax_sum);
        return sparse_flash_attention_output;
    }

    // convert str
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    EXEC_NPU_CMD(
        aclnnSparseFlashAttention,
        query,
        key,
        value,
        sparse_indices,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        scale_value,
        sparse_block_size,
        layout_query_ptr,
        layout_kv_ptr,
        sparse_mode,
        pre_tokens,
        next_tokens,
        attention_mode,
        return_softmax_lse,
        attention_output,
        softmax_max,
        softmax_sum);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(
        attention_output, softmax_max, softmax_sum);
}
}  // namespace vllm_ascend

#endif  // SPARSE_FLASH_ATTENTION_TORCH_ADPT_H
