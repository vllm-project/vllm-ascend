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
#ifndef FUSED_INFER_ATTENTION_SCORE_TORCH_ADPT_H
#define FUSED_INFER_ATTENTION_SCORE_TORCH_ADPT_H

#include <cstdint>
#include <string>
#include <tuple>

#include "op_api/aclnn_fused_infer_attention_score_v4.h"

namespace vllm_ascend {
namespace fia {

inline std::string normalize_layout(c10::string_view input_layout)
{
    const std::string layout(input_layout);
    TORCH_CHECK(layout == "TND" || layout == "BNSD_NBSD" || layout == "BSND_NBSD",
                "npu_fused_infer_attention_score_v2 custom binding only supports TND, "
                "BNSD_NBSD and BSND_NBSD, but got ", layout);
    return layout;
}

inline int64_t value_dim(const at::Tensor &query, const at::Tensor &value,
                         const std::string &layout, int64_t num_key_value_heads)
{
    if (layout == "TND") {
        TORCH_CHECK(query.dim() == 3, "TND query must be 3-dimensional");
        if (value.dim() == 4) {
            // Paged KV: [num_blocks, num_kv_heads, block_size, value_dim].
            return value.size(3);
        }
        TORCH_CHECK(value.dim() == 3, "TND non-paged value must be 3-dimensional");
        return value.size(2) / num_key_value_heads;
    }

    TORCH_CHECK(query.dim() == 4, layout, " query must be 4-dimensional");
    if (value.dim() == 4) {
        // Paged KV: [num_blocks, num_kv_heads, block_size, value_dim].
        return value.size(3);
    }
    TORCH_CHECK(value.dim() == 4, layout, " non-paged value must be 4-dimensional");
    return value.size(3);
}

inline std::tuple<at::Tensor, at::Tensor> construct_output(
    const at::Tensor &query, const at::Tensor &value, const std::string &layout,
    int64_t num_query_heads, int64_t num_key_value_heads, bool return_softmax_lse)
{
    TORCH_CHECK(num_query_heads > 0, "num_query_heads must be positive");
    if (num_key_value_heads == 0) {
        num_key_value_heads = num_query_heads;
    }
    TORCH_CHECK(num_key_value_heads > 0, "num_key_value_heads must be positive");
    TORCH_CHECK(num_query_heads % num_key_value_heads == 0,
                "num_query_heads must be divisible by num_key_value_heads");

    const int64_t value_d = value_dim(query, value, layout, num_key_value_heads);
    at::SmallVector<int64_t, 4> output_shape;
    at::SmallVector<int64_t, 4> lse_shape;
    if (layout == "TND") {
        output_shape = {query.size(0), query.size(1), value_d};
        lse_shape = {query.size(0), query.size(1), 1};
    } else if (layout == "BNSD_NBSD") {
        output_shape = {query.size(1), query.size(0), query.size(2), value_d};
        lse_shape = {query.size(0), query.size(1), query.size(2), 1};
    } else {
        output_shape = {query.size(2), query.size(0), query.size(1), value_d};
        lse_shape = {query.size(0), query.size(2), query.size(1), 1};
    }

    at::Tensor output = at_npu::native::OpPreparation::apply_tensor_without_format(
        output_shape, query.options().dtype(query.dtype()));
    at::Tensor softmax_lse;
    if (return_softmax_lse) {
        softmax_lse = at_npu::native::OpPreparation::apply_tensor_without_format(
            lse_shape, query.options().dtype(at::kFloat));
    } else {
        softmax_lse = at_npu::native::OpPreparation::apply_tensor_without_format(
            {0}, query.options().dtype(at::kFloat));
    }
    return {output, softmax_lse};
}

}  // namespace fia

inline std::tuple<at::Tensor, at::Tensor> npu_fused_infer_attention_score_v2(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &learnable_sink,
    c10::optional<at::IntArrayRef> actual_seq_qlen,
    c10::optional<at::IntArrayRef> actual_seq_kvlen,
    const c10::optional<at::Tensor> &block_table,
    int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale,
    int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout,
    int64_t sparse_mode, int64_t block_size, int64_t query_quant_mode,
    int64_t key_quant_mode, int64_t value_quant_mode, int64_t inner_precise,
    bool return_softmax_lse)
{
    (void)query_rope;
    (void)key_rope;
    (void)pse_shift;
    (void)atten_mask;
    const std::string layout = fia::normalize_layout(input_layout);
    TORCH_CHECK(query.device() == key.device() && query.device() == value.device(),
                "query, key and value must be on the same device");
    TORCH_CHECK(block_table.has_value() && block_table.value().defined(),
                "the custom FIA binding currently requires block_table");
    TORCH_CHECK(block_size > 0, "block_size must be positive for paged FIA");

    auto outputs = fia::construct_output(query, value, layout, num_query_heads,
                                         num_key_value_heads, return_softmax_lse);
    at::Tensor attention_output = std::get<0>(outputs);
    at::Tensor softmax_lse = std::get<1>(outputs);

    // aclnnFusedInferAttentionScoreV4 takes TensorList for PA key/value.  The
    // one-element lists retain the original Tensor descriptors, including
    // view strides and storage offset; no contiguous conversion is allowed in
    // this adapter.
    at::TensorList key_list(&key, 1);
    at::TensorList value_list(&value, 1);
    std::string layout_storage = layout;
    char *layout_ptr = const_cast<char *>(layout_storage.c_str());

    c10::optional<at::Tensor> empty_tensor;
    c10::optional<at::IntArrayRef> empty_array;
    int64_t antiquant_mode = 0;
    int64_t key_antiquant_mode = 0;
    int64_t value_antiquant_mode = 0;

    TensorWrapper query_wrapper = make_wrapper(query);
    TensorListWrapper key_list_wrapper =
        {key_list, make_wrapper(key).dtype};
    TensorListWrapper value_list_wrapper =
        {value_list, make_wrapper(value).dtype};
    at::Tensor null_tensor;
    const at::Tensor &query_rope_tensor =
        query_rope.has_value() ? query_rope.value() : null_tensor;
    const at::Tensor &key_rope_tensor =
        key_rope.has_value() ? key_rope.value() : null_tensor;
    const at::Tensor &learnable_sink_tensor =
        learnable_sink.has_value() ? learnable_sink.value() : null_tensor;
    TensorWrapper query_rope_wrapper = make_wrapper(query_rope_tensor);
    TensorWrapper key_rope_wrapper = make_wrapper(key_rope_tensor);
    TensorWrapper learnable_sink_wrapper = make_wrapper(learnable_sink_tensor);
    TensorWrapper dequant_scale_query_wrapper = make_wrapper(null_tensor);
    TensorWrapper dequant_scale_key_wrapper = make_wrapper(null_tensor);
    TensorWrapper dequant_scale_value_wrapper = make_wrapper(null_tensor);
    TensorWrapper attention_output_wrapper = make_wrapper(attention_output);

    EXEC_NPU_NO_FORMAT_CHECK_CMD(
        aclnnFusedInferAttentionScoreV4,
        query_wrapper, key_list_wrapper, value_list_wrapper, pse_shift,
        atten_mask, actual_seq_qlen,
        actual_seq_kvlen, empty_tensor, empty_tensor, empty_tensor, empty_tensor,
        empty_tensor, empty_tensor, empty_tensor, block_table, empty_tensor,
        empty_tensor, dequant_scale_key_wrapper, empty_tensor,
        dequant_scale_value_wrapper, empty_tensor, empty_tensor, empty_tensor,
        empty_array, query_rope_wrapper, key_rope_wrapper, empty_tensor,
        dequant_scale_query_wrapper,
        learnable_sink_wrapper, num_query_heads,
        softmax_scale, pre_tokens, next_tokens, layout_ptr, num_key_value_heads,
        sparse_mode, inner_precise, block_size, antiquant_mode, return_softmax_lse,
        key_antiquant_mode, value_antiquant_mode,
        query_quant_mode, attention_output_wrapper, softmax_lse);

    return {attention_output, softmax_lse};
}

}  // namespace vllm_ascend

#endif  // FUSED_INFER_ATTENTION_SCORE_TORCH_ADPT_H
