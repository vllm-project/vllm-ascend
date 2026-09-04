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
#ifndef FUSED_INFER_ATTENTION_SCORE_META_H
#define FUSED_INFER_ATTENTION_SCORE_META_H

#include <cstdint>
#include <string>
#include <tuple>

#include <torch/extension.h>

namespace vllm_ascend {
namespace meta {

inline std::tuple<at::Tensor, at::Tensor> npu_fused_infer_attention_score_v2_meta(
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
    (void)key;
    (void)query_rope;
    (void)key_rope;
    (void)pse_shift;
    (void)atten_mask;
    (void)learnable_sink;
    (void)actual_seq_qlen;
    (void)actual_seq_kvlen;
    (void)block_table;
    (void)softmax_scale;
    (void)pre_tokens;
    (void)next_tokens;
    (void)sparse_mode;
    (void)block_size;
    (void)query_quant_mode;
    (void)key_quant_mode;
    (void)value_quant_mode;
    (void)inner_precise;

    const std::string layout(input_layout);
    TORCH_CHECK(layout == "TND" || layout == "BNSD_NBSD" || layout == "BSND_NBSD",
                "npu_fused_infer_attention_score_v2 custom meta only supports TND, "
                "BNSD_NBSD and BSND_NBSD, but got ", layout);
    if (num_key_value_heads == 0) {
        num_key_value_heads = num_query_heads;
    }
    TORCH_CHECK(num_query_heads > 0 && num_key_value_heads > 0,
                "FIA head counts must be positive");
    TORCH_CHECK(num_query_heads % num_key_value_heads == 0,
                "num_query_heads must be divisible by num_key_value_heads");

    c10::SymInt value_d;
    if (value.dim() == 4) {
        value_d = value.sym_size(3);
    } else if (layout == "TND") {
        value_d = value.sym_size(2) / c10::SymInt(num_key_value_heads);
    } else {
        value_d = value.sym_size(3);
    }

    c10::SymDimVector output_shape;
    c10::SymDimVector lse_shape;
    if (layout == "TND") {
        output_shape = {query.sym_size(0), query.sym_size(1), value_d};
        lse_shape = {query.sym_size(0), query.sym_size(1), c10::SymInt(1)};
    } else if (layout == "BNSD_NBSD") {
        output_shape = {query.sym_size(1), query.sym_size(0), query.sym_size(2), value_d};
        lse_shape = {query.sym_size(0), query.sym_size(1), query.sym_size(2), c10::SymInt(1)};
    } else {
        output_shape = {query.sym_size(2), query.sym_size(0), query.sym_size(1), value_d};
        lse_shape = {query.sym_size(0), query.sym_size(2), query.sym_size(1), c10::SymInt(1)};
    }

    at::Tensor output = at::empty_symint(output_shape, query.options().dtype(query.dtype()));
    at::Tensor softmax_lse = return_softmax_lse
        ? at::empty_symint(lse_shape, query.options().dtype(at::kFloat))
        : at::empty_symint({c10::SymInt(0)}, query.options().dtype(at::kFloat));
    return {output, softmax_lse};
}

}  // namespace meta
}  // namespace vllm_ascend

#endif  // FUSED_INFER_ATTENTION_SCORE_META_H
