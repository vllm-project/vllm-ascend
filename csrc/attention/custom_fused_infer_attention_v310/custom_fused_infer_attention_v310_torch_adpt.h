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
#ifndef CUSTOM_FUSED_INFER_ATTENTION_V310_TORCH_ADPT_H
#define CUSTOM_FUSED_INFER_ATTENTION_V310_TORCH_ADPT_H
namespace vllm_ascend {

at::Tensor npu_custom_fused_infer_attention_v310(
    const at::Tensor &query, at::TensorList key, at::TensorList value,
    const c10::optional<at::Tensor> &attn_mask,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_lengths_q,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &block_table,
    int64_t num_heads, double scale_value, c10::string_view input_layout,
    int64_t num_key_value_heads, int64_t block_size, int64_t inner_precise)
{
    at::Tensor output = at::empty_symint(query.sym_sizes(), query.options());

    auto actSeqLenQueryMiddle = actual_seq_lengths_q.value_or(at::ArrayRef<c10::SymInt>{});
    auto actSeqLenQuery = c10::asIntArrayRefUnchecked(actSeqLenQueryMiddle);

    auto actSeqLenKeyMiddle = actual_seq_lengths_kv.value_or(at::ArrayRef<c10::SymInt>{});
    auto actSeqLenKey = c10::asIntArrayRefUnchecked(actSeqLenKeyMiddle);

    std::string input_layout_str = std::string(input_layout);
    const char *input_layout_char = input_layout_str.c_str();
    // dispatch hostAPI
    EXEC_NPU_CMD(aclnnCustomFusedInferAttentionV310, query, key, value, attn_mask,
                 actSeqLenQuery, actSeqLenKey, block_table, num_heads, scale_value, input_layout_char,
                 num_key_value_heads, block_size, inner_precise, output);
    return output;
}

}
#endif
