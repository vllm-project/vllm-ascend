/*
 * Licensed under the BSD 3-Clause License  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 */
#ifndef NGRAM_SPEC_DECODE_TORCH_ADPT_H
#define NGRAM_SPEC_DECODE_TORCH_ADPT_H

#include <torch/extension.h>
#include <torch_npu/csrc/framework/OpCommand.h>

namespace vllm_ascend {

// N-gram 投机解码算子
// 输入：
//   token_ids: [batch_size, max_seq_len], int32, 会被 in-place 修改
//   num_tokens_no_spec: [batch_size], int32
//   sampled_token_ids: [batch_size, max_new_tokens], int32
//   discard_request_mask: [batch_size], int32
//   vocab_size, min_n, max_n, k: 标量属性
// 输出：
//   token_ids (in-place 修改), next_token_ids, draft_token_ids, num_valid_draft_tokens
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_ngram_spec_decode(
    at::Tensor &token_ids,
    const at::Tensor &num_tokens_no_spec,
    const at::Tensor &sampled_token_ids,
    const at::Tensor &discard_request_mask,
    int64_t vocab_size,
    int64_t min_n,
    int64_t max_n,
    int64_t k)
{
    int64_t batch_size = token_ids.size(0);
    auto device = token_ids.device();

    // discard_request_mask 可能是 bool 类型，算子要求 int32，做类型转换
    at::Tensor discard_mask_int = discard_request_mask.dtype() == at::kBool
        ? discard_request_mask.to(at::kInt)
        : discard_request_mask;

    // 构造输出 tensor
    at::Tensor next_token_ids = at::empty({batch_size}, at::dtype(at::kInt).device(device));
    at::Tensor draft_token_ids = at::empty({batch_size, k}, at::dtype(at::kInt).device(device));
    at::Tensor num_valid_draft_tokens = at::empty({batch_size}, at::dtype(at::kInt).device(device));

    EXEC_NPU_CMD(aclnnNgramSpecDecode,
        token_ids, num_tokens_no_spec, sampled_token_ids, discard_mask_int,
        vocab_size, min_n, max_n, k,
        next_token_ids, draft_token_ids, num_valid_draft_tokens);

    return std::make_tuple(token_ids, next_token_ids, draft_token_ids, num_valid_draft_tokens);
}

}  // namespace vllm_ascend

#endif  // NGRAM_SPEC_DECODE_TORCH_ADPT_H
