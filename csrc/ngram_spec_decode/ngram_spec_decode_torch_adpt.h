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

// N-gram spec decode op
// inputs：
//   token_ids: [batch_size, max_seq_len], int32,
//   num_tokens_no_spec: [batch_size], int32
//   sampled_token_ids: [batch_size, max_new_tokens], int32
//   discard_request_mask: [batch_size], int32
//   vocab_size, min_n, max_n, k
// outputs:
//   token_ids (in-place change), next_token_ids, draft_token_ids, num_valid_draft_tokens
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

    at::Tensor discard_mask_int = discard_request_mask.dtype() == at::kBool
        ? discard_request_mask.to(at::kInt)
        : discard_request_mask;

    // Allocate outputs with a trailing over-write cushion. The kernel's
    // CopyOut path issues DataCopyPad GM writes whose burst length can
    // be smaller than the NPU's 32-byte MTE alignment; under that
    // alignment the underlying MTE3 burst can write past the apparent
    // tensor end on the last row. Tightly-sized allocations (the original
    // ``at::empty({batch_size}, ...)``) leave no room for that
    // alignment-driven over-write, surfacing as a multi-core MTE OOB on
    // device (CI signature: fixp_error0 = 0x30266b9 across cores).
    //
    // We therefore allocate ``batch_size + OVER_WRITE_MARGIN`` rows /
    // ``(batch_size + OVER_WRITE_MARGIN) * k`` elements and ``narrow``
    // back to the user-visible shape. The narrowed view shares storage
    // with the larger allocation, so any kernel-side alignment
    // over-write lands inside owned memory rather than off the end.
    constexpr int64_t OVER_WRITE_MARGIN = 8;  // 32 bytes / sizeof(int32) = 8 ints

    at::Tensor next_token_ids_storage = at::empty(
        {batch_size + OVER_WRITE_MARGIN},
        at::dtype(at::kInt).device(device));
    at::Tensor next_token_ids = next_token_ids_storage.narrow(0, 0, batch_size);

    at::Tensor draft_token_ids_storage = at::empty(
        {batch_size + OVER_WRITE_MARGIN, k},
        at::dtype(at::kInt).device(device));
    at::Tensor draft_token_ids = draft_token_ids_storage.narrow(0, 0, batch_size);

    at::Tensor num_valid_draft_tokens_storage = at::empty(
        {batch_size + OVER_WRITE_MARGIN},
        at::dtype(at::kInt).device(device));
    at::Tensor num_valid_draft_tokens =
        num_valid_draft_tokens_storage.narrow(0, 0, batch_size);

    EXEC_NPU_CMD(aclnnNgramSpecDecode,
        token_ids, num_tokens_no_spec, sampled_token_ids, discard_mask_int,
        vocab_size, min_n, max_n, k,
        next_token_ids, draft_token_ids, num_valid_draft_tokens);

    return std::make_tuple(token_ids, next_token_ids, draft_token_ids, num_valid_draft_tokens);
}

}  // namespace vllm_ascend

#endif  // NGRAM_SPEC_DECODE_TORCH_ADPT_H
