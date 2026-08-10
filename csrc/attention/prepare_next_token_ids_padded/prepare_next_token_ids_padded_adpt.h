#ifndef PREPARE_NEXT_TOKEN_IDS_PADDED_TORCH_ADPT_H
#define PREPARE_NEXT_TOKEN_IDS_PADDED_TORCH_ADPT_H

#include <cstdint>
#include <limits>
#include <tuple>

#include <torch/extension.h>
#include "aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend {

inline std::tuple<at::Tensor, at::Tensor>
prepare_next_token_ids_padded(
    const at::Tensor& sampled_token_ids,
    const at::Tensor& discard_request_mask,
    const at::Tensor& backup_next_token_ids,
    int64_t vocab_size)
{
    // Shape checks.
    TORCH_CHECK(
        sampled_token_ids.dim() == 2,
        "sampled_token_ids must be a 2D tensor, but got ",
        sampled_token_ids.dim(),
        " dimensions.");

    TORCH_CHECK(
        discard_request_mask.dim() == 1,
        "discard_request_mask must be a 1D tensor, but got ",
        discard_request_mask.dim(),
        " dimensions.");

    TORCH_CHECK(
        backup_next_token_ids.dim() == 1,
        "backup_next_token_ids must be a 1D tensor, but got ",
        backup_next_token_ids.dim(),
        " dimensions.");

    const int64_t batch_size = sampled_token_ids.size(0);
        

    constexpr int64_t INT32_ELEMENTS_PER_BLOCK =
        32 / static_cast<int64_t>(sizeof(int32_t));

    at::Tensor next_token_ids_storage = at::empty(
        {batch_size + INT32_ELEMENTS_PER_BLOCK},
        sampled_token_ids.options().dtype(at::ScalarType::Int));

    at::Tensor valid_sampled_tokens_count_storage = at::empty(
        {batch_size + INT32_ELEMENTS_PER_BLOCK},
        sampled_token_ids.options().dtype(at::ScalarType::Int));

    at::Tensor next_token_ids =
        next_token_ids_storage.narrow(0, 0, batch_size);

    at::Tensor valid_sampled_tokens_count =
        valid_sampled_tokens_count_storage.narrow(0, 0, batch_size);

    EXEC_NPU_CMD(
        aclnnPrepareNextTokenIdsPadded,
        sampled_token_ids,
        discard_request_mask,
        backup_next_token_ids,
        vocab_size,
        next_token_ids,
        valid_sampled_tokens_count);

    return std::make_tuple(
        next_token_ids,
        valid_sampled_tokens_count);
}

}

#endif 