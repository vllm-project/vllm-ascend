#ifndef SPARSE_ATTENTION_SCORE_PREFILL_TORCH_ADPT_H
#define SPARSE_ATTENTION_SCORE_PREFILL_TORCH_ADPT_H

#include <ATen/ATen.h>
#include <torch/torch.h>
#include <acl/acl.h>

namespace vllm_ascend {

at::Tensor npu_sparse_attention_score_prefill(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &block_table,
    const at::Tensor &k2q_row_ptr,
    const at::Tensor &k2q_q_indices,
    const at::Tensor &k2q_slot_indices,
    int64_t num_key_value_heads, double scale_value, int64_t block_size,
    int64_t top_k, int64_t inner_precise,
    const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv
    )
{

    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
                                       "than 0, but shape[", i, "] is ", query.size(i));
    }

    at::Tensor output = at::empty(query.sizes(), query.options().dtype(query.dtype()));

    EXEC_NPU_CMD(
        aclnnSparseAttentionScorePrefill,
        query,
        key,
        value,
        block_table,
        k2q_row_ptr,
        k2q_q_indices,
        k2q_slot_indices,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        num_key_value_heads,
        scale_value,
        block_size,
        top_k,
        inner_precise,
        output
    );

    return output;
}
}
#endif
