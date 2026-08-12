#ifndef SPARSE_ATTENTION_SCORE_TORCH_ADPT_H
#define SPARSE_ATTENTION_SCORE_TORCH_ADPT_H

#include <ATen/ATen.h>
#include <torch/torch.h>
#include <acl/acl.h>

namespace vllm_ascend {

at::Tensor npu_sparse_attention_score(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &select_idx, const at::Tensor &block_table,
    int64_t num_key_value_heads, double scale_value, int64_t block_size,
    int64_t top_k, int64_t inner_precise,
    const c10::optional<at::Tensor> &select_num_idx,
    const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &q_dequant_scale,
    const c10::optional<at::Tensor> &k_dequant_scale,
    const c10::optional<at::Tensor> &v_dequant_scale
    )
{

    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
                                       "than 0, but shape[", i, "] is ", query.size(i));
    }

    at::Tensor output = at::empty(query.sizes(), query.options().dtype(query.dtype()));
    at::Tensor softmax_lse;

    EXEC_NPU_CMD(
        aclnnSparseAttentionScore,
        query,
        key,
        value,
        select_idx,
        block_table,
        select_num_idx,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        q_dequant_scale,
        k_dequant_scale,
        v_dequant_scale,
        num_key_value_heads,
        scale_value,
        block_size,
        top_k,
        inner_precise,
        output,
        softmax_lse);

    return output;
}
}
#endif
