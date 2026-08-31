/**
 * Torch binding for the AICPU metadata operator used by FIA v2 sink.
 * Adapted from the omni-ops ai_infra_fused_infer_attention_sink_metadata
 * binding. Attributes are compile-time constants; actual_seq_lengths and
 * actual_seq_lengths_kv are device INT64 [B] tensors. The caller supplies the
 * AIC and AIV core counts.
 */
#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace custom {
using namespace vllm_ascend::fia_v2_sink_opapi;
const int SIZE = 1024;
using namespace at_npu::native;

// 为NPU设备实现前向接口
at::Tensor npu_fused_infer_attention_score_v2_sink_metadata_npu(
    int64_t num_heads_q, int64_t num_heads_kv, int64_t heads_dim_qk, int64_t heads_dim_v,
    const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    int64_t batch_size, int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens,
    c10::string_view input_layout, c10::string_view input_layout_kv,
    int64_t sink_num, int64_t k_sink_num, int64_t rope_head_dim, int64_t block_size,
    int64_t aic_core_num, int64_t aiv_core_num, bool batch_invariant)
{
    std::string input_layout_str = std::string(input_layout);
    std::string input_layout_kv_str = std::string(input_layout_kv);

    // Match the operator ABI and Python-side graph buffer: int32[1024].
    std::vector<c10::SymInt> sym_shape = {c10::SymInt(SIZE)};
    auto options = at::TensorOptions().dtype(at::ScalarType::Int).device(at::kPrivateUse1);
    if (actual_seq_lengths.has_value() && actual_seq_lengths.value().defined()) {
        options = options.device(actual_seq_lengths.value().device());
    }
    at::Tensor meta_data = at::zeros_symint(sym_shape, options);

    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());
    char *input_layout_kv_ptr = const_cast<char *>(input_layout_kv_str.c_str());
    if (batch_invariant) {
        EXEC_NPU_CMD_V1(aclnnFusedInferAttentionScoreV2SinkMetadataV2,
                        actual_seq_lengths,
                        actual_seq_lengths_kv,
                        num_heads_q,
                        num_heads_kv,
                        heads_dim_qk,
                        heads_dim_v,
                        batch_size,
                        sparse_mode,
                        pre_tokens,
                        next_tokens,
                        input_layout_ptr,
                        input_layout_kv_ptr,
                        sink_num,
                        k_sink_num,
                        batch_invariant,
                        rope_head_dim,
                        block_size,
                        aic_core_num,
                        aiv_core_num,
                        meta_data);
    } else {
        EXEC_NPU_CMD_V1(aclnnFusedInferAttentionScoreV2SinkMetadata,
                        actual_seq_lengths,
                        actual_seq_lengths_kv,
                        num_heads_q,
                        num_heads_kv,
                        heads_dim_qk,
                        heads_dim_v,
                        batch_size,
                        sparse_mode,
                        pre_tokens,
                        next_tokens,
                        input_layout_ptr,
                        input_layout_kv_ptr,
                        sink_num,
                        k_sink_num,
                        rope_head_dim,
                        block_size,
                        aic_core_num,
                        aiv_core_num,
                        meta_data);
    }

    return meta_data;
}

// 为META设备实现前向接口
at::Tensor npu_fused_infer_attention_score_v2_sink_metadata_meta(
    int64_t num_heads_q, int64_t num_heads_kv, int64_t heads_dim_qk, int64_t heads_dim_v,
    const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    int64_t batch_size, int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens,
    c10::string_view input_layout, c10::string_view input_layout_kv,
    int64_t sink_num, int64_t k_sink_num, int64_t rope_head_dim, int64_t block_size,
    int64_t aic_core_num, int64_t aiv_core_num, bool batch_invariant)
{
    std::vector<c10::SymInt> sym_shape = {c10::SymInt(SIZE)};
    auto options = at::TensorOptions().dtype(at::ScalarType::Int).device(at::kPrivateUse1);
    if (actual_seq_lengths.has_value() && actual_seq_lengths.value().defined()) {
        options = options.device(actual_seq_lengths.value().device());
    }
    return at::zeros_symint(sym_shape, options);
}

} // namespace custom

TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, m)
{
    m.impl("_npu_fused_infer_attention_score_v2_sink_metadata",
           &custom::npu_fused_infer_attention_score_v2_sink_metadata_npu);
}

TORCH_LIBRARY_IMPL(_C_ascend, Meta, m)
{
    m.impl("_npu_fused_infer_attention_score_v2_sink_metadata",
           &custom::npu_fused_infer_attention_score_v2_sink_metadata_meta);
}
