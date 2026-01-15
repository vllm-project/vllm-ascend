#include <torch/extension.h>
#include <torch/library.h>
#include <torch/version.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/npu/Module.h>
#include "utils.h"
/*
 * How to write a meta implementation for a custom operator (meta kernel):
 *
 * Meta implementations are used for shape and dtype inference, tracing, and export.
 * They do NOT perform any real computation or allocate device memory.
 * Instead, they return empty tensors with the correct shapes, dtypes, and device types.
 *
 * Steps to write a meta implementation:
 * 1. The function signature should match the operator's schema, but only use the arguments
 *    necessary to infer output shapes and dtypes.
 * 2. Use input tensor shapes, dtypes, and any relevant arguments to compute the output shapes.
 * 3. Return empty tensors (e.g., at::empty_symint, at::empty_like) with the correct shape and dtype.
 * 4. Do NOT perform any real computation or data movement.
 * 5. Register the meta implementation with the "Meta" dispatch key using TORCH_LIBRARY_IMPL or similar.
 *
 * Example:
 *   std::tuple<at::Tensor, at::Tensor> my_op_meta(
 *       at::Tensor &input, int64_t some_param) {
 *     // Infer output shape based on input and parameters
 *     auto out_shape = ...;
 *     at::Tensor out = at::empty_symint(out_shape, input.options());
 *     // Return empty tensor(s) with correct shape/dtype
 *     return {out, ...};
 *   }
 *
 * See below for real examples.
 */

namespace vllm_ascend {
namespace meta {
const int64_t INT4_NUMS_IN_INT32 = 8;
std::tuple<at::Tensor, at::Tensor> rotary_embedding_meta(
  at::Tensor &positions,
  at::Tensor &query,
  at::Tensor &key,
  int64_t head_size,
  at::Tensor &cos_sin_cache,
  bool is_neox) {
    auto num_tokens = positions.sym_numel();
    auto query_hidden_size = query.sym_numel() / num_tokens;
    auto key_hidden_size = key.sym_numel() / num_tokens;

    auto num_heads = query_hidden_size / head_size;
    auto num_kv_heads = key_hidden_size / head_size;
    at::Tensor query_dst = at::empty_symint({num_tokens, num_heads, head_size}, query.options());
    at::Tensor key_dst = at::empty_symint({num_tokens, num_kv_heads, head_size}, key.options());

    return {query_dst, key_dst};
}

std::tuple<at::Tensor, at::Tensor> get_masked_input_and_mask_meta(
    at::Tensor &input,
    const int64_t org_vocab_start_index,
    const int64_t org_vocab_end_index,
    const int64_t num_org_vocab_padding,
    const int64_t added_vocab_start_index,
    const int64_t added_vocab_end_index) {

    at::Tensor masked_input = at::empty_like(input);
    at::Tensor mask = at::empty_like(input, input.options().dtype(at::kBool));

    return {masked_input, mask};
}

at::Tensor bgmv_expand_meta(at::Tensor &x, at::Tensor &weight, at::Tensor &indices, at::Tensor &y,
                       int64_t slice_offset, int64_t slice_size) {
    at::Tensor y_out = at::empty_like(y);
    return y_out;
}

at::Tensor sgmv_expand_meta(at::Tensor &x, at::Tensor &weight, at::Tensor &lora_indices, at::Tensor &seq_len,
                       at::Tensor &y, int64_t slice_offset, int64_t slice_size) {
    at::Tensor y_out = at::empty_like(y);
    return y_out;
}

std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &> mla_preprocess(
    const at::Tensor &hiddenState,
    const at::Tensor &wdqkv,
    const c10::optional<at::Tensor> &descale0,
    const at::Tensor &gamma1,
    const c10::optional<at::Tensor> &beta1,
    const at::Tensor &wuq,
    const c10::optional<at::Tensor> &descale1,
    const at::Tensor &gamma2,
    const at::Tensor &cos,
    const at::Tensor &sin,
    const at::Tensor &wuk,
    const at::Tensor &kv_cache,
    const at::Tensor &kv_cache_rope,
    const at::Tensor &slotmapping,
    const c10::optional<at::Tensor> &quant_scale0,
    const c10::optional<at::Tensor> &quant_offset0,
    const c10::optional<at::Tensor> &bias0,
    const c10::optional<at::Tensor> &quant_scale1,
    const c10::optional<at::Tensor> &quant_offset1,
    const c10::optional<at::Tensor> &bias1,
    const c10::optional<at::Tensor> &ctkv_scale,
    const c10::optional<at::Tensor> &q_nope_scale,
    c10::optional<c10::string_view> cache_mode,
    c10::optional<c10::string_view> quant_mode,
    c10::optional<bool> enable_inner_out,
    at::Tensor &q_out0,
    at::Tensor &kv_cache_out0,
    at::Tensor &q_out1,
    at::Tensor &kv_cache_out1,
    at::Tensor &inner_out
    )
{
    return {q_out0, kv_cache_out0, q_out1, kv_cache_out1, inner_out};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> grouped_matmul_swiglu_quant(
    const at::Tensor &x, const at::Tensor &weight, const at::Tensor &weight_scale, const at::Tensor &x_scale,
    const at::Tensor &group_list, const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &offset)
{
    int m = x.sizes()[0];
    int n = weight.sizes()[2];
    bool is_a8w4 = x.dtype() == at::kChar && weight.dtype() == at::kInt;
    if (is_a8w4) {
        n *= INT4_NUMS_IN_INT32;
    }
    at::Tensor output = at::empty({m, n/2}, x.options().dtype(c10::ScalarType::Char));
    at::Tensor output_scale = at::empty({m}, x.options().dtype(c10::ScalarType::Float));
    at::Tensor output_offset = at::empty({}, x.options().dtype(c10::ScalarType::Float));
    return {output, output_scale, output_offset};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> grouped_matmul_swiglu_quant_weight_nz_tensor_list_meta(
    const at::Tensor & x,
    const at::TensorList & weight,
    const at::TensorList & weight_scale,
    const at::Tensor & x_scale,
    const at::Tensor & group_list,
    const c10::optional<at::Tensor> & bias,
    const c10::optional<at::Tensor> & offset)
{
    auto x_size = x.sizes();
    int n = weight[0].sizes()[1];
    int m = x_size[0];
    int k = x_size[1];

    at::Tensor output = at::zeros({m, n/2}, c10::dtype(c10::ScalarType::Char));
    at::Tensor output_scale = at::zeros({m}, c10::dtype(c10::ScalarType::Float));
    at::Tensor output_offset = at::zeros({m}, c10::dtype(c10::ScalarType::Float));

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(output, output_scale, output_offset);
}

std::tuple<at::Tensor, at::Tensor> dispatch_gmm_combine_decode_meta(
    const at::Tensor &x,
    const at::Tensor &expert_ids,
    const at::TensorList &gmm1_permuted_weight,
    const at::TensorList &gmm1_permuted_weight_scale,
    const at::TensorList &gmm2_weight,
    const at::TensorList &gmm2_weight_scale,
    const at::Tensor &expert_scales,
    const c10::optional<at::Tensor> &expert_smooth_scales,
    const c10::optional<at::Tensor> &x_active_mask,
    c10::string_view group_ep,
    int64_t ep_rank_size,
    int64_t ep_rank_id,
    int64_t moe_expert_num,
    int64_t shared_expert_num,
    int64_t shared_expert_rank_num,
    int64_t quant_mode,
    int64_t global_bs)
{
    auto x_shape = x.sizes();
    int bs = x_shape[0];
    int h = x_shape[1];

    at::Tensor output = at::empty({bs, h}, x.options().device(at::kMeta));

    bool is_shared_expert = (ep_rank_id < shared_expert_rank_num);
    int64_t num_local_experts = is_shared_expert ? 1 : moe_expert_num / (ep_rank_size - shared_expert_rank_num);
    auto opts = expert_ids.options().dtype(at::kLong); 
    at::Tensor expert_token_nums = at::empty({num_local_experts}, opts.device(at::kMeta)); 
    
    return {output, expert_token_nums};
}

void batch_matmul_transpose(const at::Tensor &tensor_a, const at::Tensor &tensor_b, at::Tensor &tensor_c,
                                    c10::optional<c10::string_view> format_mode,
                                    c10::optional<c10::string_view> quant_mode)
{
    return;
}

at::Tensor& dispatch_ffn_combine_meta(
    const at::Tensor& x,
    const at::TensorList& weight1,
    const at::TensorList& weight2,
    const at::Tensor& expert_idx,
    const at::TensorList& scale1,
    const at::TensorList& scale2,
    const at::Tensor& probs,
    c10::string_view group,
    int64_t max_output_size,
    at::Tensor& out
) {
    return out;
}

at::Tensor npu_lightning_indexer_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table, c10::string_view layout_query,
    c10::string_view layout_key, int64_t sparse_count, int64_t sparse_mode)
{
    // npu tensor max size
    constexpr int32_t SIZE = 8;
    constexpr int32_t DIM_0 = 0;
    constexpr int32_t DIM_1 = 1;
    constexpr int32_t DIM_2 = 2;
    constexpr int32_t DIM_3 = 3;

    TORCH_CHECK(query.numel() > 0, "Query is empty.");
    TORCH_CHECK(key.numel() > 0, "Key is empty.");
    TORCH_CHECK(weights.numel() > 0, "Weights is empty.");
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
                                       "than 0, but shape[", i, "] is ", query.size(i));
    }
    TORCH_CHECK(sparse_count > 0, "sparse count should be greater than 0, but now is ", sparse_count);

    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);
    at::SmallVector<int64_t, SIZE> output_size;
    if (query_layout_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), key.size(DIM_2), sparse_count};
    } else {
        int n_dim_index = 0;
        n_dim_index = (key_layout_str == "TND") ? DIM_1 : DIM_2;
        output_size = {query.size(DIM_0), key.size(n_dim_index), sparse_count};
    }
    // construct the output tensor
    at::Tensor lightning_indexer_output = at::empty(output_size, query.options().dtype(at::kInt));
    return lightning_indexer_output;
}

at::Tensor npu_sparse_flash_attention_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices, double scale_value, int64_t sparse_block_size,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope, c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode)
{
    std::string layout_query_str = std::string(layout_query);
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
                                       "than 0, but shape[", i, "] is ", query.size(i));
    }
    at::Tensor output = at::empty(query.sizes(), query.options().dtype(query.dtype()));
    return output;
}
std::tuple<at::Tensor, at::Tensor> matmul_allreduce_add_rmsnorm_meta(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &residual,
    const at::Tensor &gamma,
    c10::string_view group_tp,
    int64_t tp_rank_size,
    int64_t tp_rank_id,
    double epsilon,
    bool is_trans_b,
    bool is_gather_add_out)
    {
        at::Tensor output = at::empty_like(residual);
        at::Tensor add_out = at::empty_like(residual);

        return {output, add_out};
    }

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_moe_init_routing_custom_meta(
    const at::Tensor &x, const at::Tensor &expert_idx,
    const c10::optional<at::Tensor> &scale, const c10::optional<at::Tensor> &offset, int64_t active_num,
    int64_t expert_capacity, int64_t expert_num, int64_t drop_pad_mode, int64_t expert_tokens_num_type,
    bool expert_tokens_num_flag, int64_t quant_mode, at::IntArrayRef active_expert_range, int64_t row_idx_type)
{
    constexpr int64_t DIM_X = 2;
    constexpr int64_t DIM_EXPERT_IDX = 2;
    constexpr int64_t LENGTH_ACTIVE_EXPERT_RANGE = 2;
    constexpr int64_t EXPERT_TOKENS_COUNT = 1;
    constexpr int64_t EXPERT_TOKENS_KEY_VALUE = 2;
    constexpr int64_t QUANT_MODE_UNQUANT = -1;
    constexpr int64_t QUANT_MODE_DYNAMIC_QUANT = 1;
    constexpr int64_t CUMSUM = 0;
    constexpr int64_t COUNT = 1;
    constexpr int64_t KEY_VALUE = 2;

    if (active_expert_range.empty()) {
        active_expert_range =  at::IntArrayRef({0, expert_num});
    }

    int64_t x_dim = x.dim();
    TORCH_CHECK(x_dim == DIM_X, "The x should be ", DIM_X, 
                "-Dimension, current is ", x_dim, "-Dimension.");

    int64_t expert_idx_dim = expert_idx.dim();
    TORCH_CHECK(expert_idx_dim == DIM_EXPERT_IDX, "The expert_idx should be ", DIM_EXPERT_IDX, 
                "-Dimension, current is ", expert_idx_dim, "-Dimension.");

    int64_t active_expert_range_length = active_expert_range.size();
    TORCH_CHECK(active_expert_range_length == LENGTH_ACTIVE_EXPERT_RANGE, "The active_expert_range should be ", LENGTH_ACTIVE_EXPERT_RANGE, 
                "-Dimension, current is ", expert_idx_dim, "-Dimension.");

    int expert_length = active_expert_range[1] - active_expert_range[0];
    auto x_size = x.sizes();
    auto expert_idx_size = expert_idx.sizes();

    int bs = x_size[0];
    int h = x_size[1];
    int k = expert_idx_size[1];
    int64_t expanded_scale_len = 0;
    at::Tensor expanded_x;

    if (drop_pad_mode == 1) { // Drop/Pad
        if (quant_mode == QUANT_MODE_UNQUANT) {
            expanded_x = at::empty({expert_num, expert_capacity, h}, x.options());
        } else {
            expanded_x = at::empty({expert_num, expert_capacity, h}, x.options().dtype(at::kChar));
        }
        expanded_scale_len = expert_num * expert_capacity;
    } else { // Dropless / Active
        if (active_num > 0) { // Active
            int64_t num_out_tokens = std::min((int64_t)bs * k, active_num);
            if (quant_mode == QUANT_MODE_UNQUANT) {
                expanded_x = at::empty({num_out_tokens, h}, x.options());
            } else {
                expanded_x = at::empty({num_out_tokens, h}, x.options().dtype(at::kChar));
            }
            expanded_scale_len = num_out_tokens;
        } else { // Dropless
            if (quant_mode == QUANT_MODE_UNQUANT) {
                expanded_x = at::empty({bs * k, h}, x.options());
            } else {
                expanded_x = at::empty({bs * k, h}, x.options().dtype(at::kChar));
            }
            expanded_scale_len = bs * k;
        }
    }

    at::Tensor expanded_row_idx = at::empty({bs * k}, expert_idx.options());
    at::Tensor expert_tokens_count_or_cumsum;
    if (expert_tokens_num_type >= CUMSUM && expert_tokens_num_type <= COUNT) {
        // expert_tokens_count_or_cumsum in [end-start, ]
        expert_tokens_count_or_cumsum = at::empty({expert_length}, x.options().dtype(at::kLong));
    } else if (expert_tokens_num_type == KEY_VALUE) {
        // key_value in [2, end-start]
        expert_tokens_count_or_cumsum = at::empty({expert_num, 2}, x.options().dtype(at::kLong));
    }

    at::Tensor expanded_scale = at::empty({expanded_scale_len}, x.options().dtype(at::kFloat));
    return {expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale};
}
std::tuple<at::Tensor,at::Tensor, at::Tensor> moe_gating_top_k_meta(
    const at::Tensor& x,
    int64_t k,
    int64_t k_group,
    int64_t group_count,
    int64_t group_select_mode,
    int64_t renorm,
    int64_t norm_type,
    bool out_flag,
    double routed_scaling_factor,
    double eps,
    const c10::optional<at::Tensor>& bias_opt
    
    )
{
    TORCH_CHECK(x.dim() == 2, "The x should be 2D");
    TORCH_CHECK(
        x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16,
        "float16、float32 or bfloat16 tensor expected but got a tensor with dtype: ",
        x.scalar_type());

    auto x_size = x.sizes();
    auto rows = x_size[0];
    auto expert_num = x_size[1];
    const at::Tensor &bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    if (bias.defined()) {
        TORCH_CHECK(x.scalar_type() == bias.scalar_type(), "The dtype of x and bias should be same");
        TORCH_CHECK(bias.dim() == 1, "The bias should be 1D");
        auto bias_size = bias.sizes();
        TORCH_CHECK(bias_size[0] == expert_num, "The bias first dim should be same as x second dim");
    }
    at::Tensor y = at::empty({rows, k}, x.options());
    at::Tensor expert_idx = at::empty({rows, k}, x.options().dtype(at::kInt));
    at::Tensor out = at::empty({rows, expert_num}, x.options().dtype(at::kFloat));

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y,expert_idx,out);
}
} // namespace meta
} // namespace vllm_ascend

namespace {
// Register the meta implementations of the custom kernels for symbolic tracing, this will also
// the custom kernel been captured into aclgraph
TORCH_LIBRARY_IMPL_EXPAND(CONCAT(_C, _ascend), Meta, ops) {

    // Rotary embedding meta implementation
    ops.impl("rotary_embedding", &vllm_ascend::meta::rotary_embedding_meta);
    // Masked input and mask meta implementation
    ops.impl("get_masked_input_and_mask", &vllm_ascend::meta::get_masked_input_and_mask_meta);
    // Bgmv expand
    ops.impl("bgmv_expand", &vllm_ascend::meta::bgmv_expand_meta);
    // Sgmv expand
    ops.impl("sgmv_expand", &vllm_ascend::meta::sgmv_expand_meta);
    // MLA preprocess
    ops.impl("mla_preprocess", &vllm_ascend::meta::mla_preprocess);
    // grouped_matmul_swiglu_quant meta implementation
    ops.impl("grouped_matmul_swiglu_quant", &vllm_ascend::meta::grouped_matmul_swiglu_quant);
    // Grouped matmul swiglu quant weight nz tensor list
    ops.impl("grouped_matmul_swiglu_quant_weight_nz_tensor_list", &vllm_ascend::meta::grouped_matmul_swiglu_quant_weight_nz_tensor_list_meta);
    // dispatch_gmm_combine_decode meta implementation
    ops.impl("dispatch_gmm_combine_decode", &vllm_ascend::meta::dispatch_gmm_combine_decode_meta);
    // batch_matmul_transpose
    ops.impl("batch_matmul_transpose", &vllm_ascend::meta::batch_matmul_transpose);
    // Lightning indexer
    ops.impl("npu_lightning_indexer", &vllm_ascend::meta::npu_lightning_indexer_meta);
    // Sparse flash attention
    ops.impl("npu_sparse_flash_attention", &vllm_ascend::meta::npu_sparse_flash_attention_meta);
    // MoE dispatch-ffn-combine
    ops.impl("dispatch_ffn_combine", &vllm_ascend::meta::dispatch_ffn_combine_meta);
    // matmul allreduce add rmsnorm
    ops.impl("matmul_allreduce_add_rmsnorm", &vllm_ascend::meta::matmul_allreduce_add_rmsnorm_meta);
    // moe_init_routing_custom
    ops.impl("npu_moe_init_routing_custom", &vllm_ascend::meta::npu_moe_init_routing_custom_meta);
    // Moe_gating_top_k
    ops.impl("moe_gating_top_k", &vllm_ascend::meta::moe_gating_top_k_meta);
}
}
