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

#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
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

void batch_matmul_transpose(const at::Tensor &tensor_a, const at::Tensor &tensor_b, at::Tensor &tensor_c,
                                    c10::optional<c10::string_view> format_mode,
                                    c10::optional<c10::string_view> quant_mode)
{
    return;
}
#endif

void device_print_meta(c10::string_view msg)
{
    (void)msg;
}

void device_print_tensor_meta(const at::Tensor& tensor)
{
    (void)tensor;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> grouped_matmul_swiglu_quant(
    const at::Tensor &x, const at::Tensor &weight, const at::Tensor &weight_scale, const at::Tensor &x_scale,
    const at::Tensor &group_list, const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &offset,
    double swiglu_limit)
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
    const c10::optional<at::Tensor> & offset,
    double swiglu_limit)
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

std::tuple<at::Tensor, at::Tensor> grouped_matmul_swiglu_quant_v2_meta(
    const at::Tensor & x,
    const at::TensorList &weight,
    const at::TensorList &weight_scale,
    const at::Tensor & x_scale,
    const at::Tensor & group_list,
    const c10::optional<at::Tensor> & smooth_scale,
    const c10::optional<at::TensorList> weight_assist_matrix,
    const c10::optional<at::Tensor> & bias,
    c10::optional<int64_t> dequant_mode,
    c10::optional<int64_t> dequant_dtype,
    c10::optional<int64_t> quant_mode,
    c10::optional<int64_t> quant_dtype,
    bool transpose_weight,
    int64_t group_list_type,
    at::IntArrayRef tuning_config,
    double swiglu_limit)
{

    auto x_size = x.sizes();
    int n = weight_scale[0].sizes().back();
    int m = x_size[0];
    int k = x_size[1];

    at::Tensor output =  at::empty({m, n/2}, x.options().dtype(at::kChar));
    at::Tensor output_scale =  at::empty({m}, x.options().dtype(at::kFloat));



    return std::tuple<at::Tensor, at::Tensor>(output, output_scale);
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

std::tuple<at::Tensor&, at::Tensor&> dispatch_ffn_combine_meta(
    const at::Tensor& x,
    const at::TensorList& weight1,
    const at::TensorList& weight2,
    const at::Tensor& expert_idx,
    const at::TensorList& scale1,
    const at::TensorList& scale2,
    const at::TensorList& bias1,
    const at::TensorList& bias2,
    const at::Tensor& probs,
    c10::string_view group,
    int64_t max_output_size,
    at::Tensor& out,
    at::Tensor& expert_token_nums,
    const c10::optional<at::Tensor> &x_active_mask,
    double swiglu_limit
) {
    return {out, expert_token_nums};
}

std::tuple<at::Tensor, at::Tensor> npu_lightning_indexer_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table, c10::string_view layout_query,
    c10::string_view layout_key, int64_t sparse_count, int64_t sparse_mode,
    int64_t pre_tokens, int64_t next_tokens, bool return_value)
{
    // npu tensor max size
    constexpr int64_t SIZE = 8;
    constexpr int64_t DIM_0 = 0;
    constexpr int64_t DIM_1 = 1;
    constexpr int64_t DIM_2 = 2;

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
    at::Tensor sparse_indices_out = at::empty(output_size, query.options().dtype(at::kInt));
    at::Tensor sparse_values_out;
    if (return_value) {
        sparse_values_out = at::empty(output_size, query.options().dtype(query.dtype()));
    } else {
        sparse_values_out = at::empty({0}, query.options().dtype(query.dtype()));
    }
    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out, sparse_values_out);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_sparse_flash_attention_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices, double scale_value,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope, int64_t sparse_block_size,
    c10::string_view layout_query, c10::string_view layout_kv,
    int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens,
    int64_t attention_mode, bool return_softmax_lse)
{
    constexpr int64_t SIZE = 8;
    constexpr int64_t DIM_0 = 0;
    constexpr int64_t DIM_1 = 1;
    constexpr int64_t DIM_2 = 2;
    constexpr int64_t DIM_3 = 3;
    constexpr int64_t DIM_4 = 4;

    std::string layout_query_str = std::string(layout_query);
    TORCH_CHECK(layout_query_str == "BSND" || layout_query_str == "TND",
                "The layout of query only support BSND and TND, but got ",
                layout_query_str);
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
                                       "than 0, but shape[", i, "] is ", query.size(i));
    }

    at::SmallVector<int64_t, SIZE> output_size;
    if (layout_query_str == "TND") {
        TORCH_CHECK(query.dim() == DIM_3,
                    "When the layout of query is TND, the query dimension must be 3, but got ",
                    query.dim());
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2)};
    } else {
        TORCH_CHECK(query.dim() == DIM_4,
                    "When the layout of query is BSND, the query dimension must be 4, but got ",
                    query.dim());
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), query.size(DIM_3)};
    }

    at::Tensor output = at::empty(output_size, query.options().dtype(query.dtype()));
    at::SmallVector<int64_t, SIZE> softmax_size;
    if (return_softmax_lse) {
        if (query.dim() == DIM_3) {
            softmax_size = {key.size(DIM_1), query.size(DIM_0), query.size(DIM_1) / key.size(DIM_1)};
        } else {
            softmax_size = {
                query.size(DIM_0), key.size(DIM_2), query.size(DIM_1), query.size(DIM_2) / key.size(DIM_2)};
        }
    } else {
        softmax_size = {0};
    }

    at::Tensor softmax_max = at::empty(softmax_size, query.options().dtype(at::kFloat));
    at::Tensor softmax_sum = at::empty(softmax_size, query.options().dtype(at::kFloat));
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(output, softmax_max, softmax_sum);
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

std::tuple<at::Tensor,at::Tensor, at::Tensor> npu_add_rms_norm_bias_meta(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& gamma,
    const c10::optional<at::Tensor> &beta,
    double epsilon)
{
    int64_t dim_x = x1.dim();
    int64_t dim_gamma = gamma.dim();
    int64_t diff = dim_x - dim_gamma;
    c10::SymDimVector new_shape;
    at::Tensor rstd;

    if (diff > 0) {
        new_shape.reserve(dim_x);
        auto x1_sizes = x1.sym_sizes();
        for (int64_t i = 0; i < diff; ++i) {
            new_shape.push_back(x1_sizes[i]);
        }
        for (int64_t i = 0; i < dim_gamma; ++i) {
            new_shape.push_back(c10::SymInt(1));
        }
    } else {
        new_shape.assign(dim_x, c10::SymInt(1));
    }
    rstd = at::empty_symint(new_shape, x1.options().dtype(at::kFloat));
    at::Tensor y = at::empty_symint(x1.sym_sizes(), x1.options());
    at::Tensor x = at::empty_symint(x1.sym_sizes(), x1.options());
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, rstd, x);
}

at::Tensor npu_reshape_and_cache_bnsd_meta(const at::Tensor& hashq,
                                           const at::Tensor& hashkCache,
                                           const at::Tensor& slotMapping,
                                           const at::Tensor& seqLen,
                                           const at::Tensor& hashkCacheOut) {
    at::Tensor output = at::empty(hashkCache.sizes(), hashkCache.options().dtype(hashkCache.dtype()).device(hashkCache.device()));
    return output;
}


at::Tensor npu_hamming_dist_top_k_meta(const at::Tensor &hashq,
                                       const at::Tensor &hashkCache,
                                       const at::Tensor& hashkCacheRope,
                                       const at::Tensor &topN,
                                       const at::Tensor &seqLen,
                                       const c10::optional<at::Tensor> &chunkSize,
                                       const c10::optional<int64_t> maxSeqLen,
                                       const c10::optional<int64_t> sink,
                                       const c10::optional<int64_t> recent,
                                       const c10::optional<int64_t> supportOffload,
                                       const c10::optional<at::Tensor> &blockTable,
                                       const c10::optional<at::Tensor> &mask,
                                       const c10::optional<at::Tensor>& indices) {
    if (indices.has_value()) {
        return at::empty_like(indices.value());
    }
    uint32_t MAX_BLOCK_PER_REQ_INHSA = 512;

    auto n_bs = hashq.size(0);
    auto n_kv_heads = hashkCache.size(1);
    auto n_max_kv = MAX_BLOCK_PER_REQ_INHSA;
    at::Tensor out = at::empty({n_bs, n_kv_heads, n_max_kv}, torch::TensorOptions().dtype(torch::kInt32).device(hashq.device()));
    return out;
}

at::Tensor npu_sign_bits_pack_meta(const at::Tensor& input,
                                   const int64_t size) {
    int64_t ySize = (input.size(0) + 7) / 8;
    int64_t outDim = 0;
    if (size != 0) {
        outDim = ySize / size;
    }

    at::Tensor out = torch::empty({size, outDim}, torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
    return out;
}

std::tuple<at::Tensor, at::Tensor> npu_gemma_rms_norm_meta(
    const at::Tensor& x,
    const at::Tensor& gamma,
    double epsilon)
{
    int64_t dim_x = x.dim();
    int64_t dim_gamma = gamma.dim();
    int64_t diff = dim_x - dim_gamma;
    c10::SymDimVector new_shape;
    at::Tensor rstd;
    if (diff > 0) {
        new_shape.reserve(dim_x);
        auto x_sizes = x.sym_sizes();
        for (int64_t i = 0; i < diff; ++i) {
            new_shape.push_back(x_sizes[i]);
        }
        for (int64_t i = 0; i < dim_gamma; ++i) {
            new_shape.push_back(c10::SymInt(1));
        }
    } else {
        new_shape.assign(dim_x, c10::SymInt(1));
    }
    rstd = at::empty_symint(new_shape, x.options().dtype(at::kFloat));
    at::Tensor y = at::empty_symint(x.sym_sizes(), x.options());
    return std::tuple<at::Tensor, at::Tensor>(y, rstd);
}

void transpose_kv_cache_by_block_meta(
    const at::TensorList &k_cache,
    const at::TensorList &v_cache,
    const at::Tensor &block_ids,
    int64_t block_size,
    int64_t head_num,
    int64_t head_dim,
    int64_t split_num,
    int64_t layer_num)
{
    return;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_copy_and_expand_eagle_inputs_meta(
    const at::Tensor &target_token_ids,
    const at::Tensor &target_positions,
    const at::Tensor &next_token_ids,
    const at::Tensor &query_start_loc,
    const at::Tensor &query_end_loc,
    int64_t padding_token_id,
    int64_t parallel_drafting_token_id,
    int64_t num_padding_slots_per_request,
    bool shift_input_ids,
    int64_t total_draft_tokens)
{
    int64_t total_input_tokens = target_token_ids.size(0);
    int64_t num_reqs = query_start_loc.size(0) - 1;

    at::Tensor out_input_ids = at::empty({total_draft_tokens}, target_token_ids.options());
    at::Tensor out_positions = at::empty({total_draft_tokens}, target_token_ids.options());
    at::Tensor out_is_rejected_token_mask = at::empty({total_draft_tokens}, target_token_ids.options().dtype(at::kChar));
    at::Tensor out_is_masked_token_mask = at::empty({total_draft_tokens}, target_token_ids.options().dtype(at::kChar));
    at::Tensor out_new_token_indices = at::empty({num_reqs * num_padding_slots_per_request}, target_token_ids.options());
    at::Tensor out_hidden_state_mapping = at::empty({total_input_tokens}, target_token_ids.options());

    return {out_input_ids, out_positions, out_is_rejected_token_mask, out_is_masked_token_mask,
            out_new_token_indices, out_hidden_state_mapping};
}

at::Tensor npu_causal_conv1d_custom_meta(
    const at::Tensor& output,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_state,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef query_start_loc_opt,
    at::IntArrayRef cache_indices_opt,
    at::IntArrayRef initial_state_mode_opt,
    at::IntArrayRef num_accepted_tokens_opt,
    int64_t  activation_mode,
    int64_t  pad_slot_id,
    int64_t  run_mode)
{
    return output;
}

at::Tensor npu_causal_conv1d_310_meta(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& conv_states,
    const c10::optional<at::Tensor>& query_start_loc,
    const c10::optional<at::Tensor>& cache_indices,
    const c10::optional<at::Tensor>& initial_state_mode,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    int64_t activation_mode,
    int64_t pad_slot_id,
    int64_t run_mode)
{

    at::Tensor output = at::empty_symint(x.sym_sizes(), x.options());
    return output;
}

at::Tensor npu_recurrent_gated_delta_rule_310_meta(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& beta,
    at::Tensor& state,
    const at::Tensor& actual_seq_lengths,
    const at::Tensor& ssm_state_indices,
    const c10::optional<at::Tensor>& g,
    const c10::optional<at::Tensor>& gk,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    double scale_value)
{

    at::Tensor output = at::empty_symint(value.sym_sizes(), value.options());
    return output;
}

at::Tensor npu_recurrent_gated_delta_rule_meta(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& state,
    const c10::optional<at::Tensor>& beta,
    const c10::optional<double> scale,
    const c10::optional<at::Tensor>& actual_seq_lengths,
    const c10::optional<at::Tensor>& ssm_state_indices,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    const c10::optional<at::Tensor>& g,
    const c10::optional<at::Tensor>& gk)
{

    auto options = value.options().dtype(at::ScalarType::BFloat16);
    at::Tensor output = at::empty_symint(value.sym_sizes(), options);
    return output;
}

std::tuple<at::Tensor, at::Tensor> npu_fused_gdn_gating_meta(
    const at::Tensor& A_log,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& dt_bias,
    double beta,
    double threshold)
{
    (void)beta;
    (void)threshold;
    int64_t batch = a.size(0);
    int64_t num_heads = a.size(1);

    at::Tensor g = at::empty_symint(
        {1, batch, num_heads}, a.options().dtype(c10::kFloat));
    at::Tensor beta_output = at::empty_symint(
        {1, batch, num_heads}, b.options());

    return std::make_tuple(g, beta_output);
}

std::vector<at::Tensor> moe_grouped_matmul_meta(
    at::Tensor x,
    at::Tensor weight,
    const at::Tensor& group_list,
    int64_t split_item,
    int64_t group_type,
    int64_t group_list_type
)
{
    bool transpose_weight = false;
    bool weight_nz = true;

    at::TensorList x_list = at::TensorList(x);
    at::TensorList weight_list = at::TensorList(weight);
    std::vector<at::Tensor> y;
    c10::TensorOptions options = x[0].options().dtype(x[0].scalar_type());
    auto m = x[0].sizes()[0];
    auto n = weight[0].sizes()[1];
    if (!transpose_weight) {
        n = weight[0].sizes()[2];
    }
    at::Tensor y_0 = at::zeros(at::IntArrayRef{m, n}, options);
    y.emplace_back(y_0);
    at::TensorList result = at::TensorList(y);

    return y;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_gating_top_k_hash_meta(
    const at::Tensor& x,
    int64_t k,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& input_ids_opt,
    const c10::optional<at::Tensor>& tid2eid_opt,
    int64_t k_group,
    int64_t group_count,
    double routed_scaling_factor,
    double eps,
    int64_t group_select_mode,
    int64_t renorm,
    int64_t norm_type,
    bool out_flag)
{
    TORCH_CHECK(x.dim() == 2, "x must be 2D, but got dim=", x.dim());
    TORCH_CHECK(
        x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16,
        "x dtype must be float16/float32/bfloat16, but got ", x.scalar_type());

    TORCH_CHECK(k > 0, "k must be > 0, but got k=", k);
    TORCH_CHECK(k_group >= 1, "k_group must be >= 1, but got k_group=", k_group);
    TORCH_CHECK(group_count >= 1, "group_count must be >= 1, but got group_count=", group_count);

    TORCH_CHECK(group_select_mode == 0 || group_select_mode == 1,
                "group_select_mode must be 0 or 1, but got ", group_select_mode);
    TORCH_CHECK(renorm == 0,
                "renorm can only be 0 currently, but got ", renorm);
    TORCH_CHECK(norm_type == 0 || norm_type == 1 || norm_type ==2,
                "norm_type must be 0 (softmax) or 1 (sigmoid) or 2 (softplus), but got ", norm_type);

    TORCH_CHECK(eps > 0.0, "eps must be > 0, but got ", eps);
    TORCH_CHECK(routed_scaling_factor > 0.0,
                "routed_scaling_factor must be > 0, but got ", routed_scaling_factor);

    const auto sizes = x.sizes();
    const int64_t rows = sizes[0];
    const int64_t expert_num = sizes[1];

    TORCH_CHECK(expert_num > 0, "expert_num must be > 0");
    TORCH_CHECK(expert_num <= 2048,
                "expert_num (E) must be <= 2048, but got ", expert_num);

    if (bias_opt.has_value() && bias_opt->defined()) {
        const auto& bias = *bias_opt;
        TORCH_CHECK(bias.dim() == 1, "bias must be 1D, but got dim=", bias.dim());
        TORCH_CHECK(bias.size(0) == expert_num,
                    "bias.size(0) must equal expert_num. bias.size(0)=",
                    bias.size(0), ", expert_num=", expert_num);
        TORCH_CHECK(bias.scalar_type() == x.scalar_type(),
                    "bias dtype must equal x dtype. x=", x.scalar_type(),
                    ", bias=", bias.scalar_type());
    }

    if (input_ids_opt.has_value() && input_ids_opt->defined()) {
        const auto& input_ids = *input_ids_opt;
        TORCH_CHECK(input_ids.scalar_type() == at::kInt || input_ids.scalar_type() == at::kLong,
                    "input_ids dtype must be int32 or int64, but got ", input_ids.scalar_type());
        TORCH_CHECK(input_ids.numel() == rows,
                    "input_ids.numel() must equal x.size(0). input_ids.numel()=",
                    input_ids.numel(), ", rows=", rows);
    }

    if (tid2eid_opt.has_value() && tid2eid_opt->defined()) {
        const auto& tid2eid = *tid2eid_opt;
        TORCH_CHECK(tid2eid.scalar_type() == at::kInt || tid2eid.scalar_type() == at::kLong,
                    "tid2eid dtype must be int32 or int64, but got ", tid2eid.scalar_type());
        TORCH_CHECK(tid2eid.dim() >= 1, "tid2eid must have dim>=1, but got dim=", tid2eid.dim());
    }

    at::Tensor y = at::empty({rows, k}, x.options());
    at::Tensor expert_idx = at::empty({rows, k}, x.options().dtype(at::kInt));
    at::Tensor out = at::empty({rows, expert_num}, x.options().dtype(at::kFloat));

    return {y, expert_idx, out};
}

std::tuple<at::Tensor> construct_compressor_output_tensor(const at::Tensor &x, const at::Tensor &norm_weight,
                                                          const at::Tensor &rope_sin, int64_t cmp_ratio, int64_t coff)
{
    constexpr int DIM_3 = 3;
    auto x_dim = x.dim();
    at::SmallVector<int64_t, 8> cmp_kv_size;
    at::Tensor cmp_kv;
    auto cmp_s = 0;
    if (x_dim == DIM_3) {
        cmp_s = (x.size(1) + cmp_ratio - 1) / cmp_ratio;
        cmp_kv_size = {x.size(0), cmp_s, norm_weight.size(0)};
    } else {
        cmp_s = rope_sin.size(0);
        cmp_kv_size = {cmp_s, norm_weight.size(0)};
    }

    cmp_kv = at::empty(cmp_kv_size, x.options().dtype(x.dtype()));

    return std::tuple<at::Tensor>(cmp_kv);
}

std::tuple<at::Tensor>
compressor_meta(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate, at::Tensor &state_cache,
                const at::Tensor &ape, const at::Tensor &norm_weight, const at::Tensor &rope_sin,
                const at::Tensor &rope_cos, const c10::optional<at::Tensor> &state_block_table,
                const c10::optional<at::Tensor> &cu_seqlens, const c10::optional<at::Tensor> &seqused,
                const c10::optional<at::Tensor> &start_pos, int64_t rope_head_dim, int64_t cmp_ratio, int64_t coff,
                double norm_eps, int64_t rotary_mode, int64_t cache_mode)
{
    // construct the output tensor
    auto x_dim = x.dim();
    auto norm_weight_dim = norm_weight.dim();
    auto rope_sin_dim = rope_sin.dim();

    std::tuple<at::Tensor> output = construct_compressor_output_tensor(x, norm_weight, rope_sin, cmp_ratio, coff);

    return output;
}

std::tuple<at::Tensor, at::Tensor> construct_quant_lightning_indexer_output_tensor(const at::Tensor& query, const at::Tensor& key,
                                                           int64_t sparse_count, std::string query_layout_str,
                                                           std::string key_layout_str, bool return_value)
{
    constexpr int64_t SIZE = 8;
    constexpr int64_t DIM_0 = 0;
    constexpr int64_t DIM_1 = 1;
    constexpr int64_t DIM_2 = 2;
    constexpr int64_t DIM_3 = 3;
    at::SmallVector<int64_t, SIZE> output_size;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
            "than 0, but shape[", i, "] is ", query.size(i));
    }
    for (size_t i = 0; i < key.sizes().size(); i++) {
        TORCH_CHECK(key.size(i) > 0, "All values within key's shape should be greater "
            "than 0, but shape[", i, "] is ", key.size(i));
    }
    TORCH_CHECK(sparse_count > 0, "sparse count should be greater than 0, but now is ", sparse_count);
    int64_t keyHeadNum = (key_layout_str == "TND")? key.size(DIM_1) : key.size(DIM_2);
    if (query_layout_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), keyHeadNum, sparse_count};
    } else {
        output_size = {query.size(DIM_0), keyHeadNum, sparse_count};
    }
    at::Tensor sparse_indices_out = at::empty(output_size, query.options().dtype(at::kInt));
    at::Tensor sparse_values_out;
    if (return_value) {
        sparse_values_out = at::empty(output_size, query.options().dtype(at::kFloat));
    } else {
        sparse_values_out = at::empty({0}, query.options().dtype(at::kFloat));
    }

    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out, sparse_values_out);
}

std::tuple<at::Tensor, at::Tensor> npu_quant_lightning_indexer_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &query_dequant_scale, const at::Tensor &key_dequant_scale,
    int64_t query_quant_mode, int64_t key_quant_mode,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &metadata,
    c10::string_view layout_query, c10::string_view layout_key, int64_t sparse_count,
    int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, int64_t cmp_ratio, bool return_value)
{
    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);
    std::tuple<at::Tensor, at::Tensor> quant_lightning_indexer_output = construct_quant_lightning_indexer_output_tensor(
            query, key, sparse_count, query_layout_str, key_layout_str, return_value);
    at::Tensor sparse_indices_out = std::get<0>(quant_lightning_indexer_output);
    at::Tensor sparse_values_out = std::get<1>(quant_lightning_indexer_output);

    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out, sparse_values_out);
}

std::tuple<at::Tensor, at::Tensor> construct_output_tensor(const at::Tensor &q, std::string layout,
    bool return_softmax_lse)
{
    for (size_t i = 0; i < q.sizes().size(); i++) {
        TORCH_CHECK(q.size(i) > 0,
            "All values within query's shape should be greater "
            "than 0, but shape[",
            i,
            "] is ",
            q.size(i));
    }
    at::Tensor output = at::empty(q.sizes(), q.options().dtype(q.dtype()));
    at::Tensor softmax_lse;
    if (return_softmax_lse) {
        std::vector<int64_t> lse_sizes(q.sizes().begin(), q.sizes().end());
        lse_sizes.back() = 1;
        softmax_lse = at::empty(lse_sizes, q.options().dtype(c10::ScalarType::Float));
    } else {
        softmax_lse = at::empty({0}, q.options().dtype(c10::ScalarType::Float));
    }
    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

std::tuple<at::Tensor, at::Tensor> npu_sparse_attn_sharedkv_meta(const at::Tensor &q, const c10::optional<at::Tensor> &ori_kv,
    const c10::optional<at::Tensor> &cmp_kv, const c10::optional<at::Tensor> &ori_sparse_indices,
    const c10::optional<at::Tensor> &cmp_sparse_indices, const c10::optional<at::Tensor> &ori_block_table,
    const c10::optional<at::Tensor> &cmp_block_table, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv, const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &metadata,
    double softmax_scale, int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode, int64_t ori_win_left,
    int64_t ori_win_right, c10::string_view layout_q, c10::string_view layout_kv, bool return_softmax_lse)
{
    std::string layout_q_str = std::string(layout_q);
    std::tuple<at::Tensor, at::Tensor> output = construct_output_tensor(q, layout_q_str, return_softmax_lse);

    return output;
}

at::Tensor npu_sparse_attn_sharedkv_metadata_meta(
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv,
    const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_kv,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    int64_t ori_topk,
    int64_t cmp_topk,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    bool has_ori_kv,
    bool has_cmp_kv,
    const c10::string_view device)
{
    constexpr int64_t OUTPUT_SIZE = 1024;
    at::Tensor output;
    if (cu_seqlens_q.has_value()) {
        output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(cu_seqlens_q.value().device()));
    } else if (cu_seqlens_ori_kv.has_value()) {
        output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(cu_seqlens_ori_kv.value().device()));
    } else if (cu_seqlens_cmp_kv.has_value()) {
        output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(cu_seqlens_cmp_kv.value().device()));
    } else if (seqused_q.has_value()) {
        output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(seqused_q.value().device()));
    } else if (seqused_kv.has_value()) {
        output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(seqused_kv.value().device()));
    } else {
        auto deviceOri = at::Device(std::string(device));
        std::string device_str = "meta";
        if (deviceOri.has_index()) {
            device_str += ":";
            device_str += std::to_string(deviceOri.index());
        }
        output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(at::Device(device_str)));
    }
    return output;
}

at::Tensor npu_quant_lightning_indexer_metadata_meta(
    int64_t num_heads_q, int64_t num_heads_k, int64_t head_dim, int64_t query_quant_mode, int64_t key_quant_mode,
    const c10::optional<at::Tensor> &actual_seq_lengths_query, const c10::optional<at::Tensor> &actual_seq_lengths_key, int64_t batch_size,
    int64_t max_seqlen_q, int64_t max_seqlen_k, const c10::string_view layout_query, c10::string_view layout_key, int64_t sparse_count,
    int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, int64_t cmp_ratio, const c10::string_view device)
{
    constexpr int64_t OUTPUT_SIZE = 1024;
    at::Tensor output;
    if (actual_seq_lengths_query.has_value()) {
        output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(actual_seq_lengths_query.value().device()));
    } else if (actual_seq_lengths_key.has_value()) {
        output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(actual_seq_lengths_key.value().device()));
    } else {
        auto deviceOri = at::Device(std::string(device));
        std::string device_str = "meta";
        if (deviceOri.has_index()) {
            device_str += ":";
            device_str += std::to_string(deviceOri.index());
        }
        output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(at::Device(device_str)));
    }

    return output;
}

at::Tensor construct_hc_post_output_tensor(const at::Tensor& residual)
{
    c10::SymIntArrayRef output_size = residual.sym_sizes();
    at::Tensor out = at::empty_symint(output_size, residual.options().dtype(residual.dtype()));
    return out;
}

at::Tensor npu_hc_post_meta(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post,
    const at::Tensor& comb)
{
    at::Tensor outputs = construct_hc_post_output_tensor(residual);
    return outputs;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_hc_pre_output_tensor(const at::Tensor& x, int64_t hc_mult)
{
    auto xDims = x.dim();
    at::SmallVector<c10::SymInt, 8> y_size;
    at::SmallVector<c10::SymInt, 8> post_size;
    at::SmallVector<c10::SymInt, 8> comb_frag_size;
    if (xDims == 4) {
        auto batch = x.sym_size(0);
        auto size = x.sym_size(1);
        auto d = x.sym_size(3);
        y_size = {batch, size, d};
        post_size = {batch, size, hc_mult};
        comb_frag_size = {batch, size, hc_mult, hc_mult};
    } else if (xDims == 3){
        auto bs = x.sym_size(0);
        auto d = x.sym_size(2);
        y_size = {bs, d};
        post_size = {bs, hc_mult};
        comb_frag_size = {bs, hc_mult, hc_mult};
    }

    at::Tensor y = at::empty_symint(c10::SymIntArrayRef(y_size), x.options().dtype(at::kBFloat16));
    at::Tensor post = at::empty_symint(c10::SymIntArrayRef(post_size), x.options().dtype(at::kFloat));
    at::Tensor comb_frag = at::empty_symint(c10::SymIntArrayRef(comb_frag_size), x.options().dtype(at::kFloat));

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
}

at::Tensor construct_hc_pre_rsqrt_output_tensor(const at::Tensor& x, float epsilon=1e-6)
{
    constexpr int64_t SIZE = 8;
    TORCH_CHECK(epsilon >= 0, "epsilon should be greater than 0.");

    auto options = x.options();
    auto xDims = x.dim();
    c10::SmallVector<int64_t, SIZE> yOut_shape;
    for (size_t i = 0; i < xDims - 2; i++) {
        yOut_shape.push_back(x.sizes()[i]);
    }
    yOut_shape.push_back(1);
    at::Tensor yOut = at::empty(yOut_shape, options.dtype(at::kFloat));

    return yOut;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_meta(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale, const at::Tensor& hc_base,
    int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps, double hc_eps)
{
    auto output_tensors = construct_hc_pre_output_tensor(x, hc_mult);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor post = std::get<1>(output_tensors);
    at::Tensor comb_frag = std::get<2>(output_tensors);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
}

at::Tensor construct_hc_pre_inv_rms_output_tensor(const at::Tensor& x, float epsilon=1e-20)
{
    constexpr int64_t SIZE = 8;
    TORCH_CHECK(epsilon >= 0, "epsilon should be greater than 0.");

    auto options = x.options();
    auto xDims = x.dim();
    c10::SmallVector<int64_t, SIZE> yOut_shape;
    for (auto i = 0; i < xDims - 2; i++) {
        yOut_shape.push_back(x.sizes()[i]);
    }
    yOut_shape.push_back(1);
    at::Tensor yOut = at::empty(yOut_shape, options.dtype(at::kFloat));

    return yOut;
}

at::Tensor npu_hc_pre_inv_rms_meta(const at::Tensor& x, double epsilon=1e-20)
{
    TORCH_CHECK(x.numel() > 0, "Input tensor x should not be empty.");
    TORCH_CHECK(epsilon >= 0, "epsilon should be greater than 0.");

    at::Tensor yOut;
    yOut = construct_hc_pre_inv_rms_output_tensor(x, epsilon);

    return yOut;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_hc_pre_sinkhorn_output_tensor(const at::Tensor& mixes, const at::Tensor& x, int64_t hc_mult)
{
    auto xDims = x.dim();
    at::SmallVector<int64_t, 8> y_size;
    at::SmallVector<int64_t, 8> post_size;
    at::SmallVector<int64_t, 8> comb_frag_size;
    if (xDims == 4) {
        auto batch = x.size(0);
        auto size = x.size(1);
        auto d = x.size(3);
        y_size = {batch, size, d};
        post_size = {batch, size, hc_mult};
        comb_frag_size = {batch, size, hc_mult, hc_mult};
    } else if (xDims == 3){
        auto bs = x.size(0);
        auto d = x.size(2);
        y_size = {bs, d};
        post_size = {bs, hc_mult};
        comb_frag_size = {bs, hc_mult, hc_mult};
    }

    at::Tensor y = at::empty(y_size, x.options().dtype(at::kBFloat16));
    at::Tensor post = at::empty(post_size, x.options().dtype(at::kFloat));
    at::Tensor comb_frag = at::empty(comb_frag_size, x.options().dtype(at::kFloat));

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_sinkhorn_meta(
    const at::Tensor& mixes, const at::Tensor& rsqrt, const at::Tensor& hc_scale, const at::Tensor& hc_base,
    const at::Tensor& x, int64_t hc_mult, int64_t hc_sinkhorn_iters, double hc_eps)
{
    auto output_tensors = construct_hc_pre_sinkhorn_output_tensor(mixes, x, hc_mult);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor post = std::get<1>(output_tensors);
    at::Tensor comb_frag = std::get<2>(output_tensors);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
}

void inplace_partial_rotary_mul_meta(
    at::Tensor &x,
    const at::Tensor &r1,
    const at::Tensor &r2,
    c10::string_view rotary_mode,
    at::IntArrayRef partial_slice)
{
    auto origin_dim_num = x.dim();
    return;
}

std::tuple<at::Tensor, at::Tensor> npu_rms_norm_dynamic_quant_meta(
    const at::Tensor& x,
    const at::Tensor& gamma,
    const c10::optional<at::Tensor>& smooth_scale,
    const c10::optional<at::Tensor>& beta,
    double epsilon)
{
    constexpr int32_t SIZE = 8;
    at::Tensor y_out = at::empty_like(x);
    auto options = x.options();
    c10::SmallVector<int64_t, SIZE> scale_out_shape;
    for (size_t i = 0; i < x.sizes().size() - 1; i++) {
        scale_out_shape.push_back(x.sizes()[i]);
    }
    at::Tensor scale_out = at::empty(scale_out_shape, options.dtype(at::kFloat));

    return std::make_tuple(y_out, scale_out);
}

void indexer_compress_epilog_meta(
    at::Tensor& indexer_compress_cache,
    at::Tensor& indexer_compress_cache_scale,
    const at::Tensor& x,
    const at::Tensor& slot_mapping,
    int64_t quant_mode = 1,
    bool round_scale = true)
{
    return;
}

void kv_compress_epilog_meta(
    at::Tensor& kv_compress_cache,
    const at::Tensor& x,
    const at::Tensor& slot_mapping,
    int64_t quant_group_size,
    int64_t quant_mode,
    bool round_scale_flag,
    int64_t layout)
{
    return;
}

std::tuple<at::Tensor, at::Tensor> npu_kv_quant_sparse_attn_sharedkv_meta(
    const at::Tensor& q,
    int64_t kv_quant_mode,
    const c10::optional<at::Tensor>& ori_kv,
    const c10::optional<at::Tensor>& cmp_kv,
    const c10::optional<at::Tensor>& ori_sparse_indices,
    const c10::optional<at::Tensor>& cmp_sparse_indices,
    const c10::optional<at::Tensor>& ori_block_table,
    const c10::optional<at::Tensor>& cmp_block_table,
    const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& cu_seqlens_ori_kv,
    const c10::optional<at::Tensor>& cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& seqused_kv,
    const c10::optional<at::Tensor>& sinks,
    const c10::optional<at::Tensor>& metadata,
    int64_t tile_size,
    int64_t rope_head_dim,
    double softmax_scale,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    bool return_softmax_lse)
{
    std::string layout_q_str = std::string(layout_q);
    return construct_output_tensor(q, layout_q_str, return_softmax_lse);
}

at::Tensor npu_kv_quant_sparse_attn_sharedkv_metadata_meta(
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim,
    int64_t kv_quant_mode,
    const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& cu_seqlens_ori_kv,
    const c10::optional<at::Tensor>& cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& seqused_kv,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    int64_t ori_topk,
    int64_t cmp_topk,
    int64_t tile_size,
    int64_t rope_head_dim,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    bool has_ori_kv,
    bool has_cmp_kv,
    const c10::string_view device)
{
    constexpr int64_t OUTPUT_SIZE = 1024;
    if (cu_seqlens_q.has_value()) {
        return torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(cu_seqlens_q.value().device()));
    }
    if (cu_seqlens_ori_kv.has_value()) {
        return torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(cu_seqlens_ori_kv.value().device()));
    }
    if (cu_seqlens_cmp_kv.has_value()) {
        return torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(cu_seqlens_cmp_kv.value().device()));
    }
    if (seqused_q.has_value()) {
        return torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(seqused_q.value().device()));
    }
    if (seqused_kv.has_value()) {
        return torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(seqused_kv.value().device()));
    }

    auto device_ori = at::Device(std::string(device));
    std::string device_str = "meta";
    if (device_ori.has_index()) {
        device_str += ":";
        device_str += std::to_string(device_ori.index());
    }
    return torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(at::Device(device_str)));
}

int64_t get_type_code(at::ScalarType dst_type)
{
    switch (dst_type) {
        case at::ScalarType::Float8_e5m2:
            return 35;
        case at::ScalarType::Float8_e4m3fn:
            return 36;
        case at::ScalarType::Half:
            return 1;
        case at::ScalarType::BFloat16:
            return 27;
        default:
            TORCH_CHECK(false, "Unsupported dtype: ", dst_type);
    }
    return 0;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_swiglu_group_quant_output_tensor(
    const at::Tensor& x,
    int64_t dst_type,
    int64_t quant_mode,
    bool ue8m0_scale)
{
    constexpr int64_t SIZE = 8;
    constexpr int64_t SWIGLU_FACTOR = 2;
    constexpr int64_t PER_BLOCK_FP16 = 128;
    constexpr int64_t PER_MX_FP16 = 32;
    constexpr int64_t MX_SCALE_ALIGN_FACTOR = 2;
    constexpr int64_t GROUP_QUANT = 1;
    constexpr int64_t MX_QUANT = 2;
    constexpr int64_t FP8_QUANT = 3;

    at::SmallVector<int64_t, SIZE> y_size(x.sizes().begin(), x.sizes().end());
    TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kBFloat16,
                "x should be FLOAT16 or BFLOAT16.");
    int64_t x_last_dim = x.sizes().back();
    TORCH_CHECK(quant_mode == GROUP_QUANT || quant_mode == MX_QUANT || quant_mode == FP8_QUANT,
                "Unsupported quant mode, only support ", GROUP_QUANT, " or ", MX_QUANT, " or ", FP8_QUANT, ".");
    if (quant_mode == GROUP_QUANT || quant_mode == FP8_QUANT) {
        TORCH_CHECK(x_last_dim % 256 == 0,
                    "In group quant, the last dim of x should be divisible by 256, actual ", x_last_dim, ".");
    } else {
        TORCH_CHECK(x_last_dim % 128 == 0,
                    "In mx quant, the last dim of x should be divisible by 128, actual ", x_last_dim, ".");
    }

    y_size.back() = y_size.back() / SWIGLU_FACTOR;
    int64_t y_last_dim = y_size.back();
    auto y_dtype = dst_type == 35 ? at::kFloat8_e5m2 : at::kFloat8_e4m3fn;
    at::Tensor y = at::empty(y_size, x.options().dtype(y_dtype));

    at::SmallVector<int64_t, SIZE> scale_size(y_size.begin(), y_size.end());
    if (quant_mode == GROUP_QUANT || quant_mode == FP8_QUANT) {
        scale_size.back() = (y_last_dim + PER_BLOCK_FP16 - 1) / PER_BLOCK_FP16;
    } else if (quant_mode == MX_QUANT) {
        int64_t scale_last_dim = (y_last_dim + PER_MX_FP16 - 1) / PER_MX_FP16;
        scale_last_dim = (scale_last_dim + MX_SCALE_ALIGN_FACTOR - 1) / MX_SCALE_ALIGN_FACTOR;
        scale_size.back() = scale_last_dim;
        scale_size.push_back(MX_SCALE_ALIGN_FACTOR);
    }

    auto scale_type = at::kFloat;
    if (quant_mode == MX_QUANT || (quant_mode == FP8_QUANT && ue8m0_scale)) {
        scale_type = at::kFloat8_e8m0fnu;
    }
    at::Tensor scale = at::empty(scale_size, x.options().dtype(scale_type));
    at::Tensor y_origin = at::empty(y_size, x.options().dtype(x.dtype()));

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, scale, y_origin);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_swiglu_group_quant_meta(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& topk_weight,
    const c10::optional<at::Tensor>& group_index,
    at::ScalarType dst_type = at::ScalarType::Float8_e4m3fn,
    int64_t quant_mode = 1,
    int64_t group_size = 128,
    bool round_scale = false,
    bool ue8m0_scale = false,
    bool output_origin = false,
    int64_t group_list_type = 0,
    double clamp_value = 0.0)
{
    int64_t dst_type_code = get_type_code(dst_type);
    return construct_swiglu_group_quant_output_tensor(x, dst_type_code, quant_mode, ue8m0_scale);
}

std::tuple<at::Tensor, at::Tensor> construct_load_index_kv_cache_output_tensor(
    const at::Tensor& kv_cache,
    const at::Tensor& slot_mapping)
{
    constexpr int64_t KV_LAST_DIM = 128;
    int64_t n = slot_mapping.size(0);

    at::Tensor kv = at::empty({n, KV_LAST_DIM}, kv_cache.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor kv_scale = at::empty({n}, kv_cache.options().dtype(at::kFloat));

    return std::tuple<at::Tensor, at::Tensor>(kv, kv_scale);
}

std::tuple<at::Tensor, at::Tensor> npu_load_index_kv_cache_meta(
    const at::Tensor& kv_cache,
    const at::Tensor& slot_mapping)
{
    return construct_load_index_kv_cache_output_tensor(kv_cache, slot_mapping);
}

void indexer_compress_epilog_v2_meta(
    at::Tensor& indexer_compress_cache,
    const at::Tensor& x,
    const at::Tensor& slot_mapping,
    int64_t layout = 2)
{
    return;
}

std::tuple<at::Tensor, at::Tensor> npu_dequant_swiglu_quant_meta(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& weight_scale,
    const c10::optional<at::Tensor>& activation_scale,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& quant_scale,
    const c10::optional<at::Tensor>& quant_offset,
    const c10::optional<at::Tensor>& group_index,
    bool activate_left,
    int64_t quant_mode,
    int64_t swiglu_mode,
    double clamp_limit,
    double glu_alpha,
    double glu_bias)
{
    c10::SmallVector<int64_t, 8> y_size;
    c10::SmallVector<int64_t, 8> scale_size;
    for (int64_t i = 0; i < x.dim() - 1; ++i) {
        y_size.push_back(x.size(i));
        scale_size.push_back(x.size(i));
    }
    y_size.push_back(x.size(x.dim() - 1) / 2);

    at::Tensor y = at::empty(y_size, x.options().dtype(c10::ScalarType::Char));
    at::Tensor scale = at::empty(scale_size, x.options().dtype(c10::ScalarType::Float));
    return {y, scale};
}

at::Tensor npu_lightning_indexer_quant_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &query_dequant_scale, const at::Tensor &key_dequant_scale,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table, int64_t query_quant_mode, int64_t key_quant_mode,
    c10::string_view layout_query, c10::string_view layout_key, int64_t sparse_count, int64_t sparse_mode)
{
    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);

    const int SIZE = 8;
    const int DIM_0 = 0;
    const int DIM_1 = 1;
    const int DIM_2 = 2;
    const int DIM_3 = 3;

    at::SmallVector<int64_t, SIZE> output_size;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
            "than 0, but shape[", i, "] is ", query.size(i));
    }
    for (size_t i = 0; i < key.sizes().size(); i++) {
        TORCH_CHECK(key.size(i) > 0, "All values within key's shape should be greater "
            "than 0, but shape[", i, "] is ", key.size(i));
    }
    TORCH_CHECK(sparse_count > 0, "sparse count should be greater than 0, but now is ", sparse_count);
    int64_t keyHeadNum = (key_layout_str == "TND")? key.size(DIM_1) : key.size(DIM_2);
    if (query_layout_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), keyHeadNum, sparse_count};
    } else {
        output_size = {query.size(DIM_0), keyHeadNum, sparse_count};
    }
    at::Tensor lightning_indexer_quant_output = at::empty(output_size, query.options().dtype(at::kInt));

    return lightning_indexer_quant_output;
}

void npu_scatter_nd_update_v2_meta(
    at::Tensor& var,
    const at::Tensor& indices,
    const at::Tensor& update)
{
    return;
}

// N-gram spec decode meta
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_ngram_spec_decode_meta(
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
    at::Tensor next_token_ids = at::empty({batch_size}, token_ids.options());
    at::Tensor draft_token_ids = at::empty({batch_size, k}, token_ids.options());
    at::Tensor num_valid_draft_tokens = at::empty({batch_size}, token_ids.options());
    return std::make_tuple(token_ids, next_token_ids, draft_token_ids, num_valid_draft_tokens);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk_gated_delta_rule_fwd_h_meta(
    const at::Tensor & k,
    const at::Tensor & w,
    const at::Tensor & u,
    const c10::optional<at::Tensor> & g,
    const c10::optional<at::Tensor> & gk,
    const c10::optional<at::Tensor> & initial_state,
    c10::optional<bool> output_final_state,
    c10::optional<int64_t> chunk_size,
    c10::optional<bool> save_new_value,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices,
    c10::optional<bool> use_exp2,
    c10::optional<bool> transpose_state_layout)
{
    bool output_final_state_ = output_final_state.has_value() ? output_final_state.value() : false;
    const at::Tensor &initial_state_ = c10::value_or_else(initial_state, [] { return at::Tensor(); });
    int64_t chunk_size_ = chunk_size.has_value() ? chunk_size.value() : 64;
    const at::Tensor &g_ = c10::value_or_else(g, [] { return at::Tensor(); });
    const at::Tensor &gk_ = c10::value_or_else(gk, [] { return at::Tensor(); });

    auto k_sizes = k.sizes();
    auto u_sizes = u.sizes();
    int K = k_sizes[3];
    int B = k_sizes[0];
    int T = k_sizes[2];
    int HV = u_sizes[1];
    int V = u_sizes[3];

    int NT = 0;
    if (chunk_indices.has_value()) {
        auto chunk_indices_ref = chunk_indices.value();
        NT = chunk_indices_ref.size() / 2;
    } else {
        NT = (T + chunk_size_ - 1) / chunk_size_;
    }

    at::Tensor h_out = at::zeros({B, HV, NT, K, V}, k.options());
    at::Tensor v_new_out = at::zeros(u.sizes(), u.options());
    at::Tensor final_state_out;
    if (output_final_state_) {
        int N = cu_seqlens.has_value() ? cu_seqlens->size() - 1 : B;
        auto state_options = initial_state.has_value() ? initial_state->options() : h_out.options();
        final_state_out = at::empty({N, HV, K, V}, state_options);
    } else {
        final_state_out = at::empty({1}, k.options());
    }

    bool save_new_value_ = save_new_value.value_or(true);
    bool use_exp2_ = use_exp2.value_or(false);
    bool transpose_state_layout_ = transpose_state_layout.value_or(false);

    if (output_final_state_) {
        return std::make_tuple(h_out, v_new_out, final_state_out);
    } else {
        return std::make_tuple(h_out, v_new_out, at::Tensor());
    }
}

at::Tensor chunk_fwd_o_meta(
    const at::Tensor & q,
    const at::Tensor & k,
    const at::Tensor & v,
    const at::Tensor & h,
    double scale,
    const c10::optional<at::Tensor> & g,
    const c10::optional<at::Tensor> & g_gamma,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices,
    c10::optional<int64_t> chunk_size,
    c10::optional<bool> transpose_state_layout)
{
    at::Tensor o = at::zeros(v.sizes(), v.options());
    int64_t chunk_size_ = chunk_size.has_value() ? chunk_size.value() : 64;
    const at::Tensor &g_ = c10::value_or_else(g, [] { return at::Tensor(); });
    (void)g_gamma;
    (void)transpose_state_layout;

    return o;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> store_kv_block_pre(
    const at::Tensor &slot_mapping_npu,
    at::IntArrayRef slot_mapping_list,
    int64_t block_size)
{
    auto s_size = slot_mapping_npu.sizes();
    at::Tensor group_len = at::empty({s_size[0]}, slot_mapping_npu.options());
    at::Tensor group_key_idx = at::empty({s_size[0]}, slot_mapping_npu.options());
    at::Tensor group_key_cache_idx = at::empty({s_size[0]}, slot_mapping_npu.options());
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(group_len, group_key_idx, group_key_cache_idx);

}

void store_kv_block(
    const at::Tensor &key_in,
    const at::Tensor &key_cache_in,
    const at::Tensor &group_len,
    const at::Tensor &group_key_idx,
    const at::Tensor &group_key_cache_idx,
    int64_t block_size)
{
    return;

} 

} // namespace meta
} // namespace vllm_ascend

// Register the meta implementations of the custom kernels for symbolic tracing, this will also
// the custom kernel been captured into aclgraph
#ifdef ASCEND_PLATFORM_310P
// Pybind on Ascend 310P
namespace {
TORCH_LIBRARY_IMPL_EXPAND(CONCAT(_C, _ascend), Meta, ops) {
    // causal_conv1d_310
    ops.impl("npu_causal_conv1d_310", &vllm_ascend::meta::npu_causal_conv1d_310_meta);
    // npu_recurrent_gated_delta_rule_310
    ops.impl("npu_recurrent_gated_delta_rule_310", &vllm_ascend::meta::npu_recurrent_gated_delta_rule_310_meta);
    // chunk_gated_delta_rule_fwd_h
    ops.impl("chunk_gated_delta_rule_fwd_h", &vllm_ascend::meta::chunk_gated_delta_rule_fwd_h_meta);
    // chunk_fwd_o
    ops.impl("chunk_fwd_o", &vllm_ascend::meta::chunk_fwd_o_meta);
}
}
#else
// Pybind on other platform
namespace {
TORCH_LIBRARY_IMPL_EXPAND(CONCAT(_C, _ascend), Meta, ops) {
    //Gemma rmsnorm meta implementation
    ops.impl("npu_gemma_rms_norm", &vllm_ascend::meta::npu_gemma_rms_norm_meta);
    // recurrent_gated_delta_rule meta implementation
    ops.impl("npu_recurrent_gated_delta_rule", &vllm_ascend::meta::npu_recurrent_gated_delta_rule_meta);
    // Launch host print from device
    ops.impl("device_print", &vllm_ascend::meta::device_print_meta);
    // launch host print from device for tensors
    ops.impl("device_print_tensor", &vllm_ascend::meta::device_print_tensor_meta);
#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
    // Direct kernel meta implementations
    ops.impl("get_masked_input_and_mask", &vllm_ascend::meta::get_masked_input_and_mask_meta);
    // Bgmv expand
    ops.impl("bgmv_expand", &vllm_ascend::meta::bgmv_expand_meta);
    // Sgmv expand
    ops.impl("sgmv_expand", &vllm_ascend::meta::sgmv_expand_meta);
    // MLA preprocess
    ops.impl("mla_preprocess", &vllm_ascend::meta::mla_preprocess);
    // batch_matmul_transpose
    ops.impl("batch_matmul_transpose", &vllm_ascend::meta::batch_matmul_transpose);
#endif
    // grouped_matmul_swiglu_quant_weight_nz meta implementation
    ops.impl("grouped_matmul_swiglu_quant_weight_nz", &vllm_ascend::meta::grouped_matmul_swiglu_quant);
    // grouped_matmul_swiglu_quant meta implementation
    ops.impl("grouped_matmul_swiglu_quant", &vllm_ascend::meta::grouped_matmul_swiglu_quant);
    // Grouped matmul swiglu quant weight nz tensor list
    ops.impl("grouped_matmul_swiglu_quant_weight_nz_tensor_list", &vllm_ascend::meta::grouped_matmul_swiglu_quant_weight_nz_tensor_list_meta);
    // Grouped matmul swiglu quant v2
    ops.impl("grouped_matmul_swiglu_quant_v2", &vllm_ascend::meta::grouped_matmul_swiglu_quant_v2_meta);
    // dispatch_gmm_combine_decode meta implementation
    ops.impl("dispatch_gmm_combine_decode", &vllm_ascend::meta::dispatch_gmm_combine_decode_meta);
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
    // Add_Rms_Norm_Bias
    ops.impl("npu_add_rms_norm_bias", &vllm_ascend::meta::npu_add_rms_norm_bias_meta);
    // transpose_kv_cache_by_block
    ops.impl("transpose_kv_cache_by_block", &vllm_ascend::meta::transpose_kv_cache_by_block_meta);
    // hamming_dist_top_k
    ops.impl("npu_hamming_dist_top_k", &vllm_ascend::meta::npu_hamming_dist_top_k_meta);
    // reshape_and_cache_bnsd
    ops.impl("npu_reshape_and_cache_bnsd", &vllm_ascend::meta::npu_reshape_and_cache_bnsd_meta);
    // npu_sign_bits_pack
    ops.impl("npu_sign_bits_pack", &vllm_ascend::meta::npu_sign_bits_pack_meta);
    // CopyAndExpandEagleInputs
    ops.impl("npu_copy_and_expand_eagle_inputs", &vllm_ascend::meta::npu_copy_and_expand_eagle_inputs_meta);
    // causal_conv1d_fn
    ops.impl("npu_causal_conv1d_custom", &vllm_ascend::meta::npu_causal_conv1d_custom_meta);
    // moe_grouped_matmul
    ops.impl("moe_grouped_matmul", &vllm_ascend::meta::moe_grouped_matmul_meta);
    ops.impl("moe_gating_top_k_hash", &vllm_ascend::meta::moe_gating_top_k_hash_meta);
    ops.impl("compressor", &vllm_ascend::meta::compressor_meta);
    ops.impl("npu_quant_lightning_indexer", &vllm_ascend::meta::npu_quant_lightning_indexer_meta);
    ops.impl("npu_quant_lightning_indexer_metadata", &vllm_ascend::meta::npu_quant_lightning_indexer_metadata_meta);
    ops.impl("npu_sparse_attn_sharedkv", &vllm_ascend::meta::npu_sparse_attn_sharedkv_meta);
    ops.impl("npu_sparse_attn_sharedkv_metadata", &vllm_ascend::meta::npu_sparse_attn_sharedkv_metadata_meta);
    ops.impl("npu_hc_post", &vllm_ascend::meta::npu_hc_post_meta);
    ops.impl("npu_hc_pre", &vllm_ascend::meta::npu_hc_pre_meta);
    ops.impl("npu_hc_pre_v2", &vllm_ascend::meta::npu_hc_pre_meta);
    ops.impl("npu_hc_pre_inv_rms", &vllm_ascend::meta::npu_hc_pre_inv_rms_meta);
    ops.impl("npu_hc_pre_sinkhorn", &vllm_ascend::meta::npu_hc_pre_sinkhorn_meta);
    ops.impl("inplace_partial_rotary_mul", &vllm_ascend::meta::inplace_partial_rotary_mul_meta);
    ops.impl("npu_rms_norm_dynamic_quant", &vllm_ascend::meta::npu_rms_norm_dynamic_quant_meta);
    ops.impl("indexer_compress_epilog", &vllm_ascend::meta::indexer_compress_epilog_meta);
    ops.impl("kv_compress_epilog", &vllm_ascend::meta::kv_compress_epilog_meta);
    ops.impl("npu_kv_quant_sparse_attn_sharedkv", &vllm_ascend::meta::npu_kv_quant_sparse_attn_sharedkv_meta);
    ops.impl("npu_kv_quant_sparse_attn_sharedkv_metadata",
             &vllm_ascend::meta::npu_kv_quant_sparse_attn_sharedkv_metadata_meta);
    ops.impl("npu_swiglu_group_quant", &vllm_ascend::meta::npu_swiglu_group_quant_meta);
    ops.impl("npu_load_index_kv_cache", &vllm_ascend::meta::npu_load_index_kv_cache_meta);
    ops.impl("indexer_compress_epilog_v2", &vllm_ascend::meta::indexer_compress_epilog_v2_meta);
    ops.impl("npu_dequant_swiglu_quant", &vllm_ascend::meta::npu_dequant_swiglu_quant_meta);
    ops.impl("npu_scatter_nd_update_v2", &vllm_ascend::meta::npu_scatter_nd_update_v2_meta);
    // Lightning indexer quant
    ops.impl("npu_lightning_indexer_quant", &vllm_ascend::meta::npu_lightning_indexer_quant_meta);
    // N-gram spec decode
    ops.impl("npu_ngram_spec_decode", &vllm_ascend::meta::npu_ngram_spec_decode_meta);
    // chunk_gated_delta_rule_fwd_h
    ops.impl("chunk_gated_delta_rule_fwd_h", &vllm_ascend::meta::chunk_gated_delta_rule_fwd_h_meta);
    // chunk_fwd_o
    ops.impl("chunk_fwd_o", &vllm_ascend::meta::chunk_fwd_o_meta);
     // store_kv_block
    ops.impl("store_kv_block_pre", &vllm_ascend::meta::store_kv_block_pre);
    ops.impl("store_kv_block", &vllm_ascend::meta::store_kv_block);
    // npu_fused_gdn_gating
    ops.impl("npu_fused_gdn_gating", &vllm_ascend::meta::npu_fused_gdn_gating_meta);
}
}
#endif
