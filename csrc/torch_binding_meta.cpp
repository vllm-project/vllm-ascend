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

std::tuple<at::Tensor, at::Tensor, at::Tensor> grouped_matmul_swiglu_quant_weight_nz(
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
    double swiglu_limit,
    at::Tensor& out
) {
    return out;
}

at::Tensor npu_lightning_indexer_custom_meta(
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
        output_size = {query.size(DIM_0), query.size(1), key.size(2), sparse_count};
    } else {
        int n_dim_index = 0;
        n_dim_index = (key_layout_str == "TND") ? 1 : 2;
        output_size = {query.size(DIM_0), key.size(n_dim_index), sparse_count};
    }
    // construct the output tensor
    at::Tensor lightning_indexer_custom_output = at::empty(output_size, query.options().dtype(at::kInt));
    return lightning_indexer_custom_output;
}

at::Tensor npu_sparse_flash_attention_custom_meta(
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

    // bias: optional, 1D [E], dtype same as x
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

    // input_ids: optional, int32/int64; numel must match rows（最稳的约束）
    if (input_ids_opt.has_value() && input_ids_opt->defined()) {
        const auto& input_ids = *input_ids_opt;
        TORCH_CHECK(input_ids.scalar_type() == at::kInt || input_ids.scalar_type() == at::kLong,
                    "input_ids dtype must be int32 or int64, but got ", input_ids.scalar_type());
        TORCH_CHECK(input_ids.numel() == rows,
                    "input_ids.numel() must equal x.size(0). input_ids.numel()=",
                    input_ids.numel(), ", rows=", rows);
    }

    // tid2eid: optional, int32/int64;
    if (tid2eid_opt.has_value() && tid2eid_opt->defined()) {
        const auto& tid2eid = *tid2eid_opt;
        TORCH_CHECK(tid2eid.scalar_type() == at::kInt || tid2eid.scalar_type() == at::kLong,
                    "tid2eid dtype must be int32 or int64, but got ", tid2eid.scalar_type());
        TORCH_CHECK(tid2eid.dim() >= 1, "tid2eid must have dim>=1, but got dim=", tid2eid.dim());
    }


    // outputs:
    // y: [rows, k] dtype same as x
    // expert_idx: [rows, k] int32
    // out: [rows, expert_num] float32（OpDef 固定输出 float）
    at::Tensor y = at::empty({rows, k}, x.options());
    at::Tensor expert_idx = at::empty({rows, k}, x.options().dtype(at::kInt));
    at::Tensor out = at::empty({rows, expert_num}, x.options().dtype(at::kFloat));

    return {y, expert_idx, out};
}



std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> 
construct_compressor_output_tensor(const at::Tensor &x, const at::Tensor &norm_weight, const at::Tensor &rope_sin, 
                                   int64_t cmp_ratio, int64_t coff, bool enable_grad)
{
    constexpr int32_t DIM_1 = 1;
    constexpr int32_t DIM_2 = 2;
    constexpr int32_t DIM_3 = 3;
    constexpr int32_t VALUE_0 = 0;
    auto x_dim = x.dim();
    at::SmallVector<int64_t, 8> cmp_kv_size;
    at::SmallVector<int64_t, 8> wkv_proj_size;
    at::SmallVector<int64_t, 8> softmax_res_size;
    at::SmallVector<int64_t, 8> norm_x_size;
    at::SmallVector<int64_t, 8> norm_rstd_size;
    at::Tensor cmp_kv;
    at::Tensor wkv_proj;
    at::Tensor softmax_res;
    at::Tensor norm_x;
    at::Tensor norm_rstd;
    auto cmp_s = 0;
    if (x_dim == DIM_3) {
        cmp_s = (x.size(1) + cmp_ratio - 1) / cmp_ratio;
        cmp_kv_size = {x.size(0), cmp_s, norm_weight.size(0)};
        if (enable_grad) {
            wkv_proj_size = {x.size(0), x.size(1), coff * norm_weight.size(0)};
            softmax_res_size = {x.size(0), cmp_s, coff * cmp_ratio, norm_weight.size(0)};
            norm_x_size = {x.size(0), cmp_s, norm_weight.size(0)};
            norm_rstd_size = {x.size(0), cmp_s};
        }
    } else {
        cmp_s = rope_sin.size(0);
        cmp_kv_size = {cmp_s, norm_weight.size(0)};
        if (enable_grad) {
            wkv_proj_size = {x.size(0), coff * norm_weight.size(0)};
            softmax_res_size = {cmp_s, coff * cmp_ratio, norm_weight.size(0)};
            norm_x_size = {cmp_s, norm_weight.size(0)};
            norm_rstd_size = {cmp_s};
        }
    }

    cmp_kv = at::empty(cmp_kv_size, x.options().dtype(x.dtype()));
    if (enable_grad) {
        wkv_proj = at::empty(wkv_proj_size, x.options().dtype(x.dtype()));
        softmax_res = at::empty(softmax_res_size, x.options().dtype(x.dtype()));
        norm_x = at::empty(norm_x_size, x.options().dtype(x.dtype()));
        norm_rstd = at::empty(norm_rstd_size, x.options().dtype(x.dtype()));
    } else {
        wkv_proj = at::empty({0}, x.options().dtype(x.dtype()));
        softmax_res = at::empty({0}, x.options().dtype(x.dtype()));
        norm_x = at::empty({0}, x.options().dtype(x.dtype()));
        norm_rstd = at::empty({0}, x.options().dtype(x.dtype()));
    }

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
        cmp_kv, wkv_proj, softmax_res, norm_x, norm_rstd);
}


// 为META设备实现前向接口
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
compressor_meta(
    const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate, 
    at::Tensor &kv_state, at::Tensor &score_state, 
    const at::Tensor &ape, const at::Tensor &norm_weight, 
    const at::Tensor &rope_sin, const at::Tensor &rope_cos, 
    const c10::optional<at::Tensor> &kv_block_table, const c10::optional<at::Tensor> &score_block_table, 
    const c10::optional<at::Tensor> &cu_seqlens, const c10::optional<at::Tensor> &seqused, 
    const c10::optional<at::Tensor> &start_pos, int64_t rope_head_dim, int64_t cmp_ratio, 
    int64_t coff, double norm_eps, int64_t rotary_mode, bool enable_grad)
{
    // construct the output tensor
    auto x_dim = x.dim();
    auto norm_weight_dim = norm_weight.dim();
    auto rope_sin_dim = rope_sin.dim();

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> output = construct_compressor_output_tensor(
        x, norm_weight, rope_sin, cmp_ratio, coff, enable_grad);

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
    // construct the output tensor
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
    // construct the output tensor
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

// 工具函数，推导输出hc_pre_inv_rms_shape
at::Tensor construct_hc_pre_rsqrt_output_tensor(const at::Tensor& x, float epsilon=1e-6)
{
    constexpr int64_t SIZE = 8;
    // Check input tensor validity
    TORCH_CHECK(epsilon >= 0, "epsilon should be greater than 0.");

    // Get input tensor options
    auto options = x.options();

    // Construct yOut output tensor
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
    // construct the output tensor
    auto output_tensors = construct_hc_pre_output_tensor(x, hc_mult);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor post = std::get<1>(output_tensors);
    at::Tensor comb_frag = std::get<2>(output_tensors);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
}

at::Tensor construct_hc_pre_inv_rms_output_tensor(const at::Tensor& x, float epsilon=1e-20)
{
    constexpr int64_t SIZE = 8;
    // Check input tensor validity
    TORCH_CHECK(epsilon >= 0, "epsilon should be greater than 0.");

    // Get input tensor options
    auto options = x.options();

    // Construct yOut output tensor
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

    // construct the output tensors
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
    // construct the output tensor
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
    // construct the output tensors
    at::Tensor y_out = at::empty_like(x);
    auto options = x.options();
    c10::SmallVector<int64_t, SIZE> scale_out_shape;
    for (size_t i = 0; i < x.sizes().size() - 1; i++) {
        scale_out_shape.push_back(x.sizes()[i]);
    }
    at::Tensor scale_out = at::empty(scale_out_shape, options.dtype(at::kFloat));

    return std::make_tuple(y_out, scale_out);
}

} // namespace meta
} // namespace vllm_ascend

namespace {
// Register the meta implementations of the custom kernels for symbolic tracing, this will also
// the custom kernel been captured into aclgraph
TORCH_LIBRARY_IMPL_EXPAND(CONCAT(_C, _ascend), Meta, ops) {

    // Masked input and mask meta implementation
    ops.impl("get_masked_input_and_mask", &vllm_ascend::meta::get_masked_input_and_mask_meta);
    // Bgmv expand
    ops.impl("bgmv_expand", &vllm_ascend::meta::bgmv_expand_meta);
    // Sgmv expand
    ops.impl("sgmv_expand", &vllm_ascend::meta::sgmv_expand_meta);
    // MLA preprocess
    ops.impl("mla_preprocess", &vllm_ascend::meta::mla_preprocess);
    // grouped_matmul_swiglu_quant_weight_nz meta implementation
    ops.impl("grouped_matmul_swiglu_quant_weight_nz", &vllm_ascend::meta::grouped_matmul_swiglu_quant_weight_nz);
    // Grouped matmul swiglu quant weight nz tensor list
    ops.impl("grouped_matmul_swiglu_quant_weight_nz_tensor_list", &vllm_ascend::meta::grouped_matmul_swiglu_quant_weight_nz_tensor_list_meta);
    // dispatch_gmm_combine_decode meta implementation
    ops.impl("dispatch_gmm_combine_decode", &vllm_ascend::meta::dispatch_gmm_combine_decode_meta);
    // batch_matmul_transpose
    ops.impl("batch_matmul_transpose", &vllm_ascend::meta::batch_matmul_transpose);
    // Lightning indexer
    ops.impl("npu_lightning_indexer_custom", &vllm_ascend::meta::npu_lightning_indexer_custom_meta);
    // Sparse flash attention
    ops.impl("npu_sparse_flash_attention_custom", &vllm_ascend::meta::npu_sparse_flash_attention_custom_meta);
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
    // Moe_gating_top_k_hash
    ops.impl("moe_gating_top_k_hash", &vllm_ascend::meta::moe_gating_top_k_hash_meta);
    // compressor
    ops.impl("compressor", &vllm_ascend::meta::compressor_meta);
    // npu_quant_lightning_indexer
    ops.impl("npu_quant_lightning_indexer", &vllm_ascend::meta::npu_quant_lightning_indexer_meta);
    // npu_quant_lightning_indexer_metadata
    ops.impl("npu_quant_lightning_indexer_metadata", &vllm_ascend::meta::npu_quant_lightning_indexer_metadata_meta);
    // npu_sparse_attn_sharedkv
    ops.impl("npu_sparse_attn_sharedkv", &vllm_ascend::meta::npu_sparse_attn_sharedkv_meta);
    // npu_sparse_attn_sharedkv_metadata
    ops.impl("npu_sparse_attn_sharedkv_metadata", &vllm_ascend::meta::npu_sparse_attn_sharedkv_metadata_meta);
    // npu_quant_lightning_indexer_metadata
    ops.impl("npu_quant_lightning_indexer_metadata", &vllm_ascend::meta::npu_quant_lightning_indexer_metadata_meta);
    // npu_hc_post
    ops.impl("npu_hc_post", &vllm_ascend::meta::npu_hc_post_meta);
    // npu_hc_pre
    ops.impl("npu_hc_pre", &vllm_ascend::meta::npu_hc_pre_meta);
    // npu_hc_pre_inv_rms
    ops.impl("npu_hc_pre_inv_rms", &vllm_ascend::meta::npu_hc_pre_inv_rms_meta);
    // npu_hc_pre_inv_rms
    ops.impl("npu_hc_pre_sinkhorn", &vllm_ascend::meta::npu_hc_pre_sinkhorn_meta);
    // inplace_partial_rotary_mul
    ops.impl("inplace_partial_rotary_mul", &vllm_ascend::meta::inplace_partial_rotary_mul_meta);
    // npu_rms_norm_dynamic_quant
    ops.impl("npu_rms_norm_dynamic_quant", &vllm_ascend::meta::npu_rms_norm_dynamic_quant_meta);
}
}
