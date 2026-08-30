/**
 * Torch binding for the AICore FIA v2 sink operator, driven by GM metadata.
 * Adapted from the omni-ops npu_fused_infer_attention_sink_v2 binding:
 *   - unused quantization, PSE, and sink parameters are omitted
 *   - returns the (attention_out, softmax_lse) tuple
 *   - calls aclnnFusedInferAttentionScoreV2SinkV3 with device sequence lengths
 *     and metadata tensors
 */
#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace custom {
const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t DIM_3 = 3;
const static int64_t PA_BBH_DIMS = 3;
const static int64_t PA_BNBD_DIMS = 4;
using namespace at_npu::native;

// 推导输出 shape（TND 族 + BSH/BSND/BNSD 常用路径；quant 分支已裁剪——schema 不含量化参数）
static std::tuple<at::Tensor, at::Tensor> construct_output_tensor(
    const at::Tensor &query,
    const at::Tensor &value,
    const std::string &input_layout_str,
    const c10::optional<at::Tensor> &block_table,
    int64_t num_query_heads,
    int64_t num_key_value_heads,
    bool return_softmax_lse)
{
    at::Tensor output;
    c10::SymInt batchSize = 1;
    c10::SymInt qsSize = 1;
    if (input_layout_str == "TND_NTD") {
        // out [N1, T, D_v] (NTD)
        output = at::empty_symint({query.sym_size(DIM_1), query.sym_size(DIM_0), query.sym_size(DIM_2)},
                                  query.options().dtype(query.dtype()));
    } else if (input_layout_str == "NTD_TND" || input_layout_str == "NTD") {
        const int64_t kv_dim = value.dim();
        if (block_table.has_value() && kv_dim == PA_BNBD_DIMS) {
            output = at::empty_symint({query.sym_size(DIM_1), query.sym_size(DIM_0), value.sym_size(DIM_3)},
                                      query.options().dtype(query.dtype()));
        } else {
            output = at::empty_symint({query.sym_size(DIM_1), query.sym_size(DIM_0), value.sym_size(DIM_2)},
                                      query.options().dtype(query.dtype()));
        }
    } else if (input_layout_str == "TND") {
        const int64_t kv_dim = value.dim();
        if (block_table.has_value()) {
            if (kv_dim == PA_BBH_DIMS) {
                output = at::empty_symint(
                    {query.sym_size(DIM_0), query.sym_size(DIM_1), value.sym_size(DIM_2) / num_key_value_heads},
                    query.options().dtype(query.dtype()));
            } else if (kv_dim == PA_BNBD_DIMS) {
                output = at::empty_symint({query.sym_size(DIM_0), query.sym_size(DIM_1), value.sym_size(DIM_3)},
                                          query.options().dtype(query.dtype()));
            } else {
                output = at::empty_symint({query.sym_size(DIM_0), query.sym_size(DIM_1), value.sym_size(DIM_2)},
                                          query.options().dtype(query.dtype()));
            }
        } else {
            output = at::empty_symint({query.sym_size(DIM_0), query.sym_size(DIM_1), value.sym_size(DIM_2)},
                                      query.options().dtype(query.dtype()));
        }
    } else if (input_layout_str == "BSH" || input_layout_str == "BSND") {
        batchSize = query.sym_size(DIM_0);
        qsSize = query.sym_size(DIM_1);
        output = at::empty_symint(query.sym_sizes(), query.options().dtype(query.dtype()));
    } else if (input_layout_str == "BNSD") {
        batchSize = query.sym_size(DIM_0);
        qsSize = query.sym_size(DIM_2);
        output = at::empty_symint(query.sym_sizes(), query.options().dtype(query.dtype()));
    } else {
        TORCH_CHECK(false, "fia_v2_sink: unsupported input_layout ", input_layout_str,
                    " (supported: TND, TND_NTD, NTD, NTD_TND, BSH, BSND, BNSD)");
    }

    auto options = at::TensorOptions().dtype(c10::ScalarType::Float).device(output.device());
    at::Tensor softmax_lse;
    if (input_layout_str == "TND" || input_layout_str == "TND_NTD") {
        if (block_table.has_value()) {
            if (query.sym_size(DIM_2) == 0) {
                softmax_lse = at::empty_symint({query.sym_size(DIM_0), num_query_heads, 0}, options);
            } else {
                softmax_lse = at::empty_symint({query.sym_size(DIM_0), num_query_heads, 1}, options);
            }
        } else {
            softmax_lse = at::empty_symint({query.sym_size(DIM_0), query.sym_size(DIM_1), 1}, options);
        }
    } else if (input_layout_str == "NTD" || input_layout_str == "NTD_TND") {
        if (block_table.has_value()) {
            softmax_lse = at::empty_symint({query.sym_size(DIM_1), query.sym_size(DIM_0), 1}, options);
        } else {
            softmax_lse = at::empty_symint({query.sym_size(DIM_1), query.sym_size(DIM_0), 1}, options);
        }
    } else {
        softmax_lse = at::empty_symint({batchSize, num_query_heads, qsSize, 1}, options);
    }
    if (!return_softmax_lse) {
        softmax_lse = at::empty_symint({0}, options);
    }
    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

// 为NPU设备实现前向接口
std::tuple<at::Tensor, at::Tensor> npu_fused_infer_attention_score_v2_sink_npu(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &actual_seq_qlen,
    const c10::optional<at::Tensor> &actual_seq_kvlen,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &meta_data,
    int64_t num_query_heads,
    int64_t num_key_value_heads,
    double softmax_scale,
    int64_t pre_tokens,
    int64_t next_tokens,
    c10::string_view input_layout,
    int64_t sparse_mode,
    int64_t block_size,
    int64_t inner_precise,
    bool return_softmax_lse)
{
    std::string input_layout_str = std::string(input_layout);

    auto fia_output = custom::construct_output_tensor(query, value, input_layout_str, block_table,
                                                      num_query_heads, num_key_value_heads,
                                                      return_softmax_lse);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);

    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    // Unused slots in the non-quantized path are passed as undefined tensors.
    at::Tensor dequant_scale1;
    at::Tensor quant_scale1;
    at::Tensor dequant_scale2;
    at::Tensor quant_scale_out;
    at::Tensor quant_offset_out;
    at::Tensor antiquant_scale;
    at::Tensor antiquant_offset;
    at::Tensor query_padding_size;
    at::Tensor kv_padding_size;
    at::Tensor dequant_scale_key;
    at::Tensor dequant_offset_key;
    at::Tensor dequant_scale_value;
    at::Tensor dequant_offset_value;
    at::Tensor dequant_scale_key_rope;
    at::Tensor dequant_scale_query;
    at::Tensor key_rope_antiquant_scale;
    at::Tensor key_sink;
    at::Tensor key_rope_sink;
    at::Tensor value_sink;
    // softmax_max/sum 不暴露（适配点 A），以空 tensor 传入
    at::Tensor softmax_max;
    at::Tensor softmax_sum;
    const bool softmax_max_sum_flag = false;
    const int64_t antiquant_mode = 0;
    const int64_t key_quant_mode = 0;
    const int64_t value_quant_mode = 0;
    const int64_t query_quant_mode = 0;
    const int64_t sink_number = 0;
    const bool batch_invariant = false;

    at::TensorList valueTensors = value;
    at::TensorList keyTensors = key;

    EXEC_NPU_CMD_V1(aclnnFusedInferAttentionScoreV2SinkV3,
                    query,
                    keyTensors,
                    valueTensors,
                    pse_shift,
                    atten_mask,
                    actual_seq_qlen,
                    actual_seq_kvlen,
                    dequant_scale1,
                    quant_scale1,
                    dequant_scale2,
                    quant_scale_out,
                    quant_offset_out,
                    antiquant_scale,
                    antiquant_offset,
                    block_table,
                    query_padding_size,
                    kv_padding_size,
                    dequant_scale_key,
                    dequant_offset_key,
                    dequant_scale_value,
                    dequant_offset_value,
                    query_rope,
                    key_rope,
                    key_rope_antiquant_scale,
                    dequant_scale_query,
                    meta_data,
                    key_sink,
                    key_rope_sink,
                    value_sink,
                    num_query_heads,
                    softmax_scale,
                    pre_tokens,
                    next_tokens,
                    input_layout_ptr,
                    num_key_value_heads,
                    sparse_mode,
                    inner_precise,
                    block_size,
                    antiquant_mode,
                    return_softmax_lse,
                    key_quant_mode,
                    value_quant_mode,
                    query_quant_mode,
                    sink_number,
                    batch_invariant,
                    softmax_max_sum_flag,
                    output,
                    softmax_lse,
                    softmax_max,
                    softmax_sum);
    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

// 为META设备实现前向接口
std::tuple<at::Tensor, at::Tensor> npu_fused_infer_attention_score_v2_sink_meta(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &actual_seq_qlen,
    const c10::optional<at::Tensor> &actual_seq_kvlen,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &meta_data,
    int64_t num_query_heads,
    int64_t num_key_value_heads,
    double softmax_scale,
    int64_t pre_tokens,
    int64_t next_tokens,
    c10::string_view input_layout,
    int64_t sparse_mode,
    int64_t block_size,
    int64_t inner_precise,
    bool return_softmax_lse)
{
    std::string input_layout_str = std::string(input_layout);
    return custom::construct_output_tensor(query, value, input_layout_str, block_table, num_query_heads,
                                           num_key_value_heads, return_softmax_lse);
}

} // namespace custom

TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, m)
{
    m.impl("npu_fused_infer_attention_score_v2_sink", &custom::npu_fused_infer_attention_score_v2_sink_npu);
}

TORCH_LIBRARY_IMPL(_C_ascend, Meta, m)
{
    m.impl("npu_fused_infer_attention_score_v2_sink", &custom::npu_fused_infer_attention_score_v2_sink_meta);
}
