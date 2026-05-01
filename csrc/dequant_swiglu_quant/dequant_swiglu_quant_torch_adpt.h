
std::tuple<at::Tensor, at::Tensor> dequant_swiglu_quant(
    const at::Tensor& x, const c10::optional<at::Tensor>& weight_scale,
    const c10::optional<at::Tensor>& activation_scale, const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& quant_scale, const c10::optional<at::Tensor>& quant_offset,
    const c10::optional<at::Tensor>& group_index, bool activate_left, int64_t quant_mode, int64_t swiglu_mode,
    double clamp_limit, double glu_alpha, double glu_bias)
{
    TORCH_CHECK(x.dim() > 1, "x dim should larger than 1", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(quant_mode == 0 || quant_mode == 1, "quant_mode only support 0 or 1, but got", quant_mode,
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(swiglu_mode == 0 || swiglu_mode == 1, "swiglu_mode only support 0 or 1, but got ", swiglu_mode,
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(std::isfinite(clamp_limit) && clamp_limit > 0.0, "clamp_limit should be positive finite",
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(std::isfinite(glu_alpha), "glu_alpha should be finite", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(std::isfinite(glu_bias),  "glu_bias should be finite",  OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x.size(x.dim() - 1) % 2 == 0, "x last dim should be even", OPS_ERROR(ErrCode::PARAM));

    at::SmallVector<int64_t, op_infer::SIZE> y_size;
    at::SmallVector<int64_t, op_infer::SIZE> scale_size;
    for (int i = 0; i < x.dim() - 1; i++) {
        y_size.push_back(x.size(i));
        scale_size.push_back(x.size(i));
    }
    auto last_dim = x.size(x.dim() - 1) / 2;
    y_size.push_back(last_dim);

    at::Tensor y = npu_preparation::apply_tensor_without_format(y_size, c10::dtype(c10::ScalarType::Char));
    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));

    std::string quant_mode_str = "static";
    if (quant_mode == 1) {
        quant_mode_str = "dynamic";
    }
    char* quant_mode_ptr = const_cast<char*>(quant_mode_str.c_str());

    const at::Tensor& quant_scale_opt = c10::value_or_else(quant_scale, [] { return at::Tensor(); });
    const at::Tensor& quant_offset_opt = c10::value_or_else(quant_offset, [] { return at::Tensor(); });
    const at::Tensor& group_index_opt = c10::value_or_else(group_index, [] { return at::Tensor(); });

    const at::Tensor& weight_scale_opt = c10::value_or_else(weight_scale, [] { return at::Tensor(); });
    const at::Tensor& activate_scale_opt = c10::value_or_else(activation_scale, [] { return at::Tensor(); });
    const at::Tensor& bias_opt = c10::value_or_else(bias, [] { return at::Tensor(); });

    static const bool is_v2_available = check_aclnn_kernel_available("aclnnDequantSwigluQuantV2");

    if (swiglu_mode == 0 && !is_v2_available) {
        EXEC_NPU_CMD(aclnnDequantSwigluQuant, x, weight_scale_opt, activate_scale_opt, bias_opt, quant_scale_opt,
                     quant_offset_opt, group_index_opt, activate_left, quant_mode_ptr, y, scale);
    } else {
        int64_t dst_type = 2;
        char* round_mode = "rint";
        int64_t activate_dim = -1;
        EXEC_NPU_CMD(aclnnDequantSwigluQuantV2, x, weight_scale_opt, activate_scale_opt, bias_opt, quant_scale_opt,
                     quant_offset_opt, group_index_opt, activate_left, quant_mode_ptr, dst_type, round_mode,
                     activate_dim, swiglu_mode, clamp_limit, glu_alpha, glu_bias, y, scale);
    }

    return std::tie(y, scale);
}