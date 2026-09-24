// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace vllm_ascend {
std::tuple<at::Tensor, at::Tensor, at::Tensor> attn_res_fwd_fused(
    const at::Tensor& prefix, const c10::optional<at::Tensor>& addend,
    at::Tensor blocks, const at::Tensor& proj, const at::Tensor& norm,
    double eps, int64_t valid, const c10::optional<at::Tensor>& output_norm,
    double output_eps, int64_t write_idx, bool save_materialized, bool mix)
{
    TORCH_CHECK(prefix.dim() == 2 && blocks.dim() == 3 &&
                blocks.size(0) == prefix.size(0) && blocks.size(2) == prefix.size(1),
                "attn_res_fwd.fused: invalid prefix/bank shape");
    TORCH_CHECK(proj.dim() == 2 && proj.size(0) == 1 && proj.size(1) == prefix.size(1) &&
                norm.dim() == 1 && norm.size(0) == prefix.size(1), "attn_res_fwd.fused: invalid weights");
    TORCH_CHECK(valid >= 0 && valid <= blocks.size(1) && write_idx >= -1 &&
                write_idx < blocks.size(1) && (write_idx < 0 || write_idx >= valid),
                "attn_res_fwd.fused: invalid bank extent/write slot");
    const at::Tensor* inputs[] = {&prefix, &blocks, &proj, &norm};
    for (const at::Tensor* tensor : inputs) {
        TORCH_CHECK(tensor->scalar_type() == at::kBFloat16 && tensor->device() == prefix.device(),
                    "attn_res_fwd.fused: inputs must be BF16 on the same device");
    }
    TORCH_CHECK(prefix.is_contiguous() && proj.is_contiguous() && norm.is_contiguous() &&
                blocks.stride(2) == 1 && blocks.stride(1) == prefix.size(1) &&
                blocks.stride(0) >= blocks.size(1) * prefix.size(1),
                "attn_res_fwd.fused: prefix/weights/individual bank rows must be contiguous");
    if (addend.has_value()) {
        TORCH_CHECK(addend->sizes() == prefix.sizes() && addend->dtype() == prefix.dtype() &&
                    addend->device() == prefix.device() && addend->is_contiguous(),
                    "attn_res_fwd.fused: invalid addend");
    }
    if (output_norm.has_value()) {
        TORCH_CHECK(output_norm->sizes() == norm.sizes() && output_norm->dtype() == prefix.dtype() &&
                    output_norm->device() == prefix.device() && output_norm->is_contiguous() && output_eps > 0,
                    "attn_res_fwd.fused: invalid output norm");
    }
    auto output = at::empty_like(prefix);
    auto raw_prefix = addend.has_value() ? at::empty_like(prefix) : prefix;
    auto materialized = save_materialized ? at::empty_like(prefix) : output;
    auto add_input = addend.value_or(prefix);
    auto output_norm_input = output_norm.value_or(norm);
    const auto bank_token_stride = blocks.stride(0);
    const double fused_output_eps = output_norm.has_value() ? output_eps : 0.0;
    const bool fuse_add = addend.has_value();
    EXEC_NPU_CMD(aclnnAttnResFwdFused, prefix, blocks, proj, norm,
        add_input, output_norm_input, eps, valid, bank_token_stride, write_idx,
        fused_output_eps, save_materialized, mix, fuse_add,
        output, raw_prefix, materialized);
    return {output, raw_prefix, materialized};
}
}
