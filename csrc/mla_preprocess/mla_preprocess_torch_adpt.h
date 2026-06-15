/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MLA_PREPROCESS_TORCH_ADPT_H
#define MLA_PREPROCESS_TORCH_ADPT_H


#include "op_host/mla_preprocess.h"

namespace vllm_ascend {
std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &> mla_preprocess(
    const at::Tensor &hiddenState, const at::Tensor &wdqkv,
    const c10::optional<at::Tensor> &descale0, const at::Tensor &gamma1, const c10::optional<at::Tensor> &beta1, const at::Tensor &wuq,
    const c10::optional<at::Tensor> &descale1, const at::Tensor &gamma2, const at::Tensor &cos, const at::Tensor &sin,
    const at::Tensor &wuk, const at::Tensor &kv_cache, const at::Tensor &kv_cache_rope, const at::Tensor &slotmapping,
    const c10::optional<at::Tensor> &quant_scale0, const c10::optional<at::Tensor> &quant_offset0, const c10::optional<at::Tensor> &bias0,
    const c10::optional<at::Tensor> &quant_scale1, const c10::optional<at::Tensor> &quant_offset1, const c10::optional<at::Tensor> &bias1,
    const c10::optional<at::Tensor> &ctkv_scale, const c10::optional<at::Tensor> &q_nope_scale,
    c10::optional<c10::string_view> cache_mode, c10::optional<c10::string_view> quant_mode, c10::optional<bool> enable_inner_out, at::Tensor &q_out0,
    at::Tensor &kv_cache_out0, at::Tensor &q_out1, at::Tensor &kv_cache_out1, at::Tensor &inner_out)
{
    at::Tensor Descale0 =
        descale0.has_value()
            ? descale0.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Descale1 =
        descale1.has_value()
            ? descale1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Beta1 =
        beta1.has_value()
            ? beta1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Quant_scale0 =
        quant_scale0.has_value()
            ? quant_scale0.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Quant_scale1 =
        quant_scale1.has_value()
            ? quant_scale1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Quant_offset0 =
        quant_offset0.has_value()
            ? quant_offset0.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Quant_offset1 =
        quant_offset1.has_value()
            ? quant_offset1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Bias0 =
        bias0.has_value()
            ? bias0.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Bias1 =
        bias1.has_value()
            ? bias1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor CtkvScale =
        ctkv_scale.has_value()
            ? ctkv_scale.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor QnopeScale =
        q_nope_scale.has_value()
            ? q_nope_scale.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    bool enableInnerOut =
        enable_inner_out.has_value()
            ? enable_inner_out.value()
            : false;
    
    auto [workspace_tensor, tiling, block_dim] = mlapo::mla_preprocess_tiling(
        hiddenState,
        wdqkv,
        wuk,
        gamma1,
        kv_cache_rope,
        cache_mode,
        quant_mode,
        enableInnerOut
    );

    void *hidden_state_ptr = hiddenState.data_ptr();
    void *quant_scale0_ptr = Quant_scale0.data_ptr();
    void *quant_offset0_ptr = Quant_offset0.data_ptr();
    void *wdqkv_ptr = wdqkv.data_ptr();
    void *bias0_ptr = Bias0.data_ptr();
    void *gamma1_ptr = gamma1.data_ptr();
    void *beta1_ptr = Beta1.data_ptr();
    void *quant_scale1_ptr = Quant_scale1.data_ptr();
    void *quant_offset1_ptr = Quant_offset1.data_ptr();
    void *gamma2_ptr = gamma2.data_ptr();
    void *sin_ptr = sin.data_ptr();
    void *cos_ptr = cos.data_ptr();
    void *kv_cache_ptr = kv_cache.data_ptr();
    void *slotmapping_ptr = slotmapping.data_ptr();
    void *wuq_ptr = wuq.data_ptr();
    void *bias1_ptr = Bias1.data_ptr();
    void *wuk_ptr = wuk.data_ptr();
    void *descale0_ptr = Descale0.data_ptr();
    void *descale1_ptr = Descale1.data_ptr();
    void *ctkv_scale_ptr = CtkvScale.data_ptr();
    void *qnope_scale_ptr = QnopeScale.data_ptr();
    void *q_out0_ptr = q_out0.data_ptr();
    void *kv_cache_out0_ptr = kv_cache_out0.data_ptr();
    void *q_out1_ptr = q_out1.data_ptr();
    void *kv_cache_out1_ptr = kv_cache_out1.data_ptr();
    void *inner_out_ptr = inner_out.data_ptr();
    void *workspace_ptr = workspace_tensor.data_ptr();
    void *tiling_ptr = tiling.data_ptr();

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("mla_preprocess");

    cmd.SetCustomHandler([stream, hidden_state_ptr, quant_scale0_ptr, quant_offset0_ptr, wdqkv_ptr, bias0_ptr,
                          gamma1_ptr, beta1_ptr, quant_scale1_ptr, quant_offset1_ptr, gamma2_ptr, sin_ptr, cos_ptr,
                          kv_cache_ptr, slotmapping_ptr, wuq_ptr, bias1_ptr, wuk_ptr, descale0_ptr, descale1_ptr, ctkv_scale_ptr,
                          qnope_scale_ptr, q_out0_ptr, kv_cache_out0_ptr, q_out1_ptr, kv_cache_out1_ptr, inner_out_ptr, workspace_ptr,
                          tiling_ptr, block_dim]() -> int {
        mla_preprocess_impl(stream, hidden_state_ptr, quant_scale0_ptr, quant_offset0_ptr, wdqkv_ptr, bias0_ptr,
                            gamma1_ptr, beta1_ptr, quant_scale1_ptr, quant_offset1_ptr, gamma2_ptr, sin_ptr, cos_ptr, sin_ptr, cos_ptr,
                            kv_cache_ptr, slotmapping_ptr, wuq_ptr, bias1_ptr, wuk_ptr, descale0_ptr, descale1_ptr, ctkv_scale_ptr,
                            qnope_scale_ptr, q_out0_ptr, kv_cache_out0_ptr, q_out1_ptr, kv_cache_out1_ptr,
                            inner_out_ptr, q_out0_ptr, workspace_ptr, tiling_ptr, block_dim);
        return 0;
    });
    cmd.Run();
    return std::forward_as_tuple(q_out0, kv_cache_out0, q_out1, kv_cache_out1, inner_out);
}

std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &> mla_preprocess_by_cache(
    const at::Tensor &hiddenState, const at::Tensor &wdqkv,
    const c10::optional<at::Tensor> &descale0, const at::Tensor &gamma1, const c10::optional<at::Tensor> &beta1, const at::Tensor &wuq,
    const c10::optional<at::Tensor> &descale1, const at::Tensor &gamma2, const at::Tensor &positions, const at::Tensor &cos_sin_cache,
    const at::Tensor &wuk, const at::Tensor &kv_cache, const at::Tensor &kv_cache_rope, const at::Tensor &slotmapping,
    const c10::optional<at::Tensor> &quant_scale0, const c10::optional<at::Tensor> &quant_offset0, const c10::optional<at::Tensor> &bias0,
    const c10::optional<at::Tensor> &quant_scale1, const c10::optional<at::Tensor> &quant_offset1, const c10::optional<at::Tensor> &bias1,
    const c10::optional<at::Tensor> &ctkv_scale, const c10::optional<at::Tensor> &q_nope_scale,
    c10::optional<c10::string_view> cache_mode, c10::optional<c10::string_view> quant_mode, c10::optional<bool> enable_inner_out,
    bool is_neox_style, at::Tensor &q_out0, at::Tensor &kv_cache_out0, at::Tensor &q_out1, at::Tensor &kv_cache_out1,
    at::Tensor &inner_out, c10::optional<bool> enable_raw_q_out, at::Tensor &raw_q_out)
{
    TORCH_CHECK(positions.defined(), "mla_preprocess_by_cache expects defined positions");
    TORCH_CHECK(positions.dim() == 1, "mla_preprocess_by_cache expects 1-D positions, got dim ", positions.dim());
    TORCH_CHECK(positions.scalar_type() == at::kLong || positions.scalar_type() == at::kInt,
                "mla_preprocess_by_cache expects int32/int64 positions, got ", positions.scalar_type());
    TORCH_CHECK(hiddenState.dim() >= 1 && positions.numel() == hiddenState.size(0),
                "mla_preprocess_by_cache expects positions length to match hiddenState tokens, got ",
                positions.numel(), " and ", hiddenState.dim() >= 1 ? hiddenState.size(0) : 0);
    TORCH_CHECK(cos_sin_cache.defined(), "mla_preprocess_by_cache expects defined cos_sin_cache");
    TORCH_CHECK(cos_sin_cache.dim() == 2,
                "mla_preprocess_by_cache expects cos_sin_cache with shape [max_position, rotary_dim], got dim ",
                cos_sin_cache.dim());
    TORCH_CHECK(cos_sin_cache.size(-1) % 2 == 0,
                "mla_preprocess_by_cache expects even cos_sin_cache last dim, got ", cos_sin_cache.size(-1));
    TORCH_CHECK(cos_sin_cache.stride(-1) == 1,
                "mla_preprocess_by_cache expects contiguous cos_sin_cache rows, got last-dim stride ",
                cos_sin_cache.stride(-1));
    TORCH_CHECK(is_neox_style, "mla_preprocess_by_cache currently supports only Neox-style RoPE");

    at::Tensor Descale0 =
        descale0.has_value()
            ? descale0.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Descale1 =
        descale1.has_value()
            ? descale1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Beta1 =
        beta1.has_value()
            ? beta1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Quant_scale0 =
        quant_scale0.has_value()
            ? quant_scale0.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Quant_scale1 =
        quant_scale1.has_value()
            ? quant_scale1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Quant_offset0 =
        quant_offset0.has_value()
            ? quant_offset0.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Quant_offset1 =
        quant_offset1.has_value()
            ? quant_offset1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Bias0 =
        bias0.has_value()
            ? bias0.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Bias1 =
        bias1.has_value()
            ? bias1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor CtkvScale =
        ctkv_scale.has_value()
            ? ctkv_scale.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor QnopeScale =
        q_nope_scale.has_value()
            ? q_nope_scale.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    bool enableInnerOut =
        enable_inner_out.has_value()
            ? enable_inner_out.value()
            : false;
    bool enableRawQOut =
        enable_raw_q_out.has_value()
            ? enable_raw_q_out.value()
            : false;
    if (enableRawQOut) {
        TORCH_CHECK(raw_q_out.defined(), "mla_preprocess_by_cache raw_q_out must be defined when enabled");
        TORCH_CHECK(raw_q_out.scalar_type() == hiddenState.scalar_type(),
                    "mla_preprocess_by_cache raw_q_out dtype must match hiddenState dtype, got ",
                    raw_q_out.scalar_type(), " and ", hiddenState.scalar_type());
        TORCH_CHECK(raw_q_out.is_contiguous(), "mla_preprocess_by_cache raw_q_out must be contiguous");
        TORCH_CHECK(raw_q_out.numel() >= hiddenState.size(0) * wuk.size(0) * wuk.size(1),
                    "mla_preprocess_by_cache raw_q_out is too small for [tokens, heads, qk_nope_head_dim], got ",
                    raw_q_out.numel(), " elements for ", hiddenState.size(0), " tokens, ", wuk.size(0),
                    " heads, and ", wuk.size(1), " qk_nope_head_dim");
    }
    at::Tensor Positions = positions.scalar_type() == at::kLong ? positions : positions.to(at::kLong);
    if (!Positions.is_contiguous()) {
        Positions = Positions.contiguous();
    }

    auto [workspace_tensor, tiling, block_dim] = mlapo::mla_preprocess_tiling(
        hiddenState,
        wdqkv,
        wuk,
        gamma1,
        kv_cache_rope,
        cache_mode,
        quant_mode,
        enableInnerOut,
        enableRawQOut,
        &cos_sin_cache,
        is_neox_style
    );

    void *hidden_state_ptr = hiddenState.data_ptr();
    void *quant_scale0_ptr = Quant_scale0.data_ptr();
    void *quant_offset0_ptr = Quant_offset0.data_ptr();
    void *wdqkv_ptr = wdqkv.data_ptr();
    void *bias0_ptr = Bias0.data_ptr();
    void *gamma1_ptr = gamma1.data_ptr();
    void *beta1_ptr = Beta1.data_ptr();
    void *quant_scale1_ptr = Quant_scale1.data_ptr();
    void *quant_offset1_ptr = Quant_offset1.data_ptr();
    void *gamma2_ptr = gamma2.data_ptr();
    void *positions_ptr = Positions.data_ptr();
    void *cos_sin_cache_ptr = cos_sin_cache.data_ptr();
    void *kv_cache_ptr = kv_cache.data_ptr();
    void *slotmapping_ptr = slotmapping.data_ptr();
    void *wuq_ptr = wuq.data_ptr();
    void *bias1_ptr = Bias1.data_ptr();
    void *wuk_ptr = wuk.data_ptr();
    void *descale0_ptr = Descale0.data_ptr();
    void *descale1_ptr = Descale1.data_ptr();
    void *ctkv_scale_ptr = CtkvScale.data_ptr();
    void *qnope_scale_ptr = QnopeScale.data_ptr();
    void *q_out0_ptr = q_out0.data_ptr();
    void *kv_cache_out0_ptr = kv_cache_out0.data_ptr();
    void *q_out1_ptr = q_out1.data_ptr();
    void *kv_cache_out1_ptr = kv_cache_out1.data_ptr();
    void *inner_out_ptr = inner_out.data_ptr();
    void *raw_q_out_ptr = raw_q_out.data_ptr();
    void *workspace_ptr = workspace_tensor.data_ptr();
    void *tiling_ptr = tiling.data_ptr();

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("mla_preprocess_by_cache");

    cmd.SetCustomHandler([stream, hidden_state_ptr, quant_scale0_ptr, quant_offset0_ptr, wdqkv_ptr, bias0_ptr,
                          gamma1_ptr, beta1_ptr, quant_scale1_ptr, quant_offset1_ptr, gamma2_ptr, positions_ptr,
                          cos_sin_cache_ptr, kv_cache_ptr, slotmapping_ptr, wuq_ptr, bias1_ptr, wuk_ptr, descale0_ptr,
                          descale1_ptr, ctkv_scale_ptr, qnope_scale_ptr, q_out0_ptr, kv_cache_out0_ptr, q_out1_ptr,
                          kv_cache_out1_ptr, inner_out_ptr, raw_q_out_ptr, workspace_ptr, tiling_ptr, block_dim]() -> int {
        mla_preprocess_by_cache_impl(stream, hidden_state_ptr, quant_scale0_ptr, quant_offset0_ptr, wdqkv_ptr,
                                     bias0_ptr, gamma1_ptr, beta1_ptr, quant_scale1_ptr, quant_offset1_ptr,
                                     gamma2_ptr, positions_ptr, cos_sin_cache_ptr, kv_cache_ptr, slotmapping_ptr,
                                     wuq_ptr, bias1_ptr, wuk_ptr, descale0_ptr, descale1_ptr, ctkv_scale_ptr,
                                     qnope_scale_ptr, q_out0_ptr, kv_cache_out0_ptr, q_out1_ptr, kv_cache_out1_ptr,
                                     inner_out_ptr, raw_q_out_ptr, workspace_ptr, tiling_ptr, block_dim);
        return 0;
    });
    cmd.Run();
    return std::forward_as_tuple(q_out0, kv_cache_out0, q_out1, kv_cache_out1, inner_out, raw_q_out);
}
}
#endif
