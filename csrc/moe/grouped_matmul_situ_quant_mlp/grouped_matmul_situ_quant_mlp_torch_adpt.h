#pragma once

#include <ATen/ATen.h>
#include <cmath>

#include "../grouped_matmul_situ_quant/grouped_matmul_situ_quant_torch_adpt.h"

namespace ascend_kernel {

inline at::Tensor gmm_situ_quant_mlp(
    const at::Tensor& x, const at::Tensor& xScale, const at::Tensor& groupList, const at::Tensor& upWeight,
    const at::Tensor& upScale, const at::Tensor& upCompensation, const at::Tensor& downWeight,
    const at::Tensor& downScale, const at::Tensor& downCompensation, int64_t groupListType, double beta,
    double linearBeta)
{
    TORCH_CHECK(x.is_privateuseone() && x.dim() == 2 && x.scalar_type() == at::kChar && x.is_contiguous(),
                "grouped_matmul_situ_quant_mlp expects contiguous NPU INT8 x of shape [M, K]");
    const auto device = x.device();
    const auto capacity = x.size(0);
    const auto width = x.size(1);
    TORCH_CHECK(groupList.dim() == 1 && groupList.scalar_type() == at::kLong &&
                (groupListType == 0 || groupListType == 1), "expected an INT64 group list of counts or prefix sums");
    const auto experts = groupList.size(0);
    const bool reportShape = experts == 14 && width == 7168;
    const bool routedShape = experts == 7 && width == 3584;
    TORCH_CHECK((reportShape || routedShape) && capacity <= 7168, "unsupported A2/A3 SiTU MLP dimensions");
    TORCH_CHECK(std::isfinite(beta) && beta > 0.0 && std::isfinite(linearBeta), "invalid SiTU parameters");
    for (const auto& tensor : {xScale, groupList, upScale, upCompensation, downScale, downCompensation}) {
        TORCH_CHECK(tensor.device() == device && tensor.is_contiguous(), "expected contiguous inputs on x.device");
    }
    TORCH_CHECK(xScale.dim() == 1 && xScale.size(0) == capacity && xScale.scalar_type() == at::kFloat,
                "expected FP32 x_scale with one element per input row");
    const auto upWidth = reportShape ? 7168 : 6144;
    const auto activationWidth = upWidth / 2;
    const auto checkWeight = [&](const at::Tensor& weight, int64_t inFeatures, int64_t outFeatures) {
        TORCH_CHECK(weight.device() == device && weight.dim() == 3 && weight.scalar_type() == at::kInt &&
                    weight.storage_offset() == 0 && weight.is_contiguous() && weight.size(0) == experts &&
                    weight.size(1) == inFeatures && weight.size(2) * 8 == outFeatures,
                    "expected stacked, contiguous NZ INT4 weights in an INT32 carrier");
        const auto format = at_npu::native::custom_ops::get_npu_format(weight);
        TORCH_CHECK(format == kNpuFormatFractalNz, "expected native FRACTAL_NZ storage for A2/A3 SiTU MLP");
    };
    const auto checkChannelMetadata = [&](const at::Tensor& scale, const at::Tensor& compensation, int64_t columns) {
        TORCH_CHECK(scale.scalar_type() == at::kLong && scale.numel() == experts * columns &&
                    ((scale.dim() == 2 && scale.size(0) == experts && scale.size(1) == columns) ||
                     (scale.dim() == 3 && scale.size(0) == experts && scale.size(1) == 1 && scale.size(2) == columns)),
                    "expected one packed scale per expert and output channel");
        TORCH_CHECK(compensation.scalar_type() == at::kFloat && compensation.dim() == 2 &&
                    compensation.size(0) == experts && compensation.size(1) == columns,
                    "expected FP32 compensation per expert and output channel");
    };
    checkWeight(upWeight, width, upWidth);
    checkWeight(downWeight, activationWidth, width);
    checkChannelMetadata(upScale, upCompensation, upWidth);
    checkChannelMetadata(downScale, downCompensation, width);

    auto y = at::empty_like(x, x.options().dtype(at::kBFloat16));
    if (capacity != 0) {
        EXEC_NPU_CMD(aclnnGroupedMatmulSituQuantMlpV2, x, xScale, groupList, upWeight, upScale, upCompensation,
                     downWeight, downScale, downCompensation, groupListType, beta, linearBeta, y);
    }
    return y;
}

} // namespace ascend_kernel
