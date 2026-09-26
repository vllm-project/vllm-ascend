#ifndef GMSQ_OPS_H
#define GMSQ_OPS_H

#include <ATen/ATen.h>
#include <optional>
#include <tuple>
#include <vector>

namespace ascend_kernel {

// V2-aligned dual entries (aclnnGroupedMatmulSwigluQuantV2 / WeightNzV2 API
// habits): two entry names sharing one ACLNN custom-op implementation. The ND
// weight entry converts at the entry layer, while stacked and TensorList NZ
// entries feed their existing storage to the kernel through a dynamic-input
// address table. See grouped_matmul_situ_quant_torch_adpt.h for the full
// parameter behavior matrix.
std::tuple<at::Tensor, at::Tensor> gmm_situ_quant_v2_nd(
    const at::Tensor &x, const at::Tensor &weight, const at::Tensor &weightScale,
    const std::optional<at::Tensor> &weightAssistMatrix, const std::optional<at::Tensor> &bias,
    const at::Tensor &xScale, const std::optional<at::Tensor> &smoothScale, const at::Tensor &groupList,
    int64_t dequantMode, int64_t dequantDtype, int64_t quantMode, int64_t groupListType,
    const std::optional<std::vector<int64_t>> &tuningConfigOptional, double beta, double linearBeta);

std::tuple<at::Tensor, at::Tensor> gmm_situ_quant_v2_nz(
    const at::Tensor &x, const at::Tensor &weight, const at::Tensor &weightScale,
    const std::optional<at::Tensor> &weightAssistMatrix, const std::optional<at::Tensor> &bias,
    const at::Tensor &xScale, const std::optional<at::Tensor> &smoothScale, const at::Tensor &groupList,
    int64_t dequantMode, int64_t dequantDtype, int64_t quantMode, int64_t groupListType,
    const std::optional<std::vector<int64_t>> &tuningConfigOptional, double beta, double linearBeta);

std::tuple<at::Tensor, at::Tensor> gmm_situ_quant_v2_nd_list(
    const at::Tensor &x, const std::vector<at::Tensor> &weight, const std::vector<at::Tensor> &weightScale,
    const std::optional<std::vector<at::Tensor>> &weightAssistMatrix, const std::optional<at::Tensor> &bias,
    const at::Tensor &xScale, const std::optional<at::Tensor> &smoothScale, const at::Tensor &groupList,
    int64_t dequantMode, int64_t dequantDtype, int64_t quantMode, int64_t groupListType,
    const std::optional<std::vector<int64_t>> &tuningConfigOptional, double beta, double linearBeta);

std::tuple<at::Tensor, at::Tensor> gmm_situ_quant_v2_nz_list(
    const at::Tensor &x, const std::vector<at::Tensor> &weight, const std::vector<at::Tensor> &weightScale,
    const std::optional<std::vector<at::Tensor>> &weightAssistMatrix, const std::optional<at::Tensor> &bias,
    const at::Tensor &xScale, const std::optional<at::Tensor> &smoothScale, const at::Tensor &groupList,
    int64_t dequantMode, int64_t dequantDtype, int64_t quantMode, int64_t groupListType,
    const std::optional<std::vector<int64_t>> &tuningConfigOptional, double beta, double linearBeta);
}

#endif
