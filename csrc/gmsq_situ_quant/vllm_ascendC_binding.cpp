/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Direct pybind11 exposure of the fused GroupedMatmulSituQuant AscendC
// operator (fake-A8W8 grouped GMM1 + SiTU + per-token INT8 quant, single
// launch). Importing this extension attaches the module to the `torch`
// namespace, so callers can use
//   torch.vllm_ascendC.grouped_matmul_situ_quant(...)
// without going through the torch dispatcher (no TORCH_LIBRARY registration
// on this branch — see the torch.ops._C_ascend variant branch for that exposure).

#include <optional>
#include <tuple>
#include <vector>

#include <torch/extension.h>
#include <pybind11/stl.h>

#include "grouped_matmul_situ_quant.h"

namespace py = pybind11;

PYBIND11_MODULE(vllm_ascendC, m)
{
    m.doc() = "vllm-ascend direct AscendC operator bindings (torch.vllm_ascendC.*)";

    m.def(
        "grouped_matmul_situ_quant",
        [](const at::Tensor &x, std::vector<at::Tensor> weight,
           std::vector<at::Tensor> weight_scale, const at::Tensor &x_scale,
           const at::Tensor &group_list, std::vector<at::Tensor> weight_assist_matrix, double beta,
           std::optional<double> linear_beta, int64_t group_list_type) {
            return vllm_ascend::grouped_matmul_situ_quant(x, weight, weight_scale, x_scale,
                                                             group_list, weight_assist_matrix, beta,
                                                             linear_beta, group_list_type);
        },
        "Fused grouped matmul (fake-A8W8) + SiTU + per-token INT8 quant (single launch).",
        py::arg("x"), py::arg("weight"), py::arg("weight_scale"), py::arg("x_scale"),
        py::arg("group_list"), py::arg("weight_assist_matrix"), py::arg("beta") = 1.0,
        py::arg("linear_beta") = py::none(), py::arg("group_list_type") = 1);

    // Attach as a `torch` attribute: importing vllm_ascend.vllm_ascendC makes
    // torch.vllm_ascendC.grouped_matmul_situ_quant available on the torch
    // module itself.
    py::module_ torch_mod = py::module_::import("torch");
    if (!py::hasattr(torch_mod, "vllm_ascendC")) {
        torch_mod.attr("vllm_ascendC") = m;
    }
}
