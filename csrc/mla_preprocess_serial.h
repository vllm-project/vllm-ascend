// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <torch/extension.h>
#include <optional>

namespace vllm_ascend {
at::Tensor mla_serial_norm_cache(const at::Tensor&,const at::Tensor&,const at::Tensor&,
    const std::optional<at::Tensor>&,const std::optional<at::Tensor>&,const at::Tensor&,at::Tensor&,at::Tensor&);
std::tuple<at::Tensor,at::Tensor> mla_serial_q_prepare(const at::Tensor&,int64_t,
    const std::optional<at::Tensor>&,const std::optional<at::Tensor>&);
at::Tensor mla_serial_q_pack(const at::Tensor&,int64_t);
std::tuple<at::Tensor,at::Tensor> mla_serial_q_finalize(const at::Tensor&,const at::Tensor&,
    const std::optional<at::Tensor>&,const std::optional<at::Tensor>&);

namespace meta {
at::Tensor mla_serial_norm_cache(const at::Tensor&,const at::Tensor&,const at::Tensor&,
    const std::optional<at::Tensor>&,const std::optional<at::Tensor>&,const at::Tensor&,at::Tensor&,at::Tensor&);
std::tuple<at::Tensor,at::Tensor> mla_serial_q_prepare(const at::Tensor&,int64_t,
    const std::optional<at::Tensor>&,const std::optional<at::Tensor>&);
at::Tensor mla_serial_q_pack(const at::Tensor&,int64_t);
std::tuple<at::Tensor,at::Tensor> mla_serial_q_finalize(const at::Tensor&,const at::Tensor&,
    const std::optional<at::Tensor>&,const std::optional<at::Tensor>&);
}
}
