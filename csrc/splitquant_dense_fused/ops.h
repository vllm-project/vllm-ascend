#pragma once
#include <ATen/ATen.h>
namespace splitquant {
at::Tensor splitquant_dense_fused(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const c10::optional<at::Tensor>&,int64_t,const c10::optional<at::Tensor>&,const c10::optional<at::Tensor>&,const c10::optional<at::Tensor>&);
at::Tensor multi(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,at::IntArrayRef);
}

namespace splitquant {
at::Tensor state_read(const at::Tensor&, int64_t);
at::Tensor state_write(at::Tensor, const at::Tensor&, int64_t);
}

namespace splitquant {
at::Tensor state_gather(const at::Tensor&,const at::Tensor&);
at::Tensor state_scatter(at::Tensor,const at::Tensor&,const at::Tensor&);
}
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/ATen.h>
#include <vector>
namespace ascend_kernel {
at::Tensor splitquant_moe_fused(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&);
std::vector<at::Tensor> splitquant_moe_fused_debug(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&);
}

#include <tuple>
namespace splitquant {std::tuple<at::Tensor,at::Tensor> router_finish(const at::Tensor&,const at::Tensor&,bool);}

namespace splitquant {at::Tensor compact_silu_debug(const at::Tensor&,const at::Tensor&);}

namespace splitquant {at::Tensor paged_silu_debug(const at::Tensor&,const at::Tensor&);}

namespace splitquant {at::Tensor scalar_round_debug(const at::Tensor&);}

namespace splitquant {at::Tensor raw_cube_debug(const at::Tensor&,const at::Tensor&,int64_t,int64_t);}
