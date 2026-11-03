#include "torch_kernel_helper.h"
#include "ops.h"

TORCH_LIBRARY_FRAGMENT(npu, m) {
 m.def("splitquant_raw_cube_debug(Tensor a, Tensor b, int n, int g) -> Tensor");
 m.def("splitquant_scalar_round_debug(Tensor x) -> Tensor");
 m.def("splitquant_paged_silu_debug(Tensor x, Tensor lut) -> Tensor");
 m.def("splitquant_compact_silu_debug(Tensor x, Tensor lut) -> Tensor");
 m.def("splitquant_router_finish(Tensor weights, Tensor ids, bool normalize=True) -> (Tensor, Tensor)");
 m.def("splitquant_state_gather(Tensor cache, Tensor indices) -> Tensor");
 m.def("splitquant_state_scatter(Tensor(a!) cache, Tensor source, Tensor indices) -> Tensor(a!)");
 m.def("splitquant_state_read(Tensor cache, int row) -> Tensor");
 m.def("splitquant_state_write(Tensor(a!) cache, Tensor source, int row) -> Tensor(a!)");
 m.def("splitquant_dense_fused_multi(Tensor x, Tensor packed_weight, Tensor weight_scale, Tensor weight_offset, Tensor clip_max, Tensor clip_min, Tensor rotation_transposed, int[] splits) -> Tensor");
 m.def("splitquant_dense_fused(Tensor x, Tensor packed_weight, Tensor weight_scale, Tensor weight_offset, Tensor activation_clip_max, Tensor activation_clip_min, Tensor? bias=None, int group_size=128, Tensor? group_trans=None, Tensor? right_trans=None, Tensor? left_trans=None) -> Tensor");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
 m.impl("splitquant_raw_cube_debug", TORCH_FN(splitquant::raw_cube_debug));
 m.impl("splitquant_scalar_round_debug", TORCH_FN(splitquant::scalar_round_debug));
 m.impl("splitquant_paged_silu_debug", TORCH_FN(splitquant::paged_silu_debug));
 m.impl("splitquant_compact_silu_debug", TORCH_FN(splitquant::compact_silu_debug));
 m.impl("splitquant_router_finish", TORCH_FN(splitquant::router_finish));
 m.impl("splitquant_state_gather", TORCH_FN(splitquant::state_gather));
 m.impl("splitquant_state_scatter", TORCH_FN(splitquant::state_scatter));
 m.impl("splitquant_state_read", TORCH_FN(splitquant::state_read));
 m.impl("splitquant_state_write", TORCH_FN(splitquant::state_write));
 m.impl("splitquant_dense_fused_multi", TORCH_FN(splitquant::multi));
 m.impl("splitquant_dense_fused", TORCH_FN(splitquant::splitquant_dense_fused));
}
// SPDX-License-Identifier: BSD-3-Clause
#include <torch/library.h>

#define _ARGS "(Tensor x, Tensor up_weight, Tensor up_scale, Tensor down_weight_bf16, Tensor up_clip_max, Tensor up_clip_min, Tensor hidden_clip_max, Tensor hidden_clip_min, Tensor down_rotation, Tensor expert_ids, Tensor routing_weights, Tensor silu_lut)"
TORCH_LIBRARY_FRAGMENT(npu,m){m.def("splitquant_moe_fused" _ARGS " -> Tensor");m.def("splitquant_moe_fused_debug" _ARGS " -> Tensor[]");}
TORCH_LIBRARY_IMPL(npu,PrivateUse1,m){m.impl("splitquant_moe_fused",TORCH_FN(ascend_kernel::splitquant_moe_fused));m.impl("splitquant_moe_fused_debug",TORCH_FN(ascend_kernel::splitquant_moe_fused_debug));}

TORCH_LIBRARY_FRAGMENT(npu, m) { m.def("splitquant_merged_build_info() -> str", []() -> std::string { return "splitquant-unified-generic-and-moe;20261031"; }); }
