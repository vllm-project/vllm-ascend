#if __has_include("../../common/op_kernel/sol_attn_tensorutils_cce.h")
#include "../../common/op_kernel/sol_attn_tensorutils_cce.h"
#else
#include "../common/sol_attn_tensorutils_cce.h"
#endif
using namespace ascrip;
#include "sol_attn_prepare_route_pack_kernel_vec.h"


extern "C" __global__ __aicore__ void sol_attn_prepare_route_pack_kernel(GM_ADDR raw_scores, GM_ADDR thresh_bh, GM_ADDR group_approx_mask, GM_ADDR flat_exact_cnt, GM_ADDR flat_exact_idx, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    const int32_t HC = tiling_data.HC;
    const int32_t NT = tiling_data.NT;
    const int32_t NT_Q = tiling_data.NT_Q;
    const int32_t NG = tiling_data.NG;
    const int32_t NPAD = tiling_data.NPAD;
    const int32_t NT_Q_PAD8 = tiling_data.NT_Q_PAD8;
    const int32_t TILE_M = tiling_data.TILE_M;
    const int32_t sink_s0 = tiling_data.sink_s0;
    const int32_t sink_s1 = tiling_data.sink_s1;
    const float scale_log2 = tiling_data.scale_log2;
    if ASCEND_IS_AIV {
        sol_attn_prepare_route_pack_kernel_vec(raw_scores, thresh_bh, group_approx_mask, flat_exact_cnt, flat_exact_idx, HC, NT, NT_Q, NG, NPAD, NT_Q_PAD8, TILE_M == 128 ? 2 : 1, sink_s0, sink_s1, scale_log2, workspace);
    }
}
