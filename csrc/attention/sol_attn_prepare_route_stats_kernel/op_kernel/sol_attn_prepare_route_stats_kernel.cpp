#if __has_include("../../common/op_kernel/sol_attn_tensorutils_cce.h")
#include "../../common/op_kernel/sol_attn_tensorutils_cce.h"
#else
#include "../common/sol_attn_tensorutils_cce.h"
#endif
using namespace ascrip;
#include "sol_attn_prepare_route_stats_kernel_cube.h"
#include "sol_attn_prepare_route_stats_kernel_vec.h"


extern "C" __global__ __aicore__ void sol_attn_prepare_route_stats_kernel(GM_ADDR q_bar, GM_ADDR kc, GM_ADDR threshold, GM_ADDR group_approx_mask, GM_ADDR flat_exact_cnt, GM_ADDR flat_exact_idx, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tiling_data, tiling);
    const int32_t B = tiling_data.B;
    const int32_t H = tiling_data.H;
    const int32_t NT = tiling_data.NT;
    const int32_t NT_Q = tiling_data.NT_Q;
    const int32_t NG = tiling_data.NG;
    const int32_t NPAD = tiling_data.NPAD;
    const int32_t NT_Q_PAD8 = tiling_data.NT_Q_PAD8;
    const int32_t query_blocks_per_tile = tiling_data.query_blocks_per_tile;
    const int32_t sink_start = tiling_data.sink_start;
    const int32_t sink_end = tiling_data.sink_end;
    const float scale_log2 = tiling_data.scale_log2;
    const float scale_sq = tiling_data.scale_sq;
    const float tau = tiling_data.tau;
    if ASCEND_IS_AIC {
        sol_attn_prepare_route_stats_kernel_cube(q_bar, kc, threshold, group_approx_mask, flat_exact_cnt, flat_exact_idx, B, H, NT, NT_Q, NG, NPAD, NT_Q_PAD8, query_blocks_per_tile, sink_start, sink_end, scale_log2, scale_sq, tau, workspace);
    }
    if ASCEND_IS_AIV {
        sol_attn_prepare_route_stats_kernel_vec(q_bar, kc, threshold, group_approx_mask, flat_exact_cnt, flat_exact_idx, B, H, NT, NT_Q, NG, NPAD, NT_Q_PAD8, query_blocks_per_tile, sink_start, sink_end, scale_log2, scale_sq, tau, workspace);
    }
}
