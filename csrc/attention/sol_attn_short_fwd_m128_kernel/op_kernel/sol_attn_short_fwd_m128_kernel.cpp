#if __has_include("../../common/op_kernel/sol_attn_tensorutils_cce.h")
#include "../../common/op_kernel/sol_attn_tensorutils_cce.h"
#else
#include "../common/sol_attn_tensorutils_cce.h"
#endif
using namespace ascrip;
#include "sol_attn_short_fwd_m128_kernel_cube.h"
#include "sol_attn_short_fwd_m128_kernel_vec.h"


extern "C" __global__ __aicore__ void sol_attn_short_fwd_m128_kernel(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR lse, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tiling_data, tiling);
    const int32_t B = tiling_data.B;
    const int32_t T = tiling_data.T;
    const int32_t H = tiling_data.H;
    const int32_t NT = tiling_data.NT;
    const int32_t NT_Q = tiling_data.NT_Q;
    const int32_t NT_Q_PAD8 = tiling_data.NT_Q_PAD8;
    const int32_t T_LSE = tiling_data.T_LSE;
    const int32_t sink_start = tiling_data.sink_start;
    const int32_t sink_end = tiling_data.sink_end;
    const float scale_log2 = tiling_data.scale_log2;
    const float scale_sq = tiling_data.scale_sq;
    const float tau = tiling_data.tau;
    if ASCEND_IS_AIC {
        sol_attn_short_fwd_m128_kernel_cube(q, k, v, out, lse, B, T, H, NT, NT_Q, NT_Q_PAD8, T_LSE, sink_start, sink_end, scale_log2, scale_sq, tau, workspace);
    }
    if ASCEND_IS_AIV {
        sol_attn_short_fwd_m128_kernel_vec(q, k, v, out, lse, B, T, H, NT, NT_Q, NT_Q_PAD8, T_LSE, sink_start, sink_end, scale_log2, scale_sq, tau, workspace);
    }
}
