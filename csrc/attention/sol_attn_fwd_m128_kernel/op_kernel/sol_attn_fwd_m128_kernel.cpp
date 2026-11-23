#include "../../common/sol_attn/tensorutils_cce.h"
using namespace ascrip;
#include "sol_attn_fwd_m128_kernel_cube.h"
#include "sol_attn_fwd_m128_kernel_vec.h"


extern "C" __global__ __aicore__ void sol_attn_fwd_m128_kernel(GM_ADDR q, GM_ADDR k_pad, GM_ADDR v_pad, GM_ADDR kc_padded, GM_ADDR vc_padded, GM_ADDR flat_exact_cnt, GM_ADDR flat_exact_idx, GM_ADDR group_approx_mask, GM_ADDR last_group_lens, GM_ADDR out, GM_ADDR lse, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tiling_data, tiling);
    const int32_t B = tiling_data.B;
    const int32_t T = tiling_data.T;
    const int32_t H = tiling_data.H;
    const int32_t NT = tiling_data.NT;
    const int32_t NT_Q = tiling_data.NT_Q;
    const int32_t NG = tiling_data.NG;
    const int32_t NPAD = tiling_data.NPAD;
    const int32_t T_PAD_KV = tiling_data.T_PAD_KV;
    const int32_t NT_Q_PAD8 = tiling_data.NT_Q_PAD8;
    const int32_t T_LSE = tiling_data.T_LSE;
    const float scale_log2 = tiling_data.scale_log2;
    if ASCEND_IS_AIC {
        sol_attn_fwd_m128_kernel_cube(q, k_pad, v_pad, kc_padded, vc_padded, flat_exact_cnt, flat_exact_idx, group_approx_mask, last_group_lens, out, lse, B, T, H, NT, NT_Q, NG, NPAD, T_PAD_KV, NT_Q_PAD8, T_LSE, scale_log2, workspace);
    }
    if ASCEND_IS_AIV {
        sol_attn_fwd_m128_kernel_vec(q, k_pad, v_pad, kc_padded, vc_padded, flat_exact_cnt, flat_exact_idx, group_approx_mask, last_group_lens, out, lse, B, T, H, NT, NT_Q, NG, NPAD, T_PAD_KV, NT_Q_PAD8, T_LSE, scale_log2, workspace);
    }
}
