#if __has_include("../../common/op_kernel/sol_attn_tensorutils_cce.h")
#include "../../common/op_kernel/sol_attn_tensorutils_cce.h"
#else
#include "../common/sol_attn_tensorutils_cce.h"
#endif
using namespace ascrip;
#include "sol_attn_prepare_qkv_kernel_vec.h"


extern "C" __global__ __aicore__ void sol_attn_prepare_qkv_kernel(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR k_pad, GM_ADDR v_pad, GM_ADDR kc_padded, GM_ADDR vc_padded, GM_ADDR kc_pad_f32, GM_ADDR q_bar_f32, GM_ADDR last_group_lens, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    const int32_t B = tiling_data.B;
    const int32_t T = tiling_data.T;
    const int32_t H = tiling_data.H;
    const int32_t NT = tiling_data.NT;
    const int32_t NG = tiling_data.NG;
    const int32_t NPAD = tiling_data.NPAD;
    const int32_t T_PAD_KV = tiling_data.T_PAD_KV;
    const float inv_tail_len = tiling_data.inv_tail_len;
    if ASCEND_IS_AIV {
        sol_attn_prepare_qkv_kernel_vec(q, k, v, k_pad, v_pad, kc_padded, vc_padded, kc_pad_f32, q_bar_f32, last_group_lens, B, T, H, NT, NG, NPAD, T_PAD_KV, inv_tail_len, workspace);
    }
}
