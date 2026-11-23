#include "../../common/sol_attn/tensorutils_cce.h"
using namespace ascrip;
#include "sol_attn_prepare_diag_thresh_kernel_vec.h"


extern "C" __global__ __aicore__ void sol_attn_prepare_diag_thresh_kernel(GM_ADDR q_bar_f32, GM_ADDR kc_pad_f32, GM_ADDR thresh_bh, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    const int32_t B = tiling_data.B;
    const int32_t H = tiling_data.H;
    const int32_t NT = tiling_data.NT;
    const int32_t NG = tiling_data.NG;
    const int32_t NPAD = tiling_data.NPAD;
    const float scale_log2 = tiling_data.scale_log2;
    const float tau = tiling_data.tau;
    const float inv_nt = tiling_data.inv_nt;
    if ASCEND_IS_AIV {
        sol_attn_prepare_diag_thresh_kernel_vec(q_bar_f32, kc_pad_f32, thresh_bh, B, H, NT, NG, NPAD, scale_log2, tau, inv_nt, workspace);
    }
}
