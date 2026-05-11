#include "kernel_operator.h"
#include "moe_dispatch_normal_tiling.h"
#include "moe_dispatch_normal.h"

using namespace AscendC;
using namespace MoeDispatchNormalImpl;

#define TILINGKEY_NO_QUANT 10000
#define TILINGKEY_QUANT 10002

extern "C" __global__ __aicore__ void moe_dispatch_normal(GM_ADDR x, GM_ADDR expertIds, GM_ADDR send_offset,
    GM_ADDR send_token_idx, GM_ADDR recv_offset, GM_ADDR recv_count, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR assist_info_for_combine, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDispatchNormalTilingData);
    TPipe pipe;
#if (ORIG_DTYPE_RECV_X == DT_BF16 || ORIG_DTYPE_RECV_X == DT_FLOAT16)
    if (TILING_KEY_IS(TILINGKEY_NO_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDispatchNormalTilingData, tilingData, tilingGM);
        MoeDispatchNormal<DTYPE_X, DTYPE_RECV_X, false, false, false> op;
        op.Init(x,
            expertIds,
            send_offset,
            send_token_idx,
            recv_offset,
            recv_count,
            expandXOut,
            dynamicScalesOut,
            assist_info_for_combine,
            workspaceGM,
            &pipe,
            &tilingData);
        op.Process();
        return;
    }
#elif (ORIG_DTYPE_RECV_X == DT_INT8)
    if (TILING_KEY_IS(TILINGKEY_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDispatchNormalTilingData, tilingData, tilingGM);
        MoeDispatchNormal<DTYPE_X, DTYPE_RECV_X, true, false, false> op;
        op.Init(x,
            expertIds,
            send_offset,
            send_token_idx,
            recv_offset,
            recv_count,
            expandXOut,
            dynamicScalesOut,
            assist_info_for_combine,
            workspaceGM,
            &pipe,
            &tilingData);
        op.Process();
        return;
    }
#endif
}