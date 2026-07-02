#include "kernel_operator.h"
#include "zb_moe_distribute_dispatch.h"
#include "zb_moe_distribute_dispatch_tiling.h"

using namespace AscendC;
using namespace ZbMoeDistributeDispatchImpl;

/*
 * A3 tilingkey说明
 * 5位的十进制数
 * 第1位（个位）：quantMode:
 *     0: 不量化, 1: 静态量化, 2: 动态量化
 * 第2位（十位）：是否有smoothScale:
 *     0: 无, 1: 有
 * 第3位（百位）：是否做tp域allgather:
 *     0: 不做, 1: 做
 * 第4位（千位）：是否走fullmesh_v2模板:
 *     0: 不做, 1: 做
 * 第5位（万位）：无实际意义
 */

extern "C" __global__ __aicore__ void zb_moe_distribute_dispatch(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales,
                                                                 GM_ADDR xActiveMask, GM_ADDR elasticInfo,
                                                                 GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
                                                                 GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut,
                                                                 GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut,
                                                                 GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(ZbMoeDistributeDispatchTilingData);
    TPipe pipe;
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    // PRINTF("[--------- DISPATCH ---------] GETTING TILING KEY \n");
    if (TILING_KEY_IS(10000)) {
        GET_TILING_DATA_WITH_STRUCT(ZbMoeDistributeDispatchTilingData, tilingData, tilingGM);
        ZbMoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, false, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(10100)) {
        GET_TILING_DATA_WITH_STRUCT(ZbMoeDistributeDispatchTilingData, tilingData, tilingGM);
        ZbMoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, false, false, true> op;
        op.Init(x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
#elif (ORIG_DTYPE_EXPAND_X == DT_INT8)
    if (TILING_KEY_IS(10011)) {
        GET_TILING_DATA_WITH_STRUCT(ZbMoeDistributeDispatchTilingData, tilingData, tilingGM);
        ZbMoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, true, false, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(10002)) {
        GET_TILING_DATA_WITH_STRUCT(ZbMoeDistributeDispatchTilingData, tilingData, tilingGM);
        ZbMoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, true, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(10012)) {
        GET_TILING_DATA_WITH_STRUCT(ZbMoeDistributeDispatchTilingData, tilingData, tilingGM);
        ZbMoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, true, true, false> op;
        op.Init(x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(10111)) {
        GET_TILING_DATA_WITH_STRUCT(ZbMoeDistributeDispatchTilingData, tilingData, tilingGM);
        ZbMoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, true, false, false, true> op;
        op.Init(x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(10102)) {
        GET_TILING_DATA_WITH_STRUCT(ZbMoeDistributeDispatchTilingData, tilingData, tilingGM);
        ZbMoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, true, false, true> op;
        op.Init(x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(10112)) {
        GET_TILING_DATA_WITH_STRUCT(ZbMoeDistributeDispatchTilingData, tilingData, tilingGM);
        ZbMoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, true, true, true> op;
        op.Init(x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
#endif
}
