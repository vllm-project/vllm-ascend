#include "zb_moe_distribute_combine.h"
#include "kernel_operator.h"
#include "zb_moe_distribute_combine_tiling.h"

using namespace AscendC;
using namespace MoeDistributeCombineV2Impl;

namespace {
template <TemplateMC2TypeClass>
__aicore__ inline void ExecZbMoeDistributeCombine(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine,
                                                       GM_ADDR epSendCount, GM_ADDR tpSendCount, GM_ADDR scales,
                                                       GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo,
                                                       GM_ADDR oriX, GM_ADDR constExpertAlpha1,
                                                       GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR XOut,
                                                       GM_ADDR workspaceGM, GM_ADDR tilingGM, TPipe *pipePtr)
{
    GET_TILING_DATA_WITH_STRUCT(ZbMoeDistributeCombineTilingData, tilingData, tilingGM);
    ZbMoeDistributeCombine<TemplateMC2TypeFunc> op;
    // PRINTF("[---------------- Initializing CombineV2 ------------------] \n");
    op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, scales, xActiveMask, sharedExpertX,
            elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, XOut, workspaceGM, pipePtr,
            &tilingData);
    // PRINTF("[---------------- Processing CombineV2 ------------------] \n");
    op.Process();
}
}  // namespace

/*
 * A3 tilingkey说明
 * 5位的十进制数
 * 第1位（个位）：无意义占位使用
 * 第2位（十位）：通信量化选项：
 *     0：无量化, 2:int8量化
 * 第3位（百位）：是否做tp域allgather:
 *     0: 不做, 1: 做
 * 第4位（千位）：无实际意义:
 * 第5位（万位）：无实际意义.
 */

extern "C" __global__ __aicore__ void zb_moe_distribute_combine(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR epSendCount, GM_ADDR scales,
    GM_ADDR tpSendCount, GM_ADDR xActiveMask, GM_ADDR activationScale, GM_ADDR weightScale, GM_ADDR groupList,
    GM_ADDR expandScales, GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX, GM_ADDR constExpertAlpha1,
    GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)

{
    // PRINTF("[---------------- Calling CombineV2 ------------------] \n");
    REGISTER_TILING_DEFAULT(ZbMoeDistributeCombineTilingData);
    TPipe pipe;

#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    if (TILING_KEY_IS(10100)) {  // tp=2 IsInt8Quant=0
        ExecZbMoeDistributeCombine<DTYPE_EXPAND_X, DTYPE_X, int32_t, true, false>(
            expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, scales, xActiveMask, sharedExpertX,
            elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, XOut, workspaceGM, tilingGM, &pipe);
    }
    if (TILING_KEY_IS(10000)) {  // tp=1 IsInt8Quant=0
        ExecZbMoeDistributeCombine<DTYPE_EXPAND_X, DTYPE_X, int32_t, false, false>(
            expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, scales, xActiveMask, sharedExpertX,
            elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, XOut, workspaceGM, tilingGM, &pipe);
    }
    if (TILING_KEY_IS(10120)) {  // tp=2 IsInt8Quant=1
        ExecZbMoeDistributeCombine<DTYPE_EXPAND_X, DTYPE_X, int32_t, true, true>(
            expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, scales, xActiveMask, sharedExpertX,
            elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, XOut, workspaceGM, tilingGM, &pipe);
    }
    if (TILING_KEY_IS(10020)) {  // tp=1 IsInt8Quant=1
        ExecZbMoeDistributeCombine<DTYPE_EXPAND_X, DTYPE_X, int32_t, false, true>(
            expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, scales, xActiveMask, sharedExpertX,
            elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, XOut, workspaceGM, tilingGM, &pipe);
    }
#endif
}
