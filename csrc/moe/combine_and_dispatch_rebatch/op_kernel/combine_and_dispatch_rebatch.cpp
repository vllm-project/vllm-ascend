#include "kernel_operator.h"
#include "combine_and_dispatch_rebatch.h"
#include "combine_and_dispatch_rebatch_tiling.h"

using namespace AscendC;
using namespace CombineAndDispatchRebatchImpl;

extern "C" __global__ __aicore__ void combine_and_dispatch_rebatch(
    GM_ADDR batchBeforeDispatch,
    GM_ADDR residualsBeforeDispatch,
    GM_ADDR cutIdx,
    GM_ADDR prefixIndex,
    GM_ADDR discardLabel,
    GM_ADDR discardTokens,
    GM_ADDR discardMoeReqIdTensor,
    GM_ADDR reqIdList,
    GM_ADDR discardResiduals,
    GM_ADDR runningTopk,
    GM_ADDR discardTopk,
    GM_ADDR runningTopkWeight,
    GM_ADDR discardTopkWeight,
    GM_ADDR moeCntInput,
    GM_ADDR actualBs,
    GM_ADDR historyBs,
    GM_ADDR runningBs,
    GM_ADDR originalBs,
    
    // GM_ADDR rebatchOutput,
    GM_ADDR rebatchResiduals,
    
    GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(CombineAndDispatchRebatchTilingData);
    TPipe pipe;

    GET_TILING_DATA_WITH_STRUCT(CombineAndDispatchRebatchTilingData, tilingData, tilingGM);
    CombineAndDispatchRebatch<bfloat16_t> op;
    op.Init(batchBeforeDispatch, residualsBeforeDispatch, prefixIndex,
            discardLabel,
            discardTokens, discardMoeReqIdTensor, reqIdList, discardResiduals,
            runningTopk, discardTopk, runningTopkWeight, discardTopkWeight,
            moeCntInput, actualBs, historyBs, runningBs,originalBs,rebatchResiduals,
             workspaceGM, &pipe, &tilingData);
    op.Process();
}

namespace CombineAndDispatchRebatchImpl {

template <typename T>
__aicore__ inline void CombineAndDispatchRebatch<T>::InitTilingData(
    const CombineAndDispatchRebatchTilingData *tilingData)
{
    isCombineRebatch_ = tilingData->combineAndDispatchRebatchInfo.isCombineRebatch;
    hiddenSize_       = tilingData->combineAndDispatchRebatchInfo.hiddenSize;
    maxBs_            = tilingData->combineAndDispatchRebatchInfo.maxBs;
    k_                = tilingData->combineAndDispatchRebatchInfo.k;
    // aivNum_           = tilingData->combineAndDispatchRebatchInfo.aivNum;
}

template <typename T>
__aicore__ inline void CombineAndDispatchRebatch<T>::InitGlobalBuffer(
    GM_ADDR batchBeforeDispatch,
    GM_ADDR residualsBeforeDispatch,
    GM_ADDR prefixIndex,
    GM_ADDR discardLabel,
    GM_ADDR discardTokens,
    GM_ADDR discardMoeReqIdTensor,
    GM_ADDR reqIdList,
    GM_ADDR discardResiduals,
    GM_ADDR runningTopk,
    GM_ADDR discardTopk,
    GM_ADDR runningTopkWeight,
    GM_ADDR discardTopkWeight,
    GM_ADDR moeCntInput,
    // GM_ADDR rebatchOutput,
    GM_ADDR rebatchResiduals,
    GM_ADDR actualBs,
    GM_ADDR historyBs,
    GM_ADDR runningBs,
    GM_ADDR originalBs)
{
    batchBeforeDispatchGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(batchBeforeDispatch));
    residualsBeforeDispatchGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(residualsBeforeDispatch));
    prefixIndexGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(prefixIndex));
    discardLabelGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(discardLabel));
    discardTokensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(discardTokens));
    discardMoeReqIdTensorGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(discardMoeReqIdTensor));
    reqIdListGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(reqIdList));
    discardResidualsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(discardResiduals));
    runningTopkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(runningTopk));
    discardTopkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(discardTopk));
    runningTopkWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(runningTopkWeight));
    discardTopkWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(discardTopkWeight));
    moeCntInputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(moeCntInput));
    // rebatchOutputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(rebatchOutput));
    rebatchResidualsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(rebatchResiduals));
    actualBsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(actualBs));
    historyBsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(historyBs));
    runningBsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(runningBs));
    originalBsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(originalBs));
}

template <typename T>
__aicore__ inline void CombineAndDispatchRebatch<T>::Init(
    GM_ADDR batchBeforeDispatch,
    GM_ADDR residualsBeforeDispatch,
    GM_ADDR prefixIndex,
    GM_ADDR discardLabel,
    GM_ADDR discardTokens,
    GM_ADDR discardMoeReqIdTensor,
    GM_ADDR reqIdList,
    GM_ADDR discardResiduals,
    GM_ADDR runningTopk,
    GM_ADDR discardTopk,
    GM_ADDR runningTopkWeight,
    GM_ADDR discardTopkWeight,
    GM_ADDR moeCntInput,
    GM_ADDR actualBs,
    GM_ADDR historyBs,
    GM_ADDR runningBs,
    GM_ADDR originalBs,
    // GM_ADDR rebatchOutput,
    GM_ADDR rebatchResiduals,
    
    GM_ADDR workspaceGM,
    TPipe *pipe,
    const CombineAndDispatchRebatchTilingData *tilingData)
{
    pipe_ = pipe;
    icache_preload(8);
    InitTilingData(tilingData);
    InitGlobalBuffer(batchBeforeDispatch, residualsBeforeDispatch, prefixIndex,
                     discardLabel,
                     discardTokens, discardMoeReqIdTensor, reqIdList, discardResiduals,
                      runningTopk, discardTopk, runningTopkWeight,
                      discardTopkWeight, moeCntInput, rebatchResiduals,actualBs, historyBs, runningBs,originalBs);
    InitBuffers(pipe_);
    blockIdx = GetBlockIdx();
    blockNum = GetBlockNum();
}


} // namespace CombineAndDispatchRebatchImpl
