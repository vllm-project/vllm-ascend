#include "kernel_operator.h"

#include "../../combine_and_dispatch_rebatch/op_kernel/combine_and_dispatch_rebatch.h"
#include "../../combine_and_dispatch_rebatch/op_kernel/combine_and_dispatch_rebatch_impl.h"
#undef DATA_FLUSH
#include "../../attn_metadata_update/op_kernel/attn_metadata_update.h"
#include "rebatch_and_update_tiling.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void rebatch_and_update(
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
    GM_ADDR positions,
    GM_ADDR slotMapping,
    GM_ADDR blockTable,
    GM_ADDR seqLens,
    GM_ADDR discardPositions,
    GM_ADDR discardSlotMapping,
    GM_ADDR discardContextLens,
    GM_ADDR discardBlockTables,
    GM_ADDR discardAttnCnt,
    GM_ADDR discardReqIds,
    GM_ADDR rebatchResiduals,
    GM_ADDR outPositions,
    GM_ADDR outSlots,
    GM_ADDR outSeqLens,
    GM_ADDR outBlockTable,
    GM_ADDR workspace,
    GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(RebatchAndUpdateTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        RebatchAndUpdateTilingData, tilingData, tilingGM);

    TPipe pipe;
    CombineAndDispatchRebatchImpl::CombineAndDispatchRebatch<bfloat16_t>
        rebatchOp;
    rebatchOp.Init(
        batchBeforeDispatch, residualsBeforeDispatch, prefixIndex, discardLabel,
        discardTokens, discardMoeReqIdTensor, reqIdList, discardResiduals,
        runningTopk, discardTopk, runningTopkWeight, discardTopkWeight,
        moeCntInput, actualBs, historyBs, runningBs, originalBs,
        rebatchResiduals, workspace, &pipe, &tilingData.rebatch);
    //rebatchOp.SetPrefixIndex(prefixIndex);
    //rebatchOp.SetCopyResidualBack(false);
    rebatchOp.Process();

    // Both phases use all launched AIV cores. Do not recycle UB until every
    // core has completed all GM writes from the rebatch phase.
    pipe_barrier(PIPE_ALL);
    SyncAll();
    pipe_barrier(PIPE_ALL);
    pipe.Reset();

    AttnMetadataUpdateImpl::AttnMetadataUpdate updateOp;
    updateOp.Init(
        reqIdList, discardLabel, positions, slotMapping, blockTable, seqLens,
        discardPositions, discardSlotMapping, discardContextLens,
        discardBlockTables, discardAttnCnt, discardReqIds, originalBs,
        outPositions, outSlots, outSeqLens, outBlockTable, &pipe,
        &tilingData.update);
    updateOp.Process();
}
