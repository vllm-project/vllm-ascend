#include "kernel_operator.h"

using namespace AscendC;

class RejectionSampleGreedyV310Kernel {
public:
    __aicore__ inline RejectionSampleGreedyV310Kernel() {}

    __aicore__ inline void Init(
        GM_ADDR cuNumDraftTokens,
        GM_ADDR draftTokenIds,
        GM_ADDR targetArgmax,
        GM_ADDR bonusTokenIds,
        GM_ADDR outputTokenIds,
        const RejectionSampleGreedyV310TilingData* tilingData)
    {
        usedCoreNum = tilingData->usedCoreNum;
        batchSize = tilingData->batchSize;
        maxSpecLen = tilingData->maxSpecLen;
        alignedOutputLen = tilingData->alignedOutputLen;

        gmCumulativeCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cuNumDraftTokens), batchSize);
        gmDraftTokenIds.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(draftTokenIds), batchSize * maxSpecLen);
        gmTargetArgmax.SetGlobalBuffer(
            reinterpret_cast<__gm__ int64_t*>(targetArgmax), batchSize * maxSpecLen);
        gmBonusTokenIds.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(bonusTokenIds), batchSize);
        gmOutputTokenIds.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(outputTokenIds), batchSize * alignedOutputLen);
        pipe.InitBuffer(outputBuffer, alignedOutputLen * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIndex = GetBlockIdx();
        for (uint32_t requestIndex = blockIndex; requestIndex < batchSize; requestIndex += usedCoreNum) {
            ProcessOneRequest(requestIndex);
        }
    }

private:
    __aicore__ inline void ProcessOneRequest(uint32_t requestIndex)
    {
        LocalTensor<int32_t> outputLocal = outputBuffer.Get<int32_t>();
        Duplicate(outputLocal, static_cast<int32_t>(-1), alignedOutputLen);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);

        int32_t start = requestIndex == 0 ? 0 : gmCumulativeCounts.GetValue(requestIndex - 1);
        int32_t end = gmCumulativeCounts.GetValue(requestIndex);
        if (start < 0) {
            start = 0;
        }
        if (end < start) {
            end = start;
        }
        int32_t count = end - start;
        if (count > static_cast<int32_t>(maxSpecLen)) {
            count = static_cast<int32_t>(maxSpecLen);
        }

        bool allAccepted = true;
        for (int32_t position = 0; position < count; ++position) {
            const int32_t draftToken = gmDraftTokenIds.GetValue(start + position);
            const int64_t targetToken = gmTargetArgmax.GetValue(start + position);
            outputLocal.SetValue(position, static_cast<int32_t>(targetToken));
            if (static_cast<int64_t>(draftToken) != targetToken) {
                allAccepted = false;
                break;
            }
        }
        if (allAccepted) {
            outputLocal.SetValue(count, gmBonusTokenIds.GetValue(requestIndex));
        }

        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        DataCopy(gmOutputTokenIds[requestIndex * alignedOutputLen], outputLocal, alignedOutputLen);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
    }

private:
    GlobalTensor<int32_t> gmCumulativeCounts;
    GlobalTensor<int32_t> gmDraftTokenIds;
    GlobalTensor<int64_t> gmTargetArgmax;
    GlobalTensor<int32_t> gmBonusTokenIds;
    GlobalTensor<int32_t> gmOutputTokenIds;

    uint32_t usedCoreNum;
    uint32_t batchSize;
    uint32_t maxSpecLen;
    uint32_t alignedOutputLen;
    TPipe pipe;
    TBuf<QuePosition::VECCALC> outputBuffer;
};

extern "C" __global__ __aicore__ void rejection_sample_greedy_v310(
    GM_ADDR cuNumDraftTokens,
    GM_ADDR draftTokenIds,
    GM_ADDR targetArgmax,
    GM_ADDR bonusTokenIds,
    GM_ADDR outputTokenIds,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (GetBlockIdx() >= tilingData.usedCoreNum) {
        return;
    }
    if (TILING_KEY_IS(1)) {
        RejectionSampleGreedyV310Kernel op;
        op.Init(
            cuNumDraftTokens,
            draftTokenIds,
            targetArgmax,
            bonusTokenIds,
            outputTokenIds,
            &tilingData);
        op.Process();
    }
}
