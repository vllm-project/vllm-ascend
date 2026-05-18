/**
 * CopyAndExpandEagleInputs 算子 Kernel 实现 (DataCopy 版)
 *
 * 多核策略：
 *   所有 GM 读写通过 DataCopy 完成（不使用 GlobalTensor::SetValue/GetValue 访问 GM）。
 *   UB (LocalTensor) 上使用 SetValue/GetValue 构建数据，再 DataCopy 到 GM。
 *   对齐处理参考 CANN 内置算子的 DataCopyCustom 模式。
 */

#include "kernel_operator.h"

using namespace AscendC;

// ONE_BLK_SIZE comes from AscendC namespace (32 bytes per block)

class CopyAndExpandEagleInputsKernel {
public:
    __aicore__ inline CopyAndExpandEagleInputsKernel() {}

    __aicore__ inline void Init(GM_ADDR targetTokenIds, GM_ADDR targetPositions,
                                GM_ADDR nextTokenIds, GM_ADDR queryStartLoc,
                                GM_ADDR queryEndLoc,
                                GM_ADDR outInputIds, GM_ADDR outPositions,
                                GM_ADDR outIsRejectedTokenMask, GM_ADDR outIsMaskedTokenMask,
                                GM_ADDR outNewTokenIndices, GM_ADDR outHiddenStateMapping,
                                const CopyAndExpandEagleInputsTilingData* tilingData)
    {
        usedCoreNum = tilingData->usedCoreNum;
        numReqs = tilingData->numReqs;
        reqsPerCore = tilingData->reqsPerCore;
        remainderReqs = tilingData->remainderReqs;
        paddingTokenId = tilingData->paddingTokenId;
        parallelDraftingTokenId = tilingData->parallelDraftingTokenId;
        numPaddingSlotsPerReq = tilingData->numPaddingSlotsPerReq;
        totalInputTokens = tilingData->totalInputTokens;
        totalDraftTokens = tilingData->totalDraftTokens;

        uint32_t coreId = GetBlockIdx();
        if (coreId < remainderReqs) {
            myStartReq = coreId * (reqsPerCore + 1);
            myNumReqs = reqsPerCore + 1;
        } else {
            myStartReq = remainderReqs * (reqsPerCore + 1) + (coreId - remainderReqs) * reqsPerCore;
            myNumReqs = reqsPerCore;
        }

        // 绑定 GM Tensor
        gmTargetTokenIds.SetGlobalBuffer((__gm__ int32_t*)targetTokenIds, totalInputTokens);
        gmTargetPositions.SetGlobalBuffer((__gm__ int32_t*)targetPositions, totalInputTokens);
        gmNextTokenIds.SetGlobalBuffer((__gm__ int32_t*)nextTokenIds, numReqs);
        gmQueryStartLoc.SetGlobalBuffer((__gm__ int32_t*)queryStartLoc, numReqs + 1);
        gmQueryEndLoc.SetGlobalBuffer((__gm__ int32_t*)queryEndLoc, numReqs);
        gmOutInputIds.SetGlobalBuffer((__gm__ int32_t*)outInputIds, totalDraftTokens);
        gmOutPositions.SetGlobalBuffer((__gm__ int32_t*)outPositions, totalDraftTokens);
        gmOutIsRejectedTokenMask.SetGlobalBuffer((__gm__ int8_t*)outIsRejectedTokenMask, totalDraftTokens);
        gmOutIsMaskedTokenMask.SetGlobalBuffer((__gm__ int8_t*)outIsMaskedTokenMask, totalDraftTokens);
        gmOutNewTokenIndices.SetGlobalBuffer((__gm__ int32_t*)outNewTokenIndices, numPaddingSlotsPerReq * numReqs);
        gmOutHiddenStateMapping.SetGlobalBuffer((__gm__ int32_t*)outHiddenStateMapping, totalInputTokens);

        // 分配 UB 缓冲区 —— 每个 TBuf 的基地址自动 32 字节对齐
        // 元数据各自独立 TBuf，避免 UB 地址不对齐
        uint32_t metaAligned = AlignUp((myNumReqs + 1) * sizeof(int32_t), ONE_BLK_SIZE);
        pipe.InitBuffer(qsBuf, metaAligned);
        pipe.InitBuffer(qeBuf, AlignUp(myNumReqs * sizeof(int32_t), ONE_BLK_SIZE));
        pipe.InitBuffer(ntBuf, AlignUp(myNumReqs * sizeof(int32_t), ONE_BLK_SIZE));

        // I/O 缓冲区
        constexpr uint32_t MAX_PER_REQ = 4096;
        pipe.InitBuffer(inputBuf,  AlignUp(MAX_PER_REQ * sizeof(int32_t), ONE_BLK_SIZE));
        pipe.InitBuffer(outIdsBuf, AlignUp(MAX_PER_REQ * sizeof(int32_t), ONE_BLK_SIZE));
        pipe.InitBuffer(outPosBuf, AlignUp(MAX_PER_REQ * sizeof(int32_t), ONE_BLK_SIZE));
        pipe.InitBuffer(outRejBuf, AlignUp(MAX_PER_REQ * sizeof(int8_t),  ONE_BLK_SIZE));
        pipe.InitBuffer(outMskBuf, AlignUp(MAX_PER_REQ * sizeof(int8_t),  ONE_BLK_SIZE));
        pipe.InitBuffer(ntiBuf,    AlignUp(64 * sizeof(int32_t), ONE_BLK_SIZE));
        pipe.InitBuffer(hsmBuf,    AlignUp(MAX_PER_REQ * sizeof(int32_t), ONE_BLK_SIZE));

        // DataCopy 元数据到各自 UB
        if (myNumReqs > 0) {
            LocalTensor<int32_t> lqs = qsBuf.Get<int32_t>();
            DataCopyIn(lqs, gmQueryStartLoc, (int32_t)myStartReq, (int32_t)(myNumReqs + 1));

            LocalTensor<int32_t> lqe = qeBuf.Get<int32_t>();
            DataCopyIn(lqe, gmQueryEndLoc, (int32_t)myStartReq, (int32_t)myNumReqs);

            LocalTensor<int32_t> lnt = ntBuf.Get<int32_t>();
            DataCopyIn(lnt, gmNextTokenIds, (int32_t)myStartReq, (int32_t)myNumReqs);
        }
    }

    __aicore__ inline void ProcessShiftFalse()
    {
        for (uint32_t rLocal = 0; rLocal < myNumReqs; rLocal++) {
            ProcessOneRequestShiftFalse(myStartReq + rLocal, rLocal);
        }
    }

    __aicore__ inline void ProcessShiftTrue()
    {
        for (uint32_t rLocal = 0; rLocal < myNumReqs; rLocal++) {
            ProcessOneRequestShiftTrue(myStartReq + rLocal, rLocal);
        }
    }

private:
    // ============================================================
    // AlignUp 辅助
    // ============================================================
    static __aicore__ inline uint32_t AlignUp(uint32_t x, uint32_t a)
    {
        return (x + a - 1) / a * a;
    }

    // ============================================================
    // GM → UB: 标准 DataCopy，count 自动 round-up 到 block 对齐
    // 多读的元素在 UB 中不会被使用，安全无害
    // ============================================================
    __aicore__ inline void DataCopyIn(LocalTensor<int32_t>& dst,
                                       GlobalTensor<int32_t>& src,
                                       int32_t gmOffset, int32_t count)
    {
        if (count <= 0) return;
        constexpr int32_t ELEMS_PER_BLK = ONE_BLK_SIZE / (int32_t)sizeof(int32_t);  // 8
        int32_t aligned = (count + ELEMS_PER_BLK - 1) / ELEMS_PER_BLK * ELEMS_PER_BLK;
        DataCopy(dst, src[gmOffset], aligned);
        pipe_barrier(PIPE_ALL);
    }

    // ============================================================
    // UB → GM: DataCopyPad + DataCopyExtParams（C220 支持任意字节数）
    // 精确写入 count 个元素，不越界覆盖相邻数据
    // ============================================================
    __aicore__ inline void DataCopyOut_int32(GlobalTensor<int32_t>& dst,
                                              LocalTensor<int32_t>& src,
                                              int32_t gmOffset, int32_t count)
    {
        if (count <= 0) return;
        uint32_t totalBytes = static_cast<uint32_t>(count) * static_cast<uint32_t>(sizeof(int32_t));
        pipe_barrier(PIPE_ALL);
        DataCopyPad(dst[gmOffset], src, DataCopyExtParams(1, totalBytes, 0, 0, 0));
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void DataCopyOut_int8(GlobalTensor<int8_t>& dst,
                                             LocalTensor<int8_t>& src,
                                             int32_t gmOffset, int32_t count)
    {
        if (count <= 0) return;
        uint32_t totalBytes = static_cast<uint32_t>(count) * static_cast<uint32_t>(sizeof(int8_t));
        pipe_barrier(PIPE_ALL);
        DataCopyPad(dst[gmOffset], src, DataCopyExtParams(1, totalBytes, 0, 0, 0));
        pipe_barrier(PIPE_ALL);
    }

    // ============================================================
    // 元数据读取 (从各自 UB 缓冲区)
    // ============================================================
    __aicore__ inline int32_t ReadQS(uint32_t rLocal) {
        return qsBuf.Get<int32_t>().GetValue(rLocal);
    }
    __aicore__ inline int32_t ReadNextQS(uint32_t rLocal) {
        return qsBuf.Get<int32_t>().GetValue(rLocal + 1);
    }
    __aicore__ inline int32_t ReadQE(uint32_t rLocal) {
        return qeBuf.Get<int32_t>().GetValue(rLocal);
    }
    __aicore__ inline int32_t ReadNT(uint32_t rLocal) {
        return ntBuf.Get<int32_t>().GetValue(rLocal);
    }

    // ============================================================
    // shift_input_ids = false
    // ============================================================
    __aicore__ inline void ProcessOneRequestShiftFalse(uint32_t r, uint32_t rLocal)
    {
        int32_t queryStart = ReadQS(rLocal);
        int32_t nextQueryStart = ReadNextQS(rLocal);
        int32_t queryEnd = ReadQE(rLocal);

        int32_t numRejected = nextQueryStart - queryEnd - 1;
        if (numRejected < 0) numRejected = 0;
        int32_t numValid = queryEnd - queryStart + 1;
        if (numValid < 0) numValid = 0;

        int32_t outputStart = queryStart + (int32_t)r * (int32_t)numPaddingSlotsPerReq;
        int32_t outputLen = numValid + (int32_t)numPaddingSlotsPerReq + numRejected;

        // 读取输入 token 到 UB
        int32_t numInputTokensForReq = nextQueryStart - queryStart;
        LocalTensor<int32_t> localInput = inputBuf.Get<int32_t>();
        if (numInputTokensForReq > 0) {
            DataCopyIn(localInput, gmTargetTokenIds, queryStart, numInputTokensForReq);
        }

        // 读取起始 position
        LocalTensor<int32_t> localTmpPos = hsmBuf.Get<int32_t>();
        DataCopyIn(localTmpPos, gmTargetPositions, queryStart, 1);
        int32_t startPos = localTmpPos.GetValue(0);

        int32_t nextTokenId = ReadNT(rLocal);

        // 构建输出到 UB
        LocalTensor<int32_t> lIds = outIdsBuf.Get<int32_t>();
        LocalTensor<int32_t> lPos = outPosBuf.Get<int32_t>();
        LocalTensor<int8_t>  lRej = outRejBuf.Get<int8_t>();
        LocalTensor<int8_t>  lMsk = outMskBuf.Get<int8_t>();

        for (int32_t j = 0; j < numValid; j++) {
            int32_t inIdx = j;
            if (inIdx >= numInputTokensForReq) inIdx = numInputTokensForReq - 1;
            lIds.SetValue(j, localInput.GetValue(inIdx));
            lPos.SetValue(j, startPos + j);
            lRej.SetValue(j, (int8_t)0);
            lMsk.SetValue(j, (int8_t)0);
        }
        // Bonus
        lIds.SetValue(numValid, nextTokenId);
        lPos.SetValue(numValid, startPos + numValid);
        lRej.SetValue(numValid, (int8_t)0);
        lMsk.SetValue(numValid, (int8_t)0);
        // Parallel Draft
        for (int32_t k = 1; k < (int32_t)numPaddingSlotsPerReq; k++) {
            int32_t j = numValid + k;
            lIds.SetValue(j, parallelDraftingTokenId);
            lPos.SetValue(j, startPos + j);
            lRej.SetValue(j, (int8_t)0);
            lMsk.SetValue(j, (int8_t)1);
        }
        // Rejected
        for (int32_t k = 0; k < numRejected; k++) {
            int32_t j = numValid + (int32_t)numPaddingSlotsPerReq + k;
            lIds.SetValue(j, paddingTokenId);
            lPos.SetValue(j, (int32_t)0);
            lRej.SetValue(j, (int8_t)1);
            lMsk.SetValue(j, (int8_t)0);
        }

        // UB → GM
        DataCopyOut_int32(gmOutInputIds, lIds, outputStart, outputLen);
        DataCopyOut_int32(gmOutPositions, lPos, outputStart, outputLen);
        DataCopyOut_int8(gmOutIsRejectedTokenMask, lRej, outputStart, outputLen);
        DataCopyOut_int8(gmOutIsMaskedTokenMask, lMsk, outputStart, outputLen);

        // NTI
        LocalTensor<int32_t> lNti = ntiBuf.Get<int32_t>();
        lNti.SetValue(0, outputStart + numValid);
        for (int32_t k = 1; k < (int32_t)numPaddingSlotsPerReq; k++) {
            lNti.SetValue(k, outputStart + numValid + k);
        }
        int32_t ntiOff = (int32_t)r * (int32_t)numPaddingSlotsPerReq;
        DataCopyOut_int32(gmOutNewTokenIndices, lNti, ntiOff, (int32_t)numPaddingSlotsPerReq);
    }

    // ============================================================
    // shift_input_ids = true
    // ============================================================
    __aicore__ inline void ProcessOneRequestShiftTrue(uint32_t r, uint32_t rLocal)
    {
        int32_t queryStart = ReadQS(rLocal);
        int32_t nextQueryStart = ReadNextQS(rLocal);
        int32_t queryEnd = ReadQE(rLocal);

        int32_t numRejected = nextQueryStart - queryEnd - 1;
        if (numRejected < 0) numRejected = 0;
        int32_t numValid = queryEnd - queryStart;
        if (numValid < 0) numValid = 0;

        int32_t outputStart = queryStart + (int32_t)r * ((int32_t)numPaddingSlotsPerReq - 1);
        int32_t outputLen = numValid + (int32_t)numPaddingSlotsPerReq + numRejected;

        int32_t numInputTokensForReq = nextQueryStart - queryStart;
        LocalTensor<int32_t> localInput = inputBuf.Get<int32_t>();
        int32_t readStart = queryStart + 1;
        int32_t readCount = numValid;
        if (readStart + readCount > (int32_t)totalInputTokens) {
            readCount = (int32_t)totalInputTokens - readStart;
            if (readCount < 0) readCount = 0;
        }
        if (readCount > 0) {
            DataCopyIn(localInput, gmTargetTokenIds, readStart, readCount);
        }

        LocalTensor<int32_t> localTmpPos = hsmBuf.Get<int32_t>();
        DataCopyIn(localTmpPos, gmTargetPositions, queryStart, 1);
        int32_t startPos = localTmpPos.GetValue(0);

        int32_t nextTokenId = ReadNT(rLocal);

        LocalTensor<int32_t> lIds = outIdsBuf.Get<int32_t>();
        LocalTensor<int32_t> lPos = outPosBuf.Get<int32_t>();
        LocalTensor<int8_t>  lRej = outRejBuf.Get<int8_t>();
        LocalTensor<int8_t>  lMsk = outMskBuf.Get<int8_t>();

        for (int32_t j = 0; j < numValid; j++) {
            int32_t inIdx = j;
            if (inIdx >= readCount && readCount > 0) inIdx = readCount - 1;
            lIds.SetValue(j, readCount > 0 ? localInput.GetValue(inIdx) : (int32_t)0);
            lPos.SetValue(j, startPos + j);
            lRej.SetValue(j, (int8_t)0);
            lMsk.SetValue(j, (int8_t)0);
        }
        lIds.SetValue(numValid, nextTokenId);
        lPos.SetValue(numValid, startPos + numValid);
        lRej.SetValue(numValid, (int8_t)0);
        lMsk.SetValue(numValid, (int8_t)0);
        for (int32_t k = 1; k < (int32_t)numPaddingSlotsPerReq; k++) {
            int32_t j = numValid + k;
            lIds.SetValue(j, parallelDraftingTokenId);
            lPos.SetValue(j, startPos + j);
            lRej.SetValue(j, (int8_t)0);
            lMsk.SetValue(j, (int8_t)1);
        }
        for (int32_t k = 0; k < numRejected; k++) {
            int32_t j = numValid + (int32_t)numPaddingSlotsPerReq + k;
            lIds.SetValue(j, paddingTokenId);
            lPos.SetValue(j, (int32_t)0);
            lRej.SetValue(j, (int8_t)1);
            lMsk.SetValue(j, (int8_t)0);
        }

        DataCopyOut_int32(gmOutInputIds, lIds, outputStart, outputLen);
        DataCopyOut_int32(gmOutPositions, lPos, outputStart, outputLen);
        DataCopyOut_int8(gmOutIsRejectedTokenMask, lRej, outputStart, outputLen);
        DataCopyOut_int8(gmOutIsMaskedTokenMask, lMsk, outputStart, outputLen);

        LocalTensor<int32_t> lNti = ntiBuf.Get<int32_t>();
        lNti.SetValue(0, outputStart + numValid);
        for (int32_t k = 1; k < (int32_t)numPaddingSlotsPerReq; k++) {
            lNti.SetValue(k, outputStart + numValid + k);
        }
        int32_t ntiOff = (int32_t)r * (int32_t)numPaddingSlotsPerReq;
        DataCopyOut_int32(gmOutNewTokenIndices, lNti, ntiOff, (int32_t)numPaddingSlotsPerReq);

        // hidden_state_mapping
        LocalTensor<int32_t> lHsm = hsmBuf.Get<int32_t>();
        for (int32_t j = 0; j < numInputTokensForReq; j++) {
            lHsm.SetValue(j, outputStart + j);
        }
        DataCopyOut_int32(gmOutHiddenStateMapping, lHsm, queryStart, numInputTokensForReq);
    }

private:
    GlobalTensor<int32_t> gmTargetTokenIds, gmTargetPositions, gmNextTokenIds;
    GlobalTensor<int32_t> gmQueryStartLoc, gmQueryEndLoc;
    GlobalTensor<int32_t> gmOutInputIds, gmOutPositions;
    GlobalTensor<int8_t>  gmOutIsRejectedTokenMask, gmOutIsMaskedTokenMask;
    GlobalTensor<int32_t> gmOutNewTokenIndices, gmOutHiddenStateMapping;

    uint32_t usedCoreNum, numReqs, reqsPerCore, remainderReqs;
    int32_t  paddingTokenId, parallelDraftingTokenId;
    uint32_t numPaddingSlotsPerReq, totalInputTokens, totalDraftTokens;
    uint32_t myStartReq, myNumReqs;

    TPipe pipe;
    TBuf<QuePosition::VECCALC> qsBuf, qeBuf, ntBuf;
    TBuf<QuePosition::VECCALC> inputBuf, outIdsBuf, outPosBuf;
    TBuf<QuePosition::VECCALC> outRejBuf, outMskBuf, ntiBuf, hsmBuf;
};

extern "C" __global__ __aicore__ void copy_and_expand_eagle_inputs(
    GM_ADDR targetTokenIds, GM_ADDR targetPositions,
    GM_ADDR nextTokenIds, GM_ADDR queryStartLoc,
    GM_ADDR queryEndLoc,
    GM_ADDR outInputIds, GM_ADDR outPositions,
    GM_ADDR outIsRejectedTokenMask, GM_ADDR outIsMaskedTokenMask,
    GM_ADDR outNewTokenIndices, GM_ADDR outHiddenStateMapping,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    if (GetBlockIdx() >= tilingData.usedCoreNum) {
        return;
    }

    if (TILING_KEY_IS(1)) {
        CopyAndExpandEagleInputsKernel op;
        op.Init(targetTokenIds, targetPositions, nextTokenIds, queryStartLoc, queryEndLoc,
                outInputIds, outPositions, outIsRejectedTokenMask, outIsMaskedTokenMask,
                outNewTokenIndices, outHiddenStateMapping, &tilingData);

        if (tilingData.shiftInputIds == 0) {
            op.ProcessShiftFalse();
        } else {
            op.ProcessShiftTrue();
        }
    }
}
