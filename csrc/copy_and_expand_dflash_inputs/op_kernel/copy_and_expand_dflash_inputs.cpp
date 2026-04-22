/**
 * CopyAndExpandDflashInputs 算子 Kernel 实现
 *
 * 主要功能：
 * - 复制 context positions
 * - 计算 query positions (last_context_pos + 1 + offset)
 * - 设置 input_ids (next_token + mask tokens)
 * - 计算 context 和 query 的 slot_mapping
 * - 输出 token_indices_to_sample (跳过 bonus token)
 */

#include "kernel_operator.h"

using namespace AscendC;

class CopyAndExpandDflashInputsKernel {
public:
    __aicore__ inline CopyAndExpandDflashInputsKernel() {}

    __aicore__ inline void Init(
        GM_ADDR nextTokenIds,        // [num_reqs]
        GM_ADDR targetPositions,     // [num_context]
        GM_ADDR queryStartLoc,       // [num_reqs + 1]
        GM_ADDR numRejectedTokens,   // [num_reqs] or null
        GM_ADDR blockTable,          // [num_reqs, max_blocks]
        GM_ADDR outInputIds,         // [total_query_tokens] (in-place output)
        GM_ADDR outContextPositions, // [num_context] (in-place output)
        GM_ADDR outQueryPositions,   // [total_query_tokens] (in-place output)
        GM_ADDR outContextSlotMapping, // [num_context] (in-place output)
        GM_ADDR outQuerySlotMapping, // [total_query_tokens] (in-place output)
        GM_ADDR outTokenIndices,     // [num_reqs * num_speculative_tokens] (in-place output)
        const CopyAndExpandDflashInputsTilingData* tilingData)
    {
        usedCoreNum = tilingData->usedCoreNum;
        numReqs = tilingData->numReqs;
        reqsPerCore = tilingData->reqsPerCore;
        remainderReqs = tilingData->remainderReqs;
        parallelDraftingTokenId = tilingData->parallelDraftingTokenId;
        numQueryPerReq = tilingData->numQueryPerReq;
        numSpeculativeTokens = tilingData->numSpeculativeTokens;
        blockSize = tilingData->blockSize;
        totalInputTokens = tilingData->totalInputTokens;
        hasNumRejected = tilingData->hasNumRejected;
        blockTableStride = tilingData->blockTableStride;
        totalQueryTokens = tilingData->totalQueryTokens;

        uint32_t coreId = GetBlockIdx();
        if (coreId < remainderReqs) {
            myStartReq = coreId * (reqsPerCore + 1);
            myNumReqs = reqsPerCore + 1;
        } else {
            myStartReq = remainderReqs * (reqsPerCore + 1) + (coreId - remainderReqs) * reqsPerCore;
            myNumReqs = reqsPerCore;
        }

        // 绑定 GM Tensor
        gmNextTokenIds.SetGlobalBuffer((__gm__ int32_t*)nextTokenIds, numReqs);
        gmTargetPositions.SetGlobalBuffer((__gm__ int64_t*)targetPositions, totalInputTokens);
        gmQueryStartLoc.SetGlobalBuffer((__gm__ int32_t*)queryStartLoc, numReqs + 1);
        if (hasNumRejected && numRejectedTokens != nullptr) {
            gmNumRejectedTokens.SetGlobalBuffer((__gm__ int32_t*)numRejectedTokens, numReqs);
        }
        gmBlockTable.SetGlobalBuffer((__gm__ int32_t*)blockTable, numReqs * blockTableStride);
        gmOutInputIds.SetGlobalBuffer((__gm__ int32_t*)outInputIds, totalQueryTokens);
        gmOutContextPositions.SetGlobalBuffer((__gm__ int32_t*)outContextPositions, totalInputTokens);
        gmOutQueryPositions.SetGlobalBuffer((__gm__ int32_t*)outQueryPositions, totalQueryTokens);
        gmOutContextSlotMapping.SetGlobalBuffer((__gm__ int32_t*)outContextSlotMapping, totalInputTokens);
        gmOutQuerySlotMapping.SetGlobalBuffer((__gm__ int32_t*)outQuerySlotMapping, totalQueryTokens);
        gmOutTokenIndices.SetGlobalBuffer((__gm__ int32_t*)outTokenIndices, numReqs * numSpeculativeTokens);

        // 分配 UB 缓冲区
        constexpr uint32_t MAX_PER_REQ = 4096;
        pipe.InitBuffer(qsBuf, AlignUp((myNumReqs + 1) * sizeof(int32_t), ONE_BLK_SIZE));
        pipe.InitBuffer(ntBuf, AlignUp(myNumReqs * sizeof(int32_t), ONE_BLK_SIZE));
        pipe.InitBuffer(nrBuf, AlignUp(myNumReqs * sizeof(int32_t), ONE_BLK_SIZE));
        pipe.InitBuffer(inputBuf, AlignUp(MAX_PER_REQ * sizeof(int64_t), ONE_BLK_SIZE));
        pipe.InitBuffer(outCtxPosBuf, AlignUp(MAX_PER_REQ * sizeof(int32_t), ONE_BLK_SIZE));
        pipe.InitBuffer(outQueryPosBuf, AlignUp(numQueryPerReq * sizeof(int32_t), ONE_BLK_SIZE));
        pipe.InitBuffer(outCtxSlotBuf, AlignUp(MAX_PER_REQ * sizeof(int32_t), ONE_BLK_SIZE));
        pipe.InitBuffer(outQuerySlotBuf, AlignUp(numQueryPerReq * sizeof(int32_t), ONE_BLK_SIZE));
        pipe.InitBuffer(tmpBuf, AlignUp(64 * sizeof(int32_t), ONE_BLK_SIZE));

        // DataCopy 元数据到 UB
        if (myNumReqs > 0) {
            LocalTensor<int32_t> lqs = qsBuf.Get<int32_t>();
            DataCopyIn(lqs, gmQueryStartLoc, (int32_t)myStartReq, (int32_t)(myNumReqs + 1));

            LocalTensor<int32_t> lnt = ntBuf.Get<int32_t>();
            DataCopyIn(lnt, gmNextTokenIds, (int32_t)myStartReq, (int32_t)myNumReqs);

            if (hasNumRejected) {
                LocalTensor<int32_t> lnr = nrBuf.Get<int32_t>();
                DataCopyIn(lnr, gmNumRejectedTokens, (int32_t)myStartReq, (int32_t)myNumReqs);
            }
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t rLocal = 0; rLocal < myNumReqs; rLocal++) {
            ProcessOneRequest(myStartReq + rLocal, rLocal);
        }
    }

private:
    static __aicore__ inline uint32_t AlignUp(uint32_t x, uint32_t a)
    {
        return (x + a - 1) / a * a;
    }

    __aicore__ inline void DataCopyIn(LocalTensor<int32_t>& dst,
                                       GlobalTensor<int32_t>& src,
                                       int32_t gmOffset, int32_t count)
    {
        if (count <= 0) return;
        constexpr int32_t ELEMS_PER_BLK = ONE_BLK_SIZE / (int32_t)sizeof(int32_t);
        int32_t aligned = (count + ELEMS_PER_BLK - 1) / ELEMS_PER_BLK * ELEMS_PER_BLK;
        DataCopy(dst, src[gmOffset], aligned);
        pipe_barrier(PIPE_ALL);
    }


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

    __aicore__ inline int32_t ReadQS(uint32_t rLocal) {
        return qsBuf.Get<int32_t>().GetValue(rLocal);
    }
    __aicore__ inline int32_t ReadNextQS(uint32_t rLocal) {
        return qsBuf.Get<int32_t>().GetValue(rLocal + 1);
    }
    __aicore__ inline int32_t ReadNT(uint32_t rLocal) {
        return ntBuf.Get<int32_t>().GetValue(rLocal);
    }
    __aicore__ inline int32_t ReadNR(uint32_t rLocal) {
        if (hasNumRejected) {
            return nrBuf.Get<int32_t>().GetValue(rLocal);
        }
        return 0;
    }

    __aicore__ inline void ProcessOneRequest(uint32_t r, uint32_t rLocal)
    {
        int32_t ctxStart = ReadQS(rLocal);
        int32_t ctxEnd = ReadNextQS(rLocal);
        int32_t numRejected = ReadNR(rLocal);

        // 处理所有 context tokens（包括 rejected tokens），用于输出
        int32_t numCtxTokens = ctxEnd - ctxStart;

        // 有效 context 结束位置（排除 rejected tokens），仅用于计算 last position
        int32_t validCtxEnd = ctxEnd - numRejected;
        if (validCtxEnd < ctxStart) validCtxEnd = ctxStart;

        int32_t queryOffset = r * numQueryPerReq;

        // 获取最后一个有效的 context position（排除 rejected tokens），用于计算 query positions
        // 只需要读取最后一个有效token，不需要读取所有tokens
        int64_t lastCtxPos = 0;
        if (validCtxEnd > ctxStart) {
            int32_t lastValidIdx = validCtxEnd - 1;
            // 限制索引不超过 totalInputTokens - 1
            lastValidIdx = (lastValidIdx < static_cast<int32_t>(totalInputTokens)) ? lastValidIdx : static_cast<int32_t>(totalInputTokens) - 1;
            if (lastValidIdx >= 0) {
                lastCtxPos = gmTargetPositions.GetValue(lastValidIdx);
            }
        }

        // 分批处理 context tokens 以避免缓冲区溢出
        constexpr uint32_t BATCH_SIZE = 4096;  // 保持原有缓冲区大小
        LocalTensor<int64_t> ctxPosLocal = inputBuf.Get<int64_t>();
        LocalTensor<int32_t> outCtxPos = outCtxPosBuf.Get<int32_t>();
        LocalTensor<int32_t> outCtxSlot = outCtxSlotBuf.Get<int32_t>();

        int32_t processedTokens = 0;
        while (processedTokens < numCtxTokens) {
            int32_t currentBatchSize = numCtxTokens - processedTokens;
            if (currentBatchSize > static_cast<int32_t>(BATCH_SIZE)) {
                currentBatchSize = static_cast<int32_t>(BATCH_SIZE);
            }

            // 使用 DataCopy 批量读取当前批次的 context positions，避免 scalar Global Memory reads
            int32_t ctxPosIdx = ctxStart + processedTokens;
            // 确保索引范围合法
            ctxPosIdx = (ctxPosIdx < 0) ? 0 : ctxPosIdx;
            ctxPosIdx = (ctxPosIdx >= static_cast<int32_t>(totalInputTokens)) ? static_cast<int32_t>(totalInputTokens) - 1 : ctxPosIdx;

            // 检查是否会越界，如果会则调整当前批次大小
            int32_t remainingGlobalTokens = static_cast<int32_t>(totalInputTokens) - ctxPosIdx;
            if (currentBatchSize > remainingGlobalTokens) {
                currentBatchSize = remainingGlobalTokens;
            }

            if (currentBatchSize > 0) {
                // 使用 DataCopy 批量读取，性能更好
                constexpr int32_t ELEMS_PER_BLK = ONE_BLK_SIZE / (int32_t)sizeof(int64_t);
                int32_t aligned = (currentBatchSize + ELEMS_PER_BLK - 1) / ELEMS_PER_BLK * ELEMS_PER_BLK;
                DataCopy(ctxPosLocal, gmTargetPositions[ctxPosIdx], aligned);
                pipe_barrier(PIPE_ALL);

                // 如果实际读取的数据少于需要的，用 0 填充
                for (int32_t i = currentBatchSize; i < aligned; i++) {
                    ctxPosLocal.SetValue(i, 0);
                }
            }

            // 处理当前批次的输出
            if (currentBatchSize > 0) {
                for (int32_t i = 0; i < currentBatchSize; i++) {
                    int32_t outIdx = processedTokens + i;
                    // 输出 context positions
                    outCtxPos.SetValue(i, static_cast<int32_t>(ctxPosLocal.GetValue(i)));
                    // 计算 context slot mapping
                    int64_t pos = ctxPosLocal.GetValue(i);
                    int32_t blockNum = static_cast<int32_t>(pos / blockSize);
                    // clamp blockNum避免OOB
                    blockNum = (blockNum < static_cast<int32_t>(blockTableStride)) ? blockNum : static_cast<int32_t>(blockTableStride) - 1;
                    int32_t blockId = gmBlockTable.GetValue(r * blockTableStride + blockNum);
                    int32_t blockOffset = static_cast<int32_t>(pos % blockSize);
                    outCtxSlot.SetValue(i, blockId * blockSize + blockOffset);
                }

                // 输出当前批次的结果
                DataCopyOut_int32(gmOutContextPositions, outCtxPos, ctxStart + processedTokens, currentBatchSize);
                DataCopyOut_int32(gmOutContextSlotMapping, outCtxSlot, ctxStart + processedTokens, currentBatchSize);
            }

            processedTokens += currentBatchSize;
        }

        // 计算 query positions 和 input_ids
        LocalTensor<int32_t> queryPosLocal = outQueryPosBuf.Get<int32_t>();
        LocalTensor<int32_t> inputIdsLocal = tmpBuf.Get<int32_t>();
        int32_t nextTokenId = ReadNT(rLocal);

        for (uint32_t qIdx = 0; qIdx < numQueryPerReq; qIdx++) {
            int64_t queryPos = lastCtxPos + 1 + qIdx;
            int32_t queryGlobalIdx = queryOffset + qIdx;

            queryPosLocal.SetValue(qIdx, static_cast<int32_t>(queryPos));

            // Input ID: next_token for first query, mask token for rest
            if (qIdx == 0) {
                inputIdsLocal.SetValue(qIdx, nextTokenId);
            } else {
                inputIdsLocal.SetValue(qIdx, parallelDraftingTokenId);
            }
        }

        DataCopyOut_int32(gmOutQueryPositions, queryPosLocal, queryOffset, numQueryPerReq);
        DataCopyOut_int32(gmOutInputIds, inputIdsLocal, queryOffset, numQueryPerReq);

        // 计算 query slot mapping
        LocalTensor<int32_t> querySlotLocal = outQuerySlotBuf.Get<int32_t>();
        for (uint32_t qIdx = 0; qIdx < numQueryPerReq; qIdx++) {
            int64_t queryPos = static_cast<int64_t>(queryPosLocal.GetValue(qIdx));
            int32_t queryGlobalIdx = queryOffset + qIdx;

            int32_t blockNum = static_cast<int32_t>(queryPos / blockSize);
            // clamp blockNum避免OOB
            blockNum = (blockNum < static_cast<int32_t>(blockTableStride)) ? blockNum : static_cast<int32_t>(blockTableStride) - 1;
            int32_t blockId = gmBlockTable.GetValue(r * blockTableStride + blockNum);
            int32_t blockOffset = static_cast<int32_t>(queryPos % blockSize);
            querySlotLocal.SetValue(qIdx, blockId * blockSize + blockOffset);
        }
        DataCopyOut_int32(gmOutQuerySlotMapping, querySlotLocal, queryOffset, numQueryPerReq);

        // Token indices to sample (跳过 bonus token at qIdx=0)
        LocalTensor<int32_t> tokenIndicesLocal = tmpBuf.Get<int32_t>();
        for (uint32_t specIdx = 0; specIdx < numSpeculativeTokens; specIdx++) {
            tokenIndicesLocal.SetValue(specIdx, queryOffset + 1 + specIdx);
        }
        DataCopyOut_int32(gmOutTokenIndices, tokenIndicesLocal, r * numSpeculativeTokens, numSpeculativeTokens);
    }

private:
    GlobalTensor<int32_t> gmNextTokenIds, gmQueryStartLoc, gmNumRejectedTokens, gmBlockTable;
    GlobalTensor<int64_t> gmTargetPositions;
    GlobalTensor<int32_t> gmOutInputIds, gmOutTokenIndices;
    GlobalTensor<int32_t> gmOutContextPositions, gmOutQueryPositions;
    GlobalTensor<int32_t> gmOutContextSlotMapping, gmOutQuerySlotMapping;

    uint32_t usedCoreNum, numReqs, reqsPerCore, remainderReqs;
    int32_t parallelDraftingTokenId;
    uint32_t numQueryPerReq, numSpeculativeTokens, blockSize;
    uint32_t totalInputTokens, hasNumRejected, blockTableStride, totalQueryTokens;
    uint32_t myStartReq, myNumReqs;

    TPipe pipe;
    TBuf<QuePosition::VECCALC> qsBuf, ntBuf, nrBuf;
    TBuf<QuePosition::VECCALC> inputBuf;
    TBuf<QuePosition::VECCALC> outCtxPosBuf, outQueryPosBuf;  // Separate buffers for positions
    TBuf<QuePosition::VECCALC> outCtxSlotBuf, outQuerySlotBuf;  // Separate buffers for slot mappings
    TBuf<QuePosition::VECCALC> tmpBuf;
};

extern "C" __global__ __aicore__ void copy_and_expand_dflash_inputs(
    GM_ADDR nextTokenIds,
    GM_ADDR targetPositions,
    GM_ADDR queryStartLoc,
    GM_ADDR numRejectedTokens,
    GM_ADDR blockTable,
    GM_ADDR outInputIds,
    GM_ADDR outContextPositions,
    GM_ADDR outQueryPositions,
    GM_ADDR outContextSlotMapping,
    GM_ADDR outQuerySlotMapping,
    GM_ADDR outTokenIndices,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    if (GetBlockIdx() >= tilingData.usedCoreNum) {
        return;
    }

    if (TILING_KEY_IS(1)) {
        CopyAndExpandDflashInputsKernel op;
        op.Init(nextTokenIds, targetPositions, queryStartLoc, numRejectedTokens, blockTable,
                outInputIds, outContextPositions, outQueryPositions,
                outContextSlotMapping, outQuerySlotMapping, outTokenIndices, &tilingData);
        op.Process();
    }
}