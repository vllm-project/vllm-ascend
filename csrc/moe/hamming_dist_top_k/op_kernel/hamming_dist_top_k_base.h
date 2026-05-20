/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file hamming_dist_top_k_base.h
 * \brief
 */

#ifndef HAMMING_DIST_TOP_K_BASE_H
#define HAMMING_DIST_TOP_K_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

namespace AscendC {

#define YF_LOG(format, ...)                                                                                            \
if (false) {                                                                                          \
    printf("CoreIdx: %d on CoreType %d, " format, GetBlockIdx(), g_coreType, ##__VA_ARGS__);                       \
}

//constexpr uint32_t SKIP_HEAD_BLOCK_NUM = 1;
//constexpr uint32_t SKIP_TAIL_BLOCK_NUM = 2;
//constexpr uint32_t SKIP_HEAD_TOKEN_NUM = 128;
//constexpr uint32_t SKIP_TAIL_TOKEN_NUM = 256;

constexpr uint32_t MAX_FP16_PROCESS_NUM = 128;
constexpr uint32_t MAX_INT32_PROCESS_NUM = 64;
constexpr float MIN_HALF_VALUE = -65535;
constexpr half MAX_HALF_VALUE = (half) 65504;

// datablock bytes = 32bytes
constexpr uint32_t DATABLOCK_BYTES = 32;
// half 4 datablocks element size
constexpr uint32_t FOUR_DATABLOCKS_ELEMENT_SIZE = 64;
// half 8 datablocks element size
constexpr uint32_t EIGHT_DATABLOCKS_ELEMENT_SIZE = 128;

constexpr MatmulConfig MM_CFG_NO_PRELOAD{false, false, true, 0, 0, 0, false, false, false, false, false,
                                         0, 0, 0, 0, 0, 0, 0, true};

struct TilingParam {
    uint32_t usedCoreNum = 0;
    uint32_t preCoreNum = 0;
    uint32_t isBias = 0;
    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t singleCoreM = 0;
    uint32_t singleCoreN = 0;
    uint32_t singleCoreK = 0;
    uint32_t ka = 0;
    uint32_t kb = 0;
    uint32_t rope_ka = 0;
    uint32_t rope_kb = 0;
    // tiling data for select
    uint32_t layer = 0;
    uint32_t batch = 0;
    uint32_t head = 0;
    uint32_t batchN = 0;
    uint32_t selectUsedCoreNum = 0;
    uint32_t layerSize = 0;
    uint32_t layerSizeRope = 0;
    uint32_t seqLen = 0;
    uint32_t dimension = 0;
    uint32_t nope_dimension = 0;
    uint32_t rope_dimension = 0;
    uint32_t reducedBatch = 0;
    uint32_t tileN1 = 0;
    uint32_t tileN2 = 0;
    uint32_t singleCoreBatch = 0;
    uint32_t singleCoreSeqLen = 0;
    bool supportKeyRope;
    // tiling data for matmul
    uint32_t matmulResultSize = 0;
    // tiling data for topk
    uint32_t maxK = 0;
    uint32_t maxSeqLen = 0;
    uint32_t sink = 0;
    uint32_t recent = 0;
    uint32_t topKInnerSize = 0;
    uint32_t topKValueSize = 0;
    uint32_t topKIdexSize = 0;
    uint32_t kNopeUnpackGmOffset = 0;
    uint32_t mmGmOffset = 0;
    uint32_t qHead = 0;
    uint32_t headGroupNum = 0;
    uint64_t qUnpackGmOffset = 0;
    uint64_t blockCount = 0;
    // support offload
    bool supportOffload = false;
};

template <typename T>
__aicore__ inline T Min(const T a, const T b)
{
    return a < b ? a : b;
}

template <typename T>
__aicore__ inline T Max(const T a, const T b)
{
    return a > b ? a : b;
}

template <typename T>
__aicore__ inline void SelectCustom(const LocalTensor<T> &dstLocal, const LocalTensor<uint8_t> &keyCompressed, const LocalTensor<T> &src0Local, uint8_t repeatTimes)
{
    AscendC::BinaryRepeatParams repeatParams = {1, 1, 1, 8, 0, 8}; // {dstBlkStride, src0BlkStride, src1BlkStride, dstRepStride, src0RepStride, src1RepStride} src0 is reused, set repeat stride to 0.
    uint64_t mask = MAX_FP16_PROCESS_NUM;
    // DumpTensor(keyCompressed, 123, 256);
    // DumpTensor(src0Local, 124, 256);
    Select(dstLocal, keyCompressed, src0Local, static_cast<T>(-1), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, mask, repeatTimes, repeatParams);
    // DumpTensor(dstLocal, 126, 256);
}

template <typename T>
__aicore__ inline void TopKCustom(const LocalTensor<T> &dstValueLocal, const LocalTensor<int32_t> &dstIndexLocal,
    const LocalTensor<T> &srcValueLocal, const LocalTensor<int32_t> &srcIndexLocal, const int32_t k, const HammingDistTopKTilingData &tiling, uint32_t n)
{
    LocalTensor<bool> finishLocal;
    AscendC::TopKInfo topkInfo;
    topkInfo.outter = tiling.params.outer;
    topkInfo.n = n;
    topkInfo.inner = matmul::CeilDiv(n, 32) * 32; /* 32: inner must be aligned to 32 */
    TopK<half, true, false, false, TopKMode::TOPK_NORMAL>(dstValueLocal, dstIndexLocal, srcValueLocal, srcIndexLocal, finishLocal, k, tiling.topkTiling, topkInfo, true);
}

__aicore__ inline void ReduceMaxCustom(const GlobalTensor<half> &inputGm, const LocalTensor<half> &reduceInputLocal,
    const LocalTensor<half> &reduceOutputLocal, const uint16_t chunkNum, const uint8_t chunkSize)
{
    uint32_t dataBlockNum = (static_cast<uint32_t>(chunkNum) * static_cast<uint32_t>(chunkSize) + 15) / 16;  // Each dataBlock contains 16 half elements
    uint32_t blockLen = static_cast<uint32_t>(16 * sizeof(half));  // Determine blockLen (more robust for copying by dataBlock unit), 32 bytes per dataBlock
    // uint32_t blockLen = static_cast<uint32_t>(chunkSize << 1);   // Equivalent to blockLen=chunkSize*2=16*2=32, half type occupies 2 bytes, multiply by 2 to get the length in bytes
    // DataCopyExtParams: Copy dataBlockCount dataBlocks to local, keep layout as continuous dataBlock list
    DataCopyExtParams copyInParams{static_cast<uint16_t>(dataBlockNum), blockLen, 0, 0, 0};  // {255, 32, 0, 0, 0}={number of blocks to copy, length per block, 0, 0, 0}, total data size 8160

    if (chunkSize == 16 || chunkSize == 64 || chunkSize == 128) { // When chunkSize=16, 64, 128, copy one block directly at a time
        copyInParams.blockCount = 1;
        // copyInParams.blockLen = static_cast<uint32_t>(chunkSize * chunkNum * sizeof(half));  // blockLen=16*255*2=8160
        copyInParams.blockLen = static_cast<uint32_t>(dataBlockNum * blockLen); // Equivalent
    }

    DataCopyPadExtParams<half> copyInPadParams{false, 0, 0, 0};  // No padding
    DataCopyPad(reduceInputLocal, inputGm, copyInParams, copyInPadParams);  // DataCopyPad is an internal operator
    // DumpTensor(reduceInputLocal, 158, 6 * chunkSize);
    /* For positions where chunkNum tail is less than 8, fill with half minimum value,
       so that the output of BlockReduceMax at corresponding positions will also be half minimum value,
       which will not affect the subsequent TopK calculation */
    uint32_t dataBlockNumAligned = matmul::CeilDiv(dataBlockNum, 8) * 8; // dataBlockNumAligned=256, /* 8: BlockReduceMax processes 8 dataBlocks in parallel at one time */
    if (dataBlockNumAligned > dataBlockNum) {  // 256>255
        Duplicate(reduceInputLocal[dataBlockNum * 16], static_cast<half>(MIN_HALF_VALUE), (dataBlockNumAligned - dataBlockNum) * 16); /* 16: Each dataBlock is 32Bytes, containing 16 half values */
    }  // Copy to 4080
    // printf("base.h dataBlockNumAligned: %d\n", dataBlockNumAligned);

    SetFlag<HardEvent::MTE2_V>(1);   // Wait for copy completion
    WaitFlag<HardEvent::MTE2_V>(1);
    PipeBarrier<PIPE_V>();
    PipeBarrier<PIPE_ALL>();

    if (chunkSize == 64) {
        int32_t totalRepeat = dataBlockNumAligned / 8;
        int32_t repeat = Min(MAX_REPEAT_TIMES, totalRepeat);
        int32_t loopNum = matmul::CeilDiv(totalRepeat, repeat);  // loopNum=1
        int32_t tailRepeat = totalRepeat - (loopNum - 1) * repeat;
        uint64_t mask[2] = {0, 0}; /* 2: Set mask bit by bit, 2 64bit variables required */

        mask[0] = UINT64_MAX;
        uint32_t srcOffset = 0;
        uint32_t dstOffset = 0;

        for (int32_t i = 0; i < loopNum - 1; i++) {
            WholeReduceMax<half>(reduceOutputLocal[dstOffset], reduceInputLocal[srcOffset], mask, repeat * 2, 1, 1, 4, ReduceOrder::ORDER_ONLY_VALUE); // (..., repeat, dstRepStride, srcBlkStride, srcRepStride)
            srcOffset += repeat * 8 * 16; // Move repeat segments of 128-element: repeat * 128
            dstOffset += repeat * 2;      // Advance dstOffset by output count: 1 value output per repeat
        }
        // First 4 datablocks of each repeat, 64 elements
        // Iteration count tailRepeat * 2
        WholeReduceMax<half>(reduceOutputLocal[dstOffset], reduceInputLocal[srcOffset], mask, tailRepeat * 2, 1, 1, 4, ReduceOrder::ORDER_ONLY_VALUE);
        return;
    }

    if (chunkSize == 128) {
        int32_t totalRepeat = dataBlockNumAligned / 8;  // (dataBlockNumAligned * 16) / 128
        int32_t repeat = Min(MAX_REPEAT_TIMES, totalRepeat);
        int32_t loopNum = matmul::CeilDiv(totalRepeat, repeat);  // loopNum=1
        int32_t tailRepeat = totalRepeat - (loopNum - 1) * repeat;
        uint64_t mask[2]; /* 2: Set mask bit by bit, 2 64bit variables required */
        /* For chunkSize==128, we need to cover 128 consecutive half elements in each repeat,
           so set all 128 bits to 1 (continuous participation in reduction) */
        mask[0] = UINT64_MAX;
        mask[1] = UINT64_MAX;

        uint32_t srcOffset = 0;
        uint32_t dstOffset = 0;
        /* Explanation:
        - One dataBlock contains 16 half elements (32 bytes)
        - Process 8 dataBlocks in parallel at one time, so 1 repeat corresponds to 8 * 16 = 128 half elements
        - WholeReduceMax(mask=128, repeat=k) outputs 1 maximum value for each repeat (128 half elements)
            —— So dstOffset should only advance by 1 per repeat (not 8)
        */
        for (int32_t i = 0; i < loopNum - 1; i++) {
            WholeReduceMax<half>(reduceOutputLocal[dstOffset], reduceInputLocal[srcOffset], mask, repeat, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE); // (..., repeat, dstRepStride, srcBlkStride, srcRepStride)
            srcOffset += repeat * 8 * 16; /* Move repeat segments of 128-element: repeat * 128 */
            dstOffset += repeat;  // Advance dstOffset by output count: 1 value output per repeat
        }
        WholeReduceMax<half>(reduceOutputLocal[dstOffset], reduceInputLocal[srcOffset], mask, tailRepeat, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE); // (..., repeat, dstRepStride, srcBlkStride, srcRepStride)
        // DumpTensor(reduceOutputLocal, 218, chunkNum);
    } else {
        int32_t totalRepeat = dataBlockNumAligned / 8;          // totalRepeat=32, /* 8: BlockReduceMax processes 8 dataBlocks in parallel at one time */, repeat 32 times to complete
        int32_t repeat = Min(MAX_REPEAT_TIMES, totalRepeat);    // Internal parameter MAX_REPEAT_TIMES, unknown? Assume repeat=32
        int32_t loopNum = matmul::CeilDiv(totalRepeat, repeat); // loopNum=1
        int32_t tailRepeat = totalRepeat - (loopNum - 1) * repeat;  // tailRepeat=32
        uint64_t mask[2]; /* 2: Set mask bit by bit, 2 64bit variables required */

        if (chunkSize == 16) { /* chunkSize only supports 1, 8, 16 */
            mask[0] = UINT64_MAX;  // 0xffffffffffffffff, bitwise mask, all 1, all participate in calculation
            mask[1] = UINT64_MAX;  // 0xffffffffffffffff
        } else if (chunkSize == 8) { /* chunkSize only supports 1, 8, 16 */
            mask[0] = 0x00ff00ff00ff00ff;
            mask[1] = 0x00ff00ff00ff00ff;
        }

        uint32_t srcOffset = 0;
        uint32_t dstOffset = 0;
        for (int32_t i = 0; i < loopNum - 1; i++) {
            BlockReduceMax<half>(reduceOutputLocal[dstOffset], reduceInputLocal[srcOffset], repeat, mask, 1, 1, 8); // (..., mask, dstRepStride, srcBlkStride, srcRepStride)
            srcOffset += repeat * 8 * 16; /* 8: BlockReduceMax processes 8 dataBlocks in parallel at one time, 16: Each dataBlock is 32Bytes, containing 16 half values */
            dstOffset += repeat * 8; /* 8: BlockReduceMax processes 8 dataBlocks in parallel at one time, outputs 8 points */
        }
        BlockReduceMax<half>(reduceOutputLocal[dstOffset], reduceInputLocal[srcOffset], tailRepeat, mask, 1, 1, 8); // (..., mask, dstRepStride, srcBlkStride, srcRepStride)
        // repeat = 32, 8 elements one repeat, 256 elements total
        // srcBlkStride = 1, no gap between blocks in one repeat
        // dstRepStride = 1, srcRepStride = 8, no gap between repeats
    }
}

__aicore__ inline void SortInt32AscendingUB(LocalTensor<int32_t>& buf, uint32_t len) {
    if ASCEND_IS_AIC { return; }
    if (len <= 1) { return; }
    __ubuf__ int32_t* data = reinterpret_cast<__ubuf__ int32_t*>(buf.GetPhyAddr());
    for (uint32_t i = 1; i < len; ++i) {
        int32_t key = data[i];
        int32_t j = static_cast<int32_t>(i) - 1;
        while (j >= 0 && data[j] > key) {
            data[j + 1] = data[j];
            --j;
        }
        data[j + 1] = key;
    }
}

__aicore__ inline void WriteBlockTableFromTopK(
    uint32_t curBatchIdx,
    LocalTensor<int32_t>&       topKIndexUb,   // UB: Chunk index obtained by TopK (length ≥ curKScalar)
    LocalTensor<int32_t>&       blockIdUb,     // UB: Temporary buffer allocated by the caller (length ≥ curKScalar)
    uint32_t                    curKScalar,
    uint64_t                    outGmOffset,   // Offset for writing back to GM(indices)
    LocalTensor<int32_t>&       tableBlockTensor,
    const GlobalTensor<int32_t>& indicesGm,
    bool                        isContinuousBatch,
    uint32_t                    blockCount     // Number of blocks per batch (calculated by tileN1 or fixed BLOCK_SIZE)
) {
    if ASCEND_IS_AIC { return; }

    //YF_LOG("ldeng WriteBlockTableFromTopK 245 curBatchIdx=%d, curKScalar=%d,blockCount=%d,\n", curBatchIdx,curKScalar,blockCount)
    // DumpTensor(topKIndexUb, 246, topKIndexUb.GetSize());

    // Sort chunk_id in ascending order in UB
    SortInt32AscendingUB(topKIndexUb, curKScalar);

    // DumpTensor(topKIndexUb, 251, topKIndexUb.GetSize());

    // Directly read and write in UB in scalar mode
    __ubuf__ const int32_t* in_ptr = reinterpret_cast<__ubuf__ const int32_t*>(topKIndexUb.GetPhyAddr());
    __ubuf__ int32_t* out_ptr = reinterpret_cast<__ubuf__ int32_t*>(blockIdUb.GetPhyAddr());

    for (uint32_t i = 0; i < curKScalar; ++i) {
        const int32_t idx = in_ptr[i];
        out_ptr[i] = isContinuousBatch
            ? tableBlockTensor.GetValue(static_cast<uint32_t>(idx))
            : (idx + 1); // When no block_table exists,约定 block_id = idx + 1（1-based）
    }

    // DumpTensor(blockIdUb, 264, blockIdUb.GetSize());
    // UB -> GM(indices)
    DataCopyExtParams cpOut{1, static_cast<uint32_t>(curKScalar * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(indicesGm[outGmOffset], blockIdUb, cpOut);

    // DumpTensor(blockIdUb, 269, 64);
    
}

// Used for setting tail top-k to MAX_HALF_VALUE
// tensorSize should be less than topKValueInTensor.GetSize()
// copyLen is the total number of elements that should be set to MAX_HALF_VALUE, starting from the actual tail
__aicore__ inline void FillMaxValueFromTail(
    LocalTensor<half> &topKValueInTensor, uint32_t tensorSize, uint32_t copyLen, uint32_t curChunkSize)
{
    if ASCEND_IS_AIC {
        return;
    }

    ASCENDC_ASSERT((copyLen <= tensorSize), { KERNEL_LOG(KERNEL_ERROR, "copyLen should be less tensorSize"); });
    // case1: tensorSize - copyLen % alignedElements = 0, address 32bytes aligned
    // YF_LOG("tensorSize = %d, copyLen = %d\n", tensorSize, copyLen);
    uint32_t alignedElements = DATABLOCK_BYTES / sizeof(half);
    uint32_t offset = tensorSize - copyLen;
    if (offset % alignedElements == 0) {
        Duplicate(topKValueInTensor[offset], static_cast<half>(MAX_HALF_VALUE), copyLen);
        return;
    }

    // case2: compute aligned address
    uint32_t offsetAligned = offset / alignedElements * alignedElements; /* floor aligned for datacopy */
    ASCENDC_ASSERT((offsetAligned >= 0), { KERNEL_LOG(KERNEL_ERROR, "offsetAligned should be nonnegative"); });

    // After 32Bytes alignment, number of elements to process
    uint32_t alignedAddCopyElements = tensorSize - offsetAligned;
    // Process 128 elements in one iteration, 8 datablocks, one datablock 32Bytes
    // Use mask[] bitwise mode to control elements
    uint64_t mask[2] = {0, 0};
    uint32_t needSkipElements = alignedAddCopyElements - copyLen;
    // Directly process the remaining unprocessed elements, address is already 32bytes aligned
    int32_t lastCopyLen = alignedAddCopyElements - EIGHT_DATABLOCKS_ELEMENT_SIZE;
    // If one iteration can complete processing, use one iteration; otherwise, split into two Duplicate processes
    if (lastCopyLen > 0) {
        // Process 128 elements first
        alignedAddCopyElements = EIGHT_DATABLOCKS_ELEMENT_SIZE;
    }
    // YF_LOG("tensorSize = %d, copyLen = %d offsetAligned = %d alignedAddCopyElements = %d needSkipElements = %d lastCopyLen = %d\n", tensorSize, copyLen, offsetAligned, alignedAddCopyElements, needSkipElements, lastCopyLen);
    if (alignedAddCopyElements <= FOUR_DATABLOCKS_ELEMENT_SIZE) {
        mask[0] = (UINT64_MAX << needSkipElements) & (UINT64_MAX >> (FOUR_DATABLOCKS_ELEMENT_SIZE - alignedAddCopyElements));
    } else if (alignedAddCopyElements <= EIGHT_DATABLOCKS_ELEMENT_SIZE) {
        mask[0] = (UINT64_MAX << needSkipElements);
        mask[1] = UINT64_MAX >> (EIGHT_DATABLOCKS_ELEMENT_SIZE - alignedAddCopyElements);
    }
    // YF_LOG("mask[0] = %x mask[1] = %x \n", mask[0], mask[1]);

    Duplicate(topKValueInTensor[offsetAligned], static_cast<half>(MAX_HALF_VALUE), mask, 1, 1, 8);

    if (lastCopyLen > 0) {
        Duplicate(topKValueInTensor[offsetAligned + EIGHT_DATABLOCKS_ELEMENT_SIZE], static_cast<half>(MAX_HALF_VALUE), lastCopyLen);
    }
}

} // namespace AscendC
#endif // HAMMING_DIST_TOP_K_BASE_H
