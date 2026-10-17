/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_METADATA_Q_SEQ_UTILS_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_METADATA_Q_SEQ_UTILS_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace aicpu {
namespace generic_block_sparse_attention_metadata {

inline int64_t CeilDivQSeq(int64_t value, int64_t divisor)
{
    return value == 0 ? 0 : (value - 1) / divisor + 1;
}

inline bool BuildTndQSeqLayout(const std::vector<int64_t> &cuSeqLengths, const std::vector<int64_t> &seqUsedQ,
                               bool hasSeqUsedQ, int64_t blockShapeX, int64_t qBlockStorageNum,
                               std::vector<int64_t> &actualQSeqLens, std::vector<int64_t> &qStorageBlockStarts)
{
    if (cuSeqLengths.empty() || cuSeqLengths.front() != 0 || blockShapeX <= 0 || qBlockStorageNum < 0) {
        return false;
    }
    const std::size_t batchSize = cuSeqLengths.size() - 1U;
    if (hasSeqUsedQ && seqUsedQ.size() != batchSize) {
        return false;
    }

    actualQSeqLens.assign(batchSize, 0);
    qStorageBlockStarts.assign(batchSize, 0);
    int64_t storageBlockBase = 0;
    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
        // cuSeq is prefixSum(must increase). CuSeq is not nullptr when TND, QseqLens maybe pad in batch.
        if (cuSeqLengths[batchIdx] < 0 || cuSeqLengths[batchIdx + 1U] < cuSeqLengths[batchIdx]) {
            return false;
        }
        const int64_t storageQSeqLen = cuSeqLengths[batchIdx + 1U] - cuSeqLengths[batchIdx];
        const int64_t actualQSeqLen = hasSeqUsedQ ? seqUsedQ[batchIdx] : storageQSeqLen;
        if (actualQSeqLen < 0 || actualQSeqLen > storageQSeqLen) {
            return false;
        }
        actualQSeqLens[batchIdx] = actualQSeqLen;         // actual qTokens in batch.
        qStorageBlockStarts[batchIdx] = storageBlockBase; // Qblock base in batch.

        const int64_t storageBlockNum =
            CeilDivQSeq(storageQSeqLen, blockShapeX); // total Qblocks in batch base on cuSeq.
        if (storageBlockBase > std::numeric_limits<int64_t>::max() - storageBlockNum) {
            return false;
        }
        storageBlockBase += storageBlockNum;
    }
    // totalQBlocks in params.
    return storageBlockBase == qBlockStorageNum;
}

inline bool GatherTndValidBlockNums(const int32_t *counts, int64_t sparseHeadNum, int64_t qBlockStorageNum,
                                    int64_t blockShapeX, int64_t blockIndexStride,
                                    const std::vector<int64_t> &actualQSeqLens,
                                    const std::vector<int64_t> &qStorageBlockStarts,
                                    std::vector<int64_t> &validBlockNums)
{
    if (counts == nullptr || sparseHeadNum <= 0 || qBlockStorageNum < 0 || blockShapeX <= 0 || blockIndexStride <= 0 ||
        actualQSeqLens.size() != qStorageBlockStarts.size()) {
        return false;
    }
    validBlockNums.clear(); // only contain the base tasks have valid blocks.
    for (std::size_t batchIdx = 0; batchIdx < actualQSeqLens.size(); ++batchIdx) {
        // only support blockX=1 now.
        const int64_t actualQBlockNum = CeilDivQSeq(actualQSeqLens[batchIdx], blockShapeX);
        if (qStorageBlockStarts[batchIdx] < 0 || qStorageBlockStarts[batchIdx] > qBlockStorageNum ||
            actualQBlockNum > qBlockStorageNum - qStorageBlockStarts[batchIdx]) {
            return false;
        }
        for (int64_t qBlock = 0; qBlock < actualQBlockNum; ++qBlock) {
            const int64_t globalQBlock = qStorageBlockStarts[batchIdx] + qBlock;
            for (int64_t sparseHead = 0; sparseHead < sparseHeadNum; ++sparseHead) {
                const int64_t offset = sparseHead * qBlockStorageNum + globalQBlock;
                const int64_t count = counts[offset];
                if (count < 0 || count > blockIndexStride) {
                    return false;
                }
                validBlockNums.push_back(count);
            }
        }
    }
    return true;
}

} // namespace generic_block_sparse_attention_metadata
} // namespace aicpu

#endif // GENERIC_BLOCK_SPARSE_ATTENTION_METADATA_Q_SEQ_UTILS_H
