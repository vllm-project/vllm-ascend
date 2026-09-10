/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file parser.h
 * \brief
 */
#ifndef PARSER_H
#define PARSER_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

using AscendC::GlobalTensor;

// ----------------------------------------------ActualSeqLensParser--------------------------------
enum class ActualSeqLensMode {
    BY_BATCH = 0,
    ACCUM = 1,
};

template <ActualSeqLensMode MODE, typename ACTLEN_T = uint64_t, bool WITH_ZERO_HEAD = false>
class ActualSeqLensParser {};

template <typename ACTLEN_T>
class ActualSeqLensParser<ActualSeqLensMode::ACCUM, ACTLEN_T, false> {
public:
    __aicore__ inline ActualSeqLensParser() = default;

    // actualSeqLengthsGm:[3, 6, 8], 长度是batch_size, batch_size=3
    // actualLenDims表示actualSeqLengthsGm的长度, 上面的例子中actualLenDims=3
    // TBase(bIdx) = (bIdx == 0) ? 0 : actualSeqLengthsGm[bIdx - 1]
    // ActualSeqLength(bIdx) = actualSeqLengthsGm[bIdx] - ((bIdx == 0) ? 0 : actualSeqLengthsGm[bIdx - 1])
    // TSize = actualSeqLengthsGm[actualLenDims - 1]
    __aicore__ inline void Init(GlobalTensor<ACTLEN_T> actualSeqLengthsGm, uint32_t actualLenDims,
                                uint64_t defaultVal = 0)
    {
        this->actualSeqLengthsGm = actualSeqLengthsGm;
        this->actualLenDims = actualLenDims;
    }

    __aicore__ inline void Init(__gm__ uint8_t *gmAddr, uint32_t actualLenDims)
    {
        this->actualSeqLengthsGm.SetGlobalBuffer((__gm__ ACTLEN_T *)gmAddr, actualLenDims);
        this->actualLenDims = actualLenDims;
    }

    __aicore__ inline uint64_t GetTBase(uint32_t bIdx) const
    {
        if (bIdx == 0) {
            return 0;
        }
        return actualSeqLengthsGm.GetValue(bIdx - 1);
    }

    __aicore__ inline uint64_t GetMxVscaleTBase(uint32_t bIdx) const
    {
        if (bIdx == 0) {
            return 0;
        }
        uint64_t vScaleTBaseOffset = 0;
        for (uint32_t idx = 0; idx < bIdx; idx++) {
            vScaleTBaseOffset += ((GetActualSeqLength(idx) + 63) >> 6);
        }
        return vScaleTBaseOffset;
    }

    __aicore__ inline uint64_t GetActualSeqLength(uint32_t bIdx) const
    {
        if (bIdx == 0) {
            return actualSeqLengthsGm.GetValue(0);
        }
        return (actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1));
    }

    __aicore__ inline uint64_t GetTSize() const
    {
        return actualSeqLengthsGm.GetValue(actualLenDims - 1);
    }

    __aicore__ inline uint32_t GetActualLenDims() const
    {
        return actualLenDims;
    }

private:
    GlobalTensor<ACTLEN_T> actualSeqLengthsGm;
    uint32_t actualLenDims;
};

template <typename ACTLEN_T>
class ActualSeqLensParser<ActualSeqLensMode::ACCUM, ACTLEN_T, true> {
public:
    __aicore__ inline ActualSeqLensParser() = default;

    // actualSeqLengthsGm:[0, 3, 6, 8], 长度是batch_size+1, batch_size=3
    // actualLenDims表示actualSeqLengthsGm的长度, 上面的例子中actualLenDims=4
    // TBase(bIdx) = actualSeqLengthsGm[bIdx]
    // ActualSeqLength(bIdx) = actualSeqLengthsGm[bIdx + 1] - actualSeqLengthsGm[bIdx]
    // TSize = actualSeqLengthsGm[actualLenDims - 1]
    __aicore__ inline void Init(GlobalTensor<ACTLEN_T> cuSeqLensGm, uint32_t actualLenDims, uint64_t defaultVal = 0)
    {
        this->cuSeqLensGm = cuSeqLensGm;
        this->actualLenDims = actualLenDims;
        this->seqUsedDims = 0;
    }

    __aicore__ inline void Init(__gm__ uint8_t *cuSeqLensGmAddr, uint32_t actualLenDims,
                                __gm__ uint8_t *seqUsedGmAddr = nullptr, uint32_t seqUsedDims = 0)
    {
        this->cuSeqLensGm.SetGlobalBuffer((__gm__ ACTLEN_T *)cuSeqLensGmAddr, actualLenDims);
        this->seqUsedGm.SetGlobalBuffer((__gm__ ACTLEN_T *)seqUsedGmAddr, seqUsedDims);
        this->actualLenDims = actualLenDims;
        this->seqUsedDims = seqUsedDims;
    }

    __aicore__ inline uint64_t GetTBase(uint32_t bIdx) const
    {
        return cuSeqLensGm.GetValue(bIdx);
    }

    __aicore__ inline uint64_t GetActualSeqLength(uint32_t bIdx) const
    {
        if (seqUsedDims == 0) {
            return (cuSeqLensGm.GetValue(bIdx + 1) - cuSeqLensGm.GetValue(bIdx));
        } else {
            return seqUsedGm.GetValue(bIdx);
        }
    }

    __aicore__ inline uint64_t GetTSize() const
    {
        return cuSeqLensGm.GetValue(actualLenDims - 1);
    }

    __aicore__ inline uint32_t GetActualLenDims() const
    {
        return actualLenDims;
    }

    __aicore__ inline uint64_t GetMxVscaleTBase(uint32_t bIdx) const
    {
        if (bIdx == 0) {
            return 0;
        }
        uint64_t vScaleTBaseOffset = 0;
        for (uint32_t idx = 0; idx < bIdx; idx++) {
            vScaleTBaseOffset += ((GetActualSeqLength(idx) + 63) >> 6);
        }
        return vScaleTBaseOffset;
    }

private:
    GlobalTensor<ACTLEN_T> seqUsedGm;
    GlobalTensor<ACTLEN_T> cuSeqLensGm;
    uint32_t actualLenDims;
    uint32_t seqUsedDims;
};

template <typename ACTLEN_T>
class ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, ACTLEN_T, false> {
public:
    __aicore__ inline ActualSeqLensParser() = default;

    __aicore__ inline void Init(GlobalTensor<ACTLEN_T> actualSeqLengthsGm, uint32_t actualLenDims, uint64_t defaultVal)
    {
        this->actualSeqLengthsGm = actualSeqLengthsGm;
        this->actualLenDims = actualLenDims;
        this->defaultVal = defaultVal;
    }

    __aicore__ inline void Init(__gm__ uint8_t *gmAddr, uint32_t actualLenDims, uint64_t defaultVal)
    {
        this->actualSeqLengthsGm.SetGlobalBuffer((__gm__ ACTLEN_T *)gmAddr, actualLenDims);
        this->actualLenDims = actualLenDims;
        this->defaultVal = defaultVal;
    }

    __aicore__ inline uint64_t GetActualSeqLength(uint32_t bIdx) const
    {
        if (actualLenDims == 0) {
            return defaultVal;
        }
        if (actualLenDims == 1) {
            return actualSeqLengthsGm.GetValue(0);
        }
        return actualSeqLengthsGm.GetValue(bIdx);
    }

    __aicore__ inline uint32_t GetActualLenDims() const
    {
        return actualLenDims;
    }

private:
    GlobalTensor<ACTLEN_T> actualSeqLengthsGm;
    uint32_t actualLenDims = 0;
    uint64_t defaultVal = 0;
};

// ----------------------------------------------BlockTableParser--------------------------------
class BlockTableParser {
public:
    __aicore__ inline BlockTableParser() = default;

    __aicore__ inline void Init(GlobalTensor<int32_t> blockTableGm, uint32_t maxblockNumPerBatch)
    {
        this->blockTableGm = blockTableGm;
        this->maxblockNumPerBatch = maxblockNumPerBatch;
    }

    __aicore__ inline void Init(__gm__ uint8_t *blockTableGmAddr, uint32_t maxblockNumPerBatch)
    {
        this->blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTableGmAddr);
        this->maxblockNumPerBatch = maxblockNumPerBatch;
    }

    __aicore__ inline int32_t GetBlockIdx(uint32_t bIdx, uint32_t blockIdxInBatch) const
    {
        return blockTableGm.GetValue(bIdx * maxblockNumPerBatch + blockIdxInBatch);
    }

private:
    GlobalTensor<int32_t> blockTableGm;
    uint32_t maxblockNumPerBatch;
};

#endif
