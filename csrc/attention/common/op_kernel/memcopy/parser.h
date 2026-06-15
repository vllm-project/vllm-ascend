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
enum class ActualSeqLensMode
{
    BY_BATCH = 0,
    ACCUM = 1,
};

template <ActualSeqLensMode MODE, typename ACTLEN_T = uint64_t>
class ActualSeqLensParser {
};

template <typename ACTLEN_T>
class ActualSeqLensParser<ActualSeqLensMode::ACCUM, ACTLEN_T> {
public:
    __aicore__ inline ActualSeqLensParser() = default;

    __aicore__ inline void Init(GlobalTensor<ACTLEN_T> actualSeqLengthsGm, uint32_t actualLenDims,
                                uint64_t defaultVal = 0)
    {
        this->actualSeqLengthsGm = actualSeqLengthsGm;
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
private:
    GlobalTensor<ACTLEN_T> actualSeqLengthsGm;
    uint32_t actualLenDims;
};

template <typename ACTLEN_T>
class ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, ACTLEN_T> {
public:
    __aicore__ inline ActualSeqLensParser() = default;

    __aicore__ inline void Init(GlobalTensor<ACTLEN_T> actualSeqLengthsGm, uint32_t actualLenDims, uint64_t defaultVal)
    {
        this->actualSeqLengthsGm = actualSeqLengthsGm;
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

    __aicore__ inline int32_t GetBlockIdx(uint32_t bIdx, uint32_t blockIdxInBatch) const
    {
        return blockTableGm.GetValue(bIdx * maxblockNumPerBatch + blockIdxInBatch);
    }
private:
    GlobalTensor<int32_t> blockTableGm;
    uint32_t maxblockNumPerBatch;
};

#endif