/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_CATLASS_ITERATOR_TAIL_RESPLIT_ITERATOR_H
#define ARCH35_CATLASS_ITERATOR_TAIL_RESPLIT_ITERATOR_H

#include "../utils/math_utils.h"

namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass {
/*
 * size和step不一致
 */
template <typename Index>
struct TailResplitIterator {
    static constexpr int32_t STAGE_ONE = 0;
    static constexpr int32_t STAGE_TWO = 1;
    static constexpr int32_t STAGE_THREE = 2;
    static constexpr int32_t STAGE_END = 3;

    DEVICE TailResplitIterator()
    {
    }

    DEVICE
    TailResplitIterator(Index mainBlockCount, Index firstTailBlockCount, Index secondTailBlockCount,
                        Index mainBlockSize, Index firstTailBlockSize, Index secondTailBlockSize, Index blockDim,
                        Index stop)
        : blockDim(blockDim), stop(stop)
    {
        sizes[STAGE_ONE] = mainBlockSize;
        sizes[STAGE_TWO] = firstTailBlockSize;
        sizes[STAGE_THREE] = secondTailBlockSize;

        counts[STAGE_ONE] = mainBlockCount;
        counts[STAGE_TWO] = firstTailBlockCount;
        counts[STAGE_THREE] = secondTailBlockCount;
    }

    DEVICE void Config(Index blockIdx, bool isAiv, uint8_t subBlockIdx)
    {
        isAiv_ = isAiv;
        subBlockIdx_ = subBlockIdx;
        nBlockIdx_ = blockIdx % blockDim;

        Reset();
    }

    DEVICE
    void Reset()
    {
        idx = nBlockIdx_;
        stage = counts[STAGE_ONE] ? 0 : (counts[STAGE_TWO] ? STAGE_TWO : STAGE_THREE);

        Update();
    }

    DEVICE
    void operator++()
    {
        idx += blockDim;

        Update();
    }

    DEVICE
    bool End() const
    {
        return stage == STAGE_END;
    }

    DEVICE
    Index Size() const
    {
        return size;
    }

    DEVICE
    Index operator*() const
    {
        return curr;
    }

    DEVICE void Print(int32_t signal) const
    {
#if defined(__CCE_KT_TEST__)
        X_LOG("%d Iterator idx %d curr %d size %d stage %d count %d sizes %d", signal, idx, curr, size, stage,
              counts[stage], sizes[stage]);
#endif
    }

    DEVICE
    Index PrevHalfSize()
    {
        return prevHalfSize;
    }

    DEVICE
    Index PrevHalfSize() const
    {
        return prevHalfSize;
    }

private:
    DEVICE void Update()
    {
        if (stage == STAGE_ONE && idx >= counts[STAGE_ONE]) {
            stage = STAGE_TWO;
            idx = nBlockIdx_;
        }
        if (stage == STAGE_TWO && idx >= counts[STAGE_TWO]) {
            stage = STAGE_THREE;
        }
        if (stage == STAGE_THREE && idx >= (counts[STAGE_TWO] + counts[STAGE_THREE])) {
            stage = STAGE_END;
        }

        if (stage == STAGE_ONE) {
            curr = idx * sizes[STAGE_ONE];
        }
        if (stage == STAGE_TWO) {
            curr = counts[STAGE_ONE] * sizes[STAGE_ONE] + idx * sizes[STAGE_TWO];
        }
        if (stage == STAGE_THREE) {
            curr = counts[STAGE_ONE] * sizes[STAGE_ONE] + counts[STAGE_TWO] * sizes[STAGE_TWO] +
                   (idx - counts[STAGE_TWO]) * sizes[STAGE_THREE];
        }
        if (curr >= stop) {
            stage = STAGE_END;
        } else {
            size = Min(sizes[stage], stop - curr);

            if (isAiv_) {
                // NOTE 只用于A16场景
                prevHalfSize = size > 16 ? CeilAlign(size >> 1, static_cast<Index>(16)) : size;
                if (subBlockIdx_ == 1) {
                    curr += prevHalfSize;
                    // NOTE size可能为0，当1：2场景时，需要支持size=0时，仍然执行同步逻辑，才能保证CV间同步是正常的
                    size = Min(size - prevHalfSize, stop - curr);
                } else {
                    size = prevHalfSize;
                }
            }
        }
    }
    Index idx;
    Index stop;
    Index size;
    Index stage;
    Index curr;
    Index nBlockIdx_;
    Index blockDim;
    Index sizes[STAGE_END];
    Index counts[STAGE_END];
    Index prevHalfSize;
    bool isAiv_;
    uint8_t subBlockIdx_;
};

} // namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass
#endif
