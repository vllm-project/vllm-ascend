/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_CATLASS_ITERATOR_CONTINUOUS_ITERATOR_H
#define ARCH35_CATLASS_ITERATOR_CONTINUOUS_ITERATOR_H

#include "../utils/device_utils.h"

namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass {
/*
 * size和step一致
 */
template <typename Index>
struct ContinuousIterator {
    Index step = 1;
    Index stop = 0;
    Index start = 0;
    Index curr = 0;
    Index offset = 0;
    Index size = 1; // 避免重复计算

    DEVICE ContinuousIterator()
    {
    }

    DEVICE ContinuousIterator(Index stop) : stop(stop)
    {
    }

    DEVICE ContinuousIterator(Index stop, Index step) : step(step), stop(stop)
    {
        UpdateSize();
    }

    DEVICE ContinuousIterator(Index start, Index stop, Index step) : step(step), stop(stop), start(start)
    {
        UpdateSize();
    }

    DEVICE ContinuousIterator(Index start, Index stop, Index step, Index offset)
        : step(step), stop(stop), start(start), offset(offset)
    {
        UpdateSize();
    }

    DEVICE
    void Update(Index newStop, Index newStep)
    {
        step = newStep;
        stop = newStop;
        UpdateSize();
    }

    DEVICE
    void Reset()
    {
        curr = start;
        UpdateSize();
    }

    DEVICE
    void operator++()
    {
        curr += step;
        UpdateSize();
    }

    DEVICE
    bool End()
    {
        return curr >= stop;
    }

    DEVICE
    bool End() const
    {
        return curr >= stop;
    }

    DEVICE
    Index Size()
    {
        return size;
    }

    DEVICE
    Index Size() const
    {
        return size;
    }

    DEVICE
    Index operator*()
    {
        return curr + offset;
    }

    DEVICE
    Index operator*() const
    {
        return curr + offset;
    }

    DEVICE void Print(int32_t signal) const
    {
#if defined(__CCE_KT_TEST__)
        X_LOG("%d Iterator curr %d start %d step %d stop %d size %d offset %d", signal, curr, start, step, stop, size,
              offset);
#endif
    }

    DEVICE
    void UpdateSize()
    {
        size = curr + step < stop ? step : stop - curr;
    }
};
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass
#endif
