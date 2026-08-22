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
 * \file axis.h
 * \brief
 */
#ifndef AXIS_H
#define AXIS_H

#include <type_traits>
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif


struct AxisSlices;

struct Axis {
    uint32_t start;
    uint32_t sizeAct;

    __aicore__ inline Axis(uint32_t sizeAct_) : start(0), sizeAct(sizeAct_) {}
    __aicore__ inline Axis(uint32_t start_, uint32_t sizeAct_) : start(start_), sizeAct(sizeAct_) {}

    __aicore__ inline AxisSlices Split(uint32_t splitSize) const;

    template <uint32_t ALIGN_SIZE=BLOCK_CUBE>
    __aicore__ inline uint32_t AlignedSize() const
    {
        return (((ALIGN_SIZE) == 0) ? 0 : (((sizeAct) + (ALIGN_SIZE)-1) / (ALIGN_SIZE) * (ALIGN_SIZE)));
    }

    __aicore__ inline bool IsTailOf(const Axis& parent) const
    {
        return (start + sizeAct) >= parent.sizeAct;
    }
};

struct AxisSlices {
    struct Sentinel {
        uint32_t end_;
    };
    struct Iterator {
        Axis cur_;
        uint32_t tSizeAct_;
        uint32_t splitSize_;

        __aicore__ inline Iterator(const AxisSlices& slices)
            : cur_(Min(slices.splitSize_, slices.sizeAct_)),
              tSizeAct_(slices.sizeAct_), splitSize_(slices.splitSize_) {}

        __aicore__ inline Axis& operator*()
        {
            return cur_;
        }

        __aicore__ inline Iterator& operator++()
        {
            cur_.start += splitSize_;
            cur_.sizeAct = Min(splitSize_, tSizeAct_ - cur_.start);
            return *this;
        }

        __aicore__ inline bool operator!=(const Sentinel& end) const
        {
            return cur_.start < end.end_;
        }
    };

    uint32_t sizeAct_;
    uint32_t splitSize_;

    __aicore__ inline AxisSlices(uint32_t sizeAct, uint32_t splitSize) : sizeAct_(sizeAct), splitSize_(splitSize) {}

    __aicore__ inline Iterator begin()
    {
        return {*this};
    }

    __aicore__ inline Sentinel end()
    {
        return Sentinel{sizeAct_};
    }
    
    __aicore__ inline uint32_t size() const
    {
        return (sizeAct_ + (splitSize_ - 1)) / splitSize_;
    }
};

__aicore__ inline AxisSlices Axis::Split(uint32_t splitSize) const
{
    return {sizeAct, splitSize};
}
#endif // AXIS_H
