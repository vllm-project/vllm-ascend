/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_CATLASS_PIPELINE_PIPELINE_STATE_H
#define ARCH35_CATLASS_PIPELINE_PIPELINE_STATE_H

#include <cstdint>

#include "../utils/device_utils.h"

namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass {

namespace detail {
constexpr bool IsPowerOfTwo(int n)
{
    return (n & (n - 1)) == 0;
}
} // namespace detail
template <uint8_t Stages_>
struct PipelineState {
    static constexpr uint8_t Stages = Stages_;

    uint64_t index_ = 0;
    uint64_t count_ = 0;

    DEVICE
    uint64_t index() const
    {
        return index_;
    }

    DEVICE
    uint64_t count() const
    {
        return count_;
    }

    DEVICE
    void operator++()
    {
        static_assert(Stages > 0);

        ++count_;
        if constexpr (detail::IsPowerOfTwo(Stages)) {
            index_ = (index_ + 1) & (Stages - 1);
        } else {
            ++index_;
            if (index_ == Stages) {
                index_ = 0;
            }
        }
    }
};
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass
#endif
