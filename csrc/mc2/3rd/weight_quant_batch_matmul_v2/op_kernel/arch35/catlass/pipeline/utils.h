/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_CATLASS_PIPELINE_UTILS_H
#define ARCH35_CATLASS_PIPELINE_UTILS_H

#include "../utils/device_utils.h"
#include "kernel_operator.h"

using AscendC::HardEvent;
using AscendC::Hardware;

namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass {
namespace detail {
template <Hardware Src, Hardware Dst, bool FwdDirect>
DEVICE constexpr HardEvent GetQueEvt()
{
    if (Src == Hardware::GM) {
        if (Dst == Hardware::UB) {
            return FwdDirect ? HardEvent::MTE2_V : HardEvent::V_MTE2;
        } else if (Dst == Hardware::L1) {
            return FwdDirect ? HardEvent::MTE2_MTE1 : HardEvent::MTE1_MTE2;
        }
    } else if (Src == Hardware::UB) {
        if (Dst == Hardware::L1 || Dst == Hardware::GM) {
            return FwdDirect ? HardEvent::V_MTE3 : HardEvent::MTE3_V;
        }
    }
    return HardEvent::MAX;
}
} // namespace detail
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass
#endif
