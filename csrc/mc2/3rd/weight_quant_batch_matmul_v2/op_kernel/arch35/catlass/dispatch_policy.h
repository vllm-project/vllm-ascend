/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_CATLASS_DISPATCH_POLICY_H
#define ARCH35_CATLASS_DISPATCH_POLICY_H

namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass {
struct KernelWqbmm {};

template <int Stages_, typename TileShapeUb_, typename TileShapeReg_, int32_t CoreType_, int32_t SubBlockNum_,
          uint8_t StageWeightIn_, uint8_t StageVfOut_, int32_t Kub_, typename KernelSchedule = KernelWqbmm>
struct MainloopDavidWqbmmUbAntiquantScmc {
    constexpr static int Stages = Stages_;
    constexpr static int32_t SubBlockNum = SubBlockNum_;
    using TileShapeUb = TileShapeUb_;
    using TileShapeReg = TileShapeReg_;
    uint8_t StageWeightIn = StageWeightIn_;
    uint8_t StageVfOut = StageVfOut_;
    int32_t Kub = Kub_;
    constexpr static int32_t CoreType = CoreType_;
    using Schedule = KernelSchedule;
};
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass

#endif
