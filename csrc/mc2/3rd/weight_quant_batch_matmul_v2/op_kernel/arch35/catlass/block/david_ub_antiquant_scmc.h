/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_CATLASS_BLOCK_DAVID_UB_ANTIQUANT_SCMC_H
#define ARCH35_CATLASS_BLOCK_DAVID_UB_ANTIQUANT_SCMC_H

#include "../dispatch_policy.h"
#include "block_decl.h"
#include "block_utils.h"
#include "../utils/constant.h"

namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass {

template <int Stages, typename TileShapeUb, typename TileShapeReg, int32_t CoreType, int32_t SubBlockDim,
          uint8_t StageWeightIn_, uint8_t StageVfOut_, int32_t Kub_, typename KernelSchedule, typename TileShapeL1_,
          typename TileShapeL0, typename DtypeA, typename StrideA, typename DtypeB, typename StrideBOptionalTuple,
          typename DtypeBias, typename StrideBias, typename DtypeC, typename StrideC>
struct DavidUbAntiquantScmc {
public:
    using TileShapeL1 = TileShapeL1_;
    using StrideB = deduce_optional_input_t<0, StrideBOptionalTuple>;
    using StrideAntiquantScale = deduce_optional_input_t<1, StrideBOptionalTuple>;
    // NOTE 存在的目的是判断是否存在AntiquantOffset。antiquantOffset的索引位置为2
    using StrideAntiquantOffset = deduce_optional_input_t<2, StrideBOptionalTuple>;
    // Bias作为模板参数
    using NonVoidStrideBias =
        AscendC::Std::conditional_t<AscendC::Std::is_same_v<StrideBias, void>, AscendC::Std::tuple<_1>, StrideBias>;

private:
    using Subclass =
        BlockMmad<MainloopDavidWqbmmUbAntiquantScmc<Stages, TileShapeUb, TileShapeReg, CoreType, SubBlockDim,
                                                    StageWeightIn_, StageVfOut_, Kub_, KernelSchedule>,
                  TileShapeL1, TileShapeL0, DtypeA, StrideA, DtypeB, StrideBOptionalTuple, DtypeBias, StrideBias,
                  DtypeC, StrideC>;

public:
    static constexpr bool isTransA =
        !AscendC::Std::is_same_v<typename AscendC::Std::tuple_element<1, StrideA>::type, _1>;
    static constexpr bool hasBias = !AscendC::Std::is_same_v<StrideBias, void>;
    static constexpr bool hasAntiQuantOffset = !AscendC::Std::is_same_v<StrideAntiquantOffset, void>;
    static constexpr bool isWeightNz = AscendC::Std::tuple_size<StrideB>::value == 4;
    static constexpr bool isTransB =
        isWeightNz ? false : AscendC::Std::is_same_v<typename AscendC::Std::tuple_element<1, StrideB>::type, _1>;
    struct Arguments {
        __gm__ DtypeA *aGmAddr = nullptr;
        StrideA strideA{};
        __gm__ DtypeB *bGmAddr = nullptr;
        StrideB strideB{};
        __gm__ DtypeA *antiquantScaleGmAddr = nullptr;
        StrideAntiquantScale strideAntiquantScale{};
        __gm__ DtypeA *antiquantOffsetGmAddr = nullptr;
        __gm__ DtypeBias *biasGmAddr = nullptr;
        NonVoidStrideBias strideBias{};
        __gm__ DtypeC *cGmAddr = nullptr;
        StrideC strideC{};
        uint64_t groupSize;
    };
};
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass
#endif
