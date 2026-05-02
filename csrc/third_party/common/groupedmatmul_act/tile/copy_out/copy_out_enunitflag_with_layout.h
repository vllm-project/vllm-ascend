/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file copy_out_enunitflag_with_layout.h
 * \brief
 */
#ifndef MATMUL_TILE_COPY_OUT_COPY_OUT_ENUNITFLAG_WITH_LAYOUT_H
#define MATMUL_TILE_COPY_OUT_COPY_OUT_ENUNITFLAG_WITH_LAYOUT_H

#include "../tile_copy_policy.h"
#include "../../utils/tensor_utils.h"

namespace Act {
namespace Gemm {
namespace Tile {
constexpr static uint8_t FIX_PIPE_UNIT_FLAG = 3;

/**
 * @struct Copy
 * @brief Copy struct for Ascend910B architecture with specific traits and layouts
 *
 * This struct is specialized for the Ascend910B architecture and is used to
 * copy data from a source tensor to a destination tensor. It is enabled when the source
 * tensor is in CO1 position and the destination tensor is in GM position, and the output
 * type is either CubeFormat::ND or CubeFormat::ND_ALIGN
 *
 * @param [in] OutputType: the type of the output tensor
 * @param [in] DstTrait: the traits of the destination tensor
 * @param [in] SrcTrait: the traits of the source tensor
 */
template <class OutputType, class DstTrait, class SrcTrait>
struct Copy<
    Arch::Ascend910B, CopyEnUnitFlagWithLayout, OutputType, DstTrait, SrcTrait,
    AscendC::Std::enable_if_t<PosIsCO1<SrcTrait::tPos>() && PosIsGM<DstTrait::tPos>() && IsNDOrAlign<OutputType>()>
> {
    /**
     * @brief Copy data from source tensor to destination tensor
     *
     * This function copies data from the source tensor to the destination tensor
     * using the specified coordinates. It is specialized for the Ascend910B architecture
     *
     * @param [in] Coord: the type of the coordinates
     * @param [in] dstND: the destination tensor
     * @param [in] src: the source tensor
     * @param [in] coord: the coordinates for copying data
     */
    template <class Coord>
    __aicore__ inline void operator()(AscendC::GlobalTensor<DstTrait>& dstND, AscendC::LocalTensor<SrcTrait>& src,
                                      const Coord& coord)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
        using SrcT = typename SrcTrait::LiteType;
        using DstT = typename DstTrait::LiteType;

        auto srcShape = src.GetTensorTrait().GetLayout().GetShape();
        auto dstShape = dstND.GetTensorTrait().GetLayout().GetShape();
        auto dstStride = dstND.GetTensorTrait().GetLayout().GetStride();

        AscendC::FixpipeParamsV220 params;
        params.mSize =
            AscendC::Std::min<int, int>(Get<0, 0>(srcShape) * Get<0, 1>(srcShape), Get<0>(dstShape) - Get<0>(coord));
        params.nSize =
            AscendC::Std::min<int, int>(Get<1, 0>(srcShape) * Get<1, 1>(srcShape), Get<1>(dstShape) - Get<1>(coord));
        params.srcStride = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        params.dstStride = Get<0>(dstStride);
        params.unitFlag = FIX_PIPE_UNIT_FLAG;
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }

        AscendC::LocalTensor<SrcT> srcTensor;
        srcTensor.SetAddr(src.address_);
        AscendC::GlobalTensor<DstT> dstTensor;
        dstTensor.SetGlobalBuffer(dstND.address_);

        auto offset = dstND.GetTensorTrait().GetLayout()(coord);
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_ROW_MAJOR>(dstTensor[offset], srcTensor, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910B"); });
#endif
    }
};

/**
 * @struct Copy
 * @brief This struct for Ascend910B architecture with specific traits and layout
 * @param [in] OutputType: the type of the output tensor
 * @param [in] DstTrait: the traits of the destination tensor
 * @param [in] SrcTrait: the traits of the source tensor
 */
template <class OutputType, class DstTrait, class SrcTrait>
struct Copy<
    Arch::Ascend910B, CopyEnUnitFlagWithLayout, OutputType, DstTrait, SrcTrait,
    AscendC::Std::enable_if_t<PosIsCO1<SrcTrait::tPos>() && PosIsGM<DstTrait::tPos>() && IsNz<OutputType>()>
> {
    /**
     * @brief Copy operator for copying data from source tensor to destination tensor
     * @param [in] Coord: the type of the coordinates
     * @param [in] dst: the destination tensor
     * @param [in] src: the source tensor
     * @param [in] coord: the coordinates for copying data
     */
    template <class Coord>
    __aicore__ inline void operator()(AscendC::GlobalTensor<DstTrait>& dst, AscendC::LocalTensor<SrcTrait>& src,
                                      const Coord& coord)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
        using DstT = typename DstTrait::LiteType;
        using SrcT = typename SrcTrait::LiteType;

        auto srcShape = src.GetTensorTrait().GetLayout().GetShape();
        auto dstShape = dst.GetTensorTrait().GetLayout().GetShape();
        auto dstStride = dst.GetTensorTrait().GetLayout().GetStride();

        AscendC::FixpipeParamsV220 params;
        params.mSize = AscendC::Std::min<int, int>(Get<0, 0>(srcShape) * Get<0, 1>(srcShape),
                                                   Get<0, 0>(dstShape) * Get<0, 1>(dstShape) - Get<0>(coord));
        params.nSize = AscendC::Std::min<int, int>(Get<1, 0>(srcShape) * Get<1, 1>(srcShape),
                                                   Get<1, 0>(dstShape) * Get<1, 1>(dstShape) - Get<1>(coord));
        params.srcStride = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        params.dstStride = Get<1, 1>(dstStride) * sizeof(DstT) / 32; // 32: c0 byte size
        params.unitFlag = FIX_PIPE_UNIT_FLAG;
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::LocalTensor<SrcT> srcTensor;
        srcTensor.SetAddr(src.address_);
        AscendC::GlobalTensor<DstT> dstTensor;
        dstTensor.SetGlobalBuffer(dst.address_);

        auto offset = dst.GetTensorTrait().GetLayout()(coord);
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_NZ>(dstTensor[offset], srcTensor, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910B"); });
#endif
    }
};

/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific traits
 *
 * This struct is specialized for Ascend910_95 architecture and is used to
 * copy data from a source tensor to a destination tensor with specific traits
 * The traits define the position and format of the tensors
 *
 * @param [in] OutputType: the type of the output tensor
 * @param [in] DstTrait: the traits of the destination tensor
 * @param [in] SrcTrait: the traits of the source tensor
 */
template <class OutputType, class DstTrait, class SrcTrait>
struct Copy<
    Arch::Ascend910_95, CopyEnUnitFlagWithLayout, OutputType, DstTrait, SrcTrait,
    AscendC::Std::enable_if_t<PosIsCO1<SrcTrait::tPos>() && PosIsGM<DstTrait::tPos>() && IsNDOrAlign<OutputType>()>
> {
    /**
     * @brief Copy data from source tensor to destination tensor
     * @param [in] Coord: the type of the coordinates
     * @param [in] dstND: the destination tensor
     * @param [in] src: the source tensor
     * @param [in] coord: the coordinates for copying data
     */
    template <class Coord>
    __aicore__ inline void operator()(AscendC::GlobalTensor<DstTrait>& dstND, AscendC::LocalTensor<SrcTrait>& src,
                                      const Coord& coord)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        using SrcT = typename SrcTrait::LiteType;
        using DstT = typename DstTrait::LiteType;

        auto srcShape = src.GetTensorTrait().GetLayout().GetShape();
        auto dstShape = dstND.GetTensorTrait().GetLayout().GetShape();
        auto dstStride = dstND.GetTensorTrait().GetLayout().GetStride();

        AscendC::FixpipeParamsC310 params;
        params.nSize =
            AscendC::Std::min<int, int>(Get<1, 0>(srcShape) * Get<1, 1>(srcShape), Get<1>(dstShape) - Get<1>(coord));
        params.mSize =
            AscendC::Std::min<int, int>(Get<0, 0>(srcShape) * Get<0, 1>(srcShape), Get<0>(dstShape) - Get<0>(coord));
        params.srcStride = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        params.dstStride = Get<0>(dstStride);
        params.unitFlag = FIX_PIPE_UNIT_FLAG;
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }

        AscendC::GlobalTensor<DstT> dstTensor;
        dstTensor.SetGlobalBuffer(dstND.address_);
        AscendC::LocalTensor<SrcT> srcTensor;
        srcTensor.SetAddr(src.address_);

        auto offset = dstND.GetTensorTrait().GetLayout()(coord);
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_ROW_MAJOR>(dstTensor[offset], srcTensor, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};

/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific traits
 *
 * This struct is specialized for Ascend910_95 architecture and is used to
 * copy data from a source tensor to a destination tensor with specific traits
 *
 * @param [in] OutputType: the type of the output tensor
 * @param [in] DstTrait: the traits of the destination tensor
 * @param [in] SrcTrait: the traits of the source tensor
 */
template <class OutputType, class DstTrait, class SrcTrait>
struct Copy<
    Arch::Ascend910_95, CopyEnUnitFlagWithLayout, OutputType, DstTrait, SrcTrait,
    AscendC::Std::enable_if_t<PosIsCO1<SrcTrait::tPos>() && PosIsGM<DstTrait::tPos>() && IsNz<OutputType>()>
> {
    /**
     * @brief Copy data from source tensor to destination tensor
     * @param [in] Coord: the type of the coordinates
     * @param [in] dst: the destination tensor
     * @param [in] src: the source tensor
     * @param [in] coord: the coordinates for copying data
     */
    template <class Coord>
    __aicore__ inline void operator()(AscendC::GlobalTensor<DstTrait>& dst, AscendC::LocalTensor<SrcTrait>& src,
                                      const Coord& coord)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        using SrcT = typename SrcTrait::LiteType;
        using DstT = typename DstTrait::LiteType;

        auto dstShape = dst.GetTensorTrait().GetLayout().GetShape();
        auto dstStride = dst.GetTensorTrait().GetLayout().GetStride();
        auto srcShape = src.GetTensorTrait().GetLayout().GetShape();

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> params;
        params.unitFlag = FIX_PIPE_UNIT_FLAG;
        params.nSize = AscendC::Std::min<int, int>(Get<1, 0>(srcShape) * Get<1, 1>(srcShape),
                                                   Get<1, 0>(dstShape) * Get<1, 1>(dstShape) - Get<1>(coord));
        params.mSize = AscendC::Std::min<int, int>(Get<0, 0>(srcShape) * Get<0, 1>(srcShape),
                                                   Get<0, 0>(dstShape) * Get<0, 1>(dstShape) - Get<0>(coord));
        params.srcStride = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        params.dstStride = Get<1, 0>(dstShape) * Get<1, 1>(dstStride);
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::LocalTensor<SrcT> srcTensor;
        srcTensor.SetAddr(src.address_);
        AscendC::GlobalTensor<DstT> dstTensor;
        dstTensor.SetGlobalBuffer(dst.address_);

        auto dstOffset = dst.GetTensorTrait().GetLayout()(coord);
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_NZ>(dstTensor[dstOffset], srcTensor, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};

} // namespace Tile
} // namespace Gemm
} // namespace Act
#endif
