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
 * \file copy_out_with_layout.h
 * \brief
 */
#ifndef MATMUL_TILE_COPY_OUT_COPY_OUT_WITH_LAYOUT_H
#define MATMUL_TILE_COPY_OUT_COPY_OUT_WITH_LAYOUT_H

#include "../tile_copy_policy.h"
#include "../../utils/tensor_utils.h"

namespace Act {
namespace Gemm {
namespace Tile {
/**
 * @struct Copy
 * @brief Copy struct for Ascend910B architecture with specific tensor traits
 *
 * This struct is specialized for the Ascend910B architecture and handles copying of tensor
 * with specific traits. It supports copying from CO1 to GM position with CubeFormat::ND or CubeFormat::ND_ALIGN format
 *
 * @param [in] OutputType: the type of the output tensor
 * @param [in] DstTrait: the traits of the destination tensor
 * @param [in] SrcTrait: the traits of the source tensor
 */
template <class OutputType, class DstTrait, class SrcTrait>
struct Copy<
    Arch::Ascend910B, CopyWithLayout, OutputType, DstTrait, SrcTrait,
    AscendC::Std::enable_if_t<PosIsCO1<SrcTrait::tPos>() && PosIsGM<DstTrait::tPos>() && IsNDOrAlign<OutputType>()>
> {
    /**
     * @brief Copy operator for copying tensors
     * @param [in] Coord: the type of the coordinate
     * @param [in] dst: the destination tensor
     * @param [in] src: the source tensor
     * @param [in] coord: the coordinates for copying
     */
    template <class Coord>
    __aicore__ inline void operator()(AscendC::GlobalTensor<DstTrait>& dst, AscendC::LocalTensor<SrcTrait>& src,
                                      const Coord& coord)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
        using SrcT = typename SrcTrait::LiteType;
        using DstT = typename DstTrait::LiteType;

        auto dstShape = dst.GetTensorTrait().GetLayout().GetShape();
        auto srcShape = src.GetTensorTrait().GetLayout().GetShape();
        auto dstStride = dst.GetTensorTrait().GetLayout().GetStride();

        AscendC::FixpipeParamsV220 params;
        params.nSize =
            AscendC::Std::min<int, int>(Get<1, 0>(srcShape) * Get<1, 1>(srcShape), Get<1>(dstShape) - Get<1>(coord));
        params.mSize =
            AscendC::Std::min<int, int>(Get<0, 0>(srcShape) * Get<0, 1>(srcShape), Get<0>(dstShape) - Get<0>(coord));
        params.srcStride = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        params.dstStride = Get<0>(dstStride);
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }

        AscendC::GlobalTensor<DstT> dstTensor;
        dstTensor.SetGlobalBuffer(dst.address_);
        AscendC::LocalTensor<SrcT> srcTensor;
        srcTensor.SetAddr(src.address_);

        auto dstOffset = dst.GetTensorTrait().GetLayout()(coord);
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_ROW_MAJOR>(dstTensor[dstOffset], srcTensor, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910B"); });
#endif
    }
};

/**
 * @struct Copy
 * @brief Copy struct for Ascend910B architecture with specific tensor traits
 *
 * This struct is specialized for the Ascend910B architecture and handles copying of tensor
 * with specific traits. It supports copying from CO1 to GM position with CubeFormat::NZ format
 *
 * @param [in] OutputType: the type of the output tensor
 * @param [in] DstTrait: the traits of the destination tensor
 * @param [in] SrcTrait: the traits of the source tensor
 */
template <class OutputType, class DstTrait, class SrcTrait>
struct Copy<
    Arch::Ascend910B, CopyWithLayout, OutputType, DstTrait, SrcTrait,
    AscendC::Std::enable_if_t<PosIsCO1<SrcTrait::tPos>() && PosIsGM<DstTrait::tPos>() && IsNz<OutputType>()>
> {
    /**
     * @brief Copy operator for copying tensors
     * @param [in] Coord: the type of the coordinate
     * @param [in] dst: the destination tensor
     * @param [in] src: the source tensor
     * @param [in] coord: the coordinates for copying
     */
    template <class Coord>
    __aicore__ inline void operator()(AscendC::GlobalTensor<DstTrait>& dst, AscendC::LocalTensor<SrcTrait>& src,
                                      const Coord& coord)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
        using SrcT = typename SrcTrait::LiteType;
        using DstT = typename DstTrait::LiteType;

        auto srcShape = src.GetTensorTrait().GetLayout().GetShape();
        auto dstShape = dst.GetTensorTrait().GetLayout().GetShape();
        auto dstStride = dst.GetTensorTrait().GetLayout().GetStride();

        AscendC::FixpipeParamsV220 params;
        params.nSize = AscendC::Std::min<int, int>(Get<1, 0>(srcShape) * Get<1, 1>(srcShape),
                                                   Get<1, 0>(dstShape) * Get<1, 1>(dstShape) - Get<1>(coord));
        params.mSize = AscendC::Std::min<int, int>(Get<0, 0>(srcShape) * Get<0, 1>(srcShape),
                                                   Get<0, 0>(dstShape) * Get<0, 1>(dstShape) - Get<0>(coord));
        params.srcStride = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        params.dstStride = Get<1, 1>(dstStride) * sizeof(DstT) / 32; // 32: c0 byte size
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::GlobalTensor<DstT> dstTensor;
        dstTensor.SetGlobalBuffer(dst.address_);
        AscendC::LocalTensor<SrcT> srcTensor;
        srcTensor.SetAddr(src.address_);

        auto dstOff = dst.GetTensorTrait().GetLayout()(coord);
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_NZ>(dstTensor[dstOff], srcTensor, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910B"); });
#endif
    }
};

/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific tensor traits
 *
 * This struct is specialized for the Ascend910_95 architecture and CopyWithLayout,
 * and it handles copying data from a local tensor (CO1) to a global tensor (GM)
 * It supports specific cube formats(ND or ND_ALIGN)
 *
 * @param [in] OutputType: the type of the output tensor
 * @param [in] DstTrait: the traits of the destination tensor
 * @param [in] SrcTrait: the traits of the source tensor
 */
template <class OutputType, class DstTrait, class SrcTrait>
struct Copy<
    Arch::Ascend910_95, CopyWithLayout, OutputType, DstTrait, SrcTrait,
    AscendC::Std::enable_if_t<PosIsCO1<SrcTrait::tPos>() && PosIsGM<DstTrait::tPos>() && IsNDOrAlign<DstTrait>()>
> {
    /**
     * @brief Copy operator for copying tensors
     * @param [in] Coord: the type of the coordinate
     * @param [in] dst: the destination tensor
     * @param [in] src: the source tensor
     * @param [in] coord: the coordinates for copying
     */
    template <class Coord>
    __aicore__ inline void operator()(AscendC::GlobalTensor<DstTrait>& dst, AscendC::LocalTensor<SrcTrait>& src,
                                      const Coord& coord)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        using SrcT = typename SrcTrait::LiteType;
        using DstT = typename DstTrait::LiteType;

        auto dstShape = dst.GetTensorTrait().GetLayout().GetShape();
        auto srcShape = src.GetTensorTrait().GetLayout().GetShape();
        auto dstStride = dst.GetTensorTrait().GetLayout().GetStride();
        auto srcStride = src.GetTensorTrait().GetLayout().GetStride();

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
        params.nSize =
            AscendC::Std::min<int, int>(Get<1, 0>(srcShape) * Get<1, 1>(srcShape), Get<1>(dstShape) - Get<1>(coord));
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            params.nSize = AscendC::CeilAlign(params.nSize, AscendC::ONE_BLK_SIZE / sizeof(DstT));
        }
        params.mSize =
            AscendC::Std::min<int, int>(Get<0, 0>(srcShape) * Get<0, 1>(srcShape), Get<0>(dstShape) - Get<0>(coord));
        params.srcStride = Get<1, 1>(srcStride) / AscendC::BLOCK_CUBE;
        params.dstStride = Get<0>(dstStride);
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            params.dstStride = AscendC::CeilAlign(params.dstStride, AscendC::ONE_BLK_SIZE / sizeof(DstT));
        }
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }

        AscendC::LocalTensor<SrcT> srcTensor;
        AscendC::GlobalTensor<DstT> dstTensor;
        srcTensor.SetAddr(src.address_);
        dstTensor.SetGlobalBuffer(dst.address_);

        auto dstOff = dst.GetTensorTrait().GetLayout()(coord);
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_ROW_MAJOR>(dstTensor[dstOff], srcTensor, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};

/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific tensor traits
 *
 * This struct is specialized for the Ascend910_95 architecture and CopyWithLayout,
 * and it handles copying data from a local tensor (CO1) to a global tensor (GM)
 * It supports specific cube formats(NZ)
 *
 * @param [in] OutputType: the type of the output tensor
 * @param [in] DstTrait: the traits of the destination tensor
 * @param [in] SrcTrait: the traits of the source tensor
 */
template <class OutputType, class DstTrait, class SrcTrait>
struct Copy<
    Arch::Ascend910_95, CopyWithLayout, OutputType, DstTrait, SrcTrait,
    AscendC::Std::enable_if_t<PosIsCO1<SrcTrait::tPos>() && PosIsGM<DstTrait::tPos>() && IsNz<DstTrait>()>
> {
    /**
     * @brief Copy operator to perform the actual operation
     * @param [in] Coord: the type of the coordinate
     * @param [in] dst: the destination tensor
     * @param [in] src: the source tensor
     * @param [in] coord: the coordinates for copying
     */
    template <class Coord>
    __aicore__ inline void operator()(AscendC::GlobalTensor<DstTrait>& dst, AscendC::LocalTensor<SrcTrait>& src,
                                      const Coord& coord)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        using SrcT = typename SrcTrait::LiteType;
        using DstT = typename DstTrait::LiteType;

        auto srcShape = src.GetTensorTrait().GetLayout().GetShape();
        auto dstShape = dst.GetTensorTrait().GetLayout().GetShape();
        auto srcStride = src.GetTensorTrait().GetLayout().GetStride();
        auto dstStride = dst.GetTensorTrait().GetLayout().GetStride();

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> params;
        params.mSize = AscendC::Std::min<int, int>(Get<0, 0>(srcShape) * Get<0, 1>(srcShape),
                                                   Get<0, 0>(dstShape) * Get<0, 1>(dstShape) - Get<0>(coord));
        params.nSize = AscendC::Std::min<int, int>(Get<1, 0>(srcShape) * Get<1, 1>(srcShape),
                                                   Get<1, 0>(dstShape) * Get<1, 1>(dstShape) - Get<1>(coord));
        params.srcStride = Get<1, 1>(srcStride) / AscendC::BLOCK_CUBE;
        params.dstStride = Get<1, 1>(dstStride) * AscendC::BLOCK_CUBE / (AscendC::ONE_BLK_SIZE / sizeof(DstT));
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::LocalTensor<SrcT> srcTensor;
        AscendC::GlobalTensor<DstT> dstTensor;
        srcTensor.SetAddr(src.address_);
        dstTensor.SetGlobalBuffer(dst.address_);

        auto offset = dst.GetTensorTrait().GetLayout()(coord);
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_NZ>(dstTensor[offset], srcTensor, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};

/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific tensor traits
 *
 * This struct is specialized for the Ascend910_95 architecture and CopyWithLayout
 * when source tensor position is L0C, destination tensor position is UB
 * and output format is CubeFormat::ND
 *
 * @param [in] OutputType: the type of the output tensor
 * @param [in] DstTrait: the traits of the destination tensor
 * @param [in] SrcTrait: the traits of the source tensor
 */
template <class OutputType, class DstTrait, class SrcTrait>
struct Copy<
    Arch::Ascend910_95, CopyWithLayout, OutputType, DstTrait, SrcTrait,
    AscendC::Std::enable_if_t<
        PosIsL0C<SrcTrait::tPos>() && PosIsUB<DstTrait::tPos>() && IsNDOrAlign<DstTrait>()>
> {
    using DstTensor = AscendC::LocalTensor<DstTrait>;
    using SrcTensor = AscendC::LocalTensor<SrcTrait>;

    /**
     * @brief Overloaded operator() for performing the copy operation
     * @param [in] Coord: coordinate type
     * @param [in] dst: destination tensor
     * @param [in] src: source tensor
     * @param [in] coord: coordinate
     * @param [in] subIdx: sub-index, default is 0
     */
    template <class Coord>
    __aicore__ inline void operator()(DstTensor& dst, SrcTensor& src, const Coord& coord, uint8_t subIdx = 0)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        using SrcT = typename SrcTrait::LiteType;
        using DstT = typename DstTrait::LiteType;
        static constexpr AscendC::FixpipeConfig cfgUb = {AscendC::CO2Layout::ROW_MAJOR, true}; // isToUB is true

        auto srcShape = src.GetTensorTrait().GetLayout().GetShape();
        auto dstShape = dst.GetTensorTrait().GetLayout().GetShape();
        auto srcStride = src.GetTensorTrait().GetLayout().GetStride();
        auto dstStride = dst.GetTensorTrait().GetLayout().GetStride();

        AscendC::FixpipeParamsC310<cfgUb.format> params;
        params.nSize =
            AscendC::Std::min<int, int>(Get<1, 0>(srcShape) * Get<1, 1>(srcShape), Get<1>(dstShape) - Get<1>(coord));
        params.mSize =
            AscendC::Std::min<int, int>(Get<0, 0>(srcShape) * Get<0, 1>(srcShape), Get<0>(dstShape) - Get<0>(coord));
        params.srcStride = Get<1, 1>(srcStride) / AscendC::BLOCK_CUBE;
        params.dstStride = Get<0>(dstStride);
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        params.subBlockId = subIdx;
        params.dualDstCtl = 0;

        AscendC::LocalTensor<SrcT> srcTensor;
        srcTensor.SetAddr(src.address_);
        AscendC::LocalTensor<DstT> dstTensor;
        dstTensor.SetAddr(dst.address_);

        auto offset = dst.GetTensorTrait().GetLayout()(coord);
        AscendC::Fixpipe<DstT, SrcT, cfgUb>(dstTensor[offset], srcTensor, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};

/**
 * @struct Copy
 * @brief Copy struct for implementing CO1 to UB copy operations
 *
 * This struct is specialized for the Ascend910_95 architecture and CopyWithLayout
 * when source tensor position is L0C, destination tensor position is UB
 * and output format is CubeFormat::NZ
 *
 * @param [in] OutputType: the type of the output tensor
 * @param [in] DstTrait: the traits of the destination tensor
 * @param [in] SrcTrait: the traits of the source tensor
 */
template <class OutputType, class DstTrait, class SrcTrait>
struct Copy<
    Arch::Ascend910_95, CopyWithLayout, OutputType, DstTrait, SrcTrait,
    AscendC::Std::enable_if_t<PosIsL0C<SrcTrait::tPos>() && PosIsUB<DstTrait::tPos>() && IsNz<DstTrait>()>
> {
    using DstTensor = AscendC::LocalTensor<DstTrait>;
    using SrcTensor = AscendC::LocalTensor<SrcTrait>;

    /**
     * @brief Overloaded operator() for performing the copy operation
     * @param [in] Coord: coordinate type
     * @param [in] dst: destination tensor
     * @param [in] src: source tensor
     * @param [in] coord: coordinate
     * @param [in] subIdx: sub-index, default is 0
     */
    template <class Coord>
    __aicore__ inline void operator()(DstTensor& dst, SrcTensor& src, const Coord& coord, uint8_t subIdx = 0)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        using SrcT = typename SrcTrait::LiteType;
        using DstT = typename DstTrait::LiteType;
        static constexpr AscendC::FixpipeConfig cfgUb = {AscendC::CO2Layout::NZ, true}; // isToUB is true

        auto srcShape = src.GetTensorTrait().GetLayout().GetShape();
        auto dstShape = dst.GetTensorTrait().GetLayout().GetShape();
        auto srcStride = src.GetTensorTrait().GetLayout().GetStride();
        auto dstStride = dst.GetTensorTrait().GetLayout().GetStride();

        AscendC::FixpipeParamsC310<cfgUb.format> params;
        params.nSize = AscendC::Std::min<int, int>(Get<1, 0>(srcShape) * Get<1, 1>(srcShape),
                                                   Get<1, 0>(dstShape) * Get<1, 1>(dstShape) - Get<1>(coord));
        params.mSize = AscendC::Std::min<int, int>(Get<0, 0>(srcShape) * Get<0, 1>(srcShape),
                                                   Get<0, 0>(dstShape) * Get<0, 1>(dstShape) - Get<0>(coord));
        params.srcStride = Get<1, 1>(srcStride) / AscendC::BLOCK_CUBE;
        params.dstStride = Get<1, 1>(dstStride) * AscendC::BLOCK_CUBE / (AscendC::ONE_BLK_SIZE / sizeof(DstT));
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        params.dualDstCtl = 0;
        params.subBlockId = subIdx;

        AscendC::LocalTensor<DstT> dstTensor;
        dstTensor.SetAddr(dst.address_);
        AscendC::LocalTensor<SrcT> srcTensor;
        srcTensor.SetAddr(src.address_);

        auto offset = dst.GetTensorTrait().GetLayout()(coord);
        AscendC::Fixpipe<DstT, SrcT, cfgUb>(dstTensor[offset], srcTensor, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};
} // namespace Tile
} // namespace Gemm
} // namespace Act
#endif
