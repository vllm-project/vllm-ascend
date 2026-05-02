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
 * \file copy_a1_to_a2_with_layout.h
 * \brief
 */

#ifndef MATMUL_TILE_LOAD_DATA_COPY_A1_TO_A2_WITH_LAYOUT_H
#define MATMUL_TILE_LOAD_DATA_COPY_A1_TO_A2_WITH_LAYOUT_H

#include "../tile_copy_policy.h"
#include "./load_to_l0_utils.h"

namespace Act {
namespace Gemm {
namespace Tile {
/**
 * @struct Copy
 * @brief Define a Copy structure for tensor copy operations on Ascend910B architecture
 * @param [in] AType: the operation type, such as transpose or non-transpose
 * @param [in] DstTrait: the traits of the destination tensor, including data type and position
 * @param [in] SrcTrait: the traits of the source tensor, including data type and position
 */
template <class AType, class DstTrait, class SrcTrait>
struct Copy<
    Arch::Ascend910B, CopyWithLayout, AType, DstTrait, SrcTrait,
    AscendC::Std::enable_if_t<SrcTrait::tPos == AscendC::TPosition::A1 && DstTrait::tPos == AscendC::TPosition::A2>> {
public:
    using DstTensor = AscendC::LocalTensor<DstTrait>;
    using SrcTensor = AscendC::LocalTensor<SrcTrait>;

    /**
     * @brief Overloaded operator() function to perform tensor copy operations
     * @param [in] Coord: the coordinate type
     * @param [in] l0A: the destination tensor
     * @param [in] l1A: the source tensor
     * @param [in] coord: the coordinate
     */
    template <class Coord>
    __aicore__ inline void operator()(const DstTensor& l0A, const SrcTensor& l1A, const Coord& coord)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
        ASCENDC_ASSERT((AscendC::Std::is_same_v<typename DstTrait::LiteType, typename SrcTrait::LiteType>),
                       { KERNEL_LOG(KERNEL_ERROR, "Type of l0A l1A must be same"); });
        if constexpr (AType::isTrans) {
            if constexpr (AscendC::IsSameTypeV<typename AType::T, int8_t>) {
                TransposeLoadA2Int8(l0A, l1A, coord);
            } else {
                TransposeLoadA2(l0A, l1A, coord);
            }
        } else {
            NoneTransposeLoadA2(l0A, l1A, coord);
        }
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910B"); });
#endif
    }

private:
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
    /**
     * @brief Transpose load A2 to Int8 function
     *
     * This function loads data from source tensor l1A to destination tensor l0A with transpose operation
     * K_axis is m direction, and M_axis is k direction in load3d intrin
     *
     * @param [in] Coord: the coordinate type
     * @param [in] l0A: the destination tensor
     * @param [in] l1A: the source tensor
     * @param [in] coord: the coordinate
     */
    template <class Coord>
    __aicore__ inline void TransposeLoadA2Int8(const DstTensor& l0A, const SrcTensor& l1A, const Coord& coord)
    {
        // K_axis is m direction, and M_axis is k direction in load3d intrin
        auto srcShape = l1A.GetTensorTrait().GetLayout().GetShape();
        auto dstShape = l0A.GetTensorTrait().GetLayout().GetShape();
        uint16_t aL1K = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        uint16_t aL1M = Get<1, 0>(srcShape) * Get<1, 1>(srcShape);
        SetFmatrix(aL1K, aL1M);

        auto aL1MOffset = Get<1>(coord);
        auto aL1KOffset = Get<0>(coord);
        uint16_t l0ALoop = Get<1, 1>(dstShape);
        uint8_t l0ARepeat = AscendC::Ceil(Get<0, 0>(dstShape) * Get<0, 1>(dstShape), Get<1, 0>(dstShape));
        uint64_t l0ASrcAddrStride = aL1K * Get<1, 0>(dstShape);
        uint64_t c0Square = Get<1, 0>(dstShape) * Get<1, 0>(dstShape);
        uint64_t l0ADstAddrStride = l0ARepeat * c0Square;

        uint64_t l1AOffset = aL1MOffset * aL1K + aL1KOffset * Get<1, 0>(dstShape);
        uint64_t l0AOffset = 0;
        // startIndex, repeatTimes, srcStride, dstGap, dstFracGap, addrMode
        AscendC::LoadData2dTransposeParams loadData2dTransposeParams{
            0, l0ARepeat, 1, 0, static_cast<uint16_t>(l0ARepeat - 1), inc};
        AscendC::LocalTensor<typename AType::T> dstLocal;
        dstLocal.SetAddr(l0A.address_);
        AscendC::LocalTensor<typename AType::T> srcLocal;
        srcLocal.SetAddr(l1A.address_);
        for (uint16_t i = 0; i < l0ALoop; i++) {
            AscendC::LoadDataWithTranspose(dstLocal[l0AOffset], srcLocal[l1AOffset], loadData2dTransposeParams);
            l1AOffset += l0ASrcAddrStride;
            l0AOffset += l0ADstAddrStride;
        }
    }

    /**
     * @brief Transpose load A2 function
     *
     * This function loads data from source tensor l1A to destination tensor l0A with transpose operation
     * K_axis is m direction, and M_axis is k direction in load3d intrin
     *
     * @param [in] Coord: the coordinate type
     * @param [in] l0A: the destination tensor
     * @param [in] l1A: the source tensor
     * @param [in] coord: the coordinate
     */
    template <class Coord>
    __aicore__ inline void TransposeLoadA2(const DstTensor& l0A, const SrcTensor& l1A, const Coord& coord)
    {
        // K_axis is m direction, and M_axis is k direction in load3d intrin
        auto srcShape = l1A.GetTensorTrait().GetLayout().GetShape();
        auto dstShape = l0A.GetTensorTrait().GetLayout().GetShape();
        uint16_t aL1K = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        uint16_t aL1M = Get<1, 0>(srcShape) * Get<1, 1>(srcShape);
        SetFmatrix(aL1K, aL1M); // Call only on the first time
        // format(K, M), K, M need to be 16 aligned for f32
        uint16_t madMAlign = Get<1, 0>(dstShape) * Get<1, 1>(dstShape);
        uint16_t mStep = Get<0, 0>(dstShape) * Get<0, 1>(dstShape);
        AscendC::LoadData3DParamsV2Pro loadData3DV2;
        loadData3DV2.channelSize = aL1M;
        loadData3DV2.extConfig = ((uint64_t)Get<0>(coord) << M_POS_BIT) | ((uint64_t)Get<1>(coord) << K_POS_BIT) |
                                 ((uint64_t)mStep << M_STEP_BIT) | (uint64_t)madMAlign;
        loadData3DV2.enTranspose = true;

        AscendC::LocalTensor<typename AType::T> dstLocal;
        dstLocal.SetAddr(l0A[0].address_);
        AscendC::LocalTensor<typename AType::T> srcLocal;
        srcLocal.SetAddr(l1A[0].address_);
        AscendC::LoadData(dstLocal, srcLocal, loadData3DV2);
    }

    /**
     * @brief Non-transpose load A2 function
     *
     * This function loads data from source tensor l1A to destination tensor l0A without transpose operation
     * K_axis is m direction, and M_axis is k direction in load3d intrin
     *
     * @param [in] Coord: the coordinate type
     * @param [in] l0A: the destination tensor
     * @param [in] l1A: the source tensor
     * @param [in] coord: the coordinate
     */
    template <class Coord>
    __aicore__ inline void NoneTransposeLoadA2(const DstTensor& l0A, const SrcTensor& l1A, const Coord& coord)
    {
        auto srcShape = l1A.GetTensorTrait().GetLayout().GetShape();
        auto dstShape = l0A.GetTensorTrait().GetLayout().GetShape();
        uint16_t aL1K = Get<1, 0>(srcShape) * Get<1, 1>(srcShape);
        uint16_t aL1M = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        SetFmatrix(aL1K, aL1M);
        // format(M, K), K_axis is k direction, and M_axis is m direction in load3d intrin
        uint16_t madMAlign = Get<0, 0>(dstShape) * Get<0, 1>(dstShape);
        uint16_t kStep = Get<1, 0>(dstShape) * Get<1, 1>(dstShape);
        AscendC::LoadData3DParamsV2Pro loadData3DV2;
        loadData3DV2.channelSize = aL1K;
        loadData3DV2.extConfig = ((uint64_t)Get<0>(coord) << M_POS_BIT) | ((uint64_t)Get<1>(coord) << K_POS_BIT) |
                                 ((uint64_t)madMAlign << M_STEP_BIT) | (uint64_t)kStep;
        AscendC::LocalTensor<typename AType::T> dstLocal;
        dstLocal.SetAddr(l0A[0].address_);
        AscendC::LocalTensor<typename AType::T> srcLocal;
        srcLocal.SetAddr(l1A[0].address_);

        AscendC::LoadData(dstLocal, srcLocal, loadData3DV2);
    }

    /**
     * @brief Set Fmatrix function
     * @param [in] aL1K: K-axis length
     * @param [in] aL1M: M-axis length
     * @note This function sets the parameters for Fmatrix calculation
     */
    __aicore__ inline void SetFmatrix(uint16_t aL1K, uint16_t aL1M)
    {
        // fmatrix w should be 16 aligned
        uint16_t wAlign;
        if constexpr (AType::isTrans) {
            wAlign = AscendC::CeilAlign(aL1K, HW_M0);
        } else {
            wAlign = AscendC::CeilAlign(aL1M, HW_M0);
        }
        AscendC::Load3DSetFMatrixCal(1, wAlign, PAD_LIST);
    }
#endif
};

/**
 * @struct Copy
 * @brief Define a Copy structure for tensor copy operations on the Ascend910_95 architecture based on position traits
 * @param [in] AType: template parameter indicating whether to perform a transpose operation
 * @param [in] DstTrait: traits of the destination tensor
 * @param [in] SrcTrait: traits of the source tensor
 */
template <class AType, class DstTrait, class SrcTrait>
struct Copy<
    Arch::Ascend910_95, CopyWithLayout, AType, DstTrait, SrcTrait,
    AscendC::Std::enable_if_t<SrcTrait::tPos == AscendC::TPosition::A1 && DstTrait::tPos == AscendC::TPosition::A2>> {
public:
    using DstTensor = AscendC::LocalTensor<DstTrait>;
    using SrcTensor = AscendC::LocalTensor<SrcTrait>;

    /**
     * @brief Overload the operator() to perform tensor copy operations, either with or without transpose based on AType
     * @param [in] Coord: the coordinate type
     * @param [in] l0A: the destination tensor
     * @param [in] l1A: the source tensor
     * @param [in] coord: the coordinate
     */
    template <class Coord>
    __aicore__ inline void operator()(const DstTensor& l0A, const SrcTensor& l1A, const Coord& coord)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        ASCENDC_ASSERT((AscendC::Std::is_same_v<typename DstTrait::LiteType, typename SrcTrait::LiteType>),
                       { KERNEL_LOG(KERNEL_ERROR, "l0A l1A type must be same"); });
        if constexpr (AType::isTrans) {
            TransposeLoadA2(l0A, l1A, coord);
        } else {
            NoneTransposeLoadA2(l0A, l1A, coord);
        }
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }

private:
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
    constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<typename AType::T>();

    /**
     * @brief Template function for transposed loading of A2
     * @param [in] Coord: the coordinate type
     * @param [in] l0A: the destination tensor
     * @param [in] l1A: the source tensor
     * @param [in] coord: the coordinate
     * @note This function is used to load data from global memory to local memory with transposition
     */
    template <class Coord>
    __aicore__ inline void TransposeLoadA2(const DstTensor& l0A, const SrcTensor& l1A, const Coord& coord)
    {
        // K_axis is m direction, and M_axis is k direction in load3d intrin
        auto srcShape = l1A.GetTensorTrait().GetLayout().GetShape();
        auto dstShape = l0A.GetTensorTrait().GetLayout().GetShape();
        uint16_t aL1K = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        uint16_t aL1M = Get<1, 0>(srcShape) * Get<1, 1>(srcShape);
        uint16_t madK = Get<0, 0>(dstShape) * Get<0, 1>(dstShape);
        uint16_t madM = Get<1, 0>(dstShape) * Get<1, 1>(dstShape);

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = AscendC::CeilDiv(Get<0>(coord), AscendC::BLOCK_CUBE);
        loadDataParams.kStartPosition = AscendC::CeilDiv(Get<1>(coord), C0_SIZE);
        loadDataParams.kStep = AscendC::CeilDiv(madM, C0_SIZE);
        if constexpr (AscendC::IsSameType<typename AType::T, float>::value) {
            // K step must be multiples of 2 when transpose is enabled ane .type = .b32
            loadDataParams.kStep = AscendC::CeilAlign(loadDataParams.kStep, K_STEP_MIN_VAL_B32);
        }
        loadDataParams.srcStride = AscendC::CeilDiv(aL1K, ALIGN_NUM);
        loadDataParams.dstStride = AscendC::CeilDiv(madM, ALIGN_NUM);
        loadDataParams.ifTranspose = true;
        loadDataParams.mStep = AscendC::CeilDiv(madK, HW_M0);

        AscendC::LocalTensor<typename AType::T> dstLocal;
        dstLocal.SetAddr(l0A[0].address_);
        AscendC::LocalTensor<typename AType::T> srcLocal;
        srcLocal.SetAddr(l1A[0].address_);
        AscendC::LoadData(dstLocal, srcLocal, loadDataParams);
    }

    /**
     * @brief Template function for non-transposed loading of A2
     * @param [in] Coord: the coordinate type
     * @param [in] l0A: the destination tensor
     * @param [in] l1A: the source tensor
     * @param [in] coord: the coordinate
     * @note This function is used to load data from global memory to local memory without transposition
     */
    template <class Coord>
    __aicore__ inline void NoneTransposeLoadA2(const DstTensor& l0A, const SrcTensor& l1A, const Coord& coord)
    {
        auto srcShape = l1A.GetTensorTrait().GetLayout().GetShape();
        auto dstShape = l0A.GetTensorTrait().GetLayout().GetShape();
        uint16_t aL1K = Get<1, 0>(srcShape) * Get<1, 1>(srcShape);
        uint16_t aL1M = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        uint16_t madK = Get<1, 0>(dstShape) * Get<1, 1>(dstShape);
        uint16_t madM = Get<0, 0>(dstShape) * Get<0, 1>(dstShape);

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = AscendC::CeilDiv(Get<0>(coord), AscendC::BLOCK_CUBE);
        loadDataParams.kStartPosition = AscendC::CeilDiv(Get<1>(coord), C0_SIZE);
        loadDataParams.mStep = AscendC::CeilDiv(madM, HW_M0);
        loadDataParams.kStep = AscendC::CeilDiv(madK, C0_SIZE);
        loadDataParams.srcStride = AscendC::CeilDiv(aL1M, ALIGN_NUM);
        loadDataParams.dstStride = AscendC::CeilDiv(madM, ALIGN_NUM);
        loadDataParams.ifTranspose = false;

        AscendC::LocalTensor<typename AType::T> dstLocal;
        dstLocal.SetAddr(l0A[0].address_);
        AscendC::LocalTensor<typename AType::T> srcLocal;
        srcLocal.SetAddr(l1A[0].address_);

        AscendC::LoadData(dstLocal, srcLocal, loadDataParams);
    }
#endif
};
} // namespace Tile
} // namespace Gemm
} // namespace Act
#endif
