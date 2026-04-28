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
 * \file copy_b1_to_b2_with_layout.h
 * \brief
 */

#ifndef MATMUL_TILE_LOAD_DATA_COPY_B1_TO_B2_WITH_LAYOUT_H
#define MATMUL_TILE_LOAD_DATA_COPY_B1_TO_B2_WITH_LAYOUT_H

#include "../tile_copy_policy.h"
#include "./load_to_l0_utils.h"

namespace Act {
namespace Gemm {
namespace Tile {
/**
 * @struct Copy
 * @brief Template structure for copying data from B1 to B2 with layout for Ascend910B
 *
 * This template structure provides the functionality to copy data from B1 to B2 with specific layout configuration
 * It supports different architectures and provides optimized load operations based on the architectures and data layout
 *
 * @param [in] BType: the data type to be copied
 * @param [in] DstTrait: the trait of destination tensor
 * @param [in] SrcTrait: the trait of source tensor
 */
template <class BType, class DstTrait, class SrcTrait>
struct Copy<
    Arch::Ascend910B, CopyWithLayout, BType, DstTrait, SrcTrait,
    AscendC::Std::enable_if_t<SrcTrait::tPos == AscendC::TPosition::B1 && DstTrait::tPos == AscendC::TPosition::B2>> {
public:
    using DstTensor = AscendC::LocalTensor<DstTrait>;
    using SrcTensor = AscendC::LocalTensor<SrcTrait>;

    /**
     * @brief Operator to perform the copy operation for Ascend910B
     *
     * This operator performs the copy operation from B1 to B2 based on the provided coordinates
     * It checks the architecture and performs the appropriate load operation
     *
     * @param [in] Coord: the type of the coordinate
     * @param [in] l0B: the destination tensor
     * @param [in] l1B: the source tensor
     * @param [in] coord: the coordinate
     */
    template <class Coord>
    __aicore__ inline void operator()(const DstTensor& l0B, const SrcTensor& l1B, const Coord& coord)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
        if constexpr (BType::isTrans) {
            TransposeLoadB2(l0B, l1B, coord);
        } else {
            NoneTransposeLoadB2(l0B, l1B, coord);
        }
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910B"); });
#endif
    }

private:
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
    /**
     * @brief Load data from B1 to B2 with transpose for Ascend910B
     *
     * This function performs the load operation from B1 to B2 with transpose
     * It calculates the necessary parameters and performs the load operation using `LoadData2dParams`
     *
     * @param [in] Coord: the type of the coordinate
     * @param [in] l0B: the destination tensor
     * @param [in] l1B: the source tensor
     * @param [in] coord: the coordinate
     */
    template <class Coord>
    __aicore__ inline void TransposeLoadB2(const DstTensor& l0B, const SrcTensor& l1B, const Coord& coord)
    {
        auto srcShape = l1B.GetTensorTrait().GetLayout().GetShape();
        auto srcStride = l1B.GetTensorTrait().GetLayout().GetStride();
        auto dstShape = l0B.GetTensorTrait().GetLayout().GetShape();
        auto dstStride = l0B.GetTensorTrait().GetLayout().GetStride();

        // SET LOAD2D parameters , loop axis: K or M, or 1
        uint16_t madK = Get<1, 0>(dstShape) * Get<1, 1>(dstShape);
        // k is c0Size_ aligned for f32
        uint16_t kC0 = AscendC::CeilDiv(madK, Get<1, 0>(dstShape));
        uint16_t nFraC0 = Get<0, 1>(dstShape);
        uint16_t l0bLoop = 1;
        uint64_t l0bSrcAddrStride = 0;
        uint64_t l0bDstAddrStride = 0;
        uint8_t l0bRepeat = kC0 * nFraC0;
        uint16_t l0bSrcstride = 1;
        uint16_t l0bDststride = 0;

        uint16_t bL1N = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        if (nFraC0 * Get<0, 0>(dstShape) == bL1N) { // loop=1
            l0bLoop = 1;
        } else if (nFraC0 >= kC0) { // LOOP is K and repeat is n axis
            l0bLoop = kC0;
            l0bSrcAddrStride = bL1N * Get<1, 0>(srcShape);
            l0bDstAddrStride = nFraC0 * Get<1, 1>(dstStride);
            l0bRepeat = nFraC0;

            l0bSrcstride = 1;
            l0bDststride = 0;
        } else { // LOOP is N  and repeat is K axis
            l0bLoop = nFraC0;
            l0bSrcAddrStride = Get<0, 1>(srcStride);
            l0bDstAddrStride = Get<1, 1>(dstStride);
            l0bRepeat = kC0;

            l0bSrcstride = bL1N;
            l0bDststride = nFraC0 - 1;
        }
        // use load2d for L1_2_L0B
        // startIndex, repeatTimes, srcStride, sid, dstGap, ifTranspose, addrmode
        AscendC::LoadData2dParams loadDataParams{0, l0bRepeat, l0bSrcstride, 0, l0bDststride, 0, 0};
        uint16_t bL1KOffset = Get<1>(coord);
        uint64_t l1bOffset = Get<0>(coord) * Get<1, 0>(srcShape) + bL1KOffset * bL1N;
        uint64_t l0bOffset = 0;
        AscendC::LocalTensor<typename BType::T> dstLocal;
        AscendC::LocalTensor<typename BType::T> srcLocal;
        for (uint64_t i = 0; i < l0bLoop; i++) {
            dstLocal.SetAddr(l0B[l0bOffset].address_);
            srcLocal.SetAddr(l1B[l1bOffset].address_);
            AscendC::LoadData(dstLocal, srcLocal, loadDataParams);
            l1bOffset += l0bSrcAddrStride;
            l0bOffset += l0bDstAddrStride;
        }
    }

    /**
     * @brief Load data from B1 to B2 without transpose for Ascend910B
     *
     * This function performs the load operation from B1 to B2 without transpose
     * It calculates the necessary parameters and performs the load operation using `LoadData3DParamsV2Pro`
     *
     * @param [in] Coord: the type of the coordinate
     * @param [in] l0B: the destination tensor
     * @param [in] l1B: the source tensor
     * @param [in] coord: the coordinate
     */
    template <class Coord>
    __aicore__ inline void NoneTransposeLoadB2(const DstTensor& l0B, const SrcTensor& l1B, const Coord& coord)
    {
        auto srcShape = l1B.GetTensorTrait().GetLayout().GetShape();
        auto srcStride = l1B.GetTensorTrait().GetLayout().GetStride();
        auto dstShape = l0B.GetTensorTrait().GetLayout().GetShape();
        SetFmatrix(Get<0, 0>(srcShape) * Get<0, 1>(srcShape));
        // use load3dv2 for L1_2_L0B
        // n_axis is K direction, need to be 16 aligned
        uint16_t kAlign = Get<1, 0>(dstShape) * Get<1, 1>(dstShape);
        // channel size need to be 16 aligned
        uint16_t cAlign = Get<1, 0>(srcShape) * Get<1, 1>(srcShape);
        // k_axis is M direction, need to be HW_M0 aligned
        uint16_t mAlign = Get<0, 0>(dstShape) * Get<0, 1>(dstShape);
        // StepN need to be aligned
        AscendC::LoadData3DParamsV2Pro loadData3DV2;
        loadData3DV2.channelSize = cAlign;
        loadData3DV2.extConfig = ((uint64_t)Get<0>(coord) << M_POS_BIT) | ((uint64_t)Get<1>(coord) << K_POS_BIT) |
                                 ((uint64_t)mAlign << M_STEP_BIT) | (uint64_t)kAlign;
        loadData3DV2.fMatrixCtrl = true;

        AscendC::LocalTensor<typename BType::T> dstLocal;
        dstLocal.SetAddr(l0B[0].address_);
        AscendC::LocalTensor<typename BType::T> srcLocal;
        srcLocal.SetAddr(l1B[0].address_);
        AscendC::LoadData(dstLocal, srcLocal, loadData3DV2);
    }

    /**
     * @brief Set the Fmatrix for loading data for Ascend910B
     *
     * This function sets the Fmatrix for loading data based on the provided parameters
     *
     * @param [in] bL1K: the size of the block in B1
     */
    __aicore__ inline void SetFmatrix(uint16_t bL1K)
    {
        if constexpr (!BType::isTrans) {
            uint16_t wAlign = AscendC::CeilAlign(bL1K, HW_M0);
            AscendC::Load3DSetFMatrixBCal(1, wAlign, PAD_LIST);
        }
    }
#endif
};

/**
 * @struct Copy
 * @brief Template structure for copying data from B1 to B2 with layout for Ascend910_95
 *
 * This template structure provides the functionality to copy data from B1 to B2 with specific layout configuration
 * for the Ascend910_95 architecture
 *
 * @param [in] BType: the data type to be copied
 * @param [in] DstTrait: the trait of destination tensor
 * @param [in] SrcTrait: the trait of source tensor
 */
template <class BType, class DstTrait, class SrcTrait>
struct Copy<
    Arch::Ascend910_95, CopyWithLayout, BType, DstTrait, SrcTrait,
    AscendC::Std::enable_if_t<SrcTrait::tPos == AscendC::TPosition::B1 && DstTrait::tPos == AscendC::TPosition::B2>> {
public:
    using DstTensor = AscendC::LocalTensor<DstTrait>;
    using SrcTensor = AscendC::LocalTensor<SrcTrait>;

    /**
     * @brief Operator to perform the copy operation for Ascend910_95
     *
     * This operator performs the copy operation from B1 to B2 based on the provided coordinates
     * It checks the architecture and performs the appropriate load operation
     *
     * @param [in] Coord: the type of the coordinate
     * @param [in] l0B: the destination tensor
     * @param [in] l1B: the source tensor
     * @param [in] coord: the coordinate
     */
    template <class Coord>
    __aicore__ inline void operator()(const DstTensor& l0B, const SrcTensor& l1B, const Coord& coord)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
        if constexpr (BType::isTrans) {
            TransposeLoadB2(l0B, l1B, coord);
        } else {
            NoneTransposeLoadB2(l0B, l1B, coord);
        }
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }

private:
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
    constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<typename BType::T>();

    /**
     * @brief Load data from B1 to B2 with transpose for Ascend910_95
     *
     * This function performs the load operation from B1 to B2 with transpose
     * It calculates the necessary parameters and performs the load operation using `LoadData2DParamsV2`
     *
     * @param [in] Coord: the type of the coordinate
     * @param [in] l0B: the destination tensor
     * @param [in] l1B: the source tensor
     * @param [in] coord: the coordinate
     */
    template <class Coord>
    __aicore__ inline void TransposeLoadB2(const DstTensor& l0B, const SrcTensor& l1B, const Coord& coord)
    {
        auto srcShape = l1B.GetTensorTrait().GetLayout().GetShape();
        auto dstShape = l0B.GetTensorTrait().GetLayout().GetShape();
        uint16_t bL1K = Get<1, 0>(srcShape) * Get<1, 1>(srcShape);
        uint16_t bL1N = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        uint16_t madK = Get<1, 0>(dstShape) * Get<1, 1>(dstShape);
        uint16_t madN = Get<0, 0>(dstShape) * Get<0, 1>(dstShape);

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = AscendC::CeilDiv(Get<0>(coord), AscendC::BLOCK_CUBE);
        loadDataParams.kStartPosition = AscendC::CeilDiv(Get<1>(coord), C0_SIZE);
        loadDataParams.mStep = AscendC::CeilDiv(madN, HW_M0);
        loadDataParams.kStep = AscendC::CeilDiv(madK, C0_SIZE);
        loadDataParams.srcStride = AscendC::CeilDiv(bL1N, ALIGN_NUM);
        loadDataParams.dstStride = AscendC::CeilDiv(madN, ALIGN_NUM);
        loadDataParams.ifTranspose = false;

        AscendC::LocalTensor<typename BType::T> dstLocal;
        dstLocal.SetAddr(l0B[0].address_);
        AscendC::LocalTensor<typename BType::T> srcLocal;
        srcLocal.SetAddr(l1B[0].address_);
        AscendC::LoadData(dstLocal, srcLocal, loadDataParams);
    }

    /**
     * @brief Load data from B1 to B2 without transpose for Ascend910_95
     *
     * This function performs the load operation from B1 to B2 without transpose
     * It calculates the necessary parameters and performs the load operation using `LoadData2DParamsV2`
     *
     * @param [in] Coord: the type of the coordinate
     * @param [in] l0B: the destination tensor
     * @param [in] l1B: the source tensor
     * @param [in] coord: the coordinate
     */
    template <class Coord>
    __aicore__ inline void NoneTransposeLoadB2(const DstTensor& l0B, const SrcTensor& l1B, const Coord& coord)
    {
        auto srcShape = l1B.GetTensorTrait().GetLayout().GetShape();
        auto dstShape = l0B.GetTensorTrait().GetLayout().GetShape();
        uint16_t bL1N = Get<1, 0>(srcShape) * Get<1, 1>(srcShape);
        uint16_t bL1K = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        uint16_t madN = Get<1, 0>(dstShape) * Get<1, 1>(dstShape);
        uint16_t madK = Get<0, 0>(dstShape) * Get<0, 1>(dstShape);

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = AscendC::CeilDiv(Get<0>(coord), AscendC::BLOCK_CUBE);
        loadDataParams.kStartPosition = AscendC::CeilDiv(Get<1>(coord), C0_SIZE);
        loadDataParams.kStep = AscendC::CeilDiv(madN, C0_SIZE);
        if constexpr (AscendC::IsSameType<typename BType::T, float>::value) {
            // K step must be multiples of 2 when transpose is enabled ane .type = .b32
            loadDataParams.kStep = AscendC::CeilAlign(loadDataParams.kStep, K_STEP_MIN_VAL_B32);
        }
        loadDataParams.srcStride = AscendC::CeilDiv(bL1K, ALIGN_NUM);
        loadDataParams.dstStride = AscendC::CeilDiv(madN, ALIGN_NUM);
        loadDataParams.ifTranspose = true;
        loadDataParams.mStep = AscendC::CeilDiv(madK, HW_M0);

        AscendC::LocalTensor<typename BType::T> dstLocal;
        dstLocal.SetAddr(l0B[0].address_);
        AscendC::LocalTensor<typename BType::T> srcLocal;
        srcLocal.SetAddr(l1B[0].address_);
        AscendC::LoadData(dstLocal, srcLocal, loadDataParams);
    }
#endif
};
} // namespace Tile
} // namespace Gemm
} // namespace Act
#endif