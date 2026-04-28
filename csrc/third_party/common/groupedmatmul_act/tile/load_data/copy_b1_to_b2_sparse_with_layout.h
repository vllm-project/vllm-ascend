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
 * \file copy_b1_to_b2_sparse_with_layout.h
 * \brief
 */

#ifndef MATMUL_TILE_LOAD_DATA_COPY_B1_TO_B2_SPARSE_WITH_LAYOUT_H
#define MATMUL_TILE_LOAD_DATA_COPY_B1_TO_B2_SPARSE_WITH_LAYOUT_H

#include "../tile_copy_policy.h"
#include "./load_to_l0_utils.h"

namespace Act {
namespace Gemm {
namespace Tile {
/**
 * @struct Copy
 * @brief Define a Copy struct for performing sparse data copy operations on the Ascend910B architecture
 * @param [in] BType: the base type
 * @param [in] DstTrait: the destination tensor traits
 * @param [in] SrcTrait: the source tensor traits
 */
template <class BType, class DstTrait, class SrcTrait>
struct Copy<
    Arch::Ascend910B, CopySparseWithLayout, BType, DstTrait, SrcTrait,
    AscendC::Std::enable_if_t<SrcTrait::tPos == AscendC::TPosition::B1 && DstTrait::tPos == AscendC::TPosition::B2>> {
public:
    using DstTensor = AscendC::LocalTensor<DstTrait>;
    using SrcTensor = AscendC::LocalTensor<SrcTrait>;

    /**
     * @brief Overload the operator() to perform the copy operation
     * @param [in] Coord: the coordinate type
     * @param [in] SparseTrait: the sparse trait type
     * @param [in] l0B: the destination tensor
     * @param [in] l1B: the source tensor
     * @param [in] coord: the coordinate
     * @param [in] l1BIndexMatrix: the index matrix of the sparse tensor
     */
    template <class Coord, class SparseTrait>
    __aicore__ inline void operator()(const DstTensor& l0B, const SrcTensor& l1B, const Coord& coord,
                                      const AscendC::LocalTensor<SparseTrait>& l1BIndexMatrix)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
        if constexpr (BType::isTrans) {
            TransposeLoadB2(l0B, l1B, coord, l1BIndexMatrix);
        } else {
            ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Matrix B only support transpose."); });
        }
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910B"); });
#endif
    }

private:
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
    /**
     * @brief Transpose load function, loads data from source tensor to destination tensor
     * @param [in] Coord: the coordinate type
     * @param [in] SparseTrait: the sparse trait type
     * @param [in] l0B: the destination tensor
     * @param [in] l1B: the source tensor
     * @param [in] coord: the coordinate
     * @param [in] l1BIndexMatrix: the index matrix of the sparse tensor
     */
    template <class Coord, class SparseTrait>
    __aicore__ inline void TransposeLoadB2(const DstTensor& l0B, const SrcTensor& l1B, const Coord& coord,
                                           const AscendC::LocalTensor<SparseTrait>& l1BIndexMatrix)
    {
        auto dstShape = l0B.GetTensorTrait().GetLayout().GetShape();
        auto dstStride = l0B.GetTensorTrait().GetLayout().GetStride();
        auto srcShape = l1B.GetTensorTrait().GetLayout().GetShape();
        auto srcStride = l1B.GetTensorTrait().GetLayout().GetStride();

        // SET LOAD2D parameters , loop axis: K or M, or 1
        uint16_t madK = Get<1, 0>(dstShape) * Get<1, 1>(dstShape);

        // k is c0Size_ aligned for f32
        uint16_t kC0 = AscendC::CeilDiv(madK, Get<1, 0>(dstShape));
        uint16_t nFraC0 = Get<0, 1>(dstShape);
        uint16_t l0bLoop = 1;
        uint64_t l0bDstAddrStride = 0;
        uint64_t l0bSrcAddrStride = 0;
        uint8_t l0bRepeat = kC0 * nFraC0;
        uint16_t l0bDststride = 0;
        uint16_t l0bSrcstride = 1;

        uint16_t bL1N = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        if (nFraC0 * Get<0, 0>(dstShape) == bL1N) { // loop=1
            l0bLoop = 1;
        } else if (nFraC0 >= kC0) { // LOOP is K and repeat is n axis
            l0bLoop = kC0;
            l0bDstAddrStride = nFraC0 * Get<1, 1>(dstStride);
            l0bSrcAddrStride = bL1N * Get<1, 0>(srcShape);
            l0bRepeat = nFraC0;

            l0bDststride = 0;
            l0bSrcstride = 1;
        } else { // LOOP is N  and repeat is K axis
            l0bLoop = nFraC0;
            l0bDstAddrStride = Get<1, 1>(dstStride);
            l0bSrcAddrStride = Get<0, 1>(srcStride);
            l0bRepeat = kC0;

            l0bDststride = nFraC0 - 1;
            l0bSrcstride = bL1N;
        }
        // use load2d for L1_2_L0B
        // startIndex, repeatTimes, srcStride, sid, dstGap, ifTranspose, addrmode
        AscendC::LoadData2dParams loadDataParams{0, l0bRepeat, l0bSrcstride, 0, l0bDststride, 0, 0};
        uint16_t bL1KOffset = Get<1>(coord);
        uint64_t l1bOffset = Get<0>(coord) * Get<1, 0>(srcShape) + bL1KOffset * bL1N;
        uint64_t l0bOffset = 0;
        AscendC::LocalTensor<typename BType::TRANS_T> dstLocal;
        AscendC::LocalTensor<typename BType::TRANS_T> srcLocal;
        AscendC::LocalTensor<typename SparseTrait::LiteType> l1BIndex;
        for (uint64_t i = 0; i < l0bLoop; i++) {
            dstLocal.SetAddr(l0B[l0bOffset].address_);
            srcLocal.SetAddr(l1B[l1bOffset].address_);
            l1BIndex.SetAddr(l1BIndexMatrix[l1bOffset >> INDEX_SHIFT].address_);
            AscendC::LoadDataWithSparse(dstLocal, srcLocal, l1BIndex, loadDataParams);
            l1bOffset += l0bSrcAddrStride;
            l0bOffset += l0bDstAddrStride;
        }
    }
#endif
};

} // namespace Tile
} // namespace Gemm
} // namespace Act
#endif