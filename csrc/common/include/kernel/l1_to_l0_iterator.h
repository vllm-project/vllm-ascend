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
 * \file l1_to_l0_iterator.h
 * \brief
 */

#ifndef L1_TO_L0_ITERATOR_H
#define L1_TO_L0_ITERATOR_H

#include "iterator.h"

/////////////////////////////////////////////////////
// l1_to_l0_a
/////////////////////////////////////////////////////

// Partial specialization for vector
template <ArchType ArchTag, typename DataType, bool IsTransPose>
struct l1_to_l0_a<ArchTag, DataType, IsTransPose, DataFormatT::VECTOR, DataFormatT::VECTOR> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t FRACTAL_SIZE = HardwareParams::fractalSize / sizeof(DataType);

    __aicore__ l1_to_l0_a(AscendC::LocalTensor<DataType> l0Tensor,
                          AscendC::LocalTensor<DataType> l1Tensor,
                          uint32_t mTileCeil,
                          uint32_t kPartCeil,
                          uint32_t mSrcStride,
                          uint32_t kSrcStride,
                          uint32_t mDstStride,
                          uint32_t kDstStride)
    {
        AscendC::LoadData(l0Tensor,
                          l1Tensor,
                          AscendC::LoadData2dParams(0,           // baseIdx
                                                    kPartCeil,   // repeat
                                                    kSrcStride,  // srcStride
                                                    0,           // sid
                                                    kDstStride,  // dstStride
                                                    IsTransPose, // transpose
                                                    0));         // addrCalMode
    };
};

// Partial specialization for no transpose, not vector
template <ArchType ArchTag, typename DataType>
struct l1_to_l0_a<ArchTag, DataType, false, DataFormatT::ZN, DataFormatT::ZZ> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType);
    static constexpr uint32_t FRACTAL_SIZE = HardwareParams::fractalSize / sizeof(DataType);
    static constexpr uint32_t BLOCK_NUM_PER_FRACTAL = HardwareParams::fractalSize / HardwareParams::l1l0BlockSize;

    __aicore__ l1_to_l0_a(AscendC::LocalTensor<DataType> l0Tensor,
                          AscendC::LocalTensor<DataType> l1Tensor,
                          uint32_t mTileCeil,
                          uint32_t kPartCeil,
                          uint32_t mSrcStride,
                          uint32_t kSrcStride,
                          uint32_t mDstStride,
                          uint32_t kDstStride)
    {
        for (uint32_t i = 0; i < mTileCeil / BLOCK_NUM_PER_FRACTAL; i++) {
            AscendC::LoadData(l0Tensor[i * mDstStride * FRACTAL_SIZE],
                              l1Tensor[i * mSrcStride * FRACTAL_SIZE],
                              AscendC::LoadData2dParams(0,                                             // baseIdx
                                                        static_cast<uint16_t>(kPartCeil / BLOCK_SIZE), // repeat
                                                        kSrcStride,                                    // srcStride
                                                        0,                                             // sid
                                                        kDstStride - 1,                                // dstStride
                                                        false,                                         // transpose
                                                        0));                                           // addrCalMode
        }
    };
};

// Partial specialization for transpose, not vector
template <ArchType ArchTag, typename DataType>
struct l1_to_l0_a<ArchTag, DataType, true, DataFormatT::ZN, DataFormatT::ZZ> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType);
    static constexpr uint32_t FRACTAL_SIZE = HardwareParams::fractalSize / sizeof(DataType);
    static constexpr uint32_t BLOCK_NUM_PER_FRACTAL = HardwareParams::fractalSize / HardwareParams::l1l0BlockSize;

    __aicore__ l1_to_l0_a(AscendC::LocalTensor<DataType> l0Tensor,
                          AscendC::LocalTensor<DataType> l1Tensor,
                          uint32_t mTileCeil,
                          uint32_t kPartCeil,
                          uint32_t mSrcStride,
                          uint32_t kSrcStride,
                          uint32_t mDstStride,
                          uint32_t kDstStride)
    {
        for (uint32_t i = 0; i < mTileCeil / BLOCK_SIZE; i++) {
            AscendC::LoadData(l0Tensor[i * mDstStride * FRACTAL_SIZE],
                              l1Tensor[i * mSrcStride * FRACTAL_SIZE],
                              AscendC::LoadData2dParams(0,
                                                        static_cast<uint16_t>(kPartCeil / BLOCK_NUM_PER_FRACTAL),
                                                        kSrcStride,
                                                        0,
                                                        kDstStride - 1,
                                                        true,
                                                        0));
        }
    };
};

template <ArchType ArchTag, typename DataType>
struct l1_to_l0_a<ArchTag, DataType, false, DataFormatT::NZ, DataFormatT::ZZ> {
    using HardwareParams = HardwareInfo<ArchTag>;
    // 16 * 32
    static constexpr uint32_t ROW_BLOCK_SIZE = 16;
    static constexpr uint32_t COL_BLOCK_SIZE = 32 / sizeof(DataType);
    static constexpr uint32_t FRACTAL_SIZE = HardwareParams::fractalSize / sizeof(DataType);
    static constexpr uint32_t BLOCK_NUM_PER_FRACTAL = HardwareParams::fractalSize / HardwareParams::l1l0BlockSize;

    __aicore__ l1_to_l0_a(AscendC::LocalTensor<DataType> l0Tensor,
                          AscendC::LocalTensor<DataType> l1Tensor,
                          uint32_t mTileCeil,
                          uint32_t kPartCeil,
                          uint32_t mSrcStride,
                          uint32_t kSrcStride,
                          uint32_t mDstStride,
                          uint32_t kDstStride)
    {
        for (uint32_t i = 0; i < mTileCeil / ROW_BLOCK_SIZE; i++) {
            AscendC::LoadData(l0Tensor[i * ROW_BLOCK_SIZE * kPartCeil],
                              l1Tensor[i * FRACTAL_SIZE],
                              AscendC::LoadData2dParams(0,
                                                        static_cast<uint16_t>(kPartCeil / COL_BLOCK_SIZE),
                                                        mTileCeil / ROW_BLOCK_SIZE,
                                                        0,
                                                        0,
                                                        false,
                                                        0));
        }
    };
};

/////////////////////////////////////////////////////
// l1_to_l0_b
/////////////////////////////////////////////////////

// Partial specialization for vector
template <ArchType ArchTag, typename DataType, bool IsTransPose>
struct l1_to_l0_b<ArchTag, DataType, IsTransPose, DataFormatT::VECTOR, DataFormatT::VECTOR> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t FRACTAL_SIZE = HardwareParams::fractalSize / sizeof(DataType);

    __aicore__ l1_to_l0_b(AscendC::LocalTensor<DataType> l0Tensor,
                          AscendC::LocalTensor<DataType> l1Tensor,
                          uint32_t nTileCeil,
                          uint32_t kPartCeil,
                          uint32_t nSrcStride,
                          uint32_t kSrcStride,
                          uint32_t nDstStride,
                          uint32_t kDstStride)
    {
        AscendC::LoadData(
            l0Tensor, l1Tensor, AscendC::LoadData2dParams(0, kPartCeil, kSrcStride, 0, kDstStride, IsTransPose, 0));
    };
};

template <ArchType ArchTag>
struct l1_to_l0_b<ArchTag, int8_t, true, DataFormatT::NZ, DataFormatT::ZN> {
    using HardwareParams = HardwareInfo<ArchTag>;
    using DataType = int8_t;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType);

    __aicore__ l1_to_l0_b(AscendC::LocalTensor<DataType> l0Tensor,
                          AscendC::LocalTensor<DataType> l1Tensor,
                          uint32_t nTileCeil,
                          uint32_t kPartCeil,
                          uint32_t nSrcStride,
                          uint32_t kSrcStride,
                          uint32_t nDstStride,
                          uint32_t kDstStride)
    {
        for (uint32_t i = 0; i < nTileCeil / BLOCK_SIZE; i++) {
            AscendC::LoadDataWithTranspose(l0Tensor[i * kPartCeil * BLOCK_SIZE],
                                           l1Tensor[i * BLOCK_SIZE * BLOCK_SIZE],
                                           AscendC::LoadData2dTransposeParams(0,                      // startIndexIn
                                                                              kPartCeil / BLOCK_SIZE, // repeatTimesIn
                                                                              nTileCeil / BLOCK_SIZE, // srcStrideIn
                                                                              1,                      // dstGapIn
                                                                              0,                      // dstfracGapIn
                                                                              0)                      // addrModeIn
            );
        }
    };
};

// Partial specialization for no transpose, not vector
template <ArchType ArchTag, typename DataType>
struct l1_to_l0_b<ArchTag, DataType, false, DataFormatT::ZN, DataFormatT::NZ> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType);
    static constexpr uint32_t FRACTAL_SIZE = HardwareParams::fractalSize / sizeof(DataType);
    static constexpr uint32_t BLOCK_NUM_PER_FRACTAL = HardwareParams::fractalSize / HardwareParams::l1l0BlockSize;

    __aicore__ l1_to_l0_b(AscendC::LocalTensor<DataType> l0Tensor,
                          AscendC::LocalTensor<DataType> l1Tensor,
                          uint32_t nTileCeil,
                          uint32_t kPartCeil,
                          uint32_t nSrcStride,
                          uint32_t kSrcStride,
                          uint32_t nDstStride,
                          uint32_t kDstStride)
    {
        for (uint32_t i = 0; i < kPartCeil / BLOCK_NUM_PER_FRACTAL; i++) {
            AscendC::LoadData(l0Tensor[i * kDstStride * FRACTAL_SIZE],
                              l1Tensor[i * kSrcStride * FRACTAL_SIZE],
                              AscendC::LoadData2dParams(0,                                             // baseIdx
                                                        static_cast<uint16_t>(nTileCeil / BLOCK_SIZE), // repeat
                                                        nSrcStride,                                    // srcStride
                                                        0,                                             // sid
                                                        nDstStride - 1,                                // dstStride
                                                        true,                                          // transpose
                                                        0));                                           // addrCalMode
        }
    };
};

// Partial specialization for transpose, not vector
template <ArchType ArchTag, typename DataType>
struct l1_to_l0_b<ArchTag, DataType, true, DataFormatT::ZN, DataFormatT::NZ> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType);
    static constexpr uint32_t FRACTAL_SIZE = HardwareParams::fractalSize / sizeof(DataType);
    static constexpr uint32_t BLOCK_NUM_PER_FRACTAL = HardwareParams::fractalSize / HardwareParams::l1l0BlockSize;
    __aicore__ l1_to_l0_b(AscendC::LocalTensor<DataType> l0Tensor,
                          AscendC::LocalTensor<DataType> l1Tensor,
                          uint32_t nTileCeil,
                          uint32_t kPartCeil,
                          uint32_t nSrcStride,
                          uint32_t kSrcStride,
                          uint32_t nDstStride,
                          uint32_t kDstStride)
    {
        AscendC::LoadData(
            l0Tensor,
            l1Tensor,
            AscendC::LoadData2dParams(0,                                                           // baseIdx
                                      static_cast<uint16_t>(kPartCeil * nTileCeil / FRACTAL_SIZE), // repeat
                                      1,                                                           // srcStride
                                      0,                                                           // sid
                                      0,                                                           // dstStride
                                      false,                                                       // transpose
                                      0));                                                         // addr_cal_mode_t
    };
};

#endif // L1_TO_L0_ITERATOR_H