/* Adapted from
 *      https://gitee.com/ascend/ascend-transformer-boost.git
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../iterator.h"

// Partial specialization for V220, ND_in, ND_out
template <ArchType ArchTag, typename DataType>
struct gm_to_l1<ArchTag, DataType, DataFormat::ND, DataFormat::ND> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType);

    __aicore__ gm_to_l1(AscendC::LocalTensor<DataType> l1Tensor,
                        AscendC::GlobalTensor<DataType> gmTensor,
                        uint32_t nTileActual,
                        uint32_t nTileCeil,
                        uint32_t nVal,
                        uint32_t dTileActual,
                        uint32_t dTileCeil,
                        uint32_t dVal)
    {
        AscendC::DataCopy(l1Tensor,                                                               // dst
                          gmTensor,                                                               // src
                          AscendC::DataCopyParams(1,                                              // nBurst
                                                  CeilDiv<BLOCK_SIZE>(nTileActual * dTileActual), // lenBurst
                                                  0,                                              // srcGap
                                                  0));                                            // dstGap
    };
};

// Partial specialization for NZ_in, NZ_out
template <ArchType ArchTag, typename DataType>
struct gm_to_l1<ArchTag, DataType, DataFormat::NZ, DataFormat::NZ> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType);
    static constexpr uint32_t STRIDE_LIMIT = 65536;

    __aicore__ gm_to_l1(AscendC::LocalTensor<DataType> l1Tensor,
                        AscendC::GlobalTensor<DataType> gmTensor,
                        uint32_t nTileActual,
                        uint32_t nTileCeil,
                        uint32_t nVal,
                        uint32_t dTileActual,
                        uint32_t dTileCeil,
                        uint32_t dVal)
    {
        uint64_t srcStride = nVal - nTileCeil;
        if (srcStride < STRIDE_LIMIT) {
            AscendC::DataCopy(l1Tensor,                                       // dst
                              gmTensor,                                       // src
                              AscendC::DataCopyParams(dTileCeil / BLOCK_SIZE, // nBurst
                                                      nTileCeil,              // lenBurst
                                                      srcStride,              // srcGap
                                                      0));                    // dstGap
        } else {
            for (uint64_t i = 0; i < dTileCeil / BLOCK_SIZE; i++) {
                uint64_t dstOffset = i * nTileCeil * BLOCK_SIZE;
                uint64_t srcOffset = i * nVal * BLOCK_SIZE;
                AscendC::DataCopy(l1Tensor[dstOffset],               // dst
                                  gmTensor[srcOffset],               // src
                                  AscendC::DataCopyParams(1,         // nBurst
                                                          nTileCeil, // lenBurst
                                                          0,         // srcGap
                                                          0));       // dstGap
            }
        }
    };
};

// Partial specialization for V220, ND_in, ND_out
template <ArchType ArchTag, typename DataType>
struct gm_to_l1<ArchTag, DataType, DataFormat::ND, DataFormat::NZ> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType);
    static constexpr uint32_t STRIDE_LIMIT = 65536;

    __aicore__ gm_to_l1(AscendC::LocalTensor<DataType> l1Tensor,
                        AscendC::GlobalTensor<DataType> gmTensor,
                        uint32_t nTileActual,
                        uint32_t nTileCeil,
                        uint32_t nVal,
                        uint32_t dTileActual,
                        uint32_t dTileCeil,
                        uint32_t dVal)
    {
        if (dVal < STRIDE_LIMIT) {
            AscendC::DataCopy(l1Tensor,
                              gmTensor,
                              AscendC::Nd2NzParams(1,           // ndNum
                                                   nTileActual, // nValue
                                                   dTileActual, // dValue
                                                   0,           // srcNdMatrixStride, unused
                                                   dVal,        // srcDValue
                                                   nTileCeil,   // dstNzC0Stride
                                                   1,           // dstNzNStride
                                                   0));         // dstNzMatrixStride, unused
        } else {
            for (uint32_t i = 0; i < nTileActual; i++) {
                AscendC::DataCopy(l1Tensor[i * BLOCK_SIZE],
                                  gmTensor[i * dVal],
                                  AscendC::Nd2NzParams(1,           // ndNum
                                                       1,           // nValue
                                                       dTileActual, // dValue
                                                       0,           // srcNdMatrixStride, unused
                                                       0,           // srcDValue
                                                       nTileCeil,   // dstNzC0Stride
                                                       0,           // dstNzNStride
                                                       0));         // dstNzMatrixStride, unused
            }
        }
    };
};

// Partial specialization for V220, ND_in, NZ_out
template <ArchType ArchTag, typename DataType>
struct gm_to_l1<ArchTag, DataType, DataFormat::ND, DataFormat::ZN> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType);
    static constexpr uint32_t STRIDE_LIMIT = 65536;

    __aicore__ gm_to_l1(AscendC::LocalTensor<DataType> l1Tensor,
                        AscendC::GlobalTensor<DataType> gmTensor,
                        uint32_t nTileActual,
                        uint32_t nTileCeil,
                        uint32_t nVal,
                        uint32_t dTileActual,
                        uint32_t dTileCeil,
                        uint32_t dVal)
    {
        if (dVal < STRIDE_LIMIT) {
            AscendC::DataCopy(l1Tensor,
                              gmTensor,
                              AscendC::Nd2NzParams(1,           // ndNum
                                                   nTileActual, // nValue
                                                   dTileActual, // dValue
                                                   0,           // srcNdMatrixStride, unused
                                                   dVal,        // srcDValue
                                                   nTileCeil,   // dstNzC0Stride
                                                   1,           // dstNzNStride
                                                   0));         // dstNzMatrixStride, unused
        } else {
            for (uint32_t i = 0; i < nTileActual; ++i) {
                AscendC::DataCopy(l1Tensor,
                                  gmTensor,
                                  AscendC::Nd2NzParams(1,           // ndNum
                                                       1,           // nValue
                                                       dTileActual, // dValue
                                                       0,           // srcNdMatrixStride, unused
                                                       0,           // srcDValue
                                                       nTileCeil,   // dstNzC0Stride
                                                       0,           // dstNzNStride
                                                       0));         // dstNzMatrixStride, unused
            }
        }
    };
};
