/* Adapted from
 *      https://gitee.com/ascend/ascend-transformer-boost.git
 *
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../iterator.h"

constexpr uint32_t BLOCK_NUM = 16;
constexpr uint32_t BLOCK_SIZE_INT8 = 32;

template <>
struct l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, half, float> {
    /**
     * @brief Copy data from L0C buffer to global memory, partial specialized for
     *
     * @param gmTensor the destination tensor on global memory, which is stored in ND format.
     * @param l0cTensor the source tensor on L0C buffer, which is stored in FRACTAL_NZ format.
     * @param mTileActual the m-direction size of the matrix in L0C buffer.
     * @param nTileActual the n-direction size of the matrix in L0C buffer.
     * @param srcStride the source stride between the adjacent fractal matrix along n-direction in unit of C0_SIZE.
     * @param dstStride the leading dimension of the destination matrix in unit of element.
     */
    __aicore__ l0c_to_gm(AscendC::GlobalTensor<half> gmTensor,
                         AscendC::LocalTensor<float> l0cTensor,
                         uint32_t mTileActual,
                         uint32_t nTileActual,
                         uint32_t srcStride,
                         uint32_t dstStride,
                         uint8_t  unitFlag = 0)
    {
#ifdef __DAV_C220_CUBE__
        auto intriParams = AscendC::FixpipeParamsV220(nTileActual, // nSize
                                                      mTileActual, // mSize
                                                      srcStride,   // srcStride
                                                      dstStride,   // dstStride
                                                      false);      // enRelu

        intriParams.quantPre = QuantMode_t::F322F16;
        intriParams.unitFlag = unitFlag;
        AscendC::Fixpipe<half, float, AscendC::CFG_ROW_MAJOR>(gmTensor, l0cTensor, intriParams);
#else
        AscendC::FixpipeParams<float> intriParams(
            (nTileActual + BLOCK_NUM - 1) / AscendC::BLOCK_CUBE,
            static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(float) / BLOCK_SIZE_INT8),
            0,
            dstStride);
        intriParams.nz2ndParams = {true, 1, 0, 0, static_cast<uint16_t>(nTileActual)};
        intriParams.quantParams = {QuantMode_t::F322F16};
        AscendC::Fixpipe(gmTensor, l0cTensor, intriParams);
#endif
    };
};

template <>
struct l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, half, int32_t> {
    __aicore__ l0c_to_gm(AscendC::GlobalTensor<half> gmTensor,
                         AscendC::LocalTensor<int32_t> l0cTensor,
                         uint32_t mTileActual,
                         uint32_t nTileActual,
                         uint32_t srcStride,
                         uint32_t dstStride,
                         uint8_t  unitFlag = 0)
    {
#ifdef __DAV_C220_CUBE__
        auto intriParams = AscendC::FixpipeParamsV220(nTileActual, // nSize
                                                      mTileActual, // mSize
                                                      srcStride,   // srcStride
                                                      dstStride,   // dstStride
                                                      false);      // enRelu

        intriParams.quantPre = QuantMode_t::VDEQF16;
        intriParams.unitFlag = unitFlag;
        AscendC::Fixpipe<half, int32_t, AscendC::CFG_ROW_MAJOR>(gmTensor, l0cTensor, intriParams);
#else
        AscendC::FixpipeParams<int32_t> intriParams(
            (nTileActual + BLOCK_NUM - 1) / AscendC::BLOCK_CUBE,
            static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(float) / BLOCK_SIZE_INT8),
            0,
            dstStride);
        intriParams.nz2ndParams = {true, 1, 0, 0, static_cast<uint16_t>(nTileActual)};
        intriParams.quantParams = {QuantMode_t::VDEQF16};
        AscendC::Fixpipe(gmTensor, l0cTensor, intriParams);
#endif
    };
};

#ifdef __DAV_C220_CUBE__

template <>
struct l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, __bf16, float> {
    __aicore__ l0c_to_gm(AscendC::GlobalTensor<__bf16> gmTensor,
                         AscendC::LocalTensor<float> l0cTensor,
                         uint32_t mTileActual,
                         uint32_t nTileActual,
                         uint32_t srcStride,
                         uint32_t dstStride,
                         uint8_t  unitFlag = 0)
    {
#ifdef __DAV_C220_CUBE__
        auto intriParams = AscendC::FixpipeParamsV220(nTileActual, // nSize
                                                      mTileActual, // mSize
                                                      srcStride,   // srcStride
                                                      dstStride,   // dstStride
                                                      false);      // enRelu

        intriParams.quantPre = QuantMode_t::F322BF16;
        intriParams.unitFlag = unitFlag;
        AscendC::Fixpipe<__bf16, float, AscendC::CFG_ROW_MAJOR>(gmTensor, l0cTensor, intriParams);
#else
        AscendC::FixpipeParams<float> intriParams(
            (nTileActual + BLOCK_NUM - 1) / AscendC::BLOCK_CUBE,
            static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(float) / BLOCK_SIZE_INT8),
            0,
            dstStride);
        intriParams.nz2ndParams = {true, 1, 0, 0, static_cast<uint16_t>(nTileActual)};
        intriParams.quantParams = {QuantMode_t::F322BF16};
        AscendC::Fixpipe(gmTensor, l0cTensor, intriParams);
#endif
    };
};

#endif


// Partial specialization ND, float
template <>
struct l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, float, float> {
    __aicore__ l0c_to_gm(AscendC::GlobalTensor<float> gmTensor,
                         AscendC::LocalTensor<float> l0cTensor,
                         uint32_t mTileActual,
                         uint32_t nTileActual,
                         uint32_t srcStride,
                         uint32_t dstStride,
                         uint8_t  unitFlag = 0)
    {
#ifdef __DAV_C220_CUBE__
        auto intriParams = AscendC::FixpipeParamsV220(nTileActual, // nSize
                                                      mTileActual, // mSize
                                                      srcStride,   // srcStride
                                                      dstStride,   // dstStride
                                                      false);      // enRelu

        intriParams.quantPre = QuantMode_t::NoQuant;
        intriParams.unitFlag = unitFlag;
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(gmTensor, l0cTensor, intriParams);
#else
        AscendC::FixpipeParams<float> intriParams(
            (nTileActual + BLOCK_NUM - 1) / AscendC::BLOCK_CUBE,
            static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(float) / BLOCK_SIZE_INT8),
            0,
            dstStride);
        intriParams.nz2ndParams = {true, 1, 0, 0, static_cast<uint16_t>(nTileActual)};
        intriParams.quantParams = {QuantMode_t::NoQuant};
        AscendC::Fixpipe(gmTensor, l0cTensor, intriParams);
#endif
    };
};

template <>
struct l0c_to_gm<ArchType::ASCEND_V220, DataFormat::NZ, half, float> {
    __aicore__ l0c_to_gm(AscendC::GlobalTensor<half> gmTensor,
                         AscendC::LocalTensor<float> l0cTensor,
                         uint32_t mTileActual,
                         uint32_t nTileActual,
                         uint32_t srcStride,
                         uint32_t dstStride,
                         uint8_t  unitFlag = 0)
    {
#ifdef __DAV_C220_CUBE__
        auto intriParams = AscendC::FixpipeParamsV220(nTileActual, // nSize
                                                      mTileActual, // mSize
                                                      srcStride,   // srcStride
                                                      dstStride,   // dstStride
                                                      false);      // enRelu

        intriParams.quantPre = QuantMode_t::F322F16;
        intriParams.unitFlag = unitFlag;
        AscendC::Fixpipe<half, float, AscendC::CFG_NZ>(gmTensor, l0cTensor, intriParams);
#else
        AscendC::FixpipeParams<float> intriParams(
            (nTileActual + BLOCK_NUM - 1) / AscendC::BLOCK_CUBE,
            static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(float) / BLOCK_SIZE_INT8),
            0,
            dstStride - (nTileActual * sizeof(half) / sizeof(float)));
        intriParams.quantParams = {QuantMode_t::F322F16};
        AscendC::Fixpipe(gmTensor, l0cTensor, intriParams);
#endif
    };
};

template <>
struct l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, int32_t, int32_t> {
    __aicore__ l0c_to_gm(AscendC::GlobalTensor<int32_t> gmTensor,
                         AscendC::LocalTensor<int32_t> l0cTensor,
                         uint32_t mTileActual,
                         uint32_t nTileActual,
                         uint32_t srcStride,
                         uint32_t dstStride,
                         uint8_t  unitFlag = 0)
    {
#ifdef __DAV_C220_CUBE__
        auto intriParams = AscendC::FixpipeParamsV220(nTileActual, // nSize
                                                      mTileActual, // mSize
                                                      srcStride,   // srcStride
                                                      dstStride,   // dstStride
                                                      false);      // enRelu

        intriParams.quantPre = QuantMode_t::NoQuant;
        intriParams.unitFlag = unitFlag;
        AscendC::Fixpipe<int32_t, int32_t, AscendC::CFG_ROW_MAJOR>(gmTensor, l0cTensor, intriParams);
#else
        AscendC::FixpipeParams<int32_t> intriParams(
            (nTileActual + BLOCK_NUM - 1) / AscendC::BLOCK_CUBE,
            static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(float) / BLOCK_SIZE_INT8),
            0,
            dstStride);
        intriParams.nz2ndParams = {true, 1, 0, 0, static_cast<uint16_t>(nTileActual)};
        intriParams.quantParams = {QuantMode_t::VDEQF16};
        AscendC::Fixpipe(gmTensor, l0cTensor, intriParams);
#endif
    };
};
