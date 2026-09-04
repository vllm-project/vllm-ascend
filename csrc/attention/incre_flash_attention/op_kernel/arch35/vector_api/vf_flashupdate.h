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
 * \file vf_flashupdate.h
 * \brief
 */
#ifndef MY_FLASH_UPDATE_INTERFACE_H
#define MY_FLASH_UPDATE_INTERFACE_H

#include "kernel_tensor.h"

namespace AscendC {
/* **************************************************************************************************
 * FlashUpdate, fp32
 * ************************************************************************************************* */

template <typename T, typename OUTPUT_T>
__aicore__ inline void FlashUpdateBasic(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor,
                                        const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor,
                                        const uint16_t m, const uint16_t d, const uint16_t srcD)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float * curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ float * preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();

    const uint16_t floatRepSize = 64;
    const uint16_t reduceSize = 8;
    const uint16_t dLoops = srcD / floatRepSize;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vRegMax;
        MicroAPI::RegTensor<T> vRegPre;
        MicroAPI::RegTensor<T> vRegCur;
        MicroAPI::RegTensor<T> vRegMul;
        MicroAPI::RegTensor<T> vRegAdd;

        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();

        // dstTensor = preTensor * expMaxTensor + curTensor
        for (uint16_t i = 0; i < m; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vRegMax, expMaxUb + i * reduceSize); // [m,8]

            for (uint16_t j = 0; j < dLoops; ++j) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vRegPre, preUb + i * srcD + j * floatRepSize);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vRegCur, curUb + i * srcD + j * floatRepSize);

                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vRegMul, vRegMax, vRegPre, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vRegAdd, vRegMul, vRegCur, maskRegAll);

                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    dstUb + i * srcD + j * floatRepSize, vRegAdd, maskRegAll); // output0
            }
        }
    }
}

template <typename T, typename OUTPUT_T>
__aicore__ inline void FlashUpdateGeneral(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor,
                                          const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor,
                                          const uint16_t m, const uint16_t d, const uint16_t srcD)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float * curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ float * preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();

    const uint16_t floatRepSize = 64;
    const uint16_t reduceSize = 8;
    const uint16_t dLoops = d / floatRepSize;
    const uint16_t tailD = d % floatRepSize;
    uint32_t pltTailD = static_cast<uint32_t>(tailD);

    uint16_t hasTail = 0;
    if (tailD > 0) {
        hasTail = 1;
    }

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vRegMax;
        MicroAPI::RegTensor<T> vRegPre;
        MicroAPI::RegTensor<T> vRegCur;
        MicroAPI::RegTensor<T> vRegMul;
        MicroAPI::RegTensor<T> vRegAdd;

        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegTailD = MicroAPI::UpdateMask<T>(pltTailD);

        // dstTensor = preTensor * expMaxTensor + curTensor
        for (uint16_t i = 0; i < m; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vRegMax, expMaxUb + i * reduceSize); // [m,8]

            for (uint16_t j = 0; j < dLoops; ++j) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vRegPre, preUb + i * srcD + j * floatRepSize);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vRegCur, curUb + i * srcD + j * floatRepSize);

                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vRegMul, vRegMax, vRegPre, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vRegAdd, vRegMul, vRegCur, maskRegAll);

                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    dstUb + i * srcD + j * floatRepSize, vRegAdd, maskRegAll); // output0
            }
            for (uint16_t t = 0; t < hasTail; ++t) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vRegPre, preUb + i * srcD + dLoops * floatRepSize);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vRegCur, curUb + i * srcD + dLoops * floatRepSize);

                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vRegMul, vRegMax, vRegPre, maskRegTailD);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vRegAdd, vRegMul, vRegCur, maskRegTailD);

                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    dstUb + i * srcD + dLoops * floatRepSize, vRegAdd, maskRegTailD);
            }
        }
    }
}

/*
 * @ingroup FlashUpdate
 * @brief compute, dstTensor = preTensor * expMaxTensor + curTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input colums, should be 32 bytes aligned
 */
template <typename T, typename OUTPUT_T>
__aicore__ inline void FlashUpdate(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor,
                                   const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor,
                                   const uint16_t m, const uint16_t d, const uint16_t srcD)
{
    static_assert(IsSameType<T, float>::value, "VF FlashUpdate, T must be float");

    const uint16_t floatRepSize = 64;
    if (d % floatRepSize == 0) {
        FlashUpdateBasic<T, OUTPUT_T>(dstTensor, curTensor, preTensor, expMaxTensor, m, d, srcD);
    } else {
        FlashUpdateGeneral<T, OUTPUT_T>(dstTensor, curTensor, preTensor, expMaxTensor, m, d, srcD);
    }
}


template <typename T, typename OUTPUT_T>
__aicore__ inline void FlashUpdateLastBasic(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor,
                                            const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor,
                                            const LocalTensor<T>& expSumTensor, const uint16_t m,
                                            const uint16_t d, const uint16_t srcD)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float * curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ float * preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    const uint16_t floatRepSize = 64;
    uint16_t reduceSize = 8;
    uint16_t dLoops = srcD / floatRepSize;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vRegMax;
        MicroAPI::RegTensor<T> vRegSum;
        MicroAPI::RegTensor<T> vRegPre;
        MicroAPI::RegTensor<T> vRegCur;
        MicroAPI::RegTensor<T> vRegAdd;
        MicroAPI::RegTensor<T> vRegMul;
        MicroAPI::RegTensor<T> vRegDiv;

        // false: normal mode; true: higher precision mode
        static constexpr MicroAPI::DivSpecificMode mode = {MicroAPI::MaskMergeMode::ZEROING, false};
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();

        for (uint16_t i = 0; i < m; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vRegMax, expMaxUb + i * reduceSize);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vRegSum, expSumUb + i * reduceSize);

            for (uint16_t j = 0; j < dLoops; ++j) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vRegPre, preUb + i * d + j * floatRepSize);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vRegCur, curUb + i * d + j * floatRepSize);

                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vRegMul, vRegMax, vRegPre, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vRegAdd, vRegMul, vRegCur, maskRegAll);
                MicroAPI::Div<T, &mode>(vRegDiv, vRegAdd, vRegSum, maskRegAll);

                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    dstUb + i * d + j * floatRepSize, vRegDiv, maskRegAll); // output0
            }
        }
    }
}

template <typename T, typename OUTPUT_T>
__aicore__ inline void FlashUpdateLastGeneral(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor,
                                              const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor,
                                              const LocalTensor<T>& expSumTensor, const uint16_t m,
                                              const uint16_t d, const uint16_t srcD)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float * curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ float * preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    const uint16_t floatRepSize = 64;
    uint16_t reduceSize = 8;
    uint16_t dLoops = d / floatRepSize;
    uint16_t tailD = d % floatRepSize;
    uint32_t pltTailD = static_cast<uint32_t>(tailD);

    uint16_t hasTail = 0;
    if (tailD > 0) {
        hasTail = 1;
    }

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vRegMax;
        MicroAPI::RegTensor<T> vRegSum;
        MicroAPI::RegTensor<T> vRegPre;
        MicroAPI::RegTensor<T> vRegCur;
        MicroAPI::RegTensor<T> vRegAdd;
        MicroAPI::RegTensor<T> vRegMul;
        MicroAPI::RegTensor<T> vRegDiv;

        // false: normal mode; true: higher precision mode
        static constexpr MicroAPI::DivSpecificMode mode = {MicroAPI::MaskMergeMode::ZEROING, false};
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegTailD = MicroAPI::UpdateMask<T>(pltTailD);

        for (uint16_t i = 0; i < m; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vRegMax, expMaxUb + i * reduceSize);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vRegSum, expSumUb + i * reduceSize);

            for (uint16_t j = 0; j < dLoops; ++j) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vRegPre, preUb + i * srcD + j * floatRepSize);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vRegCur, curUb + i * srcD + j * floatRepSize);

                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vRegMul, vRegMax, vRegPre, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vRegAdd, vRegMul, vRegCur, maskRegAll);
                MicroAPI::Div<T, &mode>(vRegDiv, vRegAdd, vRegSum, maskRegAll);

                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    dstUb + i * srcD + j * floatRepSize, vRegDiv, maskRegAll); // output0
            }

            for (uint16_t t = 0; t < hasTail; ++t) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vRegPre, preUb + i * srcD + dLoops * floatRepSize);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vRegCur, curUb + i * srcD + dLoops * floatRepSize);

                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vRegMul, vRegMax, vRegPre, maskRegTailD);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vRegAdd, vRegMul, vRegCur, maskRegTailD);
                MicroAPI::Div<T, &mode>(vRegDiv, vRegAdd, vRegSum, maskRegTailD);

                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    dstUb + i * srcD + dLoops * floatRepSize, vRegDiv, maskRegTailD);
            }
        }
    }
}

/*
 * @ingroup FlashUpdateLast
 * @brief compute, dstTensor = (preTensor * expMaxTensor + curTensor) / expSumTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] expSumTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input colums, 32 bytes align
 */
template <typename T, typename OUTPUT_T>
__aicore__ inline void FlashUpdateLast(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor,
                                       const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor,
                                       const LocalTensor<T>& expSumTensor, uint16_t m,
                                       uint16_t d, uint16_t srcD)
{
    static_assert(IsSameType<T, float>::value, "VF FlashUpdateLast, T must be float");

    const uint16_t floatRepSize = 64;
    if (srcD % floatRepSize == 0) {
        FlashUpdateLastBasic<T, OUTPUT_T>(dstTensor, curTensor, preTensor, expMaxTensor, expSumTensor, m, d, srcD);
    } else {
        FlashUpdateLastGeneral<T, OUTPUT_T>(dstTensor, curTensor, preTensor, expMaxTensor, expSumTensor, m, d, srcD);
    }
}

} // namespace

#endif // MY_FLASH_UPDATE_INTERFACE_H