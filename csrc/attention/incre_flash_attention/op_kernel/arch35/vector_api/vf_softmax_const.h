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
 * \file vf_softmax_const.h
 * \brief
 */
#ifndef VF_SOFTMAX_FLASH_CONST_H
#define VF_SOFTMAX_FLASH_CONST_H

#include "kernel_tensor.h"
#include "adv_api/activation/softmax.h"
#include "../incre_flash_attention_pub.h"

namespace AscendC {

using MicroAPI::MaskMergeMode;
using MicroAPI::LoadDist;
using MicroAPI::StoreDist;
using MicroAPI::RegLayout;
using MicroAPI::PostLiteral;
using MicroAPI::MaskPattern;
using MicroAPI::MaskDist;
using MicroAPI::HighLowPart;
using MicroAPI::SatMode;

const float maskMinValue = SOFTMAX_MIN_NUM;
const float maskMinValue2 = -3e38;

/*
 * @ingroup UPDATE_EXPSUM_MAX
 * @brief compute max = reducemax, exp(x-max)/sum(exp(x-max))
 * @param [out] expSumUb, expSum output ub phyaddr
 * @param [out] maxUb, Max output ub phyaddr
 * @param [out] expMaxUb, expMax output ub phyaddr
 * @param [in] inExpSumUb, prev ExpSum ub phyaddr
 * @param [in] inMaxUb, prev Max ub phyaddr
 * @param [in] tmpMaxUb, current Max ub phyaddr
 * @param [in] tmpExpSumUb, current expsum ub phyaddr
 * @param [in] srcRows
 */
#define UPDATE_EXPSUM_MAX(expSumUb, maxUb, expMaxUb, inExpSumUb, inMaxUb, tmpExpSumUb, tmpMaxUb, srcRows)        \
    do {                                                                                                         \
        MicroAPI::RegTensor<T> vregInMax;                                                                        \
        MicroAPI::RegTensor<T> vregMax;                                                                          \
        MicroAPI::RegTensor<T> vregExpMax;                                                                       \
        MicroAPI::RegTensor<T> vregInExpSum;                                                                     \
        MicroAPI::RegTensor<T> vregExpSumUpdate;                                                                 \
        const uint32_t vlT = 256 / sizeof(T);                                                                    \
        const uint32_t dealRows = 8;                                                                             \
        uint16_t updateLoops = ((srcRows) + (dealRows - 1)) / dealRows;                                          \
        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MaskPattern::ALL>();                                 \
                                                                                                                 \
        for (uint16_t i = 0; i < updateLoops; ++i) {                                                             \
            MicroAPI::DataCopy<T, LoadDist::DIST_E2B_B32>(vregMax, (tmpMaxUb) + (i * dealRows));                 \
            MicroAPI::DataCopy<T, LoadDist::DIST_NORM>(vregInMax, (inMaxUb) + (i * vlT));                        \
            MicroAPI::FusedExpSub<T, T, RegLayout::ONE, MaskMergeMode::ZEROING>(vregExpMax, vregInMax, vregMax,  \
                                                                                pregAll);                        \
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM_B32>((expMaxUb) + (i * vlT), vregExpMax, pregAll);        \
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM_B32>((maxUb) + (i * vlT), vregMax, pregAll);              \
            MicroAPI::DataCopy<T, LoadDist::DIST_NORM>(vregInExpSum, (inExpSumUb) + (i * vlT));                  \
            MicroAPI::DataCopy<T, LoadDist::DIST_E2B_B32>(vregExpSumUpdate, (tmpExpSumUb) + (i * dealRows));     \
            MicroAPI::MulAddDst<T, MaskMergeMode::ZEROING>(vregExpSumUpdate, vregInExpSum, vregExpMax, pregAll); \
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM_B32>((expSumUb) + (i * vlT), vregExpSumUpdate, pregAll);  \
        }                                                                                                        \
    } while (0)

/* **************************************************************************************************
 * SoftmaxFlashV2
 * generalize to srcK = (0, 512], fp32
 * ************************************************************************************************* */

/*
 * @ingroup SoftmaxFlashV2
 * @brief compute max = reducemax, exp(x-max)/sum(exp(x-max))
 * @param [out] dstTensor, output LocalTensor
 * @param [out] expSumTensor, out sum(exp(x-max)) of last axis
 * @param [out] maxTensor, out max value of last axis
 * @param [in] srcTensor, input LocalTensor
 * @param [out] expMaxTensor, output expmax LocalTensor
 * @param [in] inExpSumTensor, in sum(exp(x-max)) of last softmax result
 * @param [in] inMaxTensor, in max value of last softmax result
 * @param [in] sharedTmpBuffer, input local temporary Tensor
 * @param [in] scale, in scale of srcTensor
 * @param [in] maskTensor, in AttenMask LocalTensor
 * @param [in] softmaxShapeInfo, in shapeInfo of softmax
 */

// 本函数IFA S方向泛化长度为(0, 64]
template <typename T, typename T1, typename PSE_SHIFT_T, bool withPseShift = false, bool isUpdate = false,
          bool withAttenMask = false, bool isBasicBlock = false, bool isBand = false, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxFlashV2_base64_update(
    const LocalTensor<T1>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const T scale,
    const uint32_t maskSizeAlignPFA, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<PSE_SHIFT_T>& pseTensor,
    const SoftMaxShapeInfo& softmaxShapeInfo = {})
{
    const uint32_t originN = softmaxShapeInfo.oriSrcK;  // N轴实际数据长度，为SInnerLoop切块的 S/act_seq_len
    const uint32_t fixSrcN = 64;
    const uint32_t floatRepSize = 64;         // 寄存器一次可以加载64个元素（fp32类型，对应256B）
    const uint32_t reduceN = 32 / sizeof(T);  // m每行的结果存为8个重复元素（32B对齐）
    const uint16_t rows = static_cast<uint16_t>(softmaxShapeInfo.srcM);  // 总行数

    __VEC_SCOPE__
    {
        __ubuf__ T1* expUb = (__ubuf__ T1*)dstTensor.GetPhyAddr();
        __ubuf__ float* expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
        __ubuf__ float* maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
        __ubuf__ float* srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
        __ubuf__ float* expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
        __ubuf__ float* inExpSumUb = (__ubuf__ T*)inExpSumTensor.GetPhyAddr();
        __ubuf__ float* inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
        __ubuf__ float* tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
        __ubuf__ float* tmpMaxUb = tmpExpSumUb + 16;  // 16 x sum + 16 x max
        __ubuf__ float* tmpExpSumUb2 = tmpExpSumUb;
        __ubuf__ float* tmpMaxUb2 = tmpMaxUb;
        __ubuf__ uint8_t* maskUb = (__ubuf__ uint8_t*)maskTensor.GetPhyAddr();
        __ubuf__ uint8_t* maskBandUb;
        if constexpr (isBand) {
            maskBandUb = (__ubuf__ uint8_t*)maskTensor[rows * maskSizeAlignPFA].GetPhyAddr();
        }
        __ubuf__ PSE_SHIFT_T* pseUb = (__ubuf__ PSE_SHIFT_T*)pseTensor.GetPhyAddr();

        uint16_t blockStride = 16 + 1;  // 加1是为了解写写ub bank冲突
        uint16_t repeatStride = 1;      //  每次nd2nz写完之后ub往后偏移1个block

        MicroAPI::UnalignReg uregMax;
        MicroAPI::UnalignReg uregExpSum;

        uint32_t tmpN, tmpN1;
        tmpN = tmpN1 = originN;
        MicroAPI::MaskReg PregNb32 = MicroAPI::UpdateMask<T>(tmpN);
        MicroAPI::MaskReg pregNb16 = MicroAPI::UpdateMask<T1>(tmpN1);

        for (uint16_t i = 0; i < rows; ++i) {
            MicroAPI::RegTensor<T> vregX0;
            MicroAPI::RegTensor<T> vregMax;

            MicroAPI::DataCopy<T, LoadDist::DIST_NORM>(vregX0, srcUb + i * fixSrcN);
            MicroAPI::Muls<T, T, MaskMergeMode::ZEROING>(vregX0, vregX0, scale, PregNb32);  // Muls(scale)

            if constexpr (withPseShift) {
                MicroAPI::RegTensor<PSE_SHIFT_T> vregPseShift0;
                MicroAPI::RegTensor<T> vregPseShiftCast0;
                MicroAPI::DataCopy<PSE_SHIFT_T, LoadDist::DIST_UNPACK_B16>(vregPseShift0, pseUb + i * fixSrcN);

                static constexpr MicroAPI::CastTrait castTraitfp16_fp32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                   MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
                MicroAPI::Cast<T, PSE_SHIFT_T, castTraitfp16_fp32>(vregPseShiftCast0, vregPseShift0, PregNb32);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregX0, vregX0, vregPseShiftCast0, PregNb32);
            }

            if constexpr (withAttenMask) {  // mask
                MicroAPI::RegTensor<T> vregMin;
                MicroAPI::MaskReg pregMask0;
                MicroAPI::Duplicate<float, float>(vregMin, maskMinValue);
                MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregMask0, maskUb + i * maskSizeAlignPFA);
                MicroAPI::Select<T>(vregX0, vregMin, vregX0, pregMask0);

                if constexpr (isBand) {
                    MicroAPI::MaskReg pregBandMask0;
                    MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregBandMask0, maskUb + i * maskSizeAlignPFA + rows * maskSizeAlignPFA);
                    MicroAPI::Select<T>(vregX0, vregX0, vregMin, pregBandMask0);
                }
            }

            {  // x_max
                MicroAPI::RegTensor<T> vregInMax;
                MicroAPI::ReduceMax<T, MaskMergeMode::ZEROING>(vregMax, vregX0, PregNb32);
                MicroAPI::Duplicate<T, HighLowPart::LOWEST, MaskMergeMode::ZEROING>(vregMax, vregMax, PregNb32);

                MicroAPI::DataCopy<T, LoadDist::DIST_BRC_B32>(vregInMax, inMaxUb + i * reduceN);
                MicroAPI::Max<T, MaskMergeMode::ZEROING>(vregMax, vregMax, vregInMax, PregNb32);
                MicroAPI::DataCopyUnAlign<T, PostLiteral::POST_MODE_UPDATE>(tmpMaxUb, vregMax, uregMax, 1);
            }

            {  // exp
                MicroAPI::FusedExpSub<T, T, RegLayout::ONE, MaskMergeMode::ZEROING>(vregX0, vregX0, vregMax, PregNb32);
                MicroAPI::RegTensor<T1> vregExp0;
                static constexpr MicroAPI::CastTrait castTrait0 = {RegLayout::ZERO, SatMode::NO_SAT,
                                                                   MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                MicroAPI::Cast<T1, T, castTrait0>(vregExp0, vregX0, PregNb32);
                MicroAPI::Pack<uint16_t, uint32_t, HighLowPart::LOWEST>((MicroAPI::RegTensor<uint16_t>&)vregExp0,
                                                                        (MicroAPI::RegTensor<uint32_t>&)vregExp0);
                MicroAPI::DataCopy<T1, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    expUb, vregExp0, blockStride, repeatStride, pregNb16);
            }

            {  // x_sum
                MicroAPI::RegTensor<T> vregExpSum;
                MicroAPI::ReduceSum<T, T, MaskMergeMode::ZEROING>(vregExpSum, vregX0, PregNb32);
                MicroAPI::DataCopyUnAlign<T, PostLiteral::POST_MODE_UPDATE>(tmpExpSumUb, vregExpSum, uregExpSum, 1);
            }
        }
        MicroAPI::DataCopyUnAlignPost<T, PostLiteral::POST_MODE_UPDATE>(tmpMaxUb, uregMax, 0);
        MicroAPI::DataCopyUnAlignPost<T, PostLiteral::POST_MODE_UPDATE>(tmpExpSumUb, uregExpSum, 0);

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        // update
        UPDATE_EXPSUM_MAX(expSumUb, maxUb, expMaxUb, inExpSumUb, inMaxUb, tmpExpSumUb2, tmpMaxUb2, rows);
    }
}

// 本函数N方向泛化长度为(64, 128]
template <typename T, typename T1, typename PSE_SHIFT_T, bool withPseShift = false, bool isUpdate = false,
          bool withAttenMask = false, bool isBasicBlock = false, bool isBand = false, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxFlashV2_base128_update(
    const LocalTensor<T1>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const T scale,
    const uint32_t maskSizeAlignPFA, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<PSE_SHIFT_T>& pseTensor,
    const SoftMaxShapeInfo& softmaxShapeInfo = {})
{
    const uint32_t originN = softmaxShapeInfo.oriSrcK;  // N轴实际数据长度，为SInnerLoop切块的 S/act_seq_len
    const uint32_t fixSrcN = 128;
    const uint32_t floatRepSize = 64;         // 寄存器一次可以加载64个元素（fp32类型，对应256B）
    const uint32_t reduceN = 32 / sizeof(T);  // m每行的结果存为8个重复元素（32B对齐）
    const uint16_t rows = static_cast<uint16_t>(softmaxShapeInfo.srcM);  // 总行数

    __VEC_SCOPE__
    {
        __ubuf__ T1* expUb = (__ubuf__ T1*)dstTensor.GetPhyAddr();
        __ubuf__ float* expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
        __ubuf__ float* maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
        __ubuf__ float* srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
        __ubuf__ float* expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
        __ubuf__ float* inExpSumUb = (__ubuf__ T*)inExpSumTensor.GetPhyAddr();
        __ubuf__ float* inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
        __ubuf__ float* tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
        __ubuf__ float* tmpMaxUb = tmpExpSumUb + 16; // 16 x sum + 16 x max
        __ubuf__ float* tmpExpSumUb2 = tmpExpSumUb;
        __ubuf__ float* tmpMaxUb2 = tmpMaxUb;
        __ubuf__ uint8_t* maskUb = (__ubuf__ uint8_t*)maskTensor.GetPhyAddr();
        __ubuf__ uint8_t* maskBandUb;
        if constexpr (isBand) {
            maskBandUb = (__ubuf__ uint8_t*)maskTensor[rows * maskSizeAlignPFA].GetPhyAddr();
        }
        __ubuf__ PSE_SHIFT_T* pseUb = (__ubuf__ PSE_SHIFT_T*)pseTensor.GetPhyAddr();

        uint16_t blockStride = 16 + 1;  // 加1是为了解写写ub bank冲突
        uint16_t repeatStride = 1;      //  每次nd2nz写完之后ub往后偏移1个block

        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T1, MaskPattern::ALL>();
        MicroAPI::RegTensor<T> vregMin;
        MicroAPI::RegTensor<T> vregMin2;
        MicroAPI::UnalignReg uregMax;
        MicroAPI::UnalignReg uregExpSum;

        MicroAPI::Duplicate<float, float>(vregMin, maskMinValue);
        MicroAPI::Duplicate<float, float>(vregMin2, maskMinValue2);
        uint32_t tmpN = originN;
        MicroAPI::MaskReg pregTail0 = MicroAPI::UpdateMask<T>(tmpN);
        MicroAPI::MaskReg pregTail1 = MicroAPI::UpdateMask<T>(tmpN);

        for (uint16_t i = 0; i < rows; ++i) {
            MicroAPI::RegTensor<T> vregX0;
            MicroAPI::RegTensor<T> vregX1;
            MicroAPI::RegTensor<T> vregMax;

            MicroAPI::DataCopy<T, LoadDist::DIST_NORM>(vregX0, srcUb + i * fixSrcN);
            MicroAPI::DataCopy<T, LoadDist::DIST_NORM>(vregX1, srcUb + 64 + i * fixSrcN);

            MicroAPI::Muls<T, T, MaskMergeMode::ZEROING>(vregX0, vregX0, scale, pregAll);  // Muls(scale)
            MicroAPI::Muls<T, T, MaskMergeMode::ZEROING>(vregX1, vregX1, scale, pregAll);

            if constexpr (withPseShift) {
                MicroAPI::RegTensor<PSE_SHIFT_T> vregPseShift0;
                MicroAPI::RegTensor<PSE_SHIFT_T> vregPseShift1;
                MicroAPI::RegTensor<T> vregPseShiftCast0;
                MicroAPI::RegTensor<T> vregPseShiftCast1;
                MicroAPI::DataCopy<PSE_SHIFT_T, LoadDist::DIST_UNPACK_B16>(vregPseShift0, pseUb + i * fixSrcN);
                MicroAPI::DataCopy<PSE_SHIFT_T, LoadDist::DIST_UNPACK_B16>(vregPseShift1, pseUb + 64 + i * fixSrcN);

                static constexpr MicroAPI::CastTrait castTraitfp16_fp32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                   MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
                MicroAPI::Cast<T, PSE_SHIFT_T, castTraitfp16_fp32>(vregPseShiftCast0, vregPseShift0, pregAll);
                MicroAPI::Cast<T, PSE_SHIFT_T, castTraitfp16_fp32>(vregPseShiftCast1, vregPseShift1, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregX0, vregX0, vregPseShiftCast0, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregX1, vregX1, vregPseShiftCast1, pregAll);
            }

            if constexpr (withAttenMask) { // mask
                MicroAPI::MaskReg pregMask0;
                MicroAPI::MaskReg pregMask1;
                MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregMask0, maskUb + i * maskSizeAlignPFA);
                MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregMask1, maskUb + 64 + i * maskSizeAlignPFA);
                MicroAPI::Select<T>(vregX0, vregMin, vregX0, pregMask0);
                MicroAPI::Select<T>(vregX1, vregMin, vregX1, pregMask1);

                if constexpr (isBand) {
                    MicroAPI::MaskReg pregBandMask0;
                    MicroAPI::MaskReg pregBandMask1;
                    MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregBandMask0, maskUb + i * maskSizeAlignPFA + rows * maskSizeAlignPFA);
                    MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregBandMask1, maskUb + 64 + i * maskSizeAlignPFA + rows * maskSizeAlignPFA);
                    MicroAPI::Select<T>(vregX0, vregX0, vregMin, pregBandMask0);
                    MicroAPI::Select<T>(vregX1, vregX1, vregMin, pregBandMask1);
                }
            }

            // select之后，当作256个数处理，不再有尾块的概念
            MicroAPI::Select<T>(vregX0, vregX0, vregMin2, pregTail0);
            MicroAPI::Select<T>(vregX1, vregX1, vregMin2, pregTail1);

            { // x_max
                MicroAPI::RegTensor<T> vregInMax;

                MicroAPI::Max<T, MaskMergeMode::ZEROING>(vregMax, vregX0, vregX1, pregAll);
                MicroAPI::ReduceMax<T, MaskMergeMode::ZEROING>(vregMax, vregMax, pregAll);
                MicroAPI::Duplicate<T, HighLowPart::LOWEST, MaskMergeMode::ZEROING>(vregMax, vregMax, pregAll);

                MicroAPI::DataCopy<T, LoadDist::DIST_BRC_B32>(vregInMax, inMaxUb + i * reduceN);
                MicroAPI::Max<T, MaskMergeMode::ZEROING>(vregMax, vregMax, vregInMax, pregAll);
                MicroAPI::DataCopyUnAlign<T, PostLiteral::POST_MODE_UPDATE>(tmpMaxUb, vregMax, uregMax, 1);
            }

            { // exp
                MicroAPI::FusedExpSub<T, T, RegLayout::ONE, MaskMergeMode::ZEROING>(vregX0, vregX0, vregMax, pregAll);
                MicroAPI::FusedExpSub<T, T, RegLayout::ONE, MaskMergeMode::ZEROING>(vregX1, vregX1, vregMax, pregAll);

                MicroAPI::RegTensor<T> vregExpEven0;
                MicroAPI::RegTensor<T> vregExpOdd0;

                MicroAPI::DeInterleave<T>(vregExpEven0, vregExpOdd0, vregX0, vregX1);
                static constexpr MicroAPI::CastTrait castTrait0 = {RegLayout::ZERO, SatMode::NO_SAT,
                                                                   MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                static constexpr MicroAPI::CastTrait castTrait1 = {RegLayout::ONE, SatMode::NO_SAT,
                                                                   MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

                MicroAPI::Cast<T1, T, castTrait0>((MicroAPI::RegTensor<T1>&)vregExpEven0, vregExpEven0, pregAll);
                MicroAPI::Cast<T1, T, castTrait1>((MicroAPI::RegTensor<T1>&)vregExpOdd0, vregExpOdd0, pregAll);

                MicroAPI::Or<uint16_t, MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vregExpEven0,
                                                               (MicroAPI::RegTensor<uint16_t>&)vregExpEven0,
                                                               (MicroAPI::RegTensor<uint16_t>&)vregExpOdd0, pregAll);

                MicroAPI::DataCopy<T1, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    expUb, (MicroAPI::RegTensor<T1>&)vregExpEven0, blockStride, repeatStride, pregAll);
            }

            { // x_sum
                MicroAPI::RegTensor<T> vregExpSum;
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregExpSum, vregX0, vregX1, pregAll);
                MicroAPI::ReduceSum<T, T, MaskMergeMode::ZEROING>(vregExpSum, vregExpSum, pregAll);
                MicroAPI::DataCopyUnAlign<T, PostLiteral::POST_MODE_UPDATE>(tmpExpSumUb, vregExpSum, uregExpSum, 1);
            }
        }
        MicroAPI::DataCopyUnAlignPost<T, PostLiteral::POST_MODE_UPDATE>(tmpMaxUb, uregMax, 0);
        MicroAPI::DataCopyUnAlignPost<T, PostLiteral::POST_MODE_UPDATE>(tmpExpSumUb, uregExpSum, 0);

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        UPDATE_EXPSUM_MAX(expSumUb, maxUb, expMaxUb, inExpSumUb, inMaxUb, tmpExpSumUb2, tmpMaxUb2, rows);
    }
}

// 本函数N方向泛化长度为(128, 256]
template <typename T, typename T1, typename PSE_SHIFT_T, bool withPseShift = false, bool isUpdate = false,
          bool withAttenMask = false, bool isBasicBlock = false, bool isBand = false, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxFlashV2_base256_update(
    const LocalTensor<T1>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const T scale,
    const uint32_t maskSizeAlignPFA, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<PSE_SHIFT_T>& pseTensor,
    const SoftMaxShapeInfo& softmaxShapeInfo = {})
{
    const uint32_t originN = softmaxShapeInfo.oriSrcK;  // N轴实际数据长度，为SInnerLoop切块的 S/act_seq_len
    const uint32_t fixSrcN = 256;
    const uint32_t floatRepSize = 64;         // 寄存器一次可以加载64个元素（fp32类型，对应256B）
    const uint32_t reduceN = 32 / sizeof(T);  // m每行的结果存为8个重复元素（32B对齐）
    const uint16_t rows = static_cast<uint16_t>(softmaxShapeInfo.srcM);  // 总行数

    __VEC_SCOPE__
    {
        __ubuf__ float* expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
        __ubuf__ float* maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
        __ubuf__ float* srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
        __ubuf__ float* expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
        __ubuf__ float* inExpSumUb = (__ubuf__ T*)inExpSumTensor.GetPhyAddr();
        __ubuf__ float* inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
        __ubuf__ float* tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
        __ubuf__ float* tmpMaxUb = tmpExpSumUb + 16; // 16 x sum + 16 x max
        __ubuf__ float* tmpExpSumUb2 = tmpExpSumUb;
        __ubuf__ float* tmpMaxUb2 = tmpMaxUb;
        __ubuf__ T1* expUb = (__ubuf__ T1*)dstTensor.GetPhyAddr();
        __ubuf__ T1* expUb1 = expUb + (16 + 1) * 128; // 第二大块NZ的偏移是(16 + 1) * 128

        __ubuf__ uint8_t* maskUb = (__ubuf__ uint8_t*)maskTensor.GetPhyAddr();
        __ubuf__ uint8_t* maskBandUb;
        if constexpr (isBand) {
            maskBandUb = (__ubuf__ uint8_t*)maskTensor[rows * maskSizeAlignPFA].GetPhyAddr();
        }
        __ubuf__ PSE_SHIFT_T* pseUb = (__ubuf__ PSE_SHIFT_T*)pseTensor.GetPhyAddr();

        uint16_t blockStride = 16 + 1;  // 加1是为了解写写ub bank冲突
        uint16_t repeatStride = 1;      //  每次nd2nz写完之后ub往后偏移1个block

        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T1, MaskPattern::ALL>();
        MicroAPI::RegTensor<T> vregMin;
        MicroAPI::RegTensor<T> vregMin2;
        MicroAPI::UnalignReg uregMax;
        MicroAPI::UnalignReg uregExpSum;

        MicroAPI::Duplicate<float, float>(vregMin, maskMinValue);
        MicroAPI::Duplicate<float, float>(vregMin2, maskMinValue2);

        uint32_t tmpN = originN;
        MicroAPI::MaskReg pregTail0 = MicroAPI::UpdateMask<T>(tmpN);
        MicroAPI::MaskReg pregTail1 = MicroAPI::UpdateMask<T>(tmpN);
        MicroAPI::MaskReg pregTail2 = MicroAPI::UpdateMask<T>(tmpN);
        MicroAPI::MaskReg pregTail3 = MicroAPI::UpdateMask<T>(tmpN);

        for (uint16_t i = 0; i < rows; ++i) {
            MicroAPI::RegTensor<T> vregX0;
            MicroAPI::RegTensor<T> vregX1;
            MicroAPI::RegTensor<T> vregX2;
            MicroAPI::RegTensor<T> vregX3;
            MicroAPI::RegTensor<T> vregMax;

            MicroAPI::DataCopy<T, LoadDist::DIST_NORM>(vregX0, srcUb + i * fixSrcN);
            MicroAPI::DataCopy<T, LoadDist::DIST_NORM>(vregX1, srcUb + 64 + i * fixSrcN);
            MicroAPI::DataCopy<T, LoadDist::DIST_NORM>(vregX2, srcUb + 128 + i * fixSrcN);
            MicroAPI::DataCopy<T, LoadDist::DIST_NORM>(vregX3, srcUb + 192 + i * fixSrcN);

            MicroAPI::Muls<T, T, MaskMergeMode::ZEROING>(vregX0, vregX0, scale, pregAll);  // Muls(scale)
            MicroAPI::Muls<T, T, MaskMergeMode::ZEROING>(vregX1, vregX1, scale, pregAll);
            MicroAPI::Muls<T, T, MaskMergeMode::ZEROING>(vregX2, vregX2, scale, pregAll);
            MicroAPI::Muls<T, T, MaskMergeMode::ZEROING>(vregX3, vregX3, scale, pregAll);

            if constexpr (withPseShift) {
                MicroAPI::RegTensor<PSE_SHIFT_T> vregPseShift0;
                MicroAPI::RegTensor<PSE_SHIFT_T> vregPseShift1;
                MicroAPI::RegTensor<PSE_SHIFT_T> vregPseShift2;
                MicroAPI::RegTensor<PSE_SHIFT_T> vregPseShift3;
                MicroAPI::RegTensor<T> vregPseShiftCast0;
                MicroAPI::RegTensor<T> vregPseShiftCast1;
                MicroAPI::RegTensor<T> vregPseShiftCast2;
                MicroAPI::RegTensor<T> vregPseShiftCast3;
                MicroAPI::DataCopy<PSE_SHIFT_T, LoadDist::DIST_UNPACK_B16>(vregPseShift0, pseUb + i * fixSrcN);
                MicroAPI::DataCopy<PSE_SHIFT_T, LoadDist::DIST_UNPACK_B16>(vregPseShift1, pseUb + 64 + i * fixSrcN);
                MicroAPI::DataCopy<PSE_SHIFT_T, LoadDist::DIST_UNPACK_B16>(vregPseShift2, pseUb + 128 + i * fixSrcN);
                MicroAPI::DataCopy<PSE_SHIFT_T, LoadDist::DIST_UNPACK_B16>(vregPseShift3, pseUb + 192 + i * fixSrcN);

                static constexpr MicroAPI::CastTrait castTraitfp16_fp32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                   MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
                MicroAPI::Cast<T, PSE_SHIFT_T, castTraitfp16_fp32>(vregPseShiftCast0, vregPseShift0, pregAll);
                MicroAPI::Cast<T, PSE_SHIFT_T, castTraitfp16_fp32>(vregPseShiftCast1, vregPseShift1, pregAll);
                MicroAPI::Cast<T, PSE_SHIFT_T, castTraitfp16_fp32>(vregPseShiftCast2, vregPseShift2, pregAll);
                MicroAPI::Cast<T, PSE_SHIFT_T, castTraitfp16_fp32>(vregPseShiftCast3, vregPseShift3, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregX0, vregX0, vregPseShiftCast0, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregX1, vregX1, vregPseShiftCast1, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregX2, vregX2, vregPseShiftCast2, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregX3, vregX3, vregPseShiftCast3, pregAll);
            }

            if constexpr (withAttenMask) { // mask
                MicroAPI::MaskReg pregMask0;
                MicroAPI::MaskReg pregMask1;
                MicroAPI::MaskReg pregMask2;
                MicroAPI::MaskReg pregMask3;
                MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregMask0, maskUb + i * maskSizeAlignPFA);
                MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregMask1, maskUb + 64 + i * maskSizeAlignPFA);
                MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregMask2, maskUb + 128 + i * maskSizeAlignPFA);
                MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregMask3, maskUb + 192 + i * maskSizeAlignPFA);
                MicroAPI::Select<T>(vregX0, vregMin, vregX0, pregMask0);
                MicroAPI::Select<T>(vregX1, vregMin, vregX1, pregMask1);
                MicroAPI::Select<T>(vregX2, vregMin, vregX2, pregMask2);
                MicroAPI::Select<T>(vregX3, vregMin, vregX3, pregMask3);

                if constexpr (isBand) {
                    MicroAPI::MaskReg pregBandMask0;
                    MicroAPI::MaskReg pregBandMask1;
                    MicroAPI::MaskReg pregBandMask2;
                    MicroAPI::MaskReg pregBandMask3;
                    MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregBandMask0, maskUb + i * maskSizeAlignPFA + rows * maskSizeAlignPFA);
                    MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregBandMask1, maskUb + 64 + i * maskSizeAlignPFA + rows * maskSizeAlignPFA);
                    MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregBandMask2, maskUb + 128 + i * maskSizeAlignPFA + rows * maskSizeAlignPFA);
                    MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregBandMask3, maskUb + 192 + i * maskSizeAlignPFA + rows * maskSizeAlignPFA);
                    MicroAPI::Select<T>(vregX0, vregX0, vregMin, pregBandMask0);
                    MicroAPI::Select<T>(vregX1, vregX1, vregMin, pregBandMask1);
                    MicroAPI::Select<T>(vregX2, vregX2, vregMin, pregBandMask2);
                    MicroAPI::Select<T>(vregX3, vregX3, vregMin, pregBandMask3);
                }
            }

            // select之后，当作256个数处理，不再有尾块的概念
            MicroAPI::Select<T>(vregX0, vregX0, vregMin2, pregTail0);
            MicroAPI::Select<T>(vregX1, vregX1, vregMin2, pregTail1);
            MicroAPI::Select<T>(vregX2, vregX2, vregMin2, pregTail2);
            MicroAPI::Select<T>(vregX3, vregX3, vregMin2, pregTail3);

            { // x_max
                MicroAPI::RegTensor<T> vregMaxTmp0;
                MicroAPI::RegTensor<T> vregMaxTmp1;
                MicroAPI::RegTensor<T> vregInMax;

                MicroAPI::Max<T, MaskMergeMode::ZEROING>(vregMaxTmp0, vregX0, vregX1, pregAll);
                MicroAPI::Max<T, MaskMergeMode::ZEROING>(vregMaxTmp1, vregX2, vregX3, pregAll);
                MicroAPI::Max<T, MaskMergeMode::ZEROING>(vregMax, vregMaxTmp0, vregMaxTmp1, pregAll);
                MicroAPI::ReduceMax<T, MaskMergeMode::ZEROING>(vregMax, vregMax, pregAll);
                MicroAPI::Duplicate<T, HighLowPart::LOWEST, MaskMergeMode::ZEROING>(vregMax, vregMax, pregAll);

                MicroAPI::DataCopy<T, LoadDist::DIST_BRC_B32>(vregInMax, inMaxUb + i * reduceN);
                MicroAPI::Max<T, MaskMergeMode::ZEROING>(vregMax, vregMax, vregInMax, pregAll);
                MicroAPI::DataCopyUnAlign<T, PostLiteral::POST_MODE_UPDATE>(tmpMaxUb, vregMax, uregMax, 1);
            }

            {  // exp
                MicroAPI::FusedExpSub<T, T, RegLayout::ONE, MaskMergeMode::ZEROING>(vregX0, vregX0, vregMax, pregAll);
                MicroAPI::FusedExpSub<T, T, RegLayout::ONE, MaskMergeMode::ZEROING>(vregX1, vregX1, vregMax, pregAll);
                MicroAPI::FusedExpSub<T, T, RegLayout::ONE, MaskMergeMode::ZEROING>(vregX2, vregX2, vregMax, pregAll);
                MicroAPI::FusedExpSub<T, T, RegLayout::ONE, MaskMergeMode::ZEROING>(vregX3, vregX3, vregMax, pregAll);

                MicroAPI::RegTensor<T> vregExpEven0;
                MicroAPI::RegTensor<T> vregExpOdd0;
                MicroAPI::RegTensor<T> vregExpEven1;
                MicroAPI::RegTensor<T> vregExpOdd1;

                MicroAPI::DeInterleave<T>(vregExpEven0, vregExpOdd0, vregX0, vregX1);
                MicroAPI::DeInterleave<T>(vregExpEven1, vregExpOdd1, vregX2, vregX3);
                static constexpr MicroAPI::CastTrait castTrait0 = {RegLayout::ZERO, SatMode::NO_SAT,
                                                                   MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                static constexpr MicroAPI::CastTrait castTrait1 = {RegLayout::ONE, SatMode::NO_SAT,
                                                                   MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

                MicroAPI::Cast<T1, T, castTrait0>((MicroAPI::RegTensor<T1>&)vregExpEven0, vregExpEven0, pregAll);
                MicroAPI::Cast<T1, T, castTrait1>((MicroAPI::RegTensor<T1>&)vregExpOdd0, vregExpOdd0, pregAll);
                MicroAPI::Cast<T1, T, castTrait0>((MicroAPI::RegTensor<T1>&)vregExpEven1, vregExpEven1, pregAll);
                MicroAPI::Cast<T1, T, castTrait1>((MicroAPI::RegTensor<T1>&)vregExpOdd1, vregExpOdd1, pregAll);

                MicroAPI::Or<uint16_t, MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vregExpEven0,
                                                               (MicroAPI::RegTensor<uint16_t>&)vregExpEven0,
                                                               (MicroAPI::RegTensor<uint16_t>&)vregExpOdd0, pregAll);
                MicroAPI::Or<uint16_t, MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vregExpEven1,
                                                               (MicroAPI::RegTensor<uint16_t>&)vregExpEven1,
                                                               (MicroAPI::RegTensor<uint16_t>&)vregExpOdd1, pregAll);

                MicroAPI::DataCopy<T1, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    expUb, (MicroAPI::RegTensor<T1>&)vregExpEven0, blockStride, repeatStride, pregAll);
                MicroAPI::DataCopy<T1, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    expUb1, (MicroAPI::RegTensor<T1>&)vregExpEven1, blockStride, repeatStride, pregAll);
            }

            { // x_sum
                MicroAPI::RegTensor<T> vregExpSum0;
                MicroAPI::RegTensor<T> vregExpSum1;
                MicroAPI::RegTensor<T> vregExpSum;

                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregExpSum0, vregX0, vregX1, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregExpSum1, vregX2, vregX3, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregExpSum, vregExpSum0, vregExpSum1, pregAll);
                MicroAPI::ReduceSum<T, T, MaskMergeMode::ZEROING>(vregExpSum, vregExpSum, pregAll);
                MicroAPI::DataCopyUnAlign<T, PostLiteral::POST_MODE_UPDATE>(tmpExpSumUb, vregExpSum, uregExpSum, 1);
            }
        }
        MicroAPI::DataCopyUnAlignPost<T, PostLiteral::POST_MODE_UPDATE>(tmpMaxUb, uregMax, 0);
        MicroAPI::DataCopyUnAlignPost<T, PostLiteral::POST_MODE_UPDATE>(tmpExpSumUb, uregExpSum, 0);

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        UPDATE_EXPSUM_MAX(expSumUb, maxUb, expMaxUb, inExpSumUb, inMaxUb, tmpExpSumUb2, tmpMaxUb2, rows);
    }
}

// 本函数N方向泛化长度为(256, 512]
template <typename T, typename T1, typename PSE_SHIFT_T, bool withPseShift = false, bool isUpdate = false,
          bool withAttenMask = false, bool isBasicBlock = false, bool isBand = false, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxFlashV2_base512_update(
    const LocalTensor<T1>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const T scale,
    const uint32_t maskSizeAlignPFA, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<PSE_SHIFT_T>& pseTensor,
    const uint32_t& profileG, const SoftMaxShapeInfo& softmaxShapeInfo = {})
{
    const uint32_t originN = softmaxShapeInfo.oriSrcK;  // N轴实际数据长度，为SInnerLoop切块的 S/act_seq_len
    const uint32_t fixSrcN = 512;
    const uint32_t floatRepSize = 64;         // 寄存器一次可以加载64个元素（fp32类型，对应256B）
    const uint32_t reduceN = 32 / sizeof(T);  // m每行的结果存为8个重复元素（32B对齐）
    const uint16_t rows = static_cast<uint16_t>(softmaxShapeInfo.srcM);  // 总行数
    const uint32_t fixSrcG = profileG;

    __VEC_SCOPE__
    {
        __ubuf__ float* expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
        __ubuf__ float* maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
        __ubuf__ float* srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
        __ubuf__ float* expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
        __ubuf__ float* inExpSumUb = (__ubuf__ T*)inExpSumTensor.GetPhyAddr();
        __ubuf__ float* inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
        __ubuf__ float* tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
        __ubuf__ float* tmpMaxUb = tmpExpSumUb + 16; // 16 x sum + 16 x max
        __ubuf__ float* tmpExpSumUb2 = tmpExpSumUb;
        __ubuf__ float* tmpMaxUb2 = tmpMaxUb;
        __ubuf__ uint8_t* maskUb = (__ubuf__ uint8_t*)maskTensor.GetPhyAddr();
        __ubuf__ uint8_t* maskBandUb;
        if constexpr (isBand) {
            maskBandUb = (__ubuf__ uint8_t*)maskTensor[rows * maskSizeAlignPFA].GetPhyAddr();
        }
        __ubuf__ PSE_SHIFT_T* pseUb = (__ubuf__ PSE_SHIFT_T*)pseTensor.GetPhyAddr();

        __ubuf__ T1* expUb = (__ubuf__ T1*)dstTensor.GetPhyAddr();
        __ubuf__ T1* expUb1 = expUb + (fixSrcG + 1) * 128;     // 第二大块NZ的偏移是(profileG + 1) * 128
        __ubuf__ T1* expUb2 = expUb + 2 * (fixSrcG + 1) * 128; // 第三大块NZ的偏移是2 * (profileG + 1) * 128
        __ubuf__ T1* expUb3 = expUb + 3 * (fixSrcG + 1) * 128; // 第四大块NZ的偏移是3 * (profileG + 1) * 128

        uint16_t blockStride = fixSrcG + 1;  // 加1是为了解写写ub bank冲突
        uint16_t repeatStride = 1;      //  每次nd2nz写完之后ub往后偏移1个block

        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T1, MaskPattern::ALL>();
        MicroAPI::RegTensor<T> vregMin;
        MicroAPI::RegTensor<T> vregMin2;
        MicroAPI::UnalignReg uregMax;
        MicroAPI::UnalignReg uregExpSum;

        MicroAPI::Duplicate<float, float>(vregMin, maskMinValue);
        MicroAPI::Duplicate<float, float>(vregMin2, maskMinValue2);
        uint32_t tmpN = originN;
        MicroAPI::MaskReg pregTail0 = MicroAPI::UpdateMask<T>(tmpN);
        MicroAPI::MaskReg pregTail1 = MicroAPI::UpdateMask<T>(tmpN);
        MicroAPI::MaskReg pregTail2 = MicroAPI::UpdateMask<T>(tmpN);
        MicroAPI::MaskReg pregTail3 = MicroAPI::UpdateMask<T>(tmpN);
        MicroAPI::MaskReg pregTail4 = MicroAPI::UpdateMask<T>(tmpN);
        MicroAPI::MaskReg pregTail5 = MicroAPI::UpdateMask<T>(tmpN);
        MicroAPI::MaskReg pregTail6 = MicroAPI::UpdateMask<T>(tmpN);
        MicroAPI::MaskReg pregTail7 = MicroAPI::UpdateMask<T>(tmpN);

        // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)
        for (uint16_t i = 0; i < rows; ++i) {
            MicroAPI::RegTensor<T> vregX0;
            MicroAPI::RegTensor<T> vregX1;
            MicroAPI::RegTensor<T> vregX2;
            MicroAPI::RegTensor<T> vregX3;
            MicroAPI::RegTensor<T> vregX4;
            MicroAPI::RegTensor<T> vregX5;
            MicroAPI::RegTensor<T> vregX6;
            MicroAPI::RegTensor<T> vregX7;
            MicroAPI::RegTensor<T> vregMax;

            MicroAPI::DataCopy<T, LoadDist::DIST_NORM>(vregX0, srcUb + i * fixSrcN);
            MicroAPI::DataCopy<T, LoadDist::DIST_NORM>(vregX1, srcUb + 64 + i * fixSrcN);
            MicroAPI::DataCopy<T, LoadDist::DIST_NORM>(vregX2, srcUb + 128 + i * fixSrcN);
            MicroAPI::DataCopy<T, LoadDist::DIST_NORM>(vregX3, srcUb + 192 + i * fixSrcN);
            MicroAPI::DataCopy<T, LoadDist::DIST_NORM>(vregX4, srcUb + 256 + i * fixSrcN);
            MicroAPI::DataCopy<T, LoadDist::DIST_NORM>(vregX5, srcUb + 320 + i * fixSrcN);
            MicroAPI::DataCopy<T, LoadDist::DIST_NORM>(vregX6, srcUb + 384 + i * fixSrcN);
            MicroAPI::DataCopy<T, LoadDist::DIST_NORM>(vregX7, srcUb + 448 + i * fixSrcN);

            MicroAPI::Muls<T, T, MaskMergeMode::ZEROING>(vregX0, vregX0, scale, pregAll);  // Muls(scale)
            MicroAPI::Muls<T, T, MaskMergeMode::ZEROING>(vregX1, vregX1, scale, pregAll);
            MicroAPI::Muls<T, T, MaskMergeMode::ZEROING>(vregX2, vregX2, scale, pregAll);
            MicroAPI::Muls<T, T, MaskMergeMode::ZEROING>(vregX3, vregX3, scale, pregAll);
            MicroAPI::Muls<T, T, MaskMergeMode::ZEROING>(vregX4, vregX4, scale, pregAll);
            MicroAPI::Muls<T, T, MaskMergeMode::ZEROING>(vregX5, vregX5, scale, pregAll);
            MicroAPI::Muls<T, T, MaskMergeMode::ZEROING>(vregX6, vregX6, scale, pregAll);
            MicroAPI::Muls<T, T, MaskMergeMode::ZEROING>(vregX7, vregX7, scale, pregAll);

            if constexpr (withPseShift) {
                MicroAPI::RegTensor<PSE_SHIFT_T> vregPseShift0;
                MicroAPI::RegTensor<PSE_SHIFT_T> vregPseShift1;
                MicroAPI::RegTensor<PSE_SHIFT_T> vregPseShift2;
                MicroAPI::RegTensor<PSE_SHIFT_T> vregPseShift3;
                MicroAPI::RegTensor<PSE_SHIFT_T> vregPseShift4;
                MicroAPI::RegTensor<PSE_SHIFT_T> vregPseShift5;
                MicroAPI::RegTensor<PSE_SHIFT_T> vregPseShift6;
                MicroAPI::RegTensor<PSE_SHIFT_T> vregPseShift7;
                MicroAPI::RegTensor<T> vregPseShiftCast0;
                MicroAPI::RegTensor<T> vregPseShiftCast1;
                MicroAPI::RegTensor<T> vregPseShiftCast2;
                MicroAPI::RegTensor<T> vregPseShiftCast3;
                MicroAPI::RegTensor<T> vregPseShiftCast4;
                MicroAPI::RegTensor<T> vregPseShiftCast5;
                MicroAPI::RegTensor<T> vregPseShiftCast6;
                MicroAPI::RegTensor<T> vregPseShiftCast7;
                MicroAPI::DataCopy<PSE_SHIFT_T, LoadDist::DIST_UNPACK_B16>(vregPseShift0, pseUb + i * fixSrcN);
                MicroAPI::DataCopy<PSE_SHIFT_T, LoadDist::DIST_UNPACK_B16>(vregPseShift1, pseUb + 64 + i * fixSrcN);
                MicroAPI::DataCopy<PSE_SHIFT_T, LoadDist::DIST_UNPACK_B16>(vregPseShift2, pseUb + 128 + i * fixSrcN);
                MicroAPI::DataCopy<PSE_SHIFT_T, LoadDist::DIST_UNPACK_B16>(vregPseShift3, pseUb + 192 + i * fixSrcN);
                MicroAPI::DataCopy<PSE_SHIFT_T, LoadDist::DIST_UNPACK_B16>(vregPseShift4, pseUb + 256 + i * fixSrcN);
                MicroAPI::DataCopy<PSE_SHIFT_T, LoadDist::DIST_UNPACK_B16>(vregPseShift5, pseUb + 320 + i * fixSrcN);
                MicroAPI::DataCopy<PSE_SHIFT_T, LoadDist::DIST_UNPACK_B16>(vregPseShift6, pseUb + 384 + i * fixSrcN);
                MicroAPI::DataCopy<PSE_SHIFT_T, LoadDist::DIST_UNPACK_B16>(vregPseShift7, pseUb + 448 + i * fixSrcN);

                static constexpr MicroAPI::CastTrait castTraitfp16_fp32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                   MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
                MicroAPI::Cast<T, PSE_SHIFT_T, castTraitfp16_fp32>(vregPseShiftCast0, vregPseShift0, pregAll);
                MicroAPI::Cast<T, PSE_SHIFT_T, castTraitfp16_fp32>(vregPseShiftCast1, vregPseShift1, pregAll);
                MicroAPI::Cast<T, PSE_SHIFT_T, castTraitfp16_fp32>(vregPseShiftCast2, vregPseShift2, pregAll);
                MicroAPI::Cast<T, PSE_SHIFT_T, castTraitfp16_fp32>(vregPseShiftCast3, vregPseShift3, pregAll);
                MicroAPI::Cast<T, PSE_SHIFT_T, castTraitfp16_fp32>(vregPseShiftCast4, vregPseShift4, pregAll);
                MicroAPI::Cast<T, PSE_SHIFT_T, castTraitfp16_fp32>(vregPseShiftCast5, vregPseShift5, pregAll);
                MicroAPI::Cast<T, PSE_SHIFT_T, castTraitfp16_fp32>(vregPseShiftCast6, vregPseShift6, pregAll);
                MicroAPI::Cast<T, PSE_SHIFT_T, castTraitfp16_fp32>(vregPseShiftCast7, vregPseShift7, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregX0, vregX0, vregPseShiftCast0, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregX1, vregX1, vregPseShiftCast1, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregX2, vregX2, vregPseShiftCast2, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregX3, vregX3, vregPseShiftCast3, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregX4, vregX4, vregPseShiftCast4, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregX5, vregX5, vregPseShiftCast5, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregX6, vregX6, vregPseShiftCast6, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregX7, vregX7, vregPseShiftCast7, pregAll);
            }

            if constexpr (withAttenMask) { // mask
                MicroAPI::MaskReg pregMask0;
                MicroAPI::MaskReg pregMask1;
                MicroAPI::MaskReg pregMask2;
                MicroAPI::MaskReg pregMask3;
                MicroAPI::MaskReg pregMask4;
                MicroAPI::MaskReg pregMask5;
                MicroAPI::MaskReg pregMask6;
                MicroAPI::MaskReg pregMask7;
                MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregMask0, maskUb + i * maskSizeAlignPFA);
                MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregMask1, maskUb + 64 + i * maskSizeAlignPFA);
                MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregMask2, maskUb + 128 + i * maskSizeAlignPFA);
                MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregMask3, maskUb + 192 + i * maskSizeAlignPFA);
                MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregMask4, maskUb + 256 + i * maskSizeAlignPFA);
                MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregMask5, maskUb + 320 + i * maskSizeAlignPFA);
                MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregMask6, maskUb + 384 + i * maskSizeAlignPFA);
                MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregMask7, maskUb + 448 + i * maskSizeAlignPFA);
                MicroAPI::Select<T>(vregX0, vregMin, vregX0, pregMask0);
                MicroAPI::Select<T>(vregX1, vregMin, vregX1, pregMask1);
                MicroAPI::Select<T>(vregX2, vregMin, vregX2, pregMask2);
                MicroAPI::Select<T>(vregX3, vregMin, vregX3, pregMask3);
                MicroAPI::Select<T>(vregX4, vregMin, vregX4, pregMask4);
                MicroAPI::Select<T>(vregX5, vregMin, vregX5, pregMask5);
                MicroAPI::Select<T>(vregX6, vregMin, vregX6, pregMask6);
                MicroAPI::Select<T>(vregX7, vregMin, vregX7, pregMask7);

                if constexpr (isBand) {
                    MicroAPI::MaskReg pregBandMask0;
                    MicroAPI::MaskReg pregBandMask1;
                    MicroAPI::MaskReg pregBandMask2;
                    MicroAPI::MaskReg pregBandMask3;
                    MicroAPI::MaskReg pregBandMask4;
                    MicroAPI::MaskReg pregBandMask5;
                    MicroAPI::MaskReg pregBandMask6;
                    MicroAPI::MaskReg pregBandMask7;
                    MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregBandMask0, maskUb + i * maskSizeAlignPFA + rows * maskSizeAlignPFA);
                    MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregBandMask1, maskUb + 64 + i * maskSizeAlignPFA + rows * maskSizeAlignPFA);
                    MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregBandMask2, maskUb + 128 + i * maskSizeAlignPFA + rows * maskSizeAlignPFA);
                    MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregBandMask3, maskUb + 192 + i * maskSizeAlignPFA + rows * maskSizeAlignPFA);
                    MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregBandMask4, maskUb + 256 + i * maskSizeAlignPFA + rows * maskSizeAlignPFA);
                    MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregBandMask5, maskUb + 320 + i * maskSizeAlignPFA + rows * maskSizeAlignPFA);
                    MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregBandMask6, maskUb + 384 + i * maskSizeAlignPFA + rows * maskSizeAlignPFA);
                    MicroAPI::DataCopy<uint8_t, MaskDist::DIST_DS>(pregBandMask7, maskUb + 448 + i * maskSizeAlignPFA + rows * maskSizeAlignPFA);
                    MicroAPI::Select<T>(vregX0, vregX0, vregMin, pregBandMask0);
                    MicroAPI::Select<T>(vregX1, vregX1, vregMin, pregBandMask1);
                    MicroAPI::Select<T>(vregX2, vregX2, vregMin, pregBandMask2);
                    MicroAPI::Select<T>(vregX3, vregX3, vregMin, pregBandMask3);
                    MicroAPI::Select<T>(vregX4, vregX4, vregMin, pregBandMask4);
                    MicroAPI::Select<T>(vregX5, vregX5, vregMin, pregBandMask5);
                    MicroAPI::Select<T>(vregX6, vregX6, vregMin, pregBandMask6);
                    MicroAPI::Select<T>(vregX7, vregX7, vregMin, pregBandMask7);
                }
            }
            // select之后，当作256个数处理，不再有尾块的概念
            MicroAPI::Select<T>(vregX0, vregX0, vregMin2, pregTail0);
            MicroAPI::Select<T>(vregX1, vregX1, vregMin2, pregTail1);
            MicroAPI::Select<T>(vregX2, vregX2, vregMin2, pregTail2);
            MicroAPI::Select<T>(vregX3, vregX3, vregMin2, pregTail3);
            MicroAPI::Select<T>(vregX4, vregX4, vregMin2, pregTail4);
            MicroAPI::Select<T>(vregX5, vregX5, vregMin2, pregTail5);
            MicroAPI::Select<T>(vregX6, vregX6, vregMin2, pregTail6);
            MicroAPI::Select<T>(vregX7, vregX7, vregMin2, pregTail7);
            { // x_max
                MicroAPI::RegTensor<T> vregMaxTmp0;
                MicroAPI::RegTensor<T> vregMaxTmp1;
                MicroAPI::RegTensor<T> vregMaxTmp2;
                MicroAPI::RegTensor<T> vregMaxTmp3;
                MicroAPI::RegTensor<T> vregInMax;

                MicroAPI::Max<T, MaskMergeMode::ZEROING>(vregMaxTmp0, vregX0, vregX1, pregAll);
                MicroAPI::Max<T, MaskMergeMode::ZEROING>(vregMaxTmp1, vregX2, vregX3, pregAll);
                MicroAPI::Max<T, MaskMergeMode::ZEROING>(vregMaxTmp2, vregX4, vregX5, pregAll);
                MicroAPI::Max<T, MaskMergeMode::ZEROING>(vregMaxTmp3, vregX6, vregX7, pregAll);
                MicroAPI::Max<T, MaskMergeMode::ZEROING>(vregMaxTmp0, vregMaxTmp0, vregMaxTmp1, pregAll);
                MicroAPI::Max<T, MaskMergeMode::ZEROING>(vregMaxTmp2, vregMaxTmp2, vregMaxTmp3, pregAll);
                MicroAPI::Max<T, MaskMergeMode::ZEROING>(vregMax, vregMaxTmp0, vregMaxTmp2, pregAll);

                MicroAPI::ReduceMax<T, MaskMergeMode::ZEROING>(vregMax, vregMax, pregAll);
                MicroAPI::Duplicate<T, HighLowPart::LOWEST, MaskMergeMode::ZEROING>(vregMax, vregMax, pregAll);

                MicroAPI::DataCopy<T, LoadDist::DIST_BRC_B32>(vregInMax, inMaxUb + i * reduceN);
                MicroAPI::Max<T, MaskMergeMode::ZEROING>(vregMax, vregMax, vregInMax, pregAll);
                MicroAPI::DataCopyUnAlign<T, PostLiteral::POST_MODE_UPDATE>(tmpMaxUb, vregMax, uregMax, 1);
            }

            { // exp
                MicroAPI::FusedExpSub<T, T, RegLayout::ONE, MaskMergeMode::ZEROING>(vregX0, vregX0, vregMax, pregAll);
                MicroAPI::FusedExpSub<T, T, RegLayout::ONE, MaskMergeMode::ZEROING>(vregX1, vregX1, vregMax, pregAll);
                MicroAPI::FusedExpSub<T, T, RegLayout::ONE, MaskMergeMode::ZEROING>(vregX2, vregX2, vregMax, pregAll);
                MicroAPI::FusedExpSub<T, T, RegLayout::ONE, MaskMergeMode::ZEROING>(vregX3, vregX3, vregMax, pregAll);
                MicroAPI::FusedExpSub<T, T, RegLayout::ONE, MaskMergeMode::ZEROING>(vregX4, vregX4, vregMax, pregAll);
                MicroAPI::FusedExpSub<T, T, RegLayout::ONE, MaskMergeMode::ZEROING>(vregX5, vregX5, vregMax, pregAll);
                MicroAPI::FusedExpSub<T, T, RegLayout::ONE, MaskMergeMode::ZEROING>(vregX6, vregX6, vregMax, pregAll);
                MicroAPI::FusedExpSub<T, T, RegLayout::ONE, MaskMergeMode::ZEROING>(vregX7, vregX7, vregMax, pregAll);

                MicroAPI::RegTensor<T> vregExpEven0;
                MicroAPI::RegTensor<T> vregExpOdd0;
                MicroAPI::RegTensor<T> vregExpEven1;
                MicroAPI::RegTensor<T> vregExpOdd1;
                MicroAPI::RegTensor<T> vregExpEven2;
                MicroAPI::RegTensor<T> vregExpOdd2;
                MicroAPI::RegTensor<T> vregExpEven3;
                MicroAPI::RegTensor<T> vregExpOdd3;

                MicroAPI::DeInterleave<T>(vregExpEven0, vregExpOdd0, vregX0, vregX1);
                MicroAPI::DeInterleave<T>(vregExpEven1, vregExpOdd1, vregX2, vregX3);
                MicroAPI::DeInterleave<T>(vregExpEven2, vregExpOdd2, vregX4, vregX5);
                MicroAPI::DeInterleave<T>(vregExpEven3, vregExpOdd3, vregX6, vregX7);

                static constexpr MicroAPI::CastTrait castTrait0 = {RegLayout::ZERO, SatMode::NO_SAT,
                                                                   MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                static constexpr MicroAPI::CastTrait castTrait1 = {RegLayout::ONE, SatMode::NO_SAT,
                                                                   MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

                MicroAPI::Cast<T1, T, castTrait0>((MicroAPI::RegTensor<T1>&)vregExpEven0, vregExpEven0, pregAll);
                MicroAPI::Cast<T1, T, castTrait1>((MicroAPI::RegTensor<T1>&)vregExpOdd0, vregExpOdd0, pregAll);
                MicroAPI::Cast<T1, T, castTrait0>((MicroAPI::RegTensor<T1>&)vregExpEven1, vregExpEven1, pregAll);
                MicroAPI::Cast<T1, T, castTrait1>((MicroAPI::RegTensor<T1>&)vregExpOdd1, vregExpOdd1, pregAll);
                MicroAPI::Cast<T1, T, castTrait0>((MicroAPI::RegTensor<T1>&)vregExpEven2, vregExpEven2, pregAll);
                MicroAPI::Cast<T1, T, castTrait1>((MicroAPI::RegTensor<T1>&)vregExpOdd2, vregExpOdd2, pregAll);
                MicroAPI::Cast<T1, T, castTrait0>((MicroAPI::RegTensor<T1>&)vregExpEven3, vregExpEven3, pregAll);
                MicroAPI::Cast<T1, T, castTrait1>((MicroAPI::RegTensor<T1>&)vregExpOdd3, vregExpOdd3, pregAll);

                MicroAPI::Or<uint16_t, MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vregExpEven0,
                                                               (MicroAPI::RegTensor<uint16_t>&)vregExpEven0,
                                                               (MicroAPI::RegTensor<uint16_t>&)vregExpOdd0, pregAll);
                MicroAPI::Or<uint16_t, MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vregExpEven1,
                                                               (MicroAPI::RegTensor<uint16_t>&)vregExpEven1,
                                                               (MicroAPI::RegTensor<uint16_t>&)vregExpOdd1, pregAll);
                MicroAPI::Or<uint16_t, MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vregExpEven2,
                                                               (MicroAPI::RegTensor<uint16_t>&)vregExpEven2,
                                                               (MicroAPI::RegTensor<uint16_t>&)vregExpOdd2, pregAll);
                MicroAPI::Or<uint16_t, MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vregExpEven3,
                                                               (MicroAPI::RegTensor<uint16_t>&)vregExpEven3,
                                                               (MicroAPI::RegTensor<uint16_t>&)vregExpOdd3, pregAll);

                MicroAPI::DataCopy<T1, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    expUb, (MicroAPI::RegTensor<T1>&)vregExpEven0, blockStride, repeatStride, pregAll);
                MicroAPI::DataCopy<T1, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    expUb1, (MicroAPI::RegTensor<T1>&)vregExpEven1, blockStride, repeatStride, pregAll);
                MicroAPI::DataCopy<T1, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    expUb2, (MicroAPI::RegTensor<T1>&)vregExpEven2, blockStride, repeatStride, pregAll);
                MicroAPI::DataCopy<T1, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    expUb3, (MicroAPI::RegTensor<T1>&)vregExpEven3, blockStride, repeatStride, pregAll);
            }

            { // x_sum
                MicroAPI::RegTensor<T> vregExpSum0;
                MicroAPI::RegTensor<T> vregExpSum1;
                MicroAPI::RegTensor<T> vregExpSum2;
                MicroAPI::RegTensor<T> vregExpSum3;
                MicroAPI::RegTensor<T> vregExpSum;

                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregExpSum0, vregX0, vregX1, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregExpSum1, vregX2, vregX3, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregExpSum2, vregX4, vregX5, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregExpSum3, vregX6, vregX7, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregExpSum0, vregExpSum0, vregExpSum1, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregExpSum2, vregExpSum2, vregExpSum3, pregAll);
                MicroAPI::Add<T, MaskMergeMode::ZEROING>(vregExpSum, vregExpSum0, vregExpSum2, pregAll);

                MicroAPI::ReduceSum<T, T, MaskMergeMode::ZEROING>(vregExpSum, vregExpSum, pregAll);
                MicroAPI::DataCopyUnAlign<T, PostLiteral::POST_MODE_UPDATE>(tmpExpSumUb, vregExpSum, uregExpSum, 1);
            }
        }
        MicroAPI::DataCopyUnAlignPost<T, PostLiteral::POST_MODE_UPDATE>(tmpMaxUb, uregMax, 0);
        MicroAPI::DataCopyUnAlignPost<T, PostLiteral::POST_MODE_UPDATE>(tmpExpSumUb, uregExpSum, 0);

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        UPDATE_EXPSUM_MAX(expSumUb, maxUb, expMaxUb, inExpSumUb, inMaxUb, tmpExpSumUb2, tmpMaxUb2, rows);
    }
}

template <typename T, typename T1, typename PSE_SHIFT_T, bool withPseShift = false, bool withAttenMask = false,
          uint32_t originN = 0, bool isBand = false, bool isPFA = false>
__aicore__ inline void SoftmaxFlashV2_VF(const LocalTensor<T1>& dstTensor, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const T scale, const LocalTensor<uint8_t>& maskTensor,
    const LocalTensor<PSE_SHIFT_T>& pseTensor, const uint32_t& profileG, const SoftMaxShapeInfo& softmaxShapeInfo = {})
{
    uint32_t maskSizeAlignPFA = 0; // 非pfa场景无需偏移
    if constexpr (isPFA) {
        maskSizeAlignPFA = softmaxShapeInfo.srcK;
    }
    if constexpr (originN == 64) {
        SoftmaxFlashV2_base64_update<T, T1, PSE_SHIFT_T, withPseShift, true, withAttenMask, true, isBand> (dstTensor,
            expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor, sharedTmpBuffer,
            scale, maskSizeAlignPFA, maskTensor, pseTensor, softmaxShapeInfo);
    } else if constexpr (originN == 128) {
        SoftmaxFlashV2_base128_update<T, T1, PSE_SHIFT_T, withPseShift, true, withAttenMask, true, isBand> (dstTensor,
            expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor, sharedTmpBuffer,
            scale, maskSizeAlignPFA, maskTensor, pseTensor, softmaxShapeInfo);
    } else if constexpr (originN == 256) {
        SoftmaxFlashV2_base256_update<T, T1, PSE_SHIFT_T, withPseShift, true, withAttenMask, true, isBand> (dstTensor,
            expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor, sharedTmpBuffer,
            scale, maskSizeAlignPFA, maskTensor, pseTensor, softmaxShapeInfo);
    } else if constexpr (originN == 512) {
        SoftmaxFlashV2_base512_update<T, T1, PSE_SHIFT_T, withPseShift, true, withAttenMask, true, isBand> (dstTensor,
            expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor, sharedTmpBuffer,
            scale, maskSizeAlignPFA, maskTensor, pseTensor, profileG, softmaxShapeInfo);
    }
}

} // namespace

#endif // SOFTMAX_FLASH_V2_ND2NZ_CONST_H
