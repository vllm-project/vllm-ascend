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
 * \file vf_flash_decode.h
 * \brief
 */
#ifndef MY_FLASH_DECODE_H
#define MY_FLASH_DECODE_H

#include "kernel_tensor.h"

constexpr float FLT_ZERO = 0;
constexpr float FLT_MAX_NEW = 3.402823466e+38F;

namespace FaVectorApi {
// bf16->fp32
static constexpr MicroAPI::CastTrait castTraitFp16_32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                   MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// 处理循环splitKVIndex=0的场景，vregDst需要置0
template <typename T>
__simd_vf__ void ReduceFinalRes_0_VF(__ubuf__ T * dstUb, __ubuf__ T * lseUb, __ubuf__ T * accumOutUb, uint16_t k, uint16_t z, 
                                    uint32_t dealNum1Reg, uint32_t repStride, const uint16_t floatRepSize, 
                                    const uint16_t dLoops, uint32_t dealRowCount, uint32_t splitKVIndex)
{
    MicroAPI::RegTensor<T> vregDst;
    MicroAPI::RegTensor<T> vregLse;
    MicroAPI::RegTensor<T> vregAccumOut;
    uint32_t n = dealNum1Reg;
    MicroAPI::MaskReg pregTailN = MicroAPI::UpdateMask<T>(n);

    for (k = 0; k < static_cast<uint16_t>(dealRowCount); k++) {  // repeat g

        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BLK>(vregLse,
                                                            (__ubuf__ float*&)lseUb + splitKVIndex * dealRowCount * 8 + k * 8);
        for (z = 0; z < dLoops; z++) {
            // splitKVIndex=0的场景，vregDst不需要load，直接置0
            MicroAPI::Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vregDst, FLT_ZERO, pregTailN);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(
                vregAccumOut, (__ubuf__ float*&)accumOutUb + k * repStride * 8 + z * floatRepSize);
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vregAccumOut, vregLse, vregAccumOut, pregTailN);
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregDst, vregDst, vregAccumOut, pregTailN);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ float*&)dstUb + k * repStride * 8 + z * floatRepSize, vregDst, pregTailN);
        }
    }
}

template <typename T>
__aicore__ inline void ReduceFinalRes_0(LocalTensor<T>& dstLocal, LocalTensor<T>& lseLocal,
                                           LocalTensor<T>& accumOutLocal, uint32_t dealRowCount,
                                           uint64_t headDimAlignFp32, uint32_t splitKVIndex)
{
    __ubuf__ T * dstUb = (__ubuf__ T *)dstLocal.GetPhyAddr();
    __ubuf__ T * lseUb = (__ubuf__ T *)lseLocal.GetPhyAddr();
    __ubuf__ T * accumOutUb = (__ubuf__ T *)accumOutLocal.GetPhyAddr();
    uint16_t z = 0;
    uint16_t k = 0;
    const uint16_t floatRepSize = 64;
    const uint16_t dLoops = headDimAlignFp32 / floatRepSize;
    uint32_t dealNum1Reg = 256 / sizeof(float);
    uint32_t repStride = headDimAlignFp32 / 8;

    ReduceFinalRes_0_VF<T>(dstUb, lseUb, accumOutUb, k, z, dealNum1Reg, repStride, floatRepSize, 
                        dLoops, dealRowCount, splitKVIndex);
}

// 处理循环splitKVIndex>0的场景，reg_dst需要先从dstUb中load之前的结果，再进行add
template <typename T>
__simd_vf__ void ReduceFinalRes_Rest_VF(__ubuf__ T * dstUb, __ubuf__ T * lseUb, __ubuf__ T * accumOutUb, uint16_t k, uint16_t z, 
                                        uint32_t dealNum1Reg, uint32_t repStride, const uint16_t floatRepSize, const uint16_t dLoops, 
                                        uint32_t dealRowCount, uint32_t splitKVIndex)
{
    MicroAPI::RegTensor<T> vregDst;
    MicroAPI::RegTensor<T> vregLse;
    MicroAPI::RegTensor<T> vregAccumOut;
    uint32_t n = dealNum1Reg;
    MicroAPI::MaskReg pregTailN = MicroAPI::UpdateMask<T>(n);
    uint32_t stride = (0x1 << 16) | 0x8;

    for (k = 0; k < static_cast<uint16_t>(dealRowCount); k++) {  // repeat g
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BLK>(vregLse,
                                                            (__ubuf__ float*&)lseUb + splitKVIndex * dealRowCount * 8 + k * 8);
        for (z = 0; z < dLoops; z++) {
            // splitKVIndex>0的场景，reg_dst需要先从dstUb中load之前的结果，再进行add
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(
                vregDst, (__ubuf__ float*&)dstUb + k * repStride * 8 + z * floatRepSize);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(
                vregAccumOut, (__ubuf__ float*&)accumOutUb + k * repStride * 8 + z * floatRepSize);
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vregAccumOut, vregLse, vregAccumOut, pregTailN);
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregDst, vregDst,
                vregAccumOut, pregTailN);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ float*&)dstUb + k * repStride * 8 + z * floatRepSize, vregDst, pregTailN);
        }
    }
}

template <typename T>
__aicore__ inline void ReduceFinalRes_Rest(LocalTensor<T>& dstLocal, LocalTensor<T>& lseLocal,
                                                 LocalTensor<T>& accumOutLocal, uint32_t dealRowCount,
                                                 uint64_t headDimAlignFp32, uint32_t splitKVIndex)
{
    __ubuf__ T * dstUb = (__ubuf__ T *)dstLocal.GetPhyAddr();
    __ubuf__ T * lseUb = (__ubuf__ T *)lseLocal.GetPhyAddr();
    __ubuf__ T * accumOutUb = (__ubuf__ T *)accumOutLocal.GetPhyAddr();
    uint16_t k = 0;
    uint16_t z = 0;
    uint32_t dealNum1Reg = 256 / sizeof(float);
    uint32_t repStride = headDimAlignFp32 / 8;
    const uint16_t floatRepSize = 64;
    const uint16_t dLoops = headDimAlignFp32 / floatRepSize;

    ReduceFinalRes_Rest_VF<T>(dstUb, lseUb, accumOutUb, k, z, dealNum1Reg, repStride, floatRepSize, dLoops, dealRowCount, splitKVIndex);
}

template <typename T>
__aicore__ inline void ReduceFinalRes_VF(LocalTensor<T>& dstLocal, LocalTensor<T>& lseLocal, LocalTensor<T>& accumOutLocal,
    uint32_t dealRowCount, uint64_t headDimAlignFp32, uint32_t splitKVIndex)
{
    if (splitKVIndex == 0) {
        ReduceFinalRes_0(dstLocal, lseLocal, accumOutLocal, dealRowCount, headDimAlignFp32, splitKVIndex);
    } else {
        ReduceFinalRes_Rest(dstLocal, lseLocal, accumOutLocal, dealRowCount, headDimAlignFp32, splitKVIndex);
    }
}

template <typename T, uint32_t headDimAlignFp32 = 0>
__aicore__ inline void ReduceFinalRes_const_VF(LocalTensor<T>& dstLocal, LocalTensor<T>& lseLocal, LocalTensor<T>& accumOutLocal,
    uint32_t dealRowCount, uint32_t splitKVIndex)
{
    if (splitKVIndex == 0) {
        ReduceFinalRes_0(dstLocal, lseLocal, accumOutLocal, dealRowCount, headDimAlignFp32, splitKVIndex);
    } else {
        ReduceFinalRes_Rest(dstLocal, lseLocal, accumOutLocal, dealRowCount, headDimAlignFp32, splitKVIndex);
    }
}

// 处理g<=8的场景
template <typename T, typename SINK_T>
__simd_vf__ void ComputeScaleValue_8_VF(__ubuf__ uint16_t * lseSink, __ubuf__ T * lseMax, __ubuf__ T * lseMaxTmp, __ubuf__ T * lseSum, 
                                        __ubuf__ T * lseSumTmp, __ubuf__ T * lseUb, uint32_t dealCount, uint16_t i, 
                                        uint32_t dealRowCount, uint32_t actualCombineLoopSize, bool softmaxLseFlag, bool learnableSinkFlag)
{
    MicroAPI::RegTensor<T> vregLseMax;
    MicroAPI::RegTensor<T> vregLseMaxTmp;
    MicroAPI::RegTensor<T> vregLseSum;
    MicroAPI::RegTensor<T> vregLseSumTmp;
    MicroAPI::RegTensor<SINK_T> vregLseSink;
    MicroAPI::RegTensor<T> vregLseSinkCast;
    MicroAPI::RegTensor<T> vregRes;
    uint32_t n = dealCount;
    MicroAPI::MaskReg pregTailN = MicroAPI::UpdateMask<T>(n);
    MicroAPI::MaskReg pregSinkTailN = MicroAPI::UpdateMask<SINK_T>(n);
    uint16_t blockStride = 0x1;
    uint16_t repeatStride = dealRowCount;

    MicroAPI::Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vregLseMax, -FLT_MAX_NEW, pregTailN);
    MicroAPI::Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vregLseSum, FLT_ZERO, pregTailN);

    for (i = 0; i < static_cast<uint16_t>(actualCombineLoopSize); ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregLseMaxTmp, 
                                                                (__ubuf__ float *&)lseMaxTmp + i * dealCount);
        MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseMax, vregLseMax, vregLseMaxTmp, pregTailN);
    }

    for (i = 0; i < static_cast<uint16_t>(actualCombineLoopSize); ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregLseMaxTmp, 
                                                              (__ubuf__ float *&)lseMaxTmp + i * dealCount);
        MicroAPI::Sub<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseMaxTmp, vregLseMaxTmp,
                                                           vregLseMax, pregTailN);
        MicroAPI::Exp<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseMaxTmp, vregLseMaxTmp, pregTailN);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregLseSumTmp, 
                                                              (__ubuf__ float *&)lseSumTmp + i * dealCount);
        MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSumTmp, vregLseSumTmp,
                                                           vregLseMaxTmp, pregTailN);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSum, vregLseSum, vregLseSumTmp, pregTailN);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ float *&)lseSumTmp + i * dealCount, 
                                                                    vregLseSumTmp, pregTailN);
    }

    if (learnableSinkFlag) {
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>((MicroAPI::RegTensor<uint16_t>&)vregLseSink, lseSink);
        MicroAPI::Cast<T, SINK_T, castTraitFp16_32>(vregLseSinkCast, vregLseSink, pregSinkTailN);
        
        MicroAPI::Sub<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSinkCast, vregLseSinkCast, vregLseMax, pregTailN);
        MicroAPI::Exp<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSinkCast, vregLseSinkCast, pregTailN);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSum, vregLseSum, vregLseSinkCast, pregTailN);
    }

    if (softmaxLseFlag) {
        MicroAPI::RegTensor<float> vregMinValue;
        MicroAPI::RegTensor<float> vregInfValue;
        MicroAPI::MaskReg pregCompare;
        constexpr float infValue = 3e+99; // 3e+99 for float inf
        constexpr uint32_t tmpMin = 0xFF167699;
        float minValue = *((float*)&tmpMin);
        MicroAPI::Duplicate<float, float>(vregMinValue, minValue);
        MicroAPI::Duplicate<float, float>(vregInfValue, infValue);

        MicroAPI::Log<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregLseSum, pregTailN);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregRes, vregLseMax, pregTailN);
        // 如果 softmaxMax 等于负无穷，则将 lse 结果置为 inf
        MicroAPI::Compare<float, CMPMODE::EQ>(pregCompare, vregLseMax, vregMinValue, pregTailN);
        MicroAPI::Select<T>(vregRes, vregInfValue, vregRes, pregCompare);
        MicroAPI::StoreAlign<T, StoreDist::DIST_NORM_B32>(lseUb, vregRes, pregTailN);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    for (i = 0; i < static_cast<uint16_t>(actualCombineLoopSize); ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregLseSumTmp, 
                                                                (__ubuf__ float *&)lseSumTmp + i * dealCount);
        MicroAPI::Div<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSumTmp, vregLseSumTmp, vregLseSum, pregTailN);
        MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>
            ((__ubuf__ float *&)lseSum, vregLseSumTmp, blockStride, repeatStride, pregTailN);
    }
}

template <typename T, typename SINK_T>
__aicore__ inline void ComputeScaleValue_8(const LocalTensor<SINK_T> &tmpSinkUb, const LocalTensor<T> &lseMaxUb,
                                           const LocalTensor<T> &lseSumUb, const LocalTensor<T> &lseOutputUb,
                                           uint32_t dealRowCount, uint32_t actualCombineLoopSize, bool softmaxLseFlag,
                                           bool learnableSinkFlag)
{
    uint32_t dealCount = dealRowCount * 8;
    uint16_t i = 0;

    __ubuf__ T *lseMax = (__ubuf__ T *)lseMaxUb.GetPhyAddr();
    __ubuf__ T *lseMaxTmp = lseMax;
    __ubuf__ T *lseSum = (__ubuf__ T *)lseSumUb.GetPhyAddr();
    __ubuf__ T *lseSumTmp = lseSum;
    __ubuf__ T *lseUb = (__ubuf__ T *)lseOutputUb.GetPhyAddr();
    __ubuf__ uint16_t *lseSink = (__ubuf__ uint16_t *)tmpSinkUb.GetPhyAddr();

    ComputeScaleValue_8_VF<T, SINK_T>(lseSink, lseMax, lseMaxTmp, lseSum, lseSumTmp, lseUb, dealCount, i, dealRowCount,
                                      actualCombineLoopSize, softmaxLseFlag, learnableSinkFlag);
}

// //lseUb作为scale最终输出
template <typename T, typename SINK_T>
__simd_vf__ void ComputeScaleValue_8_VF_FD(__ubuf__ T *lseSink, __ubuf__ T *lseMax, __ubuf__ T *lseMaxTmp,
                                           __ubuf__ T *lseSum, __ubuf__ T *lseSumTmp, __ubuf__ T *lseOutUb,
                                           __ubuf__ T *lseUb, __ubuf__ T *lseMaxReduce, uint32_t dealCount, uint16_t i,
                                           uint32_t dealRowCount, uint32_t actualCombineLoopSize,
                                           uint16_t softmaxLseFlag, uint16_t learnableSinkFlag)
{
    MicroAPI::RegTensor<T> vregLseMax;
    MicroAPI::RegTensor<T> vregLseMaxTmp;
    MicroAPI::RegTensor<T> vregRes;
    MicroAPI::RegTensor<T> vregLseSum;
    MicroAPI::RegTensor<T> vregLseSumTmp;
    MicroAPI::RegTensor<T> vregLseSinkCast;
    uint16_t blockStride = 0x1;
    uint16_t repeatStride = dealRowCount;
    uint32_t n = dealCount;
    MicroAPI::MaskReg pregTailN = MicroAPI::UpdateMask<T>(n);
    MicroAPI::MaskReg pregSinkTailN = MicroAPI::UpdateMask<SINK_T>(n);

    MicroAPI::Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vregLseSum, FLT_ZERO, pregTailN);
    MicroAPI::Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vregLseMax, -FLT_MAX_NEW, pregTailN);

    for (i = 0; i < static_cast<uint16_t>(actualCombineLoopSize); ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregLseMaxTmp, 
                                                              (__ubuf__ float *&)lseMaxTmp + i * dealCount);
        MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseMax, vregLseMax, vregLseMaxTmp, pregTailN);
    }
    MicroAPI::StoreAlign<T, StoreDist::DIST_NORM_B32>(lseMaxReduce, vregLseMax, pregTailN);

    for (i = 0; i < static_cast<uint16_t>(actualCombineLoopSize); ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregLseMaxTmp, 
                                                              (__ubuf__ float *&)lseMaxTmp + i * dealCount);
        MicroAPI::Sub<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseMaxTmp, vregLseMaxTmp, vregLseMax, pregTailN);
        MicroAPI::Exp<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseMaxTmp, vregLseMaxTmp, pregTailN);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregLseSumTmp, 
                                                              (__ubuf__ float *&)lseSumTmp + i * dealCount);
        MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSumTmp, vregLseSumTmp, vregLseMaxTmp, pregTailN);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSum, vregLseSum, vregLseSumTmp, pregTailN);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ float *&)lseSumTmp + i * dealCount, 
                                                                    vregLseSumTmp, pregTailN);
    }

    for (i = 0;i < static_cast<uint16_t>(learnableSinkFlag); ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregLseSinkCast, (__ubuf__ float *&)lseSink);
        MicroAPI::Sub<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSinkCast, vregLseSinkCast, vregLseMax, pregTailN);
        MicroAPI::Exp<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSinkCast, vregLseSinkCast, pregTailN);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSum, vregLseSum, vregLseSinkCast, pregTailN);
    }

    for (i = 0;i < static_cast<uint16_t>(softmaxLseFlag); ++i){
        MicroAPI::RegTensor<float> vregMinValue;
        MicroAPI::RegTensor<float> vregInfValue;
        MicroAPI::MaskReg pregCompare;
        constexpr float infValue = 3e+99; // 3e+99 for float inf
        constexpr uint32_t tmpMin = 0xFF167699;
        float minValue = *((float*)&tmpMin);
        MicroAPI::Duplicate<float, float>(vregInfValue, infValue);
        MicroAPI::Duplicate<float, float>(vregMinValue, minValue);

        MicroAPI::Log<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregLseSum, pregTailN);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregRes, vregLseMax, pregTailN);
        // 如果 softmaxMax 等于负无穷，则将 lse 结果置为 inf
        MicroAPI::Compare<float, CMPMODE::EQ>(pregCompare, vregLseMax, vregMinValue, pregTailN);
        MicroAPI::Select<T>(vregRes, vregInfValue, vregRes, pregCompare);
        MicroAPI::StoreAlign<T, StoreDist::DIST_NORM_B32>(lseOutUb, vregRes, pregTailN);
    }
    
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    for (i = 0; i < static_cast<uint16_t>(actualCombineLoopSize); ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregLseSumTmp, 
                                                                (__ubuf__ float *&)lseSumTmp + i * dealCount);
        MicroAPI::Div<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSumTmp, vregLseSumTmp, vregLseSum, pregTailN);
        MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>
            ((__ubuf__ float *&)lseUb, vregLseSumTmp, blockStride, repeatStride, pregTailN);
    }
}

template <typename T, typename SINK_T>
__aicore__ inline void ComputeScaleValue_8_FD(const LocalTensor<SINK_T>& tmpSinkUb, const LocalTensor<T>& lseMaxUb, const LocalTensor<T>& lseSumUb,
    const LocalTensor<T>& lseResUb, const LocalTensor<T>& lseOutputUb, const LocalTensor<T>& lseMaxReduceUb, uint32_t dealRowCount, uint32_t actualCombineLoopSize, bool softmaxLseFlag, bool learnableSinkFlag)
{
    uint32_t dealCount = dealRowCount * 8;
    uint16_t i = 0;

    __ubuf__ T * lseMax = (__ubuf__ T *)lseMaxUb.GetPhyAddr();
    __ubuf__ T * lseMaxTmp = lseMax;
    __ubuf__ T * lseSum = (__ubuf__ T *)lseSumUb.GetPhyAddr();
    __ubuf__ T * lseSumTmp = lseSum;
    __ubuf__ T * lseUb = (__ubuf__ T *)lseResUb.GetPhyAddr();
    __ubuf__ T * lseOutUb = (__ubuf__ T *)lseOutputUb.GetPhyAddr();
    __ubuf__ T * lseMaxReduce = (__ubuf__ T *)lseMaxReduceUb.GetPhyAddr();
    //lseSink在前面已经类型转换成T
    __ubuf__ T * lseSink = (__ubuf__ T *)tmpSinkUb.GetPhyAddr();
    uint16_t softmaxLseFlagUint = 0;
    uint16_t learnableSinkFlagUint = 0;
    if (softmaxLseFlag) {
        softmaxLseFlagUint = 1;
    }
    if (learnableSinkFlag) {
        learnableSinkFlagUint = 1;
    }
    ComputeScaleValue_8_VF_FD<T, SINK_T>(lseSink, lseMax, lseMaxTmp, lseSum, lseSumTmp, lseOutUb, lseUb, lseMaxReduce,dealCount, i, dealRowCount, actualCombineLoopSize, softmaxLseFlagUint, learnableSinkFlagUint);
}

// 处理8<g<=16的场景
template <typename T, typename SINK_T>
__simd_vf__ void ComputeScaleValue_16_VF(__ubuf__ uint16_t * lseSink, __ubuf__ uint16_t * lseSink2, __ubuf__ T * lseMax, __ubuf__ T * lseMax2, __ubuf__ T * lseMaxSrc, __ubuf__ T * lseSum, 
                                        __ubuf__ T * lseSum2, __ubuf__ T * lseSumSrc, __ubuf__ T * lseUb, __ubuf__ T * lseUb2, 
                                        uint32_t dealCountSum, uint32_t dealCount, uint32_t dealCount2, 
                                        uint16_t i, uint32_t dealRowCount, uint32_t actualCombineLoopSize, bool softmaxLseFlag, bool learnableSinkFlag)
{
    MicroAPI::RegTensor<T> vregLseMax;
    MicroAPI::RegTensor<T> vregLseMaxTmp;
    MicroAPI::RegTensor<T> vregLseMax2;
    MicroAPI::RegTensor<T> vregLseMaxTmp2;
    MicroAPI::RegTensor<T> vregLseSum;
    MicroAPI::RegTensor<T> vregLseSumTmp;
    MicroAPI::RegTensor<T> vregLseSum2;
    MicroAPI::RegTensor<T> vregLseSumTmp2;
    MicroAPI::RegTensor<SINK_T> vregLseSink;
    MicroAPI::RegTensor<SINK_T> vregLseSink2;
    MicroAPI::RegTensor<T> vregLseSinkCast;
    MicroAPI::RegTensor<T> vregLseSinkCast2;
    MicroAPI::RegTensor<T> vregRes;
    MicroAPI::RegTensor<T> vregRes2;
    uint32_t n = dealCount;
    uint32_t n2 = dealCount2;
    MicroAPI::MaskReg pregTailN = MicroAPI::UpdateMask<T>(n);
    MicroAPI::MaskReg pregTailN2 = MicroAPI::UpdateMask<T>(n2);
    MicroAPI::MaskReg pregSinkTailN = MicroAPI::UpdateMask<SINK_T>(n);
    MicroAPI::MaskReg pregSinkTailN2 = MicroAPI::UpdateMask<SINK_T>(n2);
    uint16_t blockStride = 0x1;
    uint16_t repeatStride = dealRowCount;

    MicroAPI::Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vregLseMax, -FLT_MAX_NEW, pregTailN);
    MicroAPI::Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vregLseMax2, -FLT_MAX_NEW, pregTailN);
    MicroAPI::Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vregLseSum, FLT_ZERO, pregTailN);
    MicroAPI::Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vregLseSum2, FLT_ZERO, pregTailN);

    for (i = 0; i < static_cast<uint16_t>(actualCombineLoopSize); ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregLseMaxTmp, 
                                                                lseMaxSrc + i * dealCountSum);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregLseMaxTmp2, 
                                                                lseMaxSrc + i * dealCountSum + dealCount);
        MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseMax, vregLseMax, vregLseMaxTmp, pregTailN);
        MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseMax2, vregLseMax2, vregLseMaxTmp2, pregTailN2);
    }

    for (i = 0; i < static_cast<uint16_t>(actualCombineLoopSize); ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregLseMaxTmp, 
                                                                lseMaxSrc + i * dealCountSum);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregLseMaxTmp2, 
                                                                lseMaxSrc + i * dealCountSum + dealCount);
        MicroAPI::Sub<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseMaxTmp, vregLseMaxTmp, vregLseMax, pregTailN);
        MicroAPI::Sub<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseMaxTmp2, vregLseMaxTmp2, vregLseMax2, pregTailN2);
        MicroAPI::Exp<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseMaxTmp, vregLseMaxTmp, pregTailN);
        MicroAPI::Exp<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseMaxTmp2, vregLseMaxTmp2, pregTailN2);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregLseSumTmp, 
                                                                lseSumSrc + i * dealCountSum);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregLseSumTmp2, 
                                                                lseSumSrc + i * dealCountSum + dealCount);
        MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSumTmp, vregLseSumTmp, vregLseMaxTmp, pregTailN);
        MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSumTmp2, vregLseSumTmp2, vregLseMaxTmp2, pregTailN2);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSum, vregLseSum, vregLseSumTmp, pregTailN);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSum2, vregLseSum2, vregLseSumTmp2, pregTailN2);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(lseSumSrc + i * dealCountSum, vregLseSumTmp, pregTailN);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(lseSumSrc + i * dealCountSum + dealCount, vregLseSumTmp2, pregTailN2);
    }

    if (learnableSinkFlag) {
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>((MicroAPI::RegTensor<uint16_t>&)vregLseSink, lseSink);
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>((MicroAPI::RegTensor<uint16_t>&)vregLseSink2, lseSink + dealCount);
        
        MicroAPI::Cast<T, SINK_T, castTraitFp16_32>(vregLseSinkCast, vregLseSink, pregSinkTailN);
        MicroAPI::Cast<T, SINK_T, castTraitFp16_32>(vregLseSinkCast2, vregLseSink2, pregSinkTailN2);

        MicroAPI::Sub<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSinkCast, vregLseSinkCast, vregLseMax, pregTailN);
        MicroAPI::Sub<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSinkCast2, vregLseSinkCast2, vregLseMax2, pregTailN2);

        MicroAPI::Exp<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSinkCast, vregLseSinkCast, pregTailN);
        MicroAPI::Exp<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSinkCast2, vregLseSinkCast2, pregTailN2);

        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSum, vregLseSum, vregLseSinkCast, pregTailN);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSum2, vregLseSum2, vregLseSinkCast2, pregTailN2);
    }

    if (softmaxLseFlag) {
        MicroAPI::RegTensor<float> vregMinValue;
        MicroAPI::RegTensor<float> vregInfValue;
        MicroAPI::MaskReg pregCompare;
        MicroAPI::MaskReg pregCompare2;
        constexpr float infValue = 3e+99; // 3e+99 for float inf
        constexpr uint32_t tmpMin = 0xFF167699;
        float minValue = *((float*)&tmpMin);
        MicroAPI::Duplicate<float, float>(vregMinValue, minValue);
        MicroAPI::Duplicate<float, float>(vregInfValue, infValue);

        MicroAPI::Log<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregLseSum, pregTailN);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregRes, vregLseMax, pregTailN);
        MicroAPI::Log<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes2, vregLseSum2, pregTailN2);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes2, vregRes2, vregLseMax2, pregTailN2);
        // 如果 softmaxMax 等于负无穷，则将 lse 结果置为 inf
        MicroAPI::Compare<float, CMPMODE::EQ>(pregCompare, vregLseMax, vregMinValue, pregTailN);
        MicroAPI::Compare<float, CMPMODE::EQ>(pregCompare2, vregLseMax2, vregMinValue, pregTailN2);
        MicroAPI::Select<T>(vregRes, vregInfValue, vregRes, pregCompare);
        MicroAPI::Select<T>(vregRes2, vregInfValue, vregRes2, pregCompare2);
        MicroAPI::StoreAlign<T, StoreDist::DIST_NORM_B32>(lseUb, vregRes, pregTailN);
        MicroAPI::StoreAlign<T, StoreDist::DIST_NORM_B32>(lseUb2, vregRes2, pregTailN2);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    for (i = 0; i < static_cast<uint16_t>(actualCombineLoopSize); ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregLseSumTmp, 
                                                                lseSumSrc + i * dealCountSum);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregLseSumTmp2, 
                                                                lseSumSrc + i * dealCountSum + dealCount);
        MicroAPI::Div<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSumTmp, vregLseSumTmp, vregLseSum, pregTailN);
        MicroAPI::Div<T, MicroAPI::MaskMergeMode::ZEROING>(vregLseSumTmp2, vregLseSumTmp2, vregLseSum2, pregTailN2);
        MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>
            (lseSum, vregLseSumTmp, blockStride, repeatStride, pregTailN);
        MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>
            (lseSum2, vregLseSumTmp2, blockStride, repeatStride, pregTailN2);
    }
}

template <typename T, typename SINK_T>
__aicore__ inline void ComputeScaleValue_16(const LocalTensor<SINK_T>& tmpSinkUb, const LocalTensor<T>& lseMaxUb, const LocalTensor<T>& lseSumUb,
    const LocalTensor<T>& lseOutputUb, uint32_t dealRowCount, uint32_t actualCombineLoopSize, bool softmaxLseFlag, bool learnableSinkFlag)
{
    uint32_t dealCountSum = dealRowCount * 8;
    uint32_t dealCount = 8 * 8;
    uint32_t dealCount2 = dealCountSum - dealCount;
    uint16_t i = 0;

    __ubuf__ T * lseMax = (__ubuf__ T *)lseMaxUb.GetPhyAddr();
    __ubuf__ T * lseMax2 = lseMax + 64;
    __ubuf__ T * lseMaxSrc = lseMax;
    __ubuf__ T * lseSum = (__ubuf__ T *)lseSumUb.GetPhyAddr();
    __ubuf__ T * lseSum2 = lseSum + 64;
    __ubuf__ T * lseSumSrc = lseSum;
    __ubuf__ T * lseUb = (__ubuf__ T *)lseOutputUb.GetPhyAddr();
    __ubuf__ T * lseUb2 = lseUb + 64;
    __ubuf__ uint16_t * lseSink = (__ubuf__ uint16_t *)tmpSinkUb.GetPhyAddr();
    __ubuf__ uint16_t * lseSink2 = lseSink + 64;

    ComputeScaleValue_16_VF<T, SINK_T>(lseSink, lseSink2, lseMax, lseMax2, lseMaxSrc, lseSum, lseSum2, lseSumSrc, lseUb, lseUb2, 
                            dealCountSum, dealCount, dealCount2, i, dealRowCount, actualCombineLoopSize, softmaxLseFlag, learnableSinkFlag);
}

template <typename T, typename SINK_T>
__aicore__ inline void ComputeScaleValue_VF(const LocalTensor<SINK_T>& tmpSinkUb, const LocalTensor<T>& lseMaxUb, const LocalTensor<T>& lseSumUb,
    const LocalTensor<T>& lseOutputUb, uint32_t dealRowCount, uint32_t actualCombineLoopSize, bool softmaxLseFlag, bool learnableSinkFlag)
{
    if (dealRowCount <= 8) {
        ComputeScaleValue_8(tmpSinkUb, lseMaxUb, lseSumUb, lseOutputUb, dealRowCount, actualCombineLoopSize, softmaxLseFlag, learnableSinkFlag);
    } else if (dealRowCount <= 16) {
        ComputeScaleValue_16(tmpSinkUb, lseMaxUb, lseSumUb, lseOutputUb, dealRowCount, actualCombineLoopSize, softmaxLseFlag, learnableSinkFlag);
    }
}

//gqa 非量化走这个模板函数，目前dealRowCount默认为8
//lseResUb为ScaleValue的计算结果UB
template <typename T, typename SINK_T>
__aicore__ inline void ComputeScaleValue_VF_FD(const LocalTensor<SINK_T>& tmpSinkUb, const LocalTensor<T>& lseMaxUb, const LocalTensor<T>& lseSumUb,
     const LocalTensor<T>& lseResUb, const LocalTensor<T>& lseOutputUb, const LocalTensor<T>& lseMaxUbTmp,uint32_t dealRowCount, uint32_t actualCombineLoopSize, bool softmaxLseFlag, bool learnableSinkFlag)
{
    ComputeScaleValue_8_FD(tmpSinkUb, lseMaxUb, lseSumUb, lseResUb, lseOutputUb, lseMaxUbTmp, dealRowCount, actualCombineLoopSize, softmaxLseFlag, learnableSinkFlag);
}

// 处理g<=8的场景
template <typename T>
__simd_vf__ void ComputeLogSumExp_8_VF(__ubuf__ T * srcSumLocalInt, __ubuf__ T * srcMaxLocalInt, __ubuf__ T * dstLocalInt, uint32_t dealCount)
{
    MicroAPI::RegTensor<T> vregSum;
    MicroAPI::RegTensor<T> vregMax;
    MicroAPI::RegTensor<T> vregRes;
    MicroAPI::MaskReg pregTailN = MicroAPI::UpdateMask<T>(dealCount);
    MicroAPI::RegTensor<float> vregMinValue;
    MicroAPI::RegTensor<float> vregInfValue;
    MicroAPI::MaskReg pregCompare;
    constexpr float infValue = 3e+99; // 3e+99 for float inf
    constexpr uint32_t tmpMin = 0xFF167699;
    float minValue = *((float*)&tmpMin);
    MicroAPI::Duplicate<float, float>(vregMinValue, minValue);
    MicroAPI::Duplicate<float, float>(vregInfValue, infValue);

    // 1.load to reg
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregSum, (__ubuf__ float *&)srcSumLocalInt);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregMax, (__ubuf__ float *&)srcMaxLocalInt);

    // 2.LogSumExp
    MicroAPI::Log<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregSum, pregTailN);
    MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregRes, vregMax, pregTailN);

    // 如果 softmaxMax 等于负无穷，则将 lse 结果置为 inf
    MicroAPI::Compare<float, CMPMODE::EQ>(pregCompare, vregMax, vregMinValue, pregTailN);
    MicroAPI::Select<T>(vregRes, vregInfValue, vregRes, pregCompare);

    // 3.copy to ub
    MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ float *&)dstLocalInt, vregRes, pregTailN);
}

template <typename T>
__aicore__ inline void ComputeLogSumExp_8(const LocalTensor<T>& dstTensor, const LocalTensor<T>& softmaxSumTensor,
    const LocalTensor<T>& softmaxMaxTensor, uint32_t dealCount)
{
    __ubuf__ T * srcSumLocalInt = (__ubuf__ T *)softmaxSumTensor.GetPhyAddr();
    __ubuf__ T * srcMaxLocalInt = (__ubuf__ T *)softmaxMaxTensor.GetPhyAddr();
    __ubuf__ T * dstLocalInt = (__ubuf__ T *)dstTensor.GetPhyAddr();

    ComputeLogSumExp_8_VF<T>(srcSumLocalInt, srcMaxLocalInt, dstLocalInt, dealCount);
}

// 处理8<g<=16的场景
template <typename T>
__simd_vf__ void ComputeLogSumExp_16_VF(__ubuf__ T * srcSumUb, __ubuf__ T * srcSumUb2, __ubuf__ T * srcMaxUb, __ubuf__ T * srcMaxUb2, 
                                        __ubuf__ T * dstUb, __ubuf__ T * dstUb2, uint32_t dealCount1, uint32_t dealCount2)
{
    MicroAPI::RegTensor<T> vregSum;
    MicroAPI::RegTensor<T> vregSum2;
    MicroAPI::RegTensor<T> vregMax;
    MicroAPI::RegTensor<T> vregMax2;
    MicroAPI::RegTensor<T> vregRes;
    MicroAPI::RegTensor<T> vregRes2;
    MicroAPI::MaskReg pregTailN = MicroAPI::UpdateMask<T>(dealCount1);
    MicroAPI::MaskReg pregTailN2 = MicroAPI::UpdateMask<T>(dealCount2);
    MicroAPI::RegTensor<float> vregMinValue;
    MicroAPI::RegTensor<float> vregInfValue;
    MicroAPI::MaskReg pregCompare;
    MicroAPI::MaskReg pregCompare2;
    constexpr float infValue = 3e+99; // 3e+99 for float inf
    constexpr uint32_t tmpMin = 0xFF167699;
    float minValue = *((float*)&tmpMin);
    MicroAPI::Duplicate<float, float>(vregMinValue, minValue);
    MicroAPI::Duplicate<float, float>(vregInfValue, infValue);

    // 1.load to reg
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregSum, srcSumUb);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregSum2, srcSumUb2);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregMax, srcMaxUb);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregMax2, srcMaxUb2);

    // 2.LogSumExp
    MicroAPI::Log<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregSum, pregTailN);
    MicroAPI::Log<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes2, vregSum2, pregTailN2);
    MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregRes, vregMax, pregTailN);
    MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes2, vregRes2, vregMax2, pregTailN2);

    // 如果 softmaxMax 等于负无穷，则将 lse 结果置为 inf
    MicroAPI::Compare<float, CMPMODE::EQ>(pregCompare, vregMax, vregMinValue, pregTailN);
    MicroAPI::Compare<float, CMPMODE::EQ>(pregCompare2, vregMax2, vregMinValue, pregTailN2);
    MicroAPI::Select<T>(vregRes, vregInfValue, vregRes, pregCompare);
    MicroAPI::Select<T>(vregRes2, vregInfValue, vregRes2, pregCompare2);

    // 3.copy to ub
    MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb, vregRes, pregTailN);
    MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb2, vregRes2, pregTailN2);
}

template <typename T>
__aicore__ inline void ComputeLogSumExp_16(const LocalTensor<T>& dstTensor, const LocalTensor<T>& softmaxSumTensor,
    const LocalTensor<T>& softmaxMaxTensor, uint32_t dealCount)
{
    __ubuf__ T * srcSumUb = (__ubuf__ T *)softmaxSumTensor.GetPhyAddr();
    __ubuf__ T * srcSumUb2 = srcSumUb + 64; // 一个寄存器最多处理64个数
    __ubuf__ T * srcMaxUb = (__ubuf__ T *)softmaxMaxTensor.GetPhyAddr();
    __ubuf__ T * srcMaxUb2 = srcMaxUb + 64;
    __ubuf__ T * dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();
    __ubuf__ T * dstUb2 = dstUb + 64;
    uint32_t dealCount1 = 8 * 8;
    uint32_t dealCount2 = dealCount - dealCount1;

    ComputeLogSumExp_16_VF<T>(srcSumUb, srcSumUb2, srcMaxUb, srcMaxUb2, dstUb, dstUb2, dealCount1, dealCount2);
}

template <typename T>
__aicore__ inline void ComputeLogSumExp_VF(const LocalTensor<T>& dstTensor, const LocalTensor<T>& softmaxSumTensor,
    const LocalTensor<T>& softmaxMaxTensor, uint32_t dealRowCount)
{
    if (dealRowCount <= 8) {
        ComputeLogSumExp_8(dstTensor, softmaxSumTensor, softmaxMaxTensor, dealRowCount * 8); // 8:FP32 in one block
    } else if (dealRowCount <= 16) {
        ComputeLogSumExp_16(dstTensor, softmaxSumTensor, softmaxMaxTensor, dealRowCount * 8); // 8:FP32 in one block
    }
}

} // namespace

#endif // MY_FLASH_DECODE_H
