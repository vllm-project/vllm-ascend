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
 * \file dropmask.h
 * \brief
 */

#ifndef DROPMASK_H
#define DROPMASK_H

#include "adv_api/math/philox.h"
#include "util_regbase.h"

using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::DropOutShapeInfo;

namespace regbaseutil {
constexpr static int32_t uint32Uint8Ratio = 4;
constexpr static int32_t uint32BlockSize = 8;
constexpr static int32_t eachOffsetRandomUint8Num = 16;
constexpr static int32_t dropMaskUint32AlignSize = 4;
constexpr static int32_t philoxRandomNumAlignSize = 16;
constexpr static int32_t halfBaseSize = 16;
constexpr static uint32_t eachRowAlignNum = 8;
constexpr static uint32_t PHILOX_CONST_MUL_0 = 0xD2511F53;
constexpr static uint32_t PHILOX_CONST_MUL_1 = 0xCD9E8D57;
constexpr static uint32_t PHILOX_CONST_KEY_ADD_0 = 0x9E3779B9;
constexpr static uint32_t PHILOX_CONST_KEY_ADD_1 = 0xBB67AE85;
constexpr static uint32_t OFFSET_64 = 64;
constexpr static uint32_t OFFSET_32 = 32;

struct DropMaskInfo {
    int64_t seed;
    int64_t offset;
    uint8_t keepProbUint8;
    uint8_t dropMaskOuter;
    bool boolMode;
};
template <bool hasDrop, bool hasRope = false>
__aicore__ inline uint64_t ComputeDropOffset(RunInfo<false> &runInfo, ConstInfo<false, hasRope> &constInfo, DropMaskInfo &dropMaskInfo)
{
    int64_t s2SizeAligned = CeilDiv(runInfo.actualS2Size, philoxRandomNumAlignSize) * philoxRandomNumAlignSize;
    // boidx * n2 * g* s1 * s2
    int64_t bOffset = runInfo.b1SSOffsetAlign * constInfo.n2G;
    // n2oIdx * g * s1 *s2
    int64_t n2Offset = runInfo.n2oIdx * constInfo.gSize * runInfo.actualS1Size * s2SizeAligned;
    // goIdx * s1 * s2
    int64_t gOffset = runInfo.goIdx * runInfo.actualS1Size * s2SizeAligned;
    // s1oIdx * s1BaseSize * s2Size + s1innerindex * vec1S1BaseSize * s2Size
    int64_t s1Offset = (runInfo.s1oIdx * constInfo.s1BaseSize + runInfo.vecCoreOffset) * s2SizeAligned;
    // s2StartIdx + s2index * s2BaseNratioSize
    int64_t s2Offset = runInfo.s2StartIdx + runInfo.s2LoopCount * constInfo.s2BaseSize;
    // bOffset + n2Offset + gOffset + s1Offset + s2Offset用int64能表示，dropMaskInfo.offset也能用int64能表示，所以两项相加用uint64_t不会出现回绕
    return static_cast<uint64_t>(bOffset + n2Offset + gOffset + s1Offset + s2Offset) +
           static_cast<uint64_t>(dropMaskInfo.offset);
}

__simd_vf__ inline void GenIndexAlign(const uint64_t indexVecDstLocalInt, const uint32_t rowNums,
                                      const uint32_t eachRowIndexNum, const uint32_t eachRowOffset)
{
    RegTensor<int32_t> inc_idx;
    MaskReg preg;
    uint32_t sreg = eachRowIndexNum;
    preg= UpdateMask<int32_t>(sreg);

    Arange(inc_idx, 0);
    for (uint16_t s1Idx = 0; s1Idx < static_cast<uint16_t>(rowNums); s1Idx++) {
        StoreAlign<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ int32_t *&)indexVecDstLocalInt, inc_idx, eachRowIndexNum, preg);
        Adds(inc_idx, inc_idx, eachRowOffset, preg);
    }
}

__simd_vf__ inline void GenIndexUnAling(const uint64_t indexVecDstLocalInt, const uint32_t rowNums,
                                        const uint32_t eachRowIndexNum, const uint32_t eachRowOffset)
{
    RegTensor<int32_t> inc_idx;
    MaskReg preg;
    UnalignRegForStore ureg;
    uint32_t sreg = eachRowIndexNum;
    preg= UpdateMask<int32_t>(sreg);

    Arange(inc_idx, 0);
    for (uint16_t s1Idx = 0; s1Idx < static_cast<uint16_t>(rowNums); s1Idx++) {
        StoreUnAlign<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            (__ubuf__ int32_t *&)indexVecDstLocalInt, inc_idx, ureg, eachRowIndexNum);
        Adds(inc_idx, inc_idx, eachRowOffset, preg);
    }
    StoreUnAlignPost<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ int32_t *&)indexVecDstLocalInt, ureg, 0);
}

/*
 * @ingroup GenIndexVec
 * @brief generate basic block index vector, eachRowIndexNum need 8 aligned
 * @param [out] dropmaskIndexVec, output LocalTensor
 * @param [in] rowNums, basic block's rows
 * @param [in] eachRowIndexNum, each row need index nums, eachRowIndexNum = CeilDiv(CeilDiv(s2RealSize, 4), 4) * 4 / 4, only support less equal 64
 * @param [in] eachRowOffset, each row index need add offset, eachRowOffset = CeilDiv(inputParams.s2Size, 16)
 */
__aicore__ inline void GenIndexVec(LocalTensor<int32_t> &dropmaskIndexVec, uint32_t rowNums, uint32_t eachRowIndexNum,
                                   uint32_t eachRowOffset)
{   
    uint64_t indexVecDstLocalInt = reinterpret_cast<uint64_t>(dropmaskIndexVec.GetPhyAddr());
    if (eachRowIndexNum % eachRowAlignNum == 0) {
        GenIndexAlign(indexVecDstLocalInt, rowNums, eachRowIndexNum, eachRowOffset);
    } else {
        GenIndexUnAling(indexVecDstLocalInt, rowNums, eachRowIndexNum, eachRowOffset);
    }
}

__simd_vf__ inline void GenMaskVF(__ubuf__ uint32_t *mask, const uint64_t indexVecLocalInt,
                                  const uint32_t key0, const uint32_t key1, const uint32_t counter0,
                                  const uint32_t counter1, const uint32_t counter2, const uint32_t counter3,
                                  const uint16_t count, const uint8_t probValueUint8Scalar, const uint16_t mainLoop)
{
    MaskReg pg = CreateMask<uint32_t, MaskPattern::ALL>();
    MaskReg preg = CreateMask<uint8_t, MaskPattern::ALL>();
    MaskReg pd;
    MaskReg pm_0, pm_1, pm_2, pm_3;
    RegTensor<uint32_t> ctr_3, ctr_2, ctr_1, ctr_0, key_1, key_0;
    Duplicate(key_0, key0);
    Duplicate(key_1, key1);
    Duplicate(ctr_0, counter0);
    Duplicate(ctr_1, counter1);
    Duplicate(ctr_2, counter2);
    Duplicate(ctr_3, counter3);

    RegTensor<int32_t> inc_idx;
    RegTensor<uint32_t> v_zero;
    RegTensor<uint32_t> v_const_mul_0, v_const_mul_1;

    Duplicate(v_zero, 0x0);
    Duplicate(v_const_mul_0, (uint32_t)PHILOX_CONST_MUL_0);
    Duplicate(v_const_mul_1, (uint32_t)PHILOX_CONST_MUL_1);

    for (uint16_t i = 0; i < mainLoop; i++) {
        LoadAlign<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            inc_idx, ((__ubuf__ int32_t *&)indexVecLocalInt), ELE_CNT_B32);

        RegTensor<uint32_t> tmp_ctr_0 = ctr_0;
        RegTensor<uint32_t> tmp_ctr_1 = ctr_1;
        RegTensor<uint32_t> tmp_ctr_2 = ctr_2;
        RegTensor<uint32_t> tmp_ctr_3 = ctr_3;
        AddCarryOut(pd, tmp_ctr_0, ctr_0, (RegTensor<uint32_t>&)inc_idx, pg);
        AddCarryOuts(pd, tmp_ctr_1, ctr_1, v_zero, pd, pg);
        AddCarryOuts(pd, tmp_ctr_2, ctr_2, v_zero, pd, pg);
        AddCarryOuts(pd, tmp_ctr_3, ctr_3, v_zero, pd, pg);

        RegTensor<uint32_t> tmp_key_0 = key_0;
        RegTensor<uint32_t> tmp_key_1 = key_1;                                                    
        for (uint16_t j = 0; j < 7; j++) {
            RegTensor<uint32_t> tmp_l0, tmp_h0, tmp_l1, tmp_h1;
            Mull(tmp_l0, tmp_h0, tmp_ctr_0, v_const_mul_0, pg);
            Mull(tmp_l1, tmp_h1, tmp_ctr_2, v_const_mul_1, pg);
            Xor(tmp_h1, tmp_h1, tmp_ctr_1, pg);
            Xor(tmp_ctr_0, tmp_h1, tmp_key_0, pg);
            Xor(tmp_h0, tmp_h0, tmp_ctr_3, pg);
            Xor(tmp_ctr_2, tmp_h0, tmp_key_1, pg);
            tmp_ctr_1 = tmp_l1;
            tmp_ctr_3 = tmp_l0;
            Adds(tmp_key_0, tmp_key_0, (uint32_t)PHILOX_CONST_KEY_ADD_0, pg);
            Adds(tmp_key_1, tmp_key_1, (uint32_t)PHILOX_CONST_KEY_ADD_1, pg);
        }
        Interleave(tmp_ctr_0, tmp_ctr_2, tmp_ctr_0, tmp_ctr_2);
        Interleave(tmp_ctr_1, tmp_ctr_3, tmp_ctr_1, tmp_ctr_3);
        Interleave(tmp_ctr_0, tmp_ctr_1, tmp_ctr_0, tmp_ctr_1);
        Interleave(tmp_ctr_2, tmp_ctr_3, tmp_ctr_2, tmp_ctr_3);

        // uint32转成4个uint8与keepprob比较得到mask
        RegTensor<uint8_t> tmp_ctr_0_u8 = (RegTensor<uint8_t>&) tmp_ctr_0;
        RegTensor<uint8_t> tmp_ctr_1_u8 = (RegTensor<uint8_t>&) tmp_ctr_1;
        RegTensor<uint8_t> tmp_ctr_2_u8 = (RegTensor<uint8_t>&) tmp_ctr_2;
        RegTensor<uint8_t> tmp_ctr_3_u8 = (RegTensor<uint8_t>&) tmp_ctr_3;

        CompareScalar<uint8_t, CMPMODE::LE>(pm_0, tmp_ctr_0_u8, probValueUint8Scalar, preg);
        CompareScalar<uint8_t, CMPMODE::LE>(pm_1, tmp_ctr_1_u8, probValueUint8Scalar, preg);
        CompareScalar<uint8_t, CMPMODE::LE>(pm_2, tmp_ctr_2_u8, probValueUint8Scalar, preg);
        CompareScalar<uint8_t, CMPMODE::LE>(pm_3, tmp_ctr_3_u8, probValueUint8Scalar, preg);
        // pm是256bit，32Byte，偏移是按照Byte来的
        StoreAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ uint32_t *&)mask, pm_0, 32);
        StoreAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ uint32_t *&)mask, pm_1, 32);
        StoreAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ uint32_t *&)mask, pm_2, 32);
        StoreAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ uint32_t *&)mask, pm_3, 32);
    }
}

/*
 * @ingroup GenMaskByIndexVec
 * @brief generate dropout mask by indexVec, use philox
 * @param [out] dstLocal, output LocalTensor
 * @param [in] indexUb, input index LocalTensor generated by GenIndexVec interface
 * @param [in] philoxkey, philox input parameters key 64bits --> two uint32
 * @param [in] philoxCounter, philox input parameters counter 128bits  --> four uint32
 * @param [in] count, The number of uint32 random numbers expected to be generated， 4 aligned
 * @param [in] probValueUint8Scalar, keepprob*MAX_UINT8
 */
__aicore__ inline void GenMaskByIndexVec(const LocalTensor<uint8_t> &dstLocal, const LocalTensor<int32_t> &indexUb,
                                         const PhiloxKey &philoxkey, const PhiloxCounter &philoxCounter, uint16_t count,
                                         uint8_t probValueUint8Scalar)
{   
    __ubuf__ uint32_t *mask = (__ubuf__ uint32_t *)dstLocal.GetPhyAddr();
    uint64_t indexVecLocalInt = reinterpret_cast<uint64_t>(indexUb.GetPhyAddr());
    
    // 一次可以生成256个uint32的随机数, 如果不够256，生成的mask后面有一部分脏数据，dropout计算中只使用前面的有效数据
    uint16_t mainLoop = CeilDiv(count, 256);

    GenMaskVF(mask, indexVecLocalInt, philoxkey[0], philoxkey[1], philoxCounter[0], philoxCounter[1], philoxCounter[2],
        philoxCounter[3], count, probValueUint8Scalar, mainLoop);
}

template <bool hasRope = false>
__aicore__ inline void GenDropMask(TBuf<> &dropMaskBuf, TBuf<> &maskIndexBuf, uint64_t maskOffset,
                                   RunInfo<false> &runInfo, ConstInfo<false, hasRope> &constInfo, DropMaskInfo &dropMaskInfo)
{
    // 一次philox算法可以生成16个uint8的随机数
    uint64_t dropMaskOffset = CeilDiv(maskOffset, philoxRandomNumAlignSize);
    // PhiloxRandom接口出来的数据是uint32类型，这里会转成4*uint8用于生成mask, 所以生成的uint32的count需要除以4;
    // 另外由于一个offset会生成4个uint32,为了offset方便计算，需要做4对齐
    uint32_t eachRowCount = CeilDiv(CeilDiv(constInfo.s2BaseSize, uint32Uint8Ratio),
                                        dropMaskUint32AlignSize) * dropMaskUint32AlignSize;
    // 一个index生成4个uint32的随机数
    uint32_t eachRowIndexNum = CeilDiv(eachRowCount, 4);
    // 每一行index之间的偏移
    int32_t eachRowOffset = CeilDiv(runInfo.actualS2Size, philoxRandomNumAlignSize);

    LocalTensor<int32_t> dropmaskIndexVec = maskIndexBuf.template Get<int32_t>();
    if (constInfo.layoutType != (uint8_t)LayOutTypeEnum::LAYOUT_TND &&
        runInfo.actualS1Size % constInfo.s1BaseSize == 0 && runInfo.actualS2Size % constInfo.s2BaseSize == 0) {
        // 如果s1和s2方向上都没有尾块，一个核内dropmaskIndexVec只需要生成一次，可以复用
        if (runInfo.taskId == 0) {
            GenIndexVec(dropmaskIndexVec, runInfo.halfS1RealSize, eachRowIndexNum, eachRowOffset);
        }
    } else if (runInfo.s2LoopCount == 0 || runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        // 如果有尾块的场景，统一在s2Idx首尾两块刷新dropmaskIndexVec
        GenIndexVec(dropmaskIndexVec, runInfo.halfS1RealSize, eachRowIndexNum, eachRowOffset);
    }

    uint64_t dropMaskSeed = static_cast<uint64_t>(dropMaskInfo.seed);
    uint32_t seedHigh = static_cast<uint32_t>(dropMaskSeed >> 32);
    uint32_t seedLow = static_cast<uint32_t>(dropMaskSeed & 0xffffffff);
    uint32_t offsetHigh = static_cast<uint32_t>(dropMaskOffset >> 32);
    uint32_t offsetLow = static_cast<uint32_t>(dropMaskOffset & 0xffffffff);
    LocalTensor<uint8_t> dropMaskUb = dropMaskBuf.template Get<uint8_t>();
    GenMaskByIndexVec(dropMaskUb, dropmaskIndexVec, {seedLow, seedHigh}, {offsetLow, offsetHigh, 0, 0},
        runInfo.halfS1RealSize * eachRowCount, dropMaskInfo.keepProbUint8);
}

template <bool hasDrop, bool hasRope = false>
__aicore__ inline void GenDropMask(TBuf<> &dropMaskBuf, TBuf<> &maskIndexBuf, RunInfo<false> &runInfo,
                                   ConstInfo<false, hasRope> &constInfo, DropMaskInfo &dropMaskInfo)
{
    if constexpr (hasDrop == true) {
        int64_t dropMaskOffset = ComputeDropOffset<hasDrop>(runInfo, constInfo, dropMaskInfo);
        GenDropMask(dropMaskBuf, maskIndexBuf, dropMaskOffset, runInfo, constInfo, dropMaskInfo);
        return;
    }
}

__aicore__ inline void BoolCopyIn(LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor,
    int64_t srcOffset, uint32_t s1Size, uint32_t s2Size, int64_t totalS2Size, int64_t s2BaseSize)
{
    if (s1Size == 0 || s2Size == 0) {
        return;
    }
    uint32_t alignedS2Size = CeilDiv(s2Size, blockBytes) * blockBytes;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = s1Size;
    dataCopyParams.blockLen = CeilDiv(s2Size, blockBytes);
    dataCopyParams.dstStride = CeilDiv(s2BaseSize, blockBytes) - dataCopyParams.blockLen;
    if (totalS2Size % blockBytes == 0) {
        dataCopyParams.srcStride = (totalS2Size - s2Size) / blockBytes;
        DataCopy(dstTensor, srcTensor[srcOffset], dataCopyParams);
    } else {
        dataCopyParams.blockLen = s2Size;
        dataCopyParams.srcStride = totalS2Size - s2Size;
        DataCopyPadParams dataCopyPadParams;
        DataCopyPad(dstTensor, srcTensor[srcOffset], dataCopyParams, dataCopyPadParams);
    }
}

__aicore__ inline void Bit2Int8CopyIn(LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor,
    int64_t srcOffset, uint32_t s1Size, uint32_t s2Size, int64_t totalS2Size, int64_t s2BaseSize)
{
    if (s1Size == 0 || s2Size == 0) {
        return;
    }
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = s1Size;
    dataCopyParams.blockLen = CeilDiv(s2Size / byteBitRatio, blockBytes);
    dataCopyParams.dstStride = CeilDiv(s2BaseSize / byteBitRatio, blockBytes) - dataCopyParams.blockLen;
    if (totalS2Size / byteBitRatio % blockBytes == 0) {
        dataCopyParams.srcStride =
            (totalS2Size / byteBitRatio - dataCopyParams.blockLen * blockBytes) / blockBytes;
        DataCopy(dstTensor, srcTensor[srcOffset / byteBitRatio], dataCopyParams);
    } else {
        dataCopyParams.blockLen = CeilDiv(s2Size, byteBitRatio);
        dataCopyParams.srcStride = (totalS2Size - s2Size) / byteBitRatio;
        DataCopyPadParams dataCopyPadParams;
        dataCopyPadParams.isPad = true;
        dataCopyPadParams.rightPadding = 0;
        dataCopyPadParams.paddingValue = 0;
        DataCopyPad(dstTensor, srcTensor[srcOffset / byteBitRatio], dataCopyParams, dataCopyPadParams);
    }
}

template <bool hasDrop, bool hasRope = false>
__aicore__ inline int64_t ComputeOuterDropOffset(RunInfo<false> &runInfo, ConstInfo<false, hasRope> &constInfo, DropMaskInfo &dropMaskInfo)
{
    if constexpr (hasDrop == true) {
        // boidx * n2 * g* s1 * s2
        int64_t bOffset = runInfo.b1SSOffset * constInfo.n2G;
        // n2oIdx * g * s1 *s2
        int64_t n2Offset = runInfo.n2oIdx * constInfo.gSize * runInfo.actualS1Size * runInfo.actualS2Size;
        // goIdx * s1 * s2
        int64_t gOffset = runInfo.goIdx * runInfo.actualS1Size * runInfo.actualS2Size;
        // s1oIdx * s1BaseSize * s2Size
        int64_t s1Offset = (runInfo.s1oIdx * constInfo.s1BaseSize + runInfo.vecCoreOffset) * runInfo.actualS2Size;
        // s2StartIdx + s2index * s2BaseNratioSize
        int64_t s2Offset = runInfo.s2StartIdx + runInfo.s2LoopCount * constInfo.s2BaseSize;
        return bOffset + n2Offset + gOffset + s1Offset + s2Offset;
    } else {
        return 0;
    }
}

__simd_vf__ inline void DropMaskBool2BitVF(const uint64_t srcUb, const uint64_t dstUb, const uint16_t loopCount)
{
    RegTensor<uint32_t> vreg_drop;
    MaskReg vreg_cmp;
    MaskReg preg_all = CreateMask<uint8_t, MaskPattern::ALL>();

    for (uint16_t i = 0; i < loopCount; ++i) {
        LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vreg_drop, (__ubuf__ uint32_t *&)srcUb, OFFSET_64);
        RegTensor<uint8_t> vreg_tmp = (RegTensor<uint8_t> &)vreg_drop;
        CompareScalar<uint8_t, CMPMODE::EQ>(vreg_cmp, vreg_tmp, 1, preg_all);
        StoreAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                (__ubuf__ uint32_t *&)dstUb, vreg_cmp, OFFSET_32);
    }
}

__aicore__ inline void DropMaskBool2Bit(LocalTensor<uint8_t> &dstTensor, LocalTensor<uint8_t> &srcTensor,
                                        int32_t halfS1RealSize, int64_t s2BaseSize)
{
    uint64_t srcUb = reinterpret_cast<uint64_t>(srcTensor.GetPhyAddr());
    uint64_t dstUb = reinterpret_cast<uint64_t>(dstTensor.GetPhyAddr());
    uint16_t rowNumEachLoop = regBytes / static_cast<uint16_t>(s2BaseSize);
    uint16_t halfS1RealSizeLoop = static_cast<uint16_t>(halfS1RealSize) + 1;
    uint16_t loopCount = halfS1RealSizeLoop / rowNumEachLoop;

    DropMaskBool2BitVF(srcUb, dstUb, loopCount);
}

__simd_vf__ inline void DropMaskPadDelVF(const uint64_t srcUb, const uint64_t dstUb, const uint16_t loopCount)
{
    MaskReg preg1;
    MaskReg preg2;
    MaskReg preg3;
    MaskReg preg4;

    for (uint16_t i = 0; i < loopCount; ++i) {
        // srcUb: 12340000 srcUb: 56780000
        // preg1: 11223344 preg2: 55667788
        LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                preg1, (__ubuf__ uint32_t *&)srcUb, OFFSET_32);
        LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                preg2, (__ubuf__ uint32_t *&)srcUb, OFFSET_32);
        // preg3: 12345678
        MaskDeInterleave<uint8_t>(preg3, preg4, preg1, preg2);
        StoreAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                (__ubuf__ uint32_t *&)dstUb, preg3, OFFSET_32);
    }
}

__aicore__ inline void DropMaskPadDel(LocalTensor<uint8_t> &dstTensor, LocalTensor<uint8_t> &srcTensor,
                                      int32_t halfS1RealSize, int64_t s2BaseSize)
{
    uint64_t srcUb = reinterpret_cast<uint64_t>(srcTensor.GetPhyAddr());
    uint64_t dstUb = reinterpret_cast<uint64_t>(dstTensor.GetPhyAddr());
    uint16_t loopCount = (static_cast<uint16_t>(halfS1RealSize) + 1) / 2;

    DropMaskPadDelVF(srcUb, dstUb, loopCount);
}

template <bool hasDrop, bool hasRope = false>
__aicore__ inline void CopyInDropOuter(TBuf<> &dropMaskBuf, TQue<QuePosition::VECIN, 1> &dropMaskInQue,
    GlobalTensor<uint8_t>& srcTensor, RunInfo<false> &runInfo, ConstInfo<false, hasRope> &constInfo, DropMaskInfo &dropMaskInfo)
{
    if constexpr (hasDrop == true) {
        int64_t dropMaskOffset = ComputeOuterDropOffset<hasDrop>(runInfo, constInfo, dropMaskInfo);
        LocalTensor<uint8_t> dropTensor = dropMaskBuf.template Get<uint8_t>();
        LocalTensor<uint8_t> dropMaskUb = dropMaskInQue.template AllocTensor<uint8_t>();
        if (unlikely(dropMaskInfo.boolMode)) {
            BoolCopyIn(dropMaskUb, srcTensor, dropMaskOffset, runInfo.halfS1RealSize, runInfo.s2RealSize,
                       runInfo.actualS2Size, constInfo.s2BaseSize);
            dropMaskInQue.template EnQue(dropMaskUb);
            dropMaskInQue.template DeQue<uint8_t>();
            // uint8 to uint1
            DropMaskBool2Bit(dropTensor, dropMaskUb, runInfo.halfS1RealSize, constInfo.s2BaseSize);
        } else {
            Bit2Int8CopyIn(dropMaskUb, srcTensor, dropMaskOffset, runInfo.halfS1RealSize, runInfo.s2RealSize,
                           runInfo.actualS2Size, constInfo.s2BaseSize);
            dropMaskInQue.template EnQue(dropMaskUb);
            dropMaskInQue.template DeQue<uint8_t>();
            DropMaskPadDel(dropTensor, dropMaskUb, runInfo.halfS1RealSize, constInfo.s2BaseSize);
        }
        dropMaskInQue.template FreeTensor(dropMaskUb);
        return;
    }
}
}

#endif // DROPMASK_H