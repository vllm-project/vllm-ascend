/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gmmsq_quant_mx_vf.h
 * \brief SIMD-VF functions for GMM SwiGLU + MX Quantization.
 */

#ifndef GMMSQ_QUANT_MX_VF_H
#define GMMSQ_QUANT_MX_VF_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "basic_block_vf_mx.h"

namespace GMMSQWeightQuant {

namespace MicroAPI = AscendC::MicroAPI;
using AscendC::BLOCK_CUBE;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::MicroAPI::AddrReg;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

static constexpr AscendC::MicroAPI::DivSpecificMode DIV_MODE = {
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    true,
};

static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_BF16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

/*!
 * \brief Computes SwiGLU activation using vector instructions.
 *
 * \param[in] firstSrc          Swish branch input in UB.
 * \param[in] secondSrc         Gate branch input in UB.
 * \param[in] gluResAddr        Output UB address for SwiGLU result.
 * \param[in] mSize             M dimension size.
 * \param[in] nSize             N dimension size.
 * \param[in] nSrcUbAligned     Aligned N stride for source UB.
 * \param[in] nDstUbAligned     Aligned N stride for destination UB.
 * \param[in] oneRowRepeatTimes Number of vector repeats per row.
 * \param[in] sizePerRepeat     Elements processed per vector repeat.
 */
template <typename DataTypeIn>
__simd_vf__ inline void GmmSwigluVf(__ubuf__ DataTypeIn *firstSrc, __ubuf__ DataTypeIn *secondSrc,
                                    __ubuf__ bfloat16_t *gluResAddr, uint16_t mSize, uint16_t nSize,
                                    uint32_t nSrcUbAligned, uint32_t nDstUbAligned, uint16_t oneRowRepeatTimes,
                                    uint16_t sizePerRepeat)
{
    const float scalarOne = 1.0;
    const float mulFactor = 64.0; // 64：GMM缩小64倍在此处补齐
    for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
        uint32_t elementNum = nSize;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<DataTypeIn>(elementNum);
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < oneRowRepeatTimes; vfBlockIdx++) {
            AscendC::MicroAPI::RegTensor<bfloat16_t> verg4;
            AscendC::MicroAPI::RegTensor<float> swishInput, gateInput;
            AscendC::MicroAPI::RegTensor<float> verg0, verg1, verg2, verg3, swishOutput;

            uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * sizePerRepeat;

            // 1. Load Swish branch
            AscendC::MicroAPI::LoadAlign(swishInput, firstSrc + l0cOutOffset);
            MicroAPI::Muls(swishInput, swishInput, mulFactor, mask);

            // 2. Swish: x / (1 + exp(-x))
            AscendC::MicroAPI::Muls(verg0, swishInput, -(scalarOne), mask);
            AscendC::MicroAPI::Exp(verg1, verg0, mask);
            AscendC::MicroAPI::Adds(verg2, verg1, scalarOne, mask);
            AscendC::MicroAPI::Div<float, &DIV_MODE>(swishOutput, swishInput, verg2, mask);

            // 3. Load Gate branch
            AscendC::MicroAPI::LoadAlign(gateInput, secondSrc + l0cOutOffset);
            MicroAPI::Muls(gateInput, gateInput, mulFactor, mask);

            // 4. SwiGLU: Swish(x) * gate
            AscendC::MicroAPI::Mul(verg3, swishOutput, gateInput, mask);

            // 5. Cast to BF16 and store
            AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(verg4, verg3, mask);
            uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * sizePerRepeat;
            AscendC::MicroAPI::StoreAlign<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                gluResAddr + dstUbOffset, verg4, mask);
        }
    }
}

/*!
 * \brief Extracts maximum exponent per 32-byte block from source UB.
 *
 * \param[in] srcAddr             Source UB address (bfloat16_t).
 * \param[in] maxExpAddr          Destination UB address for max exponents.
 * \param[in] totalCountInUB      Total element count in UB to process.
 * \param[in] loopNum             Number of vector loops required.
 * \param[in] vlForHalfNumber     Vector length for half precision elements.
 * \param[in] elementAfterReduce  Element count after reduction.
 */
__simd_vf__ inline void ComputeMaxExpVf(__ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr,
                                        uint32_t totalCountInUB, uint16_t loopNum, uint32_t vlForHalfNumber,
                                        uint16_t elementAfterReduce)
{
    AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0, vdExp1;
    AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract0, vdExpExtract1;
    AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
    AscendC::MicroAPI::Duplicate(expMaskBF16, static_cast<uint16_t>(0x7f80));

    AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
    AscendC::MicroAPI::MaskReg scaleMask1;
    AscendC::MicroAPI::UnalignReg u1;

    for (uint16_t i = 0; i < loopNum; i++) {
        scaleMask1 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(totalCountInUB);

        AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr,
                                                                                   vlForHalfNumber * 2);

        AscendC::MicroAPI::And(vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0, expMaskBF16,
                               scaleMask1);
        AscendC::MicroAPI::And(vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1, expMaskBF16,
                               scaleMask1);

        AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
        AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);

        AscendC::MicroAPI::StoreUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            maxExpAddr, vdMaxExp, u1, elementAfterReduce);
    }
    AscendC::MicroAPI::StoreUnAlignPost(maxExpAddr, u1, 0);
}

/*!
 * \brief Computes MX scale (E8M0) and half-scale from max exponents.
 *
 * \param[in] maxExpAddr          Source UB address for max exponents.
 * \param[in] mxScaleLocalAddr    Destination UB address for MX scale.
 * \param[in] halfScaleLocalAddr  Destination UB address for half-scale.
 * \param[in] totalScaleInUB      Total scale element count in UB.
 * \param[in] loopNumScale        Number of vector loops for scale computation.
 * \param[in] vlForHalfNumber     Vector length for half precision elements.
 * \param[in] fpEmax              FP format max exponent value.
 */
__simd_vf__ inline void ComputeScaleVf(__ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
                                       __ubuf__ uint16_t *halfScaleLocalAddr, uint32_t totalScaleInUB,
                                       uint16_t loopNumScale, uint32_t vlForHalfNumber, uint16_t fpEmax)
{
    AscendC::MicroAPI::RegTensor<uint16_t> expMask, sharedExp, scaleValue, scaleOffset, halfScale, fp8NanRegTensor;
    AscendC::MicroAPI::MaskReg cmpResult, zeroMask, invalidDataMask, specialDataMask, preMaskScale;
    AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
    AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue, zeroRegTensor, nanRegTensor, specialExpRegTensor;
    AscendC::MicroAPI::Duplicate(maxExpValue, fpEmax);
    AscendC::MicroAPI::Duplicate(scaleOffset, static_cast<uint16_t>(0x7f00));
    AscendC::MicroAPI::Duplicate(fp8NanRegTensor, static_cast<uint16_t>(0x00ff));
    AscendC::MicroAPI::Duplicate(zeroRegTensor, static_cast<uint16_t>(0));
    AscendC::MicroAPI::Duplicate(nanRegTensor, static_cast<uint16_t>(0x7f81));
    // 0x0040：sharedExp == 0x7f00的边界情况
    AscendC::MicroAPI::Duplicate(specialExpRegTensor, static_cast<uint16_t>(0x0040));

    for (uint16_t i = 0; i < loopNumScale; i++) {
        preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(totalScaleInUB);

        AscendC::MicroAPI::LoadAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vdMaxExp, maxExpAddr,
                                                                                                 vlForHalfNumber);
        // 检测 INF/NAN 和 zero
        AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
        AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
        AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue,
                                                                   preMaskScale);
        AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);

        AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
        // 7：右移7位
        AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, static_cast<int16_t>(7), preMaskScale);
        AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
        AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

        AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                      AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, scaleValue,
                                                                                   vlForHalfNumber >> 1, preMaskScale);

        AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleOffset,
                                                                   preMaskScale);

        AscendC::MicroAPI::Sub(halfScale, scaleOffset, sharedExp, preMaskScale);
        AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
        AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
        AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

        AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            halfScaleLocalAddr, halfScale, vlForHalfNumber, preMaskScale);
    }
}

/*!
 * \brief Transposes MX scale layout from interleaved to block format per row.
 *
 * \param[in] srcAddr     Source UB address for scale data.
 * \param[in] dstAddr     Destination UB address for transposed scale block.
 * \param[in] mSize       M dimension size.
 * \param[in] scaleBlockN Number of scale elements per block.
 */
__simd_vf__ inline void TransMxScaleLayoutVf(__ubuf__ int8_t *srcAddr, __ubuf__ int8_t *dstAddr, uint16_t mSize,
                                             uint16_t scaleBlockN)
{
    for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
        uint32_t elemNum = scaleBlockN;
        AscendC::MicroAPI::MaskReg maskScaleN = AscendC::MicroAPI::UpdateMask<int8_t>(elemNum);
        AscendC::MicroAPI::RegTensor<int8_t> vreg0;
        AscendC::MicroAPI::UnalignReg u0;

        auto srcUb = srcAddr + mIdx * scaleBlockN;
        AscendC::MicroAPI::LoadUnAlignPre(u0, srcUb);
        AscendC::MicroAPI::LoadUnAlign(vreg0, u0, srcUb);

        auto dstUb = dstAddr + mIdx * AscendC::ONE_BLK_SIZE;
        AscendC::MicroAPI::StoreAlign<int8_t, AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(dstUb, vreg0, maskScaleN);
    }
}

/*!
 * \brief Quantizes FP8 data using computed half-scale.
 *
 * \param[in] srcAddr             Source UB address (bfloat16_t).
 * \param[in] halfScaleLocalAddr  Half-scale UB address.
 * \param[in] outLocalAddr        Output UB address (int8_t).
 * \param[in] totalCountInUB      Total element count in UB to quantize.
 * \param[in] loopNum             Number of vector loops required.
 * \param[in] vlForHalfNumber     Vector length for half precision elements.
 * \param[in] elementAfterReduce  Element count after reduction.
 */
template <typename DataTypeOut>
__simd_vf__ inline void
ComputeDataForQuantTargetFp8Vf(__ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
                               __ubuf__ int8_t *outLocalAddr, uint32_t totalCountInUB, uint16_t loopNum,
                               uint32_t vlForHalfNumber, uint16_t elementAfterReduce)
{
    uint32_t totalCountInUB2 = totalCountInUB * 2;

    AscendC::MicroAPI::MaskReg dataMask1, dataMask2, dataMask3, dataMask4;
    AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
    AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0, vdExp1;
    AscendC::MicroAPI::RegTensor<float> vdExp0FP32Zero, vdExp0FP32One, vdExp1FP32Zero, vdExp1FP32One;
    AscendC::MicroAPI::RegTensor<DataTypeOut> vdExp0FP8Zero, vdExp0FP8One, vdExp1FP8Zero, vdExp1FP8One;

    static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
        AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::MicroAPI::CastTrait castTrait32to8 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

    for (uint16_t i = 0; i < loopNum; i++) {
        dataMask1 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(totalCountInUB);
        dataMask2 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(totalCountInUB);
        dataMask3 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(totalCountInUB2);
        dataMask4 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(totalCountInUB2);

        AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr,
                                                                                   vlForHalfNumber * 2);

        AscendC::MicroAPI::LoadAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                                elementAfterReduce);

        AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<bfloat16_t> &)halfScaleForMul, dataMask1);
        AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<bfloat16_t> &)halfScaleForMul, dataMask1);

        AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);

        AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
        AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
        AscendC::MicroAPI::Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);
        AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
        AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(vdExp0FP8One, vdExp0FP32One, dataMask3);

        AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
        AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);
        AscendC::MicroAPI::Interleave(vdExp1FP32Zero, vdExp1FP32One, vdExp1FP32Zero, vdExp1FP32One);
        AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
        AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(vdExp1FP8One, vdExp1FP32One, dataMask4);

        // 64：使用DIST_PACK4_B32进行搬运4个float8拼成一个float32，256 / 4 = 64
        AscendC::MicroAPI::StoreAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                      AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
            outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP8Zero, static_cast<int64_t>(64), dataMask3);
        AscendC::MicroAPI::StoreAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                      AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
            outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP8One, static_cast<int64_t>(64), dataMask3);
        AscendC::MicroAPI::StoreAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                      AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
            outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp1FP8Zero, static_cast<int64_t>(64), dataMask4);
        AscendC::MicroAPI::StoreAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                      AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
            outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp1FP8One, static_cast<int64_t>(64), dataMask4);
    }
}

// R3P2 widen port (x0_deep cross-line, round_2/parallel_4 -> vendor line).
//
// Stock production sequence (preserved below as ...VfStock for the A/B
// micro-benchmark evidence chain) spends 2 x 7 = 14 instructions per 512B of
// widened output: two [LoadAlign DIST_US_B8 (128 packed B -> 256B reg),
// shr2, shl4, shr2, Select(16b mask), and 0x9C, StoreAlign 256B] iterations,
// each iteration widening only 128 packed bytes because the US_B8 load +
// Select interleave leaves half of every candidate register unused.
//
// Ported x0_deep scheme: 9 instructions per 512B out — ONE plain 256
// packed-byte LoadAlign, then per nibble lane
//   odd  fp8 = ashr(b, 2) & 0x9C          (sign ext carries sB to bit7)
//   even fp8 = ashr(shl(b, 4), 2) & 0x9C  (shl moves sA to bit7 first)
// then a single Interleave merges the two 256B candidate registers into the
// same 512B output layout, stored with the stock DIST_NORM_B8 descriptors
// (inner indices 2j / 2j+1 -> identical destination bytes as stock).
// ShiftRight arithmetic (sign-carry) semantics are exactly what the stock
// sequence itself already depends on for its own bit-exactness, so the
// nibble math is unchanged; only the packing granularity doubles.
// nRealSizeAlign is always a multiple of BLOCK_CUBE(16), so one k-row of
// packed data (nRealSizeAlign * 16 bytes) is always a multiple of 256B and
// innerLoopNum (== nRealSizeAlign / 8) is always even.
template <typename xType, typename wType>
__simd_vf__ inline void GmmsqAntiQuantMxA8W4NzNkVf(MxA8W4NzParams<xType, wType> mxA8W4NzParams)
{
    MicroAPI::RegTensor<int8_t> wShrReg, wShlReg, wAndReg, wLoad, wShl, wOdd, wEven;
    MicroAPI::RegTensor<uint8_t> wD0, wD1;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

    MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wShrReg, E2M1_SHIFT_RIGHT_SIZE, preg);
    MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wShlReg, SHIFT_LEFT_SIZE, preg);
    MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wAndReg, E2M1_AND_MASK, preg);

    // Ring-slot walk via POST_MODE_UPDATE pointer stores (verified shape — see
    // the widen A/B microbenchmark): two AddrReg stores inside one loop
    // iteration triggered broken vintlv/VAG codegen (half-swapped lanes),
    // pointer-advance stores are bit-exact. Each store advances innerDstStride
    // (one ring slot); 2*innerPairs advances per k-row == loopKDstStride, so
    // k-rows chain with no correction term.
    __ubuf__ uint8_t *weightHighBitPtr = (__ubuf__ uint8_t *)mxA8W4NzParams.weightHighBitPhyAddr;
    const uint16_t innerPairs = static_cast<uint16_t>(mxA8W4NzParams.innerLoopNum >> 1);
    for (uint16_t loopKIdx = 0; loopKIdx < mxA8W4NzParams.loopKNum; ++loopKIdx) {
        for (uint16_t pairIdx = 0; pairIdx < innerPairs; ++pairIdx) {
            // plain 256 packed-byte load (stock: two US_B8 128B loads at 2*pairIdx)
            MicroAPI::AddrReg aregWeightB8In = MicroAPI::CreateAddrReg<uint8_t>(
                loopKIdx, (C0_SIZE_B8 * mxA8W4NzParams.nRealSizeAlign) >> 1, pairIdx, VECTOR_REG_WIDTH);
            MicroAPI::LoadAlign((MicroAPI::RegTensor<uint8_t> &)wLoad,
                                (__ubuf__ uint8_t *&)mxA8W4NzParams.weightLowBitPhyAddr, aregWeightB8In);

            MicroAPI::ShiftRight(wOdd, wLoad, wShrReg, preg);
            MicroAPI::And(wOdd, wOdd, wAndReg, preg);
            MicroAPI::ShiftLeft(wShl, wLoad, wShlReg, preg);
            MicroAPI::ShiftRight(wEven, wShl, wShrReg, preg);
            MicroAPI::And(wEven, wEven, wAndReg, preg);
            MicroAPI::Interleave(wD0, wD1, (MicroAPI::RegTensor<uint8_t> &)wEven,
                                 (MicroAPI::RegTensor<uint8_t> &)wOdd);

            MicroAPI::StoreAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                 MicroAPI::StoreDist::DIST_NORM_B8>(weightHighBitPtr, wD0,
                                                                    mxA8W4NzParams.innerDstStride, preg);
            MicroAPI::StoreAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                 MicroAPI::StoreDist::DIST_NORM_B8>(weightHighBitPtr, wD1,
                                                                    mxA8W4NzParams.innerDstStride, preg);
        }
    }
}

// Stock (verbatim) production NZ widen — kept for historical rollback evidence.
// Not called in the shipping datapath.
template <typename xType, typename wType>
__simd_vf__ inline void GmmsqAntiQuantMxA8W4NzNkVfStock(MxA8W4NzParams<xType, wType> mxA8W4NzParams)
{
    MicroAPI::RegTensor<int8_t> wShrReg, wShlReg, wAndReg, wLoad, wShl, wShr0, wShr1, wSel, wAnd;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregVsel = MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();

    MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wShrReg, E2M1_SHIFT_RIGHT_SIZE, preg);
    MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wShlReg, SHIFT_LEFT_SIZE, preg);
    MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wAndReg, E2M1_AND_MASK, preg);

    for (uint16_t loopKIdx = 0; loopKIdx < mxA8W4NzParams.loopKNum; ++loopKIdx) {
        for (uint16_t innerLoopIdx = 0; innerLoopIdx < mxA8W4NzParams.innerLoopNum; ++innerLoopIdx) {
            MicroAPI::AddrReg aregWeightB8In = MicroAPI::CreateAddrReg<uint8_t>(
                loopKIdx, (C0_SIZE_B8 * mxA8W4NzParams.nRealSizeAlign) >> 1, innerLoopIdx, VECTOR_REG_WIDTH >> 1);
            MicroAPI::LoadAlign<uint8_t, MicroAPI::LoadDist::DIST_US_B8>(
                (MicroAPI::RegTensor<uint8_t> &)wLoad, (__ubuf__ uint8_t *&)mxA8W4NzParams.weightLowBitPhyAddr,
                aregWeightB8In);

            MicroAPI::ShiftRight(wShr0, wLoad, wShrReg, preg);
            MicroAPI::ShiftLeft(wShl, wLoad, wShlReg, preg);
            MicroAPI::ShiftRight(wShr1, wShl, wShrReg, preg);
            MicroAPI::Select(wSel, wShr1, wShr0, pregVsel);
            MicroAPI::And(wAnd, wSel, wAndReg, preg);

            MicroAPI::AddrReg aregWeightB8Out = MicroAPI::CreateAddrReg<uint8_t>(
                loopKIdx, mxA8W4NzParams.loopKDstStride, innerLoopIdx, mxA8W4NzParams.innerDstStride);
            MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_NORM_B8>(
                (__ubuf__ uint8_t *&)mxA8W4NzParams.weightHighBitPhyAddr, (MicroAPI::RegTensor<uint8_t> &)wAnd,
                aregWeightB8Out, preg);
        }
    }
}

} // namespace GMMSQWeightQuant

#endif // GMMSQ_QUANT_MX_VF_H
