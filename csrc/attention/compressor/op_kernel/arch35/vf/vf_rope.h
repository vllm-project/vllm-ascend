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
 * \file vf_rope.h
 * \brief
 */

#ifndef VF_ROPE_H
#define VF_ROPE_H

#include "kernel_operator.h"
#include "../../compressor_comm.h"

using namespace AscendC;
struct RopeParam{
    uint32_t dLen;
    uint32_t dAlign;
    uint16_t repeatTimes;
    uint16_t currSNum;
    uint16_t currDNum;
};
struct TailParam{
    uint16_t tailTwoVL;
    uint16_t tailOneVL;
    uint16_t tailLen;
};

__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

__aicore__ inline constexpr uint32_t GetUbBlockSize()
{
    return 32U;
}

constexpr MicroAPI::CastTrait castTraitB162B32 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr MicroAPI::CastTrait castTraitB322B16 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

constexpr uint32_t VL_FLOAT32_SIZE = GetVRegSize() / sizeof(float);
constexpr uint32_t BLOCK_TYPE_SIZE = GetUbBlockSize();
constexpr uint32_t HALF_INTERLEAVE_COEF = 2;

template <typename T, typename ROPET>
__simd_vf__ void HalfAlignVF(
    __ubuf__ ROPET * sinUb, __ubuf__ ROPET * cosUb, __ubuf__ T * inUb, __ubuf__ ROPET * outUb, uint32_t halfD, uint32_t halfDAlign, const RopeParam ropeParam)
{
    MicroAPI::RegTensor<float> vregIn;
    MicroAPI::RegTensor<float> vregHalfIn;
    MicroAPI::RegTensor<float> vregSin;
    MicroAPI::RegTensor<float> vregHalfSin;
    MicroAPI::RegTensor<float> vregCos;
    MicroAPI::RegTensor<float> vregHalfCos;
    MicroAPI::RegTensor<float> vregOut;
    MicroAPI::RegTensor<float> vregHalfOut;
    MicroAPI::RegTensor<ROPET> sinFp16Q;
    MicroAPI::RegTensor<ROPET> sinFp16R;
    MicroAPI::RegTensor<ROPET> cosFp16Q;
    MicroAPI::RegTensor<ROPET> cosFp16R;
    MicroAPI::RegTensor<ROPET> yBf16Q;
    MicroAPI::RegTensor<ROPET> yBf16R;
    MicroAPI::MaskReg preg;

    __ubuf__ ROPET * currSinUb;
    __ubuf__ ROPET * currCosUb;
    __ubuf__ T * currInUb;
    __ubuf__ ROPET * currOutUb;
    for (uint16_t sIdx = 0; sIdx < ropeParam.currSNum; sIdx++) {
        currSinUb = sinUb + sIdx * ropeParam.dAlign;
        currCosUb = cosUb + sIdx * ropeParam.dAlign;
        for (uint16_t row = 0; row < ropeParam.currDNum; row++) {
            currInUb = inUb + (sIdx * ropeParam.currDNum + row) * ropeParam.dAlign;
            currOutUb = outUb + (sIdx * ropeParam.currDNum + row) * ropeParam.dAlign;
            uint32_t updateCnt = halfD;
            for (uint16_t i = 0; i < ropeParam.repeatTimes; i++) {
                preg = MicroAPI::UpdateMask<float>(updateCnt);
                MicroAPI::DataCopy(vregIn, currInUb + (i * VL_FLOAT32_SIZE));
                MicroAPI::DataCopy(vregHalfIn, (currInUb + (i * VL_FLOAT32_SIZE + halfDAlign)));
                MicroAPI::DataCopy<ROPET, MicroAPI::LoadDist::DIST_UNPACK_B16>(sinFp16Q, currSinUb + (i * VL_FLOAT32_SIZE));
                MicroAPI::DataCopy<ROPET, MicroAPI::LoadDist::DIST_UNPACK_B16>(sinFp16R, currSinUb + (i * VL_FLOAT32_SIZE + halfDAlign));
                MicroAPI::DataCopy<ROPET, MicroAPI::LoadDist::DIST_UNPACK_B16>(cosFp16Q, currCosUb + (i * VL_FLOAT32_SIZE));
                MicroAPI::DataCopy<ROPET, MicroAPI::LoadDist::DIST_UNPACK_B16>(cosFp16R, currCosUb + (i * VL_FLOAT32_SIZE + halfDAlign));
                Cast<float, ROPET, castTraitB162B32>(vregSin, sinFp16Q, preg);
                Cast<float, ROPET, castTraitB162B32>(vregHalfSin, sinFp16R, preg);
                Cast<float, ROPET, castTraitB162B32>(vregCos, cosFp16Q, preg);
                Cast<float, ROPET, castTraitB162B32>(vregHalfCos, cosFp16R, preg);


                MicroAPI::Mul(vregSin, vregSin, vregHalfIn, preg);
                MicroAPI::Mul(vregHalfOut, vregHalfSin, vregIn, preg);
                MicroAPI::Mul(vregCos, vregCos, vregIn, preg);
                MicroAPI::Sub(vregOut, vregCos, vregSin, preg);
                MicroAPI::Mul(vregHalfCos, vregHalfCos, vregHalfIn, preg);
                MicroAPI::Add(vregHalfOut, vregHalfOut, vregHalfCos, preg);

                MicroAPI::Cast<ROPET, float, castTraitB322B16>(yBf16Q, vregOut, preg);
                MicroAPI::DataCopy<ROPET, MicroAPI::StoreDist::DIST_PACK_B32>(currOutUb + (i * VL_FLOAT32_SIZE), yBf16Q, preg);
                MicroAPI::Cast<ROPET, float, castTraitB322B16>(yBf16R, vregHalfOut, preg);
                MicroAPI::DataCopy<ROPET, MicroAPI::StoreDist::DIST_PACK_B32>(currOutUb + (i * VL_FLOAT32_SIZE + halfDAlign), yBf16R, preg);
            }
        }
    }
}

template <typename T, typename ROPET>
__simd_vf__ void InterleaveModeVF(
    __ubuf__ ROPET * sinUb, __ubuf__ ROPET * cosUb, __ubuf__ T * inUb, __ubuf__ ROPET * outUb, const RopeParam ropeParam, const TailParam tailParam, uint16_t loopNum)
{
    MicroAPI::RegTensor<float> vregFormerCos;
    MicroAPI::RegTensor<float> vregLatterCos;
    MicroAPI::RegTensor<float> vregFormerSin;
    MicroAPI::RegTensor<float> vregLatterSin;
    MicroAPI::RegTensor<float> vregFormerIn;
    MicroAPI::RegTensor<float> vregLatterIn;
    MicroAPI::RegTensor<float> vregOdd;
    MicroAPI::RegTensor<float> vregEven;
    MicroAPI::RegTensor<float> vregFormerOut;
    MicroAPI::RegTensor<float> vregLatterOut;
    MicroAPI::RegTensor<ROPET> sinFp16Q;
    MicroAPI::RegTensor<ROPET> sinFp16R;
    MicroAPI::RegTensor<ROPET> cosFp16Q;
    MicroAPI::RegTensor<ROPET> cosFp16R;
    MicroAPI::RegTensor<ROPET> yBf16Q;
    MicroAPI::RegTensor<ROPET> yBf16R;
    MicroAPI::MaskReg pregLoop;
    MicroAPI::MaskReg pregTail;
    pregLoop = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    __ubuf__ ROPET * currSinUb;
    __ubuf__ ROPET * currCosUb;
    __ubuf__ T * currInUb;
    __ubuf__ ROPET * currOutUb;
    __ubuf__ ROPET* tailSinUb;
    __ubuf__ ROPET* tailCosUb;
    for (uint16_t sIdx = 0; sIdx < ropeParam.currSNum; sIdx++) {//sc轴
        currSinUb = sinUb + sIdx * ropeParam.dAlign;
        currCosUb = cosUb + sIdx * ropeParam.dAlign;
        for (uint16_t idxD = 0; idxD < ropeParam.currDNum; idxD++) {//D轴
            currInUb = inUb + (sIdx * ropeParam.currDNum + idxD) * ropeParam.dAlign;//将后64个element取出，因此对于SC轴上的每行数据，
            currOutUb = outUb + (sIdx * ropeParam.currDNum + idxD) * ropeParam.dAlign;//将数据64个element输出到sc轴上每行的最后64位上，
            for (uint16_t i = 0; i < loopNum; i++) {//循环次数
                //数据拷贝，两个寄存器分别拷贝
                MicroAPI::DataCopy(vregFormerIn, (currInUb + (i * 2) * VL_FLOAT32_SIZE));
                MicroAPI::DataCopy(vregLatterIn, (currInUb + (i * 2 + 1) * VL_FLOAT32_SIZE));
                MicroAPI::DataCopy<ROPET, MicroAPI::LoadDist::DIST_UNPACK_B16>(sinFp16Q, (currSinUb + (i * 2) * VL_FLOAT32_SIZE));
                MicroAPI::DataCopy<ROPET, MicroAPI::LoadDist::DIST_UNPACK_B16>(sinFp16R, (currSinUb + (i * 2 + 1) * VL_FLOAT32_SIZE));
                MicroAPI::DataCopy<ROPET, MicroAPI::LoadDist::DIST_UNPACK_B16>(cosFp16Q, (currCosUb + (i * 2) * VL_FLOAT32_SIZE));
                MicroAPI::DataCopy<ROPET, MicroAPI::LoadDist::DIST_UNPACK_B16>(cosFp16R, (currCosUb + (i * 2 + 1) * VL_FLOAT32_SIZE));

                MicroAPI::Cast<float, ROPET, castTraitB162B32>(vregFormerSin, sinFp16Q, pregLoop);
                MicroAPI::Cast<float, ROPET, castTraitB162B32>(vregLatterSin, sinFp16R, pregLoop);
                MicroAPI::Cast<float, ROPET, castTraitB162B32>(vregFormerCos, cosFp16Q, pregLoop);
                MicroAPI::Cast<float, ROPET, castTraitB162B32>(vregLatterCos, cosFp16R, pregLoop);

                MicroAPI::Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregLoop);
                MicroAPI::Mul(vregLatterCos, vregLatterCos, vregLatterIn, pregLoop);
                MicroAPI::DeInterleave<float>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                //解交织,将前64个和后64个数据的奇数位数据和偶数位数据分别拿出来，并进行拼接EVEN存放奇数位数据，ODD存放偶数位数据
                MicroAPI::Muls(vregOdd, vregOdd, float(-1.0), pregLoop);
                MicroAPI::Interleave<float>(vregFormerIn, vregLatterIn, vregOdd, vregEven);
                //交织生成与sin相乘的数据
                MicroAPI::Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregLoop);
                MicroAPI::Add(vregFormerCos, vregFormerCos, vregFormerSin, pregLoop);
                MicroAPI::Mul(vregLatterSin, vregLatterSin, vregLatterIn, pregLoop);
                MicroAPI::Add(vregLatterCos, vregLatterCos, vregLatterSin, pregLoop);
                //拷贝输出数据
                MicroAPI::Cast<ROPET, float, castTraitB322B16>(yBf16Q, vregFormerCos, pregLoop);
                MicroAPI::DataCopy<ROPET, MicroAPI::StoreDist::DIST_PACK_B32>(currOutUb +  (i * 2) * VL_FLOAT32_SIZE, yBf16Q, pregLoop);
                MicroAPI::Cast<ROPET, float, castTraitB322B16>(yBf16R, vregLatterCos, pregLoop);
                MicroAPI::DataCopy<ROPET, MicroAPI::StoreDist::DIST_PACK_B32>(currOutUb +  (i * 2 + 1) * VL_FLOAT32_SIZE, yBf16R, pregLoop);
            }

            currInUb = inUb + (sIdx * ropeParam.currDNum + idxD) * ropeParam.dAlign + (loopNum * 2 * VL_FLOAT32_SIZE);
            currOutUb = outUb + (sIdx * ropeParam.currDNum + idxD) * ropeParam.dAlign + (loopNum * 2 * VL_FLOAT32_SIZE);
            tailSinUb = currSinUb + loopNum * 2 * VL_FLOAT32_SIZE;
            tailCosUb = currCosUb + loopNum * 2 * VL_FLOAT32_SIZE;
            // 尾块大于VL时,读取一个VL，读取尾块
            for (uint16_t i = 0; i < tailParam.tailTwoVL; i++) {
                uint32_t updateCnt = tailParam.tailLen;
                pregTail = MicroAPI::UpdateMask<float>(updateCnt);
                //搬入
                MicroAPI::DataCopy(vregFormerIn, currInUb);
                MicroAPI::DataCopy(vregLatterIn, currInUb + VL_FLOAT32_SIZE);
                MicroAPI::DataCopy<ROPET, MicroAPI::LoadDist::DIST_UNPACK_B16>(sinFp16Q, tailSinUb);
                MicroAPI::DataCopy<ROPET, MicroAPI::LoadDist::DIST_UNPACK_B16>(sinFp16R, tailSinUb + VL_FLOAT32_SIZE);
                MicroAPI::DataCopy<ROPET, MicroAPI::LoadDist::DIST_UNPACK_B16>(cosFp16Q, tailCosUb);
                MicroAPI::DataCopy<ROPET, MicroAPI::LoadDist::DIST_UNPACK_B16>(cosFp16R, tailCosUb + VL_FLOAT32_SIZE);
                MicroAPI::Cast<float, ROPET, castTraitB162B32>(vregFormerSin, sinFp16Q, pregLoop);
                MicroAPI::Cast<float, ROPET, castTraitB162B32>(vregLatterSin, sinFp16R, pregTail);
                MicroAPI::Cast<float, ROPET, castTraitB162B32>(vregFormerCos, cosFp16Q, pregLoop);
                MicroAPI::Cast<float, ROPET, castTraitB162B32>(vregLatterCos, cosFp16R, pregTail);
                //计算
                MicroAPI::Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregLoop);
                MicroAPI::Mul(vregLatterCos, vregLatterCos, vregLatterIn, pregTail);
                MicroAPI::DeInterleave<float>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                MicroAPI::Muls(vregOdd, vregOdd, float(-1.0), pregLoop);
                MicroAPI::Interleave<float>(vregFormerIn, vregLatterIn, vregOdd, vregEven);
                MicroAPI::Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregLoop);
                MicroAPI::Add(vregFormerCos, vregFormerCos, vregFormerSin, pregLoop);
                MicroAPI::Mul(vregLatterSin, vregLatterSin, vregLatterIn, pregTail);
                MicroAPI::Add(vregLatterCos, vregLatterCos, vregLatterSin, pregTail);
                //搬出
                MicroAPI::Cast<ROPET, float, castTraitB322B16>(yBf16Q, vregFormerCos, pregLoop);
                MicroAPI::Cast<ROPET, float, castTraitB322B16>(yBf16R, vregLatterCos, pregTail);
                MicroAPI::DataCopy<ROPET, MicroAPI::StoreDist::DIST_PACK_B32>(currOutUb, yBf16Q, pregLoop);
                MicroAPI::DataCopy<ROPET, MicroAPI::StoreDist::DIST_PACK_B32>(currOutUb +  VL_FLOAT32_SIZE, yBf16R, pregTail);
            }

            // 尾块小于VL时,只读取VL
            for (uint16_t i = 0; i < tailParam.tailOneVL ; i++) {
                uint32_t updateCnt = tailParam.tailLen;
                pregTail = MicroAPI::UpdateMask<float>(updateCnt);
                //搬入
                MicroAPI::DataCopy(vregFormerIn, currInUb);
                MicroAPI::DataCopy<ROPET, MicroAPI::LoadDist::DIST_UNPACK_B16>(sinFp16Q, tailSinUb);
                MicroAPI::DataCopy<ROPET, MicroAPI::LoadDist::DIST_UNPACK_B16>(cosFp16Q, tailCosUb);
                MicroAPI::Cast<float, ROPET, castTraitB162B32>(vregFormerSin, sinFp16Q, pregTail);
                MicroAPI::Cast<float, ROPET, castTraitB162B32>(vregFormerCos, cosFp16Q, pregTail);
                //计算
                MicroAPI::Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregTail);
                MicroAPI::DeInterleave<float>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                MicroAPI::Muls(vregOdd, vregOdd, float(-1.0), pregTail);
                MicroAPI::Interleave<float>(vregFormerIn, vregLatterIn, vregOdd, vregEven);
                MicroAPI::Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregTail);
                MicroAPI::Add(vregFormerCos, vregFormerCos, vregFormerSin, pregTail);
                //搬出
                MicroAPI::Cast<ROPET, float, castTraitB322B16>(yBf16Q, vregFormerCos, pregTail);
                MicroAPI::DataCopy<ROPET, MicroAPI::StoreDist::DIST_PACK_B32>(currOutUb, yBf16Q, pregTail);
            }
        }
    }
}



template <typename T, typename ROPET>
__aicore__ inline void RopeVF(const LocalTensor<ROPET>& sinTensor, const LocalTensor<ROPET>& cosTensor, const LocalTensor<T>& inTensor,
    const LocalTensor<ROPET>& outTensor, uint32_t dLen, uint16_t currSNum, uint16_t currDNum, bool isInterleave)
{
    __ubuf__ ROPET* sinUb = (__ubuf__ ROPET*)sinTensor.GetPhyAddr();
    __ubuf__ ROPET* cosUb = (__ubuf__ ROPET*)cosTensor.GetPhyAddr();
    __ubuf__ T* inUb = (__ubuf__ T*)inTensor.GetPhyAddr();
    __ubuf__ ROPET* outUb = (__ubuf__ ROPET*)outTensor.GetPhyAddr();

    RopeParam ropeParam;
    ropeParam.dLen = dLen;
    ropeParam.dAlign = Compressor::Align(dLen, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    ropeParam.currSNum = currSNum;
    ropeParam.currDNum = currDNum;
    if(isInterleave) {
        ropeParam.repeatTimes = dLen / VL_FLOAT32_SIZE;
        TailParam tailParam;
        uint16_t loopNum = ropeParam.repeatTimes / 2;//(开两个寄存器)
        uint32_t tailNum = dLen - loopNum * 2 * VL_FLOAT32_SIZE;//(尾块数据数)
        tailParam.tailTwoVL = tailNum / VL_FLOAT32_SIZE;
        tailParam.tailOneVL = (tailParam.tailTwoVL == 1 && tailNum > 0) ? 0 : 1;
        tailParam.tailLen = tailNum % VL_FLOAT32_SIZE;
        InterleaveModeVF(sinUb, cosUb, inUb, outUb, ropeParam, tailParam, loopNum);
    } else {
        uint32_t halfD = dLen / HALF_INTERLEAVE_COEF;
        ropeParam.repeatTimes = Compressor::CeilDivT(halfD, VL_FLOAT32_SIZE);
        uint32_t halfDAlign = Compressor::Align(halfD, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
        HalfAlignVF(sinUb, cosUb, inUb, outUb, halfD, halfDAlign, ropeParam);
    }

}
#endif