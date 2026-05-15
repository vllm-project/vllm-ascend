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
 * \file vf_rope.h
 * \brief
 */

#ifndef VF_ROPE_H
#define VF_ROPE_H

#include "kernel_operator.h"
#include "../../compressor_comm.h"

using namespace AscendC;

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

// load 2个对齐的Tensor 到寄存器中
template <typename T>
__aicore__ inline void LoadTwoTensorForDtypeT(__local_mem__ T *src1, __local_mem__ T *src2,
                                                MicroAPI::RegTensor<float> &dst1, MicroAPI::RegTensor<float> &dst2,
                                                MicroAPI::MaskReg &dst1Preg, MicroAPI::MaskReg &dst2Preg,
                                                uint32_t src1Offset, uint32_t src2Offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16Q;
        MicroAPI::RegTensor<half> xFp16R;
        MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__local_mem__ half *)(src1) + (src1Offset)));
        MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16R, ((__local_mem__ half *)(src2) + (src2Offset)));
        Cast<float, half, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, half, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xFp16Q;
        MicroAPI::RegTensor<bfloat16_t> xFp16R;
        MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__local_mem__ bfloat16_t *)(src1) + (src1Offset)));
        MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16R, ((__local_mem__ bfloat16_t *)(src2) + (src2Offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, bfloat16_t, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else {
        MicroAPI::DataCopy(dst1, ((__local_mem__ float *)(src1) + (src1Offset)));
        MicroAPI::DataCopy(dst2, ((__local_mem__ float *)(src2) + (src2Offset)));
    }
}

// load 对齐的 bfloat16,float16,bfloat32类型的 input(ub中)数据到 float32类型的dst(寄存器)中
template <typename T>
__aicore__ inline void LoadOneTensorForDtypeT(__local_mem__ T *input, MicroAPI::RegTensor<float> &dst,
    MicroAPI::MaskReg &preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ half *)(input) + (offset)));
        MicroAPI::Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xBf16;
        MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(xBf16,
                    ((__local_mem__ bfloat16_t *)(input) + (offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else {
        MicroAPI::DataCopy(dst, ((__local_mem__ float *)(input) + (offset)));
    }
}

// store 对齐的float32类型的src(寄存器)数据到output(ub)中，output数据类型支持bfloat16,float16,bfloat32,int32_t,int16_t,int8_t,uint8_t
template <typename T>
__aicore__ inline void StoreOneTensorForDtypeT(__local_mem__ T *output, MicroAPI::RegTensor<float> &src,
    MicroAPI::MaskReg &preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> yFp16;
        MicroAPI::Cast<half, float, castTraitB322B16>(yFp16, src, preg);
        MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>(((__local_mem__ half *)output + offset), yFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xBf16;
        MicroAPI::Cast<bfloat16_t, float, castTraitB322B16>(xBf16, src, preg);
        MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(((__local_mem__ bfloat16_t *)output + offset),
                xBf16, preg);
    } 
}

template <typename T, typename ROPET>
__aicore__ inline void HalfAlignVF(
    const LocalTensor<ROPET>& sinTensor, const LocalTensor<ROPET>& cosTensor, const LocalTensor<T>& inTensor,
    const LocalTensor<ROPET>& outTensor, uint32_t dLen, uint16_t currSNum, uint16_t currDNum)
{
    __local_mem__ ROPET* sinUb = (__local_mem__ ROPET*)sinTensor.GetPhyAddr();
    __local_mem__ ROPET* cosUb = (__local_mem__ ROPET*)cosTensor.GetPhyAddr();
    __local_mem__ T* inUb = (__local_mem__ T*)inTensor.GetPhyAddr();
    __local_mem__ ROPET* outUb = (__local_mem__ ROPET*)outTensor.GetPhyAddr();
    uint32_t halfD = dLen / HALF_INTERLEAVE_COEF;

    uint32_t dAlign = Compressor::Align(dLen, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint32_t halfDAlign = Compressor::Align(halfD, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint16_t repeatTimes = Compressor::CeilDivT(halfD, VL_FLOAT32_SIZE);
    __local_mem__ T* currInUb;
    __local_mem__ ROPET* currOutUb;
    __local_mem__ ROPET* currSinUb;
    __local_mem__ ROPET* currCosUb;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregIn;
        MicroAPI::RegTensor<float> vregHalfIn;
        MicroAPI::RegTensor<float> vregSin;
        MicroAPI::RegTensor<float> vregHalfSin;
        MicroAPI::RegTensor<float> vregCos;
        MicroAPI::RegTensor<float> vregHalfCos;
        MicroAPI::RegTensor<float> vregOut;
        MicroAPI::RegTensor<float> vregHalfOut;
        MicroAPI::MaskReg preg;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * dAlign;
            currCosUb = cosUb + sIdx * dAlign;
            for (uint16_t row = 0; row < currDNum; row++) {
                currInUb = inUb + (sIdx * currDNum + row) * dAlign;
                currOutUb = outUb + (sIdx * currDNum + row) * dAlign;
                uint32_t updateCnt = halfD;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = MicroAPI::UpdateMask<float>(updateCnt);
                    uint32_t offset = i * VL_FLOAT32_SIZE;
                    uint32_t halfOffset = offset + halfDAlign;
                    LoadTwoTensorForDtypeT<T>(
                        currInUb, currInUb, vregIn, vregHalfIn, preg, preg, offset, halfOffset);
                    LoadTwoTensorForDtypeT<ROPET>(
                        currSinUb, currSinUb, vregSin, vregHalfSin, preg, preg, offset, halfOffset);
                    LoadTwoTensorForDtypeT<ROPET>(
                        currCosUb, currCosUb, vregCos, vregHalfCos, preg, preg, offset, halfOffset);

                    MicroAPI::Mul(vregSin, vregSin, vregHalfIn, preg);
                    MicroAPI::Mul(vregHalfOut, vregHalfSin, vregIn, preg);
                    MicroAPI::Mul(vregCos, vregCos, vregIn, preg);
                    MicroAPI::Sub(vregOut, vregCos, vregSin, preg);
                    MicroAPI::Mul(vregHalfCos, vregHalfCos, vregHalfIn, preg);
                    MicroAPI::Add(vregHalfOut, vregHalfOut, vregHalfCos, preg);

                    StoreOneTensorForDtypeT<ROPET>(currOutUb, vregOut, preg, offset);
                    StoreOneTensorForDtypeT<ROPET>(currOutUb, vregHalfOut, preg, halfOffset);
                }
            }
        }
    }
}

template <typename T, typename ROPET>
__aicore__ inline void InterleaveModeVF(
    const LocalTensor<ROPET>& sinTensor, const LocalTensor<ROPET>& cosTensor, const LocalTensor<T>& inTensor,
    const LocalTensor<ROPET>& outTensor, uint32_t dLen, uint16_t currSNum, uint16_t currDNum)
{
    __local_mem__ ROPET* sinUb = (__local_mem__ ROPET*)sinTensor.GetPhyAddr();
    __local_mem__ ROPET* cosUb = (__local_mem__ ROPET*)cosTensor.GetPhyAddr();
    __local_mem__ T* inUb = (__local_mem__ T*)inTensor.GetPhyAddr();
    __local_mem__ ROPET* outUb = (__local_mem__ ROPET*)outTensor.GetPhyAddr();
    uint16_t repeatTimes = dLen / VL_FLOAT32_SIZE;//(每个寄存器256字节，最多存放数据256/4=64element，因此需要使用寄存器的数量)
    uint32_t dAlignLen = Compressor::Align(dLen, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    //Dlen(8对齐、32/4)RD=64
    uint16_t loopNum = repeatTimes / 2;//(开两个寄存器)
    uint32_t tailNum = dLen - loopNum * 2 * VL_FLOAT32_SIZE;//(尾块数据数)
    uint16_t tailTwoVL = tailNum / VL_FLOAT32_SIZE;
    uint16_t tailOneVL = (tailTwoVL == 1 && tailNum > 0) ? 0 : 1;
    uint32_t tailLen = tailNum % VL_FLOAT32_SIZE;
    __local_mem__ T* currInUb;
    __local_mem__ ROPET* currOutUb;
    __local_mem__ ROPET* currSinUb;
    __local_mem__ ROPET* currCosUb;
    __local_mem__ ROPET* tailSinUb;
    __local_mem__ ROPET* tailCosUb;

    __VEC_SCOPE__
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
        MicroAPI::MaskReg pregLoop;
        MicroAPI::MaskReg pregTail;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {//sc轴
            currSinUb = sinUb + sIdx * dAlignLen;
            currCosUb = cosUb + sIdx * dAlignLen;
            for (uint16_t idxD = 0; idxD < currDNum; idxD++) {//D轴
                uint32_t updateCnt = dLen;
                currInUb = inUb + (sIdx * currDNum + idxD) * dAlignLen;//将后64个element取出，因此对于SC轴上的每行数据，
                currOutUb = outUb + (sIdx * currDNum + idxD) * dAlignLen;//将数据64个element输出到sc轴上每行的最后64位上，
                pregLoop = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
                for (uint16_t i = 0; i < loopNum; i++) {//循环次数
                    uint32_t evenOffSet = (i * 2) * VL_FLOAT32_SIZE;//前64个element
                    uint32_t oddOffset = evenOffSet + VL_FLOAT32_SIZE;//后64个element
                    //数据拷贝，两个寄存器分别拷贝
                    LoadOneTensorForDtypeT<T>(currInUb, vregFormerIn, pregLoop, evenOffSet);
                    LoadOneTensorForDtypeT<T>(currInUb, vregLatterIn, pregLoop, oddOffset);
                    LoadOneTensorForDtypeT<ROPET>(currCosUb, vregFormerCos, pregLoop, evenOffSet);
                    LoadOneTensorForDtypeT<ROPET>(currCosUb, vregLatterCos, pregLoop, oddOffset);
                    LoadOneTensorForDtypeT<ROPET>(currSinUb, vregFormerSin, pregLoop, evenOffSet);
                    LoadOneTensorForDtypeT<ROPET>(currSinUb, vregLatterSin, pregLoop, oddOffset);
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
                    StoreOneTensorForDtypeT<ROPET>(currOutUb, vregFormerCos, pregLoop, evenOffSet);//前64个元素
                    StoreOneTensorForDtypeT<ROPET>(currOutUb, vregLatterCos, pregLoop, oddOffset);//后64个元素
                }

                currInUb = inUb + (sIdx * currDNum + idxD) * dAlignLen + (loopNum * 2 * VL_FLOAT32_SIZE);
                currOutUb = outUb + (sIdx * currDNum + idxD) * dAlignLen + (loopNum * 2 * VL_FLOAT32_SIZE);
                tailSinUb = currSinUb + loopNum * 2 * VL_FLOAT32_SIZE;
                tailCosUb = currCosUb + loopNum * 2 * VL_FLOAT32_SIZE;
                // 尾块大于VL时,读取一个VL，读取尾块
                for (uint16_t i = 0; i < tailTwoVL; i++) {
                    uint32_t updateCnt = tailLen;
                    pregTail = MicroAPI::UpdateMask<float>(updateCnt);
                    LoadOneTensorForDtypeT<T>(currInUb, vregFormerIn, pregLoop, 0);
                    LoadOneTensorForDtypeT<T>(currInUb, vregLatterIn, pregTail, VL_FLOAT32_SIZE);
                    LoadOneTensorForDtypeT<ROPET>(tailCosUb, vregFormerCos, pregLoop, 0);
                    LoadOneTensorForDtypeT<ROPET>(tailCosUb, vregLatterCos, pregTail, VL_FLOAT32_SIZE);
                    LoadOneTensorForDtypeT<ROPET>(tailSinUb, vregFormerSin, pregLoop, 0);
                    LoadOneTensorForDtypeT<ROPET>(tailSinUb, vregLatterSin, pregTail, VL_FLOAT32_SIZE);
                    MicroAPI::Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregLoop);
                    MicroAPI::Mul(vregLatterCos, vregLatterCos, vregLatterIn, pregTail);
                    MicroAPI::DeInterleave<float>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                    MicroAPI::Muls(vregOdd, vregOdd, float(-1.0), pregLoop);
                    MicroAPI::Interleave<float>(vregFormerIn, vregLatterIn, vregOdd, vregEven);
                    MicroAPI::Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregLoop);
                    MicroAPI::Add(vregFormerCos, vregFormerCos, vregFormerSin, pregLoop);
                    MicroAPI::Mul(vregLatterSin, vregLatterSin, vregLatterIn, pregTail);
                    MicroAPI::Add(vregLatterCos, vregLatterCos, vregLatterSin, pregTail);
                    StoreOneTensorForDtypeT<ROPET>(currOutUb, vregFormerCos, pregLoop, 0);
                    StoreOneTensorForDtypeT<ROPET>(currOutUb, vregLatterCos, pregTail, VL_FLOAT32_SIZE);
                }

                // 尾块小于VL时,只读取VL
                for (uint16_t i = 0; i < tailOneVL ; i++) {
                    uint32_t updateCnt = tailLen;
                    pregTail = MicroAPI::UpdateMask<float>(updateCnt);
                    LoadOneTensorForDtypeT<T>(currInUb, vregFormerIn, pregTail, 0);
                    LoadOneTensorForDtypeT<ROPET>(tailCosUb, vregFormerCos, pregTail, 0);
                    LoadOneTensorForDtypeT<ROPET>(tailSinUb, vregFormerSin, pregTail, 0);
                    MicroAPI::Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregTail);
                    MicroAPI::DeInterleave<float>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                    MicroAPI::Muls(vregOdd, vregOdd, float(-1.0), pregTail);
                    MicroAPI::Interleave<float>(vregFormerIn, vregLatterIn, vregOdd, vregEven);
                    MicroAPI::Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregTail);
                    MicroAPI::Add(vregFormerCos, vregFormerCos, vregFormerSin, pregTail);
                    StoreOneTensorForDtypeT<ROPET>(currOutUb, vregFormerCos, pregTail, 0);
                }
            }
        }
    }
}

#endif