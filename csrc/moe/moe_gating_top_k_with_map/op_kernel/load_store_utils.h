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
 * \file load_store_utils.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_ASCENDC_LOAD_STORE_UTILS_H_
#define OPS_BUILT_IN_OP_ASCENDC_LOAD_STORE_UTILS_H_

#if ASC_DEVKIT_MAJOR >=9
#include "basic_api/kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace ops {
using namespace AscendC;
constexpr AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait castTraitB322B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::MicroAPI::CastTrait castTraitB322Int32 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC,
};

constexpr AscendC::MicroAPI::CastTrait castTraitB322Int16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC,
};

constexpr AscendC::MicroAPI::CastTrait castTraitB162Int8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC,
};

// load 对齐的 bfloat16,float16,bfloat32类型的 input(ub中)数据到 float32类型的dst(寄存器)中
template <typename T>
__aicore__ inline void LoadOneTensorForDtypeT(__local_mem__ T *input, MicroAPI::RegTensor<float> &dst,
    MicroAPI::MaskReg &preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ half *)(input) + (offset)));
        Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xBf16;
        DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(xBf16,
                    ((__local_mem__ bfloat16_t *)(input) + (offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else {
        DataCopy(dst, ((__local_mem__ float *)(input) + (offset)));
    }
}

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
        DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__local_mem__ half *)(src1) + (src1Offset)));
        DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16R, ((__local_mem__ half *)(src2) + (src2Offset)));
        Cast<float, half, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, half, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xFp16Q;
        MicroAPI::RegTensor<bfloat16_t> xFp16R;
        DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__local_mem__ bfloat16_t *)(src1) + (src1Offset)));
        DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16R, ((__local_mem__ bfloat16_t *)(src2) + (src2Offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, bfloat16_t, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else {
        DataCopy(dst1, ((__local_mem__ float *)(src1) + (src1Offset)));
        DataCopy(dst2, ((__local_mem__ float *)(src2) + (src2Offset)));
    }
}

// store 对齐的float32类型的src(寄存器)数据到output(ub)中，output数据类型支持bfloat16,float16,bfloat32,int32_t,int16_t,int8_t,uint8_t
template <typename T>
__aicore__ inline void StoreOneTensorForDtypeT(__local_mem__ T *output, MicroAPI::RegTensor<float> &src,
    MicroAPI::MaskReg &preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> yFp16;
        Cast<half, float, castTraitB322B16>(yFp16, src, preg);
        DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>(((__local_mem__ half *)output + offset), yFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xBf16;
        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, src, preg);
        DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(((__local_mem__ bfloat16_t *)output + offset),
                xBf16, preg);
    } else if constexpr (IsSameType<T, int32_t>::value) {
        MicroAPI::RegTensor<int32_t> zInt32;
        MicroAPI::Cast<int32_t, float, castTraitB322Int32>(zInt32, src, preg);
        DataCopy<int32_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(((__local_mem__ int32_t *)output + offset), zInt32, preg);
    } else if constexpr (IsSameType<T, int16_t>::value) {
        MicroAPI::RegTensor<int16_t> zInt16;
        MicroAPI::Cast<int16_t, float, castTraitB322Int16>(zInt16, src, preg);
        DataCopy<int16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(((__local_mem__ int16_t *)output + offset), zInt16, preg);
    } else if constexpr (IsSameType<T, int8_t>::value) {
        MicroAPI::RegTensor<half> yFp16;
        MicroAPI::RegTensor<int8_t> zInt8;
        Cast<half, float, castTraitB322Int16>(yFp16, src, preg);
        Cast<int8_t, half, castTraitB162Int8>(zInt8, yFp16, preg);
        DataCopy<int8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(((__local_mem__ int8_t *)output + offset), zInt8, preg);
    } else if constexpr (IsSameType<T, uint8_t>::value) {
        MicroAPI::RegTensor<half> yFp16;
        MicroAPI::RegTensor<uint8_t> zUint8;
        Cast<half, float, castTraitB322Int16>(yFp16, src, preg);
        Cast<uint8_t, half, castTraitB162Int8>(zUint8, yFp16, preg);
        DataCopy<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(((__local_mem__ uint8_t *)output + offset), zUint8, preg);
    } else {
        DataCopy(((__local_mem__ float *)output + offset), src, preg);
    }
}

// load 非对齐的 bfloat16,float16,bfloat32类型的 input(ub中)数据到 float32类型的dst(寄存器)中
template <typename T>
__aicore__ inline void LoadUnAlignOneTensor(__local_mem__ T *&input, MicroAPI::RegTensor<float> &dst,
    MicroAPI::UnalignReg &uSrc, MicroAPI::MaskReg &preg, uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        MicroAPI::RegTensor<half> xFp16UnPack;
        DataCopyUnAlign(xFp16, uSrc, input, postUpdateStride);
        UnPack((MicroAPI::RegTensor<uint32_t> &)xFp16UnPack, (MicroAPI::RegTensor<uint16_t> &)xFp16);
        Cast<float, half, castTraitB162B32>(dst, xFp16UnPack, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xBf16;
        MicroAPI::RegTensor<bfloat16_t> xBf16UnPack;
        DataCopyUnAlign(xBf16, uSrc, input, postUpdateStride);
        UnPack((MicroAPI::RegTensor<uint32_t> &)xBf16UnPack, (MicroAPI::RegTensor<uint16_t> &)xBf16);
        Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16UnPack, preg);
    } else {
        DataCopyUnAlign(dst, uSrc, input, postUpdateStride);
    }
}

// store 非对齐的float32类型的src(寄存器)数据到output(ub)中，output数据类型支持bfloat16,float16,bfloat32
template <typename T>
__aicore__ inline void StoreUnAlignOneTensor(__local_mem__ T *&output, MicroAPI::RegTensor<float> &src,
    MicroAPI::UnalignReg &uValue, MicroAPI::MaskReg &preg, uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        MicroAPI::RegTensor<half> xFp16Pack;
        Cast<half, float, castTraitB322B16>(xFp16, src, preg);
        Pack((MicroAPI::RegTensor<uint16_t> &)xFp16Pack, (MicroAPI::RegTensor<uint32_t> &)xFp16);
        DataCopyUnAlign(output, xFp16Pack, uValue, postUpdateStride);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xBf16;
        MicroAPI::RegTensor<bfloat16_t> xBf16Pack;
        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, src, preg);
        Pack((MicroAPI::RegTensor<uint16_t> &)xBf16Pack, (MicroAPI::RegTensor<uint32_t> &)xBf16);
        DataCopyUnAlign(output, xBf16Pack, uValue, postUpdateStride);
    } else {
        DataCopyUnAlign(output, src, uValue, postUpdateStride);
    }
}

} // namespace ops

#endif  // OPS_BUILT_IN_OP_ASCENDC_LOAD_STORE_UTILS_H_
