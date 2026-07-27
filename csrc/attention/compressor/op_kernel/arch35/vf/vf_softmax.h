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
 * \file vf_softmax.h
 * \brief
 */

#ifndef VF_SOFTMAX_H
#define VF_SOFTMAX_H
#include "kernel_tensor.h"
namespace FaVectorApi {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

template <typename T>
__simd_vf__ inline void SoftmaxDndBase128(__ubuf__ T *inputAddr, __ubuf__ float *outputAddr,
    const uint32_t RowSize, const uint32_t ReduceSize, const uint32_t vScRealSize,
    const T minValue)
{
    RegTensor<float> vregSum00;
    RegTensor<float> vregSum10;
    RegTensor<float> vregSum20;
    RegTensor<float> vregSum30;
    RegTensor<float> vregSum01;
    RegTensor<float> vregSum11;
    RegTensor<float> vregSum21;
    RegTensor<float> vregSum31;
    RegTensor<float> vregSum02;
    RegTensor<float> vregSum12;
    RegTensor<float> vregSum22;
    RegTensor<float> vregSum32;
    RegTensor<float> vregSum03;
    RegTensor<float> vregSum13;
    RegTensor<float> vregSum23;
    RegTensor<float> vregSum33;

    RegTensor<float> vregExp00;
    RegTensor<float> vregExp10;
    RegTensor<float> vregExp20;
    RegTensor<float> vregExp30;
    RegTensor<float> vregExp01;
    RegTensor<float> vregExp11;
    RegTensor<float> vregExp21;
    RegTensor<float> vregExp31;
    RegTensor<float> vregExp02;
    RegTensor<float> vregExp12;
    RegTensor<float> vregExp22;
    RegTensor<float> vregExp32;
    RegTensor<float> vregExp03;
    RegTensor<float> vregExp13;
    RegTensor<float> vregExp23;
    RegTensor<float> vregExp33;

    RegTensor<float> vregF32_00;
    RegTensor<float> vregF32_10;
    RegTensor<float> vregF32_20;
    RegTensor<float> vregF32_30;
    RegTensor<float> vregF32_01;
    RegTensor<float> vregF32_11;
    RegTensor<float> vregF32_21;
    RegTensor<float> vregF32_31;
    RegTensor<float> vregF32_02;
    RegTensor<float> vregF32_12;
    RegTensor<float> vregF32_22;
    RegTensor<float> vregF32_32;
    RegTensor<float> vregF32_03;
    RegTensor<float> vregF32_13;
    RegTensor<float> vregF32_23;
    RegTensor<float> vregF32_33;

    RegTensor<float> vregStore00;
    RegTensor<float> vregStore10;
    RegTensor<float> vregStore20;
    RegTensor<float> vregStore30;
    RegTensor<float> vregStore01;
    RegTensor<float> vregStore11;
    RegTensor<float> vregStore21;
    RegTensor<float> vregStore31;
    MaskReg pregAll;
    pregAll = CreateMask<T, MaskPattern::ALL>();
    RegTensor<float> src00, src10, src20, src30, src01, src11, src21, src31,
        src02, src12, src22, src32, src03, src13, src23, src33;
    RegTensor<float> max00, max10, max20, max30, max01, max11, max21, max31,
        max02, max12, max22, max32, max03, max13, max23, max33;

    __ubuf__ float *srcUb00 = outputAddr;
    __ubuf__ float *srcUb01 = outputAddr + RowSize / 2;
    __ubuf__ float *srcUb02 = outputAddr + RowSize;
    __ubuf__ float *srcUb03 = outputAddr + RowSize + RowSize / 2;
    __ubuf__ float *srcUb10 = srcUb00 + ReduceSize * RowSize;
    __ubuf__ float *srcUb11 = srcUb00 + ReduceSize * RowSize + RowSize / 2;
    __ubuf__ float *srcUb12 = srcUb00 + ReduceSize * RowSize + RowSize;
    __ubuf__ float *srcUb13 = srcUb00 + ReduceSize * RowSize + RowSize + RowSize / 2;
    __ubuf__ float *srcUb20 = srcUb00 + ReduceSize * RowSize * 2;
    __ubuf__ float *srcUb21 = srcUb00 + ReduceSize * RowSize * 2 + RowSize / 2;
    __ubuf__ float *srcUb22 = srcUb00 + ReduceSize * RowSize * 2 + RowSize;
    __ubuf__ float *srcUb23 = srcUb00 + ReduceSize * RowSize * 2 + RowSize + RowSize / 2;
    __ubuf__ float *srcUb30 = srcUb00 + ReduceSize * RowSize * 3;
    __ubuf__ float *srcUb31 = srcUb00 + ReduceSize * RowSize * 3 + RowSize / 2;
    __ubuf__ float *srcUb32 = srcUb00 + ReduceSize * RowSize * 3 + RowSize;
    __ubuf__ float *srcUb33 = srcUb00 + ReduceSize * RowSize * 3 + RowSize + RowSize / 2;

    __ubuf__ float *inputAddr00 = inputAddr;
    __ubuf__ float *inputAddr01 = inputAddr + RowSize / 2;
    __ubuf__ float *inputAddr10 = inputAddr + (ReduceSize * RowSize);
    __ubuf__ float *inputAddr11 = inputAddr + (ReduceSize * RowSize) + RowSize / 2;
    __ubuf__ float *inputAddr20 = inputAddr + (ReduceSize * RowSize * 2);
    __ubuf__ float *inputAddr21 = inputAddr + (ReduceSize * RowSize * 2) + RowSize / 2;
    __ubuf__ float *inputAddr30 = inputAddr + (ReduceSize * RowSize * 3);
    __ubuf__ float *inputAddr31 = inputAddr + (ReduceSize * RowSize * 3) + RowSize / 2;

    for (uint16_t loopSc = 0; loopSc < uint16_t(vScRealSize / 4); ++loopSc) {
        Duplicate(max00, minValue);
        Duplicate(max10, minValue);
        Duplicate(max20, minValue);
        Duplicate(max30, minValue);
        Duplicate(max01, minValue);
        Duplicate(max11, minValue);
        Duplicate(max21, minValue);
        Duplicate(max31, minValue);
        Duplicate(max02, minValue);
        Duplicate(max12, minValue);
        Duplicate(max22, minValue);
        Duplicate(max32, minValue);
        Duplicate(max03, minValue);
        Duplicate(max13, minValue);
        Duplicate(max23, minValue);
        Duplicate(max33, minValue);

        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum00, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum10, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum20, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum30, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum01, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum11, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum21, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum31, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum02, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum12, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum22, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum32, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum03, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum13, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum23, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum33, 0, pregAll);
        for (uint16_t loopM = 0; loopM < uint16_t(ReduceSize / 2); ++loopM) {
            LoadAlign(src00, srcUb00 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src01, srcUb01 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src02, srcUb02 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src03, srcUb03 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(src10, srcUb10 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src11, srcUb11 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src12, srcUb12 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src13, srcUb13 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(src20, srcUb20 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src21, srcUb21 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src22, srcUb22 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src23, srcUb23 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(src30, srcUb30 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src31, srcUb31 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src32, srcUb32 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src33, srcUb33 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);

            Max(max00, max00, src00, pregAll);
            Max(max01, max01, src01, pregAll);
            Max(max02, max02, src02, pregAll);
            Max(max03, max03, src03, pregAll);
            Max(max10, max10, src10, pregAll);
            Max(max11, max11, src11, pregAll);
            Max(max12, max12, src12, pregAll);
            Max(max13, max13, src13, pregAll);
            Max(max20, max20, src20, pregAll);
            Max(max21, max21, src21, pregAll);
            Max(max22, max22, src22, pregAll);
            Max(max23, max23, src23, pregAll);
            Max(max30, max30, src30, pregAll);
            Max(max31, max31, src31, pregAll);
            Max(max32, max32, src32, pregAll);
            Max(max33, max33, src33, pregAll);
        }
        Max(max00, max00, max02, pregAll);
        Max(max01, max01, max03, pregAll);
        Max(max10, max10, max12, pregAll);
        Max(max11, max11, max13, pregAll);
        Max(max20, max20, max22, pregAll);
        Max(max21, max21, max23, pregAll);
        Max(max30, max30, max32, pregAll);
        Max(max31, max31, max33, pregAll);

        for (uint16_t loopM = 0; loopM < uint16_t(ReduceSize / 2); ++loopM) {
            LoadAlign(vregF32_00, srcUb00 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_01, srcUb01 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_02, srcUb02 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_03, srcUb03 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(vregF32_10, srcUb10 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_11, srcUb11 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_12, srcUb12 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_13, srcUb13 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(vregF32_20, srcUb20 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_21, srcUb21 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_22, srcUb22 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_23, srcUb23 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(vregF32_30, srcUb30 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_31, srcUb31 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_32, srcUb32 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_33, srcUb33 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);

            FusedExpSub(vregExp00, vregF32_00, max00, pregAll);
            FusedExpSub(vregExp01, vregF32_01, max01, pregAll);
            FusedExpSub(vregExp02, vregF32_02, max00, pregAll);
            FusedExpSub(vregExp03, vregF32_03, max01, pregAll);
            FusedExpSub(vregExp10, vregF32_10, max10, pregAll);
            FusedExpSub(vregExp11, vregF32_11, max11, pregAll);
            FusedExpSub(vregExp12, vregF32_12, max10, pregAll);
            FusedExpSub(vregExp13, vregF32_13, max11, pregAll);
            FusedExpSub(vregExp20, vregF32_20, max20, pregAll);
            FusedExpSub(vregExp21, vregF32_21, max21, pregAll);
            FusedExpSub(vregExp22, vregF32_22, max20, pregAll);
            FusedExpSub(vregExp23, vregF32_23, max21, pregAll);
            FusedExpSub(vregExp30, vregF32_30, max30, pregAll);
            FusedExpSub(vregExp31, vregF32_31, max31, pregAll);
            FusedExpSub(vregExp32, vregF32_32, max30, pregAll);
            FusedExpSub(vregExp33, vregF32_33, max31, pregAll);

            Add(vregSum00, vregExp00, vregSum00, pregAll);
            Add(vregSum01, vregExp01, vregSum01, pregAll);
            Add(vregSum02, vregExp02, vregSum02, pregAll);
            Add(vregSum03, vregExp03, vregSum03, pregAll);
            Add(vregSum10, vregExp10, vregSum10, pregAll);
            Add(vregSum11, vregExp11, vregSum11, pregAll);
            Add(vregSum12, vregExp12, vregSum12, pregAll);
            Add(vregSum13, vregExp13, vregSum13, pregAll);
            Add(vregSum20, vregExp20, vregSum20, pregAll);
            Add(vregSum21, vregExp21, vregSum21, pregAll);
            Add(vregSum22, vregExp22, vregSum22, pregAll);
            Add(vregSum23, vregExp23, vregSum23, pregAll);
            Add(vregSum30, vregExp30, vregSum30, pregAll);
            Add(vregSum31, vregExp31, vregSum31, pregAll);
            Add(vregSum32, vregExp32, vregSum32, pregAll);
            Add(vregSum33, vregExp33, vregSum33, pregAll);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb00 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp00, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb01 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp01, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb02 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp02, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb03 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp03, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb10 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp10, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb11 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp11, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb12 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp12, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb13 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp13, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb20 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp20, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb21 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp21, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb22 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp22, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb23 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp23, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb30 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp30, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb31 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp31, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb32 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp32, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb33 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp33, pregAll);
        }
        Add(vregSum00, vregSum00, vregSum02, pregAll);
        Add(vregSum01, vregSum01, vregSum03, pregAll);
        Add(vregSum10, vregSum10, vregSum12, pregAll);
        Add(vregSum11, vregSum11, vregSum13, pregAll);
        Add(vregSum20, vregSum20, vregSum22, pregAll);
        Add(vregSum21, vregSum21, vregSum23, pregAll);
        Add(vregSum30, vregSum30, vregSum32, pregAll);
        Add(vregSum31, vregSum31, vregSum33, pregAll);

        LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
            LoadAlign(vregExp00, srcUb00 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregExp01, srcUb01 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregExp10, srcUb10 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregExp11, srcUb11 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregExp20, srcUb20 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregExp21, srcUb21 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregExp30, srcUb30 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregExp31, srcUb31 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);

            Div(vregStore00, vregExp00, vregSum00, pregAll);
            Div(vregStore01, vregExp01, vregSum01, pregAll);
            Div(vregStore10, vregExp10, vregSum10, pregAll);
            Div(vregStore11, vregExp11, vregSum11, pregAll);
            Div(vregStore20, vregExp20, vregSum20, pregAll);
            Div(vregStore21, vregExp21, vregSum21, pregAll);
            Div(vregStore30, vregExp30, vregSum30, pregAll);
            Div(vregStore31, vregExp31, vregSum31, pregAll);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr00 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore00, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr01 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore01, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr10 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore10, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr11 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore11, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr20 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore20, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr21 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore21, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr30 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore30, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr31 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore31, pregAll);
        }
    }
    // 尾块处理
    for (uint16_t loopSc = 0; loopSc < uint16_t(vScRealSize % 4); ++loopSc) {
        Duplicate(max00, minValue);
        Duplicate(max01, minValue);
        Duplicate(max02, minValue);
        Duplicate(max03, minValue);

        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum00, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum01, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum02, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum03, 0, pregAll);
        for (uint16_t loopM = 0; loopM < uint16_t(ReduceSize / 2); ++loopM) {
            LoadAlign(src00, srcUb00 + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
            LoadAlign(src01, srcUb01 + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
            LoadAlign(src02, srcUb02 + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
            LoadAlign(src03, srcUb03 + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));

            Max(max00, max00, src00, pregAll);
            Max(max01, max01, src01, pregAll);
            Max(max02, max02, src02, pregAll);
            Max(max03, max03, src03, pregAll);
        }
        Max(max00, max00, max02, pregAll);
        Max(max01, max01, max03, pregAll);

        for (uint16_t loopM = 0; loopM < ReduceSize / 2; ++loopM) {
            LoadAlign(vregF32_00, srcUb00 + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
            LoadAlign(vregF32_01, srcUb01 + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
            LoadAlign(vregF32_02, srcUb02 + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
            LoadAlign(vregF32_03, srcUb03 + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));

            FusedExpSub(vregExp00, vregF32_00, max00, pregAll);
            FusedExpSub(vregExp01, vregF32_01, max01, pregAll);
            FusedExpSub(vregExp02, vregF32_02, max00, pregAll);
            FusedExpSub(vregExp03, vregF32_03, max01, pregAll);

            Add(vregSum00, vregExp00, vregSum00, pregAll);
            Add(vregSum01, vregExp01, vregSum01, pregAll);
            Add(vregSum02, vregExp02, vregSum02, pregAll);
            Add(vregSum03, vregExp03, vregSum03, pregAll);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb00 + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4)),
                vregExp00, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb01 + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4)),
                vregExp01, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb02 + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4)),
                vregExp02, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb03 + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4)),
                vregExp03, pregAll);
        }
        Add(vregSum00, vregSum00, vregSum02, pregAll);
        Add(vregSum01, vregSum01, vregSum03, pregAll);

        LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
            LoadAlign(vregExp00, srcUb00 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
            LoadAlign(vregExp01, srcUb01 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));

            Div(vregStore00, vregExp00, vregSum00, pregAll);
            Div(vregStore01, vregExp01, vregSum01, pregAll);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr00 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4)),
                vregStore00, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr01 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4)),
                vregStore01, pregAll);
        }
    }
}

template <typename T>
__simd_vf__ inline void SoftmaxDndBase64(__ubuf__ T *inputAddr, __ubuf__ float *outputAddr,
    const uint32_t RowSize, const uint32_t ReduceSize, const uint32_t vScRealSize,
    const T minValue)
{
    RegTensor<float> vregSum00;
    RegTensor<float> vregSum10;
    RegTensor<float> vregSum20;
    RegTensor<float> vregSum30;
    RegTensor<float> vregSum01;
    RegTensor<float> vregSum11;
    RegTensor<float> vregSum21;
    RegTensor<float> vregSum31;

    RegTensor<float> vregExp00;
    RegTensor<float> vregExp10;
    RegTensor<float> vregExp20;
    RegTensor<float> vregExp30;
    RegTensor<float> vregExp01;
    RegTensor<float> vregExp11;
    RegTensor<float> vregExp21;
    RegTensor<float> vregExp31;

    RegTensor<float> vregF32_00;
    RegTensor<float> vregF32_10;
    RegTensor<float> vregF32_20;
    RegTensor<float> vregF32_30;
    RegTensor<float> vregF32_01;
    RegTensor<float> vregF32_11;
    RegTensor<float> vregF32_21;
    RegTensor<float> vregF32_31;

    RegTensor<float> vregStore0;
    RegTensor<float> vregStore1;
    RegTensor<float> vregStore2;
    RegTensor<float> vregStore3;
    MaskReg pregAll;
    pregAll = CreateMask<T, MaskPattern::ALL>();
    RegTensor<float> src00, src10, src20, src30, src01, src11, src21, src31;
    RegTensor<float> max00, max10, max20, max30, max01, max11, max21, max31;

    __ubuf__ float *srcUb00 = outputAddr;
    __ubuf__ float *srcUb01 = outputAddr + RowSize;
    __ubuf__ float *srcUb10 = srcUb00 + ReduceSize * RowSize;
    __ubuf__ float *srcUb11 = srcUb00 + ReduceSize * RowSize + RowSize;
    __ubuf__ float *srcUb20 = srcUb00 + ReduceSize * RowSize * 2;
    __ubuf__ float *srcUb21 = srcUb00 + ReduceSize * RowSize * 2 + RowSize;
    __ubuf__ float *srcUb30 = srcUb00 + ReduceSize * RowSize * 3;
    __ubuf__ float *srcUb31 = srcUb00 + ReduceSize * RowSize * 3 + RowSize;

    __ubuf__ float *inputAddr0 = inputAddr;
    __ubuf__ float *inputAddr1 = inputAddr + (ReduceSize * RowSize);
    __ubuf__ float *inputAddr2 = inputAddr + (ReduceSize * RowSize * 2);
    __ubuf__ float *inputAddr3 = inputAddr + (ReduceSize * RowSize * 3);

    for (uint16_t loopSc = 0; loopSc < uint16_t(vScRealSize / 4); ++loopSc) {
        Duplicate(max00, minValue);
        Duplicate(max10, minValue);
        Duplicate(max20, minValue);
        Duplicate(max30, minValue);
        Duplicate(max01, minValue);
        Duplicate(max11, minValue);
        Duplicate(max21, minValue);
        Duplicate(max31, minValue);

        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum00, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum10, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum20, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum30, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum01, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum11, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum21, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum31, 0, pregAll);
        for (uint16_t loopM = 0; loopM < uint16_t(ReduceSize / 2); ++loopM) {
            LoadAlign(src00, srcUb00 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src01, srcUb01 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(src10, srcUb10 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src11, srcUb11 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(src20, srcUb20 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src21, srcUb21 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(src30, srcUb30 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src31, srcUb31 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);

            Max(max00, max00, src00, pregAll);
            Max(max01, max01, src01, pregAll);
            Max(max10, max10, src10, pregAll);
            Max(max11, max11, src11, pregAll);
            Max(max20, max20, src20, pregAll);
            Max(max21, max21, src21, pregAll);
            Max(max30, max30, src30, pregAll);
            Max(max31, max31, src31, pregAll);
        }
        Max(max00, max00, max01, pregAll);
        Max(max10, max10, max11, pregAll);
        Max(max20, max20, max21, pregAll);
        Max(max30, max30, max31, pregAll);

        for (uint16_t loopM = 0; loopM < uint16_t(ReduceSize / 2); ++loopM) {
            LoadAlign(vregF32_00, srcUb00 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_01, srcUb01 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(vregF32_10, srcUb10 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_11, srcUb11 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(vregF32_20, srcUb20 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_21, srcUb21 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(vregF32_30, srcUb30 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_31, srcUb31 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);

            FusedExpSub(vregExp00, vregF32_00, max00, pregAll);
            FusedExpSub(vregExp01, vregF32_01, max00, pregAll);
            FusedExpSub(vregExp10, vregF32_10, max10, pregAll);
            FusedExpSub(vregExp11, vregF32_11, max10, pregAll);
            FusedExpSub(vregExp20, vregF32_20, max20, pregAll);
            FusedExpSub(vregExp21, vregF32_21, max20, pregAll);
            FusedExpSub(vregExp30, vregF32_30, max30, pregAll);
            FusedExpSub(vregExp31, vregF32_31, max30, pregAll);

            Add(vregSum00, vregExp00, vregSum00, pregAll);
            Add(vregSum01, vregExp01, vregSum01, pregAll);
            Add(vregSum10, vregExp10, vregSum10, pregAll);
            Add(vregSum11, vregExp11, vregSum11, pregAll);
            Add(vregSum20, vregExp20, vregSum20, pregAll);
            Add(vregSum21, vregExp21, vregSum21, pregAll);
            Add(vregSum30, vregExp30, vregSum30, pregAll);
            Add(vregSum31, vregExp31, vregSum31, pregAll);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb00 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp00, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb01 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp01, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb10 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp10, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb11 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp11, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb20 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp20, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb21 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp21, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb30 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp30, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb31 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp31, pregAll);
        }
        Add(vregSum00, vregSum00, vregSum01, pregAll);
        Add(vregSum10, vregSum10, vregSum11, pregAll);
        Add(vregSum20, vregSum20, vregSum21, pregAll);
        Add(vregSum30, vregSum30, vregSum31, pregAll);

        LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
            LoadAlign(vregExp00, srcUb00 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregExp10, srcUb10 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregExp20, srcUb20 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregExp30, srcUb30 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);

            Div(vregStore0, vregExp00, vregSum00, pregAll);
            Div(vregStore1, vregExp10, vregSum10, pregAll);
            Div(vregStore2, vregExp20, vregSum20, pregAll);
            Div(vregStore3, vregExp30, vregSum30, pregAll);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr0 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore0, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr1 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore1, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr2 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore2, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr3 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore3, pregAll);
        }
    }
    // 尾块处理
    for (uint16_t loopSc = 0; loopSc < uint16_t(vScRealSize % 4); ++loopSc) {
        Duplicate(max00, minValue);
        Duplicate(max01, minValue);

        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum00, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum01, 0, pregAll);
        for (uint16_t loopM = 0; loopM < uint16_t(ReduceSize / 2); ++loopM) {
            LoadAlign(src00, srcUb00 + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
            LoadAlign(src01, srcUb01 + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));

            Max(max00, max00, src00, pregAll);
            Max(max01, max01, src01, pregAll);
        }
        Max(max00, max00, max01, pregAll);

        for (uint16_t loopM = 0; loopM < ReduceSize / 2; ++loopM) {
            LoadAlign(vregF32_00, srcUb00 + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
            LoadAlign(vregF32_01, srcUb01 + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));

            FusedExpSub(vregExp00, vregF32_00, max00, pregAll);
            FusedExpSub(vregExp01, vregF32_01, max00, pregAll);

            Add(vregSum00, vregExp00, vregSum00, pregAll);
            Add(vregSum01, vregExp01, vregSum01, pregAll);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb00 + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4)),
                vregExp00, pregAll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb01 + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4)),
                vregExp01, pregAll);
        }
        Add(vregSum00, vregSum00, vregSum01, pregAll);

        LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
            LoadAlign(vregExp00, srcUb00 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
            Div(vregStore0, vregExp00, vregSum00, pregAll);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr0 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4)),
                vregStore0, pregAll);
        }
    }
}

template <typename T>
__simd_vf__ inline void SoftmaxDndBase32(__ubuf__ T *inputAddr, __ubuf__ float *outputAddr,
    const uint32_t RowSize, const uint32_t ReduceSize, const uint32_t vScRealSize,
    const T minValue)
{
    RegTensor<float> vregSum00;
    RegTensor<float> vregSum10;
    RegTensor<float> vregSum20;
    RegTensor<float> vregSum30;
    RegTensor<float> vregSum01;
    RegTensor<float> vregSum11;
    RegTensor<float> vregSum21;
    RegTensor<float> vregSum31;

    RegTensor<float> vregExp00;
    RegTensor<float> vregExp10;
    RegTensor<float> vregExp20;
    RegTensor<float> vregExp30;
    RegTensor<float> vregExp01;
    RegTensor<float> vregExp11;
    RegTensor<float> vregExp21;
    RegTensor<float> vregExp31;

    RegTensor<float> vregF32_00;
    RegTensor<float> vregF32_10;
    RegTensor<float> vregF32_20;
    RegTensor<float> vregF32_30;
    RegTensor<float> vregF32_01;
    RegTensor<float> vregF32_11;
    RegTensor<float> vregF32_21;
    RegTensor<float> vregF32_31;

    RegTensor<float> vregStore0;
    RegTensor<float> vregStore1;
    RegTensor<float> vregStore2;
    RegTensor<float> vregStore3;

    MaskReg pregLHalf;
    MaskReg pregHHalf;
    MaskReg pregAll;
    pregAll = CreateMask<T, MaskPattern::ALL>();
    pregLHalf = CreateMask<T, MaskPattern::VL32>();
    Not(pregHHalf, pregLHalf, pregAll);
    RegTensor<float> max0, max1, max2, max3;
    RegTensor<float> src00, src10, src20, src30, src01, src11, src21, src31;
    RegTensor<float> max00, max10, max20, max30, max01, max11, max21, max31;

    __ubuf__ float *srcUb00 = outputAddr;
    __ubuf__ float *srcUb01 = outputAddr + RowSize * 2;
    __ubuf__ float *srcUb10 = srcUb00 + ReduceSize * RowSize;
    __ubuf__ float *srcUb11 = srcUb00 + ReduceSize * RowSize + RowSize * 2;
    __ubuf__ float *srcUb20 = srcUb00 + ReduceSize * RowSize * 2;
    __ubuf__ float *srcUb21 = srcUb00 + ReduceSize * RowSize * 2 + RowSize * 2;
    __ubuf__ float *srcUb30 = srcUb00 + ReduceSize * RowSize * 3;
    __ubuf__ float *srcUb31 = srcUb00 + ReduceSize * RowSize * 3 + RowSize * 2;

    __ubuf__ float *inputAddr0 = inputAddr;
    __ubuf__ float *inputAddr1 = inputAddr + (ReduceSize * RowSize);
    __ubuf__ float *inputAddr2 = inputAddr + (ReduceSize * RowSize * 2);
    __ubuf__ float *inputAddr3 = inputAddr + (ReduceSize * RowSize * 3);

    for (uint16_t loopSc = 0; loopSc < uint16_t(vScRealSize / 4); ++loopSc) {
        Duplicate(max0, minValue);
        Duplicate(max1, minValue);
        Duplicate(max2, minValue);
        Duplicate(max3, minValue);
        Duplicate(max00, minValue);
        Duplicate(max10, minValue);
        Duplicate(max20, minValue);
        Duplicate(max30, minValue);
        Duplicate(max01, minValue);
        Duplicate(max11, minValue);
        Duplicate(max21, minValue);
        Duplicate(max31, minValue);

        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum00, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum10, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum20, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum30, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum01, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum11, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum21, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum31, 0, pregAll);
        for (uint16_t loopM = 0; loopM < uint16_t(ReduceSize / 4); ++loopM) {
            LoadAlign(src00, srcUb00 + loopM * RowSize * 4 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src01, srcUb01 + loopM * RowSize * 4 + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(src10, srcUb10 + loopM * RowSize * 4 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src11, srcUb11 + loopM * RowSize * 4 + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(src20, srcUb20 + loopM * RowSize * 4 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src21, srcUb21 + loopM * RowSize * 4 + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(src30, srcUb30 + loopM * RowSize * 4 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src31, srcUb31 + loopM * RowSize * 4 + ReduceSize * RowSize * loopSc * 4);

            Max(max00, max00, src00, pregAll);
            Max(max01, max01, src01, pregAll);
            Max(max10, max10, src10, pregAll);
            Max(max11, max11, src11, pregAll);
            Max(max20, max20, src20, pregAll);
            Max(max21, max21, src21, pregAll);
            Max(max30, max30, src30, pregAll);
            Max(max31, max31, src31, pregAll);
        }
        Max(max0, max00, max01, pregAll);
        Max(max1, max10, max11, pregAll);
        Max(max2, max20, max21, pregAll);
        Max(max3, max30, max31, pregAll);

        Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(max00, max0, pregLHalf);
        Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(max01, max0, pregHHalf);
        Max(max0, max00, max01, pregLHalf);

        Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(max10, max1, pregLHalf);
        Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(max11, max1, pregHHalf);
        Max(max1, max10, max11, pregLHalf);

        Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(max20, max2, pregLHalf);
        Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(max21, max2, pregHHalf);
        Max(max2, max20, max21, pregLHalf);

        Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(max30, max3, pregLHalf);
        Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(max31, max3, pregHHalf);
        Max(max3, max30, max31, pregLHalf);

        for (uint16_t loopM = 0; loopM < uint16_t(ReduceSize / 2); ++loopM) {
            LoadAlign(vregF32_00, srcUb00 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_01, (srcUb00 + RowSize) + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(vregF32_10, srcUb10 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_11, (srcUb10 + RowSize) + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(vregF32_20, srcUb20 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_21, (srcUb20 + RowSize) + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(vregF32_30, srcUb30 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_31, (srcUb30 + RowSize) + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4);

            FusedExpSub(vregExp00, vregF32_00, max0, pregLHalf);
            FusedExpSub(vregExp01, vregF32_01, max0, pregLHalf);
            FusedExpSub(vregExp10, vregF32_10, max1, pregLHalf);
            FusedExpSub(vregExp11, vregF32_11, max1, pregLHalf);
            FusedExpSub(vregExp20, vregF32_20, max2, pregLHalf);
            FusedExpSub(vregExp21, vregF32_21, max2, pregLHalf);
            FusedExpSub(vregExp30, vregF32_30, max3, pregLHalf);
            FusedExpSub(vregExp31, vregF32_31, max3, pregLHalf);

            Add(vregSum00, vregExp00, vregSum00, pregLHalf);
            Add(vregSum01, vregExp01, vregSum01, pregLHalf);
            Add(vregSum10, vregExp10, vregSum10, pregLHalf);
            Add(vregSum11, vregExp11, vregSum11, pregLHalf);
            Add(vregSum20, vregExp20, vregSum20, pregLHalf);
            Add(vregSum21, vregExp21, vregSum21, pregLHalf);
            Add(vregSum30, vregExp30, vregSum30, pregLHalf);
            Add(vregSum31, vregExp31, vregSum31, pregLHalf);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb00 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp00, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb00 + RowSize + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp01, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb10 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp10, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb10 + RowSize + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp11, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb20 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp20, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb20 + RowSize + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp21, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb30 + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp30, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb30 + RowSize + loopM * RowSize * 2 + ReduceSize * RowSize * loopSc * 4),
                vregExp31, pregLHalf);
        }
        Add(vregSum00, vregSum00, vregSum01, pregLHalf);
        Add(vregSum10, vregSum10, vregSum11, pregLHalf);
        Add(vregSum20, vregSum20, vregSum21, pregLHalf);
        Add(vregSum30, vregSum30, vregSum31, pregLHalf);

        LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
            LoadAlign(vregExp00, srcUb00 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregExp10, srcUb10 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregExp20, srcUb20 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregExp30, srcUb30 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);

            Div(vregStore0, vregExp00, vregSum00, pregLHalf);
            Div(vregStore1, vregExp10, vregSum10, pregLHalf);
            Div(vregStore2, vregExp20, vregSum20, pregLHalf);
            Div(vregStore3, vregExp30, vregSum30, pregLHalf);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr0 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore0, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr1 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore1, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr2 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore2, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr3 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore3, pregLHalf);
        }
    }
    // 尾块处理
    for (uint16_t loopSc = 0; loopSc < uint16_t(vScRealSize % 4); ++loopSc) {
        Duplicate(max0, minValue);
        Duplicate(max00, minValue);
        Duplicate(max01, minValue);

        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum00, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum01, 0, pregAll);
        for (uint16_t loopM = 0; loopM < uint16_t(ReduceSize / 4); ++loopM) {
            LoadAlign(src00, srcUb00 + loopM * RowSize * 4 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
            LoadAlign(src01, srcUb01 + loopM * RowSize * 4 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));

            Max(max00, max00, src00, pregAll);
            Max(max01, max01, src01, pregAll);
        }
        Max(max0, max00, max01, pregAll);

        Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(max00, max0, pregLHalf);
        Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(max01, max0, pregHHalf);
        Max(max0, max00, max01, pregLHalf);

        for (uint16_t loopM = 0; loopM < uint16_t(ReduceSize / 2); ++loopM) {
            LoadAlign(vregF32_00, srcUb00 + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
            LoadAlign(vregF32_01, (srcUb00 + RowSize) + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));

            FusedExpSub(vregExp00, vregF32_00, max0, pregLHalf);
            FusedExpSub(vregExp01, vregF32_01, max0, pregLHalf);

            Add(vregSum00, vregExp00, vregSum00, pregLHalf);
            Add(vregSum01, vregExp01, vregSum01, pregLHalf);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb00 + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4)),
                vregExp00, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb00 + RowSize + loopM * RowSize * 2 + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4)),
                vregExp01, pregLHalf);
        }
        Add(vregSum00, vregSum00, vregSum01, pregLHalf);

        LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
            LoadAlign(vregExp00, srcUb00 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));

            Div(vregStore0, vregExp00, vregSum00, pregLHalf);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr0 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4)),
                vregStore0, pregLHalf);
        }
    }
}

template <typename T>
__simd_vf__ inline void SoftmaxDndBase8(__ubuf__ T *inputAddr, __ubuf__ float *outputAddr,
    const uint32_t RowSize, const uint32_t ReduceSize, const uint32_t vScRealSize,
    const T minValue)
{
    RegTensor<float> vregSum0;
    RegTensor<float> vregSum1;
    RegTensor<float> vregSum2;
    RegTensor<float> vregSum3;

    RegTensor<float> vregExp0;
    RegTensor<float> vregExp1;
    RegTensor<float> vregExp2;
    RegTensor<float> vregExp3;

    RegTensor<float> vregF32_0;
    RegTensor<float> vregF32_1;
    RegTensor<float> vregF32_2;
    RegTensor<float> vregF32_3;

    RegTensor<float> vregStore0;
    RegTensor<float> vregStore1;
    RegTensor<float> vregStore2;
    RegTensor<float> vregStore3;

    MaskReg pregL8;
    pregL8 = CreateMask<T, MaskPattern::VL8>();
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;

    __ubuf__ float *srcUb0 = outputAddr;
    __ubuf__ float *srcUb1 = srcUb0 + ReduceSize * RowSize;
    __ubuf__ float *srcUb2 = srcUb0 + ReduceSize * RowSize * 2;
    __ubuf__ float *srcUb3 = srcUb0 + ReduceSize * RowSize * 3;

    __ubuf__ float *inputAddr0 = inputAddr;
    __ubuf__ float *inputAddr1 = inputAddr + (ReduceSize * RowSize);
    __ubuf__ float *inputAddr2 = inputAddr + (ReduceSize * RowSize * 2);
    __ubuf__ float *inputAddr3 = inputAddr + (ReduceSize * RowSize * 3);

    for (uint16_t loopSc = 0; loopSc < uint16_t(vScRealSize / 4); ++loopSc) {
        Duplicate(max0, minValue);
        Duplicate(max1, minValue);
        Duplicate(max2, minValue);
        Duplicate(max3, minValue);

        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum0, 0, pregL8);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum1, 0, pregL8);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum2, 0, pregL8);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum3, 0, pregL8);

        for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
            LoadAlign(src0, srcUb0 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src1, srcUb1 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src2, srcUb2 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src3, srcUb3 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);

            Max(max0, max0, src0, pregL8);
            Max(max1, max1, src1, pregL8);
            Max(max2, max2, src2, pregL8);
            Max(max3, max3, src3, pregL8);
        }

        for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
            LoadAlign(vregF32_0, srcUb0 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_1, srcUb1 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_2, srcUb2 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_3, srcUb3 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);

            FusedExpSub(vregExp0, vregF32_0, max0, pregL8);
            FusedExpSub(vregExp1, vregF32_1, max1, pregL8);
            FusedExpSub(vregExp2, vregF32_2, max2, pregL8);
            FusedExpSub(vregExp3, vregF32_3, max3, pregL8);

            Add(vregSum0, vregExp0, vregSum0, pregL8);
            Add(vregSum1, vregExp1, vregSum1, pregL8);
            Add(vregSum2, vregExp2, vregSum2, pregL8);
            Add(vregSum3, vregExp3, vregSum3, pregL8);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb0 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregExp0, pregL8);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb1 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregExp1, pregL8);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb2 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregExp2, pregL8);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb3 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregExp3, pregL8);
        }

        LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
            LoadAlign(vregExp0, srcUb0 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregExp1, srcUb1 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregExp2, srcUb2 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregExp3, srcUb3 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);

            Div(vregStore0, vregExp0, vregSum0, pregL8);
            Div(vregStore1, vregExp1, vregSum1, pregL8);
            Div(vregStore2, vregExp2, vregSum2, pregL8);
            Div(vregStore3, vregExp3, vregSum3, pregL8);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr0 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore0, pregL8);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr1 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore1, pregL8);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr2 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore2, pregL8);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr3 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore3, pregL8);
        }
    }

    for (uint16_t loopSc = 0; loopSc < uint16_t(vScRealSize % 4); ++loopSc) {
        Duplicate(max0, minValue);

        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum0, 0, pregL8);
        for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
            LoadAlign(src0, srcUb0 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
            Max(max0, max0, src0, pregL8);
        }

        for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
            LoadAlign(vregF32_0, srcUb0 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
            FusedExpSub(vregExp0, vregF32_0, max0, pregL8);
            Add(vregSum0, vregExp0, vregSum0, pregL8);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb0 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4)),
                vregExp0, pregL8);
        }

        LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
            LoadAlign(vregExp0, srcUb0 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
            Div(vregStore0, vregExp0, vregSum0, pregL8);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr0 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4)),
                vregStore0, pregL8);
        }
    }
}

template <typename T>
__simd_vf__ inline void SoftmaxDndBase16(__ubuf__ T *inputAddr, __ubuf__ float *outputAddr,
    const uint32_t RowSize, const uint32_t ReduceSize, const uint32_t vScRealSize,
    const T minValue)
{
    RegTensor<float> vregSum00;
    RegTensor<float> vregSum10;
    RegTensor<float> vregSum20;
    RegTensor<float> vregSum30;
    RegTensor<float> vregSum01;
    RegTensor<float> vregSum11;
    RegTensor<float> vregSum21;
    RegTensor<float> vregSum31;

    RegTensor<float> vregExp00;
    RegTensor<float> vregExp10;
    RegTensor<float> vregExp20;
    RegTensor<float> vregExp30;
    RegTensor<float> vregExp01;
    RegTensor<float> vregExp11;
    RegTensor<float> vregExp21;
    RegTensor<float> vregExp31;

    RegTensor<float> vregF32_00;
    RegTensor<float> vregF32_10;
    RegTensor<float> vregF32_20;
    RegTensor<float> vregF32_30;
    RegTensor<float> vregF32_01;
    RegTensor<float> vregF32_11;
    RegTensor<float> vregF32_21;
    RegTensor<float> vregF32_31;

    RegTensor<float> vregStore0;
    RegTensor<float> vregStore1;
    RegTensor<float> vregStore2;
    RegTensor<float> vregStore3;

    MaskReg pregLHalf;
    MaskReg pregHHalf;
    MaskReg pregAll;
    pregAll = CreateMask<T, MaskPattern::ALL>();
    pregLHalf = CreateMask<T, MaskPattern::VL16>();
    Not(pregHHalf, pregLHalf, pregAll);
    RegTensor<float> max0, max1, max2, max3;
    RegTensor<float> src00, src10, src20, src30, src01, src11, src21, src31;
    RegTensor<float> max00, max10, max20, max30, max01, max11, max21, max31;

    __ubuf__ float *srcUb00 = outputAddr;
    __ubuf__ float *srcUb01 = outputAddr + RowSize * (ReduceSize / 2);
    __ubuf__ float *srcUb10 = srcUb00 + ReduceSize * RowSize;
    __ubuf__ float *srcUb11 = srcUb00 + ReduceSize * RowSize + RowSize * (ReduceSize / 2);
    __ubuf__ float *srcUb20 = srcUb00 + ReduceSize * RowSize * 2;
    __ubuf__ float *srcUb21 = srcUb00 + ReduceSize * RowSize * 2 + RowSize * (ReduceSize / 2);
    __ubuf__ float *srcUb30 = srcUb00 + ReduceSize * RowSize * 3;
    __ubuf__ float *srcUb31 = srcUb00 + ReduceSize * RowSize * 3 + RowSize * (ReduceSize / 2);

    __ubuf__ float *inputAddr0 = inputAddr;
    __ubuf__ float *inputAddr1 = inputAddr + (ReduceSize * RowSize);
    __ubuf__ float *inputAddr2 = inputAddr + (ReduceSize * RowSize * 2);
    __ubuf__ float *inputAddr3 = inputAddr + (ReduceSize * RowSize * 3);

    for (uint16_t loopSc = 0; loopSc < uint16_t(vScRealSize / 4); ++loopSc) {
        Duplicate(max0, minValue);
        Duplicate(max1, minValue);
        Duplicate(max2, minValue);
        Duplicate(max3, minValue);
        Duplicate(max00, minValue);
        Duplicate(max10, minValue);
        Duplicate(max20, minValue);
        Duplicate(max30, minValue);
        Duplicate(max01, minValue);
        Duplicate(max11, minValue);
        Duplicate(max21, minValue);
        Duplicate(max31, minValue);

        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum00, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum10, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum20, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum30, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum01, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum11, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum21, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum31, 0, pregAll);
        for (uint16_t loopM = 0; loopM < uint16_t(ReduceSize / 2); ++loopM) {
            LoadAlign(src00, srcUb00 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src01, srcUb01 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(src10, srcUb10 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src11, srcUb11 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(src20, srcUb20 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src21, srcUb21 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(src30, srcUb30 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(src31, srcUb31 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);

            Max(max00, max00, src00, pregLHalf);
            Max(max01, max01, src01, pregLHalf);
            Max(max10, max10, src10, pregLHalf);
            Max(max11, max11, src11, pregLHalf);
            Max(max20, max20, src20, pregLHalf);
            Max(max21, max21, src21, pregLHalf);
            Max(max30, max30, src30, pregLHalf);
            Max(max31, max31, src31, pregLHalf);
        }
        Max(max0, max00, max01, pregAll);
        Max(max1, max10, max11, pregAll);
        Max(max2, max20, max21, pregAll);
        Max(max3, max30, max31, pregAll);

        for (uint16_t loopM = 0; loopM < uint16_t(ReduceSize / 2); ++loopM) {
            LoadAlign(vregF32_00, srcUb00 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_01, srcUb01 + + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(vregF32_10, srcUb10 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_11, srcUb11 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(vregF32_20, srcUb20 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_21, srcUb21 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);

            LoadAlign(vregF32_30, srcUb30 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregF32_31, srcUb31 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);

            FusedExpSub(vregExp00, vregF32_00, max0, pregLHalf);
            FusedExpSub(vregExp01, vregF32_01, max0, pregLHalf);
            FusedExpSub(vregExp10, vregF32_10, max1, pregLHalf);
            FusedExpSub(vregExp11, vregF32_11, max1, pregLHalf);
            FusedExpSub(vregExp20, vregF32_20, max2, pregLHalf);
            FusedExpSub(vregExp21, vregF32_21, max2, pregLHalf);
            FusedExpSub(vregExp30, vregF32_30, max3, pregLHalf);
            FusedExpSub(vregExp31, vregF32_31, max3, pregLHalf);

            Add(vregSum00, vregExp00, vregSum00, pregLHalf);
            Add(vregSum01, vregExp01, vregSum01, pregLHalf);
            Add(vregSum10, vregExp10, vregSum10, pregLHalf);
            Add(vregSum11, vregExp11, vregSum11, pregLHalf);
            Add(vregSum20, vregExp20, vregSum20, pregLHalf);
            Add(vregSum21, vregExp21, vregSum21, pregLHalf);
            Add(vregSum30, vregExp30, vregSum30, pregLHalf);
            Add(vregSum31, vregExp31, vregSum31, pregLHalf);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb00 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregExp00, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb01 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregExp01, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb10 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregExp10, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb11 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregExp11, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb20 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregExp20, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb21 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregExp21, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb30 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregExp30, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb31 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregExp31, pregLHalf);
        }
        Add(vregSum00, vregSum00, vregSum01, pregLHalf);
        Add(vregSum10, vregSum10, vregSum11, pregLHalf);
        Add(vregSum20, vregSum20, vregSum21, pregLHalf);
        Add(vregSum30, vregSum30, vregSum31, pregLHalf);

        LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
            LoadAlign(vregExp00, srcUb00 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregExp10, srcUb10 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregExp20, srcUb20 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
            LoadAlign(vregExp30, srcUb30 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);

            Div(vregStore0, vregExp00, vregSum00, pregLHalf);
            Div(vregStore1, vregExp10, vregSum10, pregLHalf);
            Div(vregStore2, vregExp20, vregSum20, pregLHalf);
            Div(vregStore3, vregExp30, vregSum30, pregLHalf);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr0 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore0, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr1 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore1, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr2 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore2, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr3 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                vregStore3, pregLHalf);
        }
    }
    // 尾块处理
    for (uint16_t loopSc = 0; loopSc < uint16_t(vScRealSize % 4); ++loopSc) {
        Duplicate(max0, minValue);
        Duplicate(max00, minValue);
        Duplicate(max01, minValue);

        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum00, 0, pregAll);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum01, 0, pregAll);
        for (uint16_t loopM = 0; loopM < uint16_t(ReduceSize / 2); ++loopM) {
            LoadAlign(src00, srcUb00 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
            LoadAlign(src01, srcUb01 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));

            Max(max00, max00, src00, pregLHalf);
            Max(max01, max01, src01, pregLHalf);
        }
        Max(max0, max00, max01, pregAll);

        for (uint16_t loopM = 0; loopM < uint16_t(ReduceSize / 2); ++loopM) {
            LoadAlign(vregF32_00, srcUb00 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
            LoadAlign(vregF32_01, srcUb01 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));

            FusedExpSub(vregExp00, vregF32_00, max0, pregLHalf);
            FusedExpSub(vregExp01, vregF32_01, max0, pregLHalf);

            Add(vregSum00, vregExp00, vregSum00, pregLHalf);
            Add(vregSum01, vregExp01, vregSum01, pregLHalf);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb00 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4)),
                vregExp00, pregLHalf);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb01 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4)),
                vregExp01, pregLHalf);
        }

        Add(vregSum00, vregSum00, vregSum01, pregLHalf);

        LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
            LoadAlign(vregExp00, srcUb00 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));

            Div(vregStore0, vregExp00, vregSum00, pregLHalf);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr0 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4)),
                vregStore0, pregLHalf);
        }
    }
}

template <typename T>
__simd_vf__ inline void SoftmaxDndBase256(__ubuf__ T *inputAddr, __ubuf__ float *outputAddr,
    const uint32_t RowSize, const uint32_t ReduceSize, const uint32_t vScRealSize,
    const T minValue)
{
    RegTensor<float> vregSum0;
    RegTensor<float> vregSum1;
    RegTensor<float> vregSum2;
    RegTensor<float> vregSum3;

    RegTensor<float> vregExp0;
    RegTensor<float> vregExp1;
    RegTensor<float> vregExp2;
    RegTensor<float> vregExp3;

    RegTensor<float> vregF32_0;
    RegTensor<float> vregF32_1;
    RegTensor<float> vregF32_2;
    RegTensor<float> vregF32_3;

    RegTensor<float> vregStore0;
    RegTensor<float> vregStore1;
    RegTensor<float> vregStore2;
    RegTensor<float> vregStore3;

    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg pregAll;
    pregAll = CreateMask<T, MaskPattern::ALL>();

    for (uint16_t loopSc = 0; loopSc < uint16_t(vScRealSize / 4); ++loopSc) {
        for (uint16_t dChunk = 0; dChunk < 4; ++dChunk) {
            uint32_t dOffset = dChunk * 64;
            __ubuf__ float *srcUb0 = outputAddr + dOffset;
            __ubuf__ float *srcUb1 = srcUb0 + ReduceSize * RowSize;
            __ubuf__ float *srcUb2 = srcUb0 + ReduceSize * RowSize * 2;
            __ubuf__ float *srcUb3 = srcUb0 + ReduceSize * RowSize * 3;

            __ubuf__ float *inputAddr0 = inputAddr + dOffset;
            __ubuf__ float *inputAddr1 = inputAddr + dOffset + (ReduceSize * RowSize);
            __ubuf__ float *inputAddr2 = inputAddr + dOffset + (ReduceSize * RowSize * 2);
            __ubuf__ float *inputAddr3 = inputAddr + dOffset + (ReduceSize * RowSize * 3);

            Duplicate(max0, minValue);
            Duplicate(max1, minValue);
            Duplicate(max2, minValue);
            Duplicate(max3, minValue);

            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum0, 0, pregAll);
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum1, 0, pregAll);
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum2, 0, pregAll);
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum3, 0, pregAll);

            for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
                LoadAlign(src0, srcUb0 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
                LoadAlign(src1, srcUb1 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
                LoadAlign(src2, srcUb2 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
                LoadAlign(src3, srcUb3 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);

                Max(max0, max0, src0, pregAll);
                Max(max1, max1, src1, pregAll);
                Max(max2, max2, src2, pregAll);
                Max(max3, max3, src3, pregAll);
            }

            for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
                LoadAlign(vregF32_0, srcUb0 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
                LoadAlign(vregF32_1, srcUb1 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
                LoadAlign(vregF32_2, srcUb2 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
                LoadAlign(vregF32_3, srcUb3 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);

                FusedExpSub(vregExp0, vregF32_0, max0, pregAll);
                FusedExpSub(vregExp1, vregF32_1, max1, pregAll);
                FusedExpSub(vregExp2, vregF32_2, max2, pregAll);
                FusedExpSub(vregExp3, vregF32_3, max3, pregAll);

                Add(vregSum0, vregExp0, vregSum0, pregAll);
                Add(vregSum1, vregExp1, vregSum1, pregAll);
                Add(vregSum2, vregExp2, vregSum2, pregAll);
                Add(vregSum3, vregExp3, vregSum3, pregAll);

                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb0 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                    vregExp0, pregAll);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb1 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                    vregExp1, pregAll);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb2 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                    vregExp2, pregAll);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb3 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                    vregExp3, pregAll);
            }

            LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
                LoadAlign(vregExp0, srcUb0 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
                LoadAlign(vregExp1, srcUb1 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
                LoadAlign(vregExp2, srcUb2 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
                LoadAlign(vregExp3, srcUb3 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);

                Div(vregStore0, vregExp0, vregSum0, pregAll);
                Div(vregStore1, vregExp1, vregSum1, pregAll);
                Div(vregStore2, vregExp2, vregSum2, pregAll);
                Div(vregStore3, vregExp3, vregSum3, pregAll);

                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr0 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                    vregStore0, pregAll);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr1 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                    vregStore1, pregAll);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr2 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                    vregStore2, pregAll);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr3 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                    vregStore3, pregAll);
            }
        }
    }

    for (uint16_t loopSc = 0; loopSc < uint16_t(vScRealSize % 4); ++loopSc) {
        for (uint16_t dChunk = 0; dChunk < 4; ++dChunk) {
            uint32_t dOffset = dChunk * 64;
            __ubuf__ float *srcUb0 = outputAddr + dOffset;
            __ubuf__ float *inputAddr0 = inputAddr + dOffset;

            Duplicate(max0, minValue);
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum0, 0, pregAll);

            for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
                LoadAlign(src0, srcUb0 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
                Max(max0, max0, src0, pregAll);
            }

            for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
                LoadAlign(vregF32_0, srcUb0 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
                FusedExpSub(vregExp0, vregF32_0, max0, pregAll);
                Add(vregSum0, vregExp0, vregSum0, pregAll);

                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb0 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4)),
                    vregExp0, pregAll);
            }

            LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
                LoadAlign(vregExp0, srcUb0 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
                Div(vregStore0, vregExp0, vregSum0, pregAll);

                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr0 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4)),
                    vregStore0, pregAll);
            }
        }
    }
}

template <typename T>
__simd_vf__ inline void SoftmaxDndBase512(__ubuf__ T *inputAddr, __ubuf__ float *outputAddr,
    const uint32_t RowSize, const uint32_t ReduceSize, const uint32_t vScRealSize,
    const T minValue)
{
    RegTensor<float> vregSum0;
    RegTensor<float> vregSum1;
    RegTensor<float> vregSum2;
    RegTensor<float> vregSum3;

    RegTensor<float> vregExp0;
    RegTensor<float> vregExp1;
    RegTensor<float> vregExp2;
    RegTensor<float> vregExp3;

    RegTensor<float> vregF32_0;
    RegTensor<float> vregF32_1;
    RegTensor<float> vregF32_2;
    RegTensor<float> vregF32_3;

    RegTensor<float> vregStore0;
    RegTensor<float> vregStore1;
    RegTensor<float> vregStore2;
    RegTensor<float> vregStore3;

    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg pregAll;
    pregAll = CreateMask<T, MaskPattern::ALL>();

    for (uint16_t loopSc = 0; loopSc < uint16_t(vScRealSize / 4); ++loopSc) {
        for (uint16_t dChunk = 0; dChunk < 8; ++dChunk) {
            uint32_t dOffset = dChunk * 64;
            __ubuf__ float *srcUb0 = outputAddr + dOffset;
            __ubuf__ float *srcUb1 = srcUb0 + ReduceSize * RowSize;
            __ubuf__ float *srcUb2 = srcUb0 + ReduceSize * RowSize * 2;
            __ubuf__ float *srcUb3 = srcUb0 + ReduceSize * RowSize * 3;

            __ubuf__ float *inputAddr0 = inputAddr + dOffset;
            __ubuf__ float *inputAddr1 = inputAddr + dOffset + (ReduceSize * RowSize);
            __ubuf__ float *inputAddr2 = inputAddr + dOffset + (ReduceSize * RowSize * 2);
            __ubuf__ float *inputAddr3 = inputAddr + dOffset + (ReduceSize * RowSize * 3);

            Duplicate(max0, minValue);
            Duplicate(max1, minValue);
            Duplicate(max2, minValue);
            Duplicate(max3, minValue);

            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum0, 0, pregAll);
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum1, 0, pregAll);
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum2, 0, pregAll);
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum3, 0, pregAll);

            for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
                LoadAlign(src0, srcUb0 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
                LoadAlign(src1, srcUb1 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
                LoadAlign(src2, srcUb2 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
                LoadAlign(src3, srcUb3 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);

                Max(max0, max0, src0, pregAll);
                Max(max1, max1, src1, pregAll);
                Max(max2, max2, src2, pregAll);
                Max(max3, max3, src3, pregAll);
            }

            for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
                LoadAlign(vregF32_0, srcUb0 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
                LoadAlign(vregF32_1, srcUb1 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
                LoadAlign(vregF32_2, srcUb2 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
                LoadAlign(vregF32_3, srcUb3 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);

                FusedExpSub(vregExp0, vregF32_0, max0, pregAll);
                FusedExpSub(vregExp1, vregF32_1, max1, pregAll);
                FusedExpSub(vregExp2, vregF32_2, max2, pregAll);
                FusedExpSub(vregExp3, vregF32_3, max3, pregAll);

                Add(vregSum0, vregExp0, vregSum0, pregAll);
                Add(vregSum1, vregExp1, vregSum1, pregAll);
                Add(vregSum2, vregExp2, vregSum2, pregAll);
                Add(vregSum3, vregExp3, vregSum3, pregAll);

                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb0 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                    vregExp0, pregAll);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb1 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                    vregExp1, pregAll);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb2 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                    vregExp2, pregAll);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb3 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                    vregExp3, pregAll);
            }

            LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
                LoadAlign(vregExp0, srcUb0 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
                LoadAlign(vregExp1, srcUb1 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
                LoadAlign(vregExp2, srcUb2 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);
                LoadAlign(vregExp3, srcUb3 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4);

                Div(vregStore0, vregExp0, vregSum0, pregAll);
                Div(vregStore1, vregExp1, vregSum1, pregAll);
                Div(vregStore2, vregExp2, vregSum2, pregAll);
                Div(vregStore3, vregExp3, vregSum3, pregAll);

                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr0 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                    vregStore0, pregAll);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr1 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                    vregStore1, pregAll);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr2 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                    vregStore2, pregAll);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr3 + loopM * RowSize + ReduceSize * RowSize * loopSc * 4),
                    vregStore3, pregAll);
            }
        }
    }

    for (uint16_t loopSc = 0; loopSc < uint16_t(vScRealSize % 4); ++loopSc) {
        for (uint16_t dChunk = 0; dChunk < 8; ++dChunk) {
            uint32_t dOffset = dChunk * 64;
            __ubuf__ float *srcUb0 = outputAddr + dOffset;
            __ubuf__ float *inputAddr0 = inputAddr + dOffset;

            Duplicate(max0, minValue);
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vregSum0, 0, pregAll);

            for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
                LoadAlign(src0, srcUb0 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
                Max(max0, max0, src0, pregAll);
            }

            for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
                LoadAlign(vregF32_0, srcUb0 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
                FusedExpSub(vregExp0, vregF32_0, max0, pregAll);
                Add(vregSum0, vregExp0, vregSum0, pregAll);

                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)srcUb0 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4)),
                    vregExp0, pregAll);
            }

            LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            for (uint16_t loopM = 0; loopM < ReduceSize; ++loopM) {
                LoadAlign(vregExp0, srcUb0 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4));
                Div(vregStore0, vregExp0, vregSum0, pregAll);

                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)inputAddr0 + loopM * RowSize + ReduceSize * RowSize * (loopSc + vScRealSize / 4 * 4)),
                    vregStore0, pregAll);
            }
        }
    }
}

/*
 * @ingroup ProcessVec1Vf
 * @brief compute max = reducemax, exp(x-max)/sum(exp(x-max))
 * @param [out] dstTensor, output LocalTensor
 * @param [in] srcTensor, input LocalTensor
 * @param [in] RowSize, input rows
 * @param [in] vScBaseSize, input columns, should be 256 bytes aligned, the value is originN aligned to 64
 * @param [in] vScRealSize, input origin columns, support range: 0 < originN <= 128
 * @param [in] scale, scale value
 * @param [in] minValue, minimum value
 */

template <typename T>
__aicore__ inline void SoftmaxDnVF(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                   const uint32_t RowSize, const uint32_t ReduceSize, const uint32_t vScRealSize,
                                   const T minValue, const uint32_t dDealSize)
{
    __ubuf__ T *inputAddr = (__ubuf__ T*) dstTensor.GetPhyAddr();
    __ubuf__ T *outputAddr = (__ubuf__ T*) srcTensor.GetPhyAddr();
    if (dDealSize == 8) {
        SoftmaxDndBase8<T>(inputAddr, outputAddr, RowSize,
            ReduceSize, vScRealSize, minValue);
    } else if (dDealSize == 16) {
        SoftmaxDndBase16<T>(inputAddr, outputAddr, RowSize,
            ReduceSize, vScRealSize, minValue);
    } else if (dDealSize == 32) {
        SoftmaxDndBase32<T>(inputAddr, outputAddr, RowSize,
            ReduceSize, vScRealSize, minValue);
    } else if (dDealSize == 64) {
        SoftmaxDndBase64<T>(inputAddr, outputAddr, RowSize,
            ReduceSize, vScRealSize, minValue);
    } else if (dDealSize == 128) {
        SoftmaxDndBase128<T>(inputAddr, outputAddr, RowSize,
            ReduceSize, vScRealSize, minValue);
    } else if (dDealSize == 256) {
        SoftmaxDndBase256<T>(inputAddr, outputAddr, RowSize,
            ReduceSize, vScRealSize, minValue);
    } else if (dDealSize == 512) {
        SoftmaxDndBase512<T>(inputAddr, outputAddr, RowSize,
            ReduceSize, vScRealSize, minValue);
    }
}
}
#endif // VF_SOFTMAX_H
