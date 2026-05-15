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
__simd_vf__ inline void SoftmaxDndBase128(__ubuf__ T *x_softmax, __ubuf__ float *input_x_local_UB,
    const uint32_t RowSize, const uint32_t ReduceSize, const uint32_t vScRealSize,
    const T minValue)
{
    RegTensor<float> vreg_x_sum_0_0;
    RegTensor<float> vreg_x_sum_1_0;
    RegTensor<float> vreg_x_sum_2_0;
    RegTensor<float> vreg_x_sum_3_0;
    RegTensor<float> vreg_x_sum_0_1;
    RegTensor<float> vreg_x_sum_1_1;
    RegTensor<float> vreg_x_sum_2_1;
    RegTensor<float> vreg_x_sum_3_1;
    RegTensor<float> vreg_x_sum_0_2;
    RegTensor<float> vreg_x_sum_1_2;
    RegTensor<float> vreg_x_sum_2_2;
    RegTensor<float> vreg_x_sum_3_2;
    RegTensor<float> vreg_x_sum_0_3;
    RegTensor<float> vreg_x_sum_1_3;
    RegTensor<float> vreg_x_sum_2_3;
    RegTensor<float> vreg_x_sum_3_3;

    RegTensor<float> vreg_x_exp_0_0;
    RegTensor<float> vreg_x_exp_1_0;
    RegTensor<float> vreg_x_exp_2_0;
    RegTensor<float> vreg_x_exp_3_0;
    RegTensor<float> vreg_x_exp_0_1;
    RegTensor<float> vreg_x_exp_1_1;
    RegTensor<float> vreg_x_exp_2_1;
    RegTensor<float> vreg_x_exp_3_1;
    RegTensor<float> vreg_x_exp_0_2;
    RegTensor<float> vreg_x_exp_1_2;
    RegTensor<float> vreg_x_exp_2_2;
    RegTensor<float> vreg_x_exp_3_2;
    RegTensor<float> vreg_x_exp_0_3;
    RegTensor<float> vreg_x_exp_1_3;
    RegTensor<float> vreg_x_exp_2_3;
    RegTensor<float> vreg_x_exp_3_3;

    RegTensor<float> vreg_x_f32_0_0;
    RegTensor<float> vreg_x_f32_1_0;
    RegTensor<float> vreg_x_f32_2_0;
    RegTensor<float> vreg_x_f32_3_0;
    RegTensor<float> vreg_x_f32_0_1;
    RegTensor<float> vreg_x_f32_1_1;
    RegTensor<float> vreg_x_f32_2_1;
    RegTensor<float> vreg_x_f32_3_1;
    RegTensor<float> vreg_x_f32_0_2;
    RegTensor<float> vreg_x_f32_1_2;
    RegTensor<float> vreg_x_f32_2_2;
    RegTensor<float> vreg_x_f32_3_2;
    RegTensor<float> vreg_x_f32_0_3;
    RegTensor<float> vreg_x_f32_1_3;
    RegTensor<float> vreg_x_f32_2_3;
    RegTensor<float> vreg_x_f32_3_3;

    RegTensor<float> vreg_x_softmax_0_0;
    RegTensor<float> vreg_x_softmax_1_0;
    RegTensor<float> vreg_x_softmax_2_0;
    RegTensor<float> vreg_x_softmax_3_0;
    RegTensor<float> vreg_x_softmax_0_1;
    RegTensor<float> vreg_x_softmax_1_1;
    RegTensor<float> vreg_x_softmax_2_1;
    RegTensor<float> vreg_x_softmax_3_1;
    MaskReg preg_all;
    MaskReg preg_136;
    preg_all = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = static_cast<uint32_t>(128ULL);
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0_0, src1_0, src2_0, src3_0, src0_1, src1_1, src2_1, src3_1,
        src0_2, src1_2, src2_2, src3_2, src0_3, src1_3, src2_3, src3_3;
    RegTensor<float> max0_0, max1_0, max2_0, max3_0, max0_1, max1_1, max2_1, max3_1,
        max0_2, max1_2, max2_2, max3_2, max0_3, max1_3, max2_3, max3_3;
    RegTensor<float> vreg_min;

    __ubuf__ float *src_ub0_0 = input_x_local_UB;
    __ubuf__ float *src_ub0_1 = input_x_local_UB + RowSize / 2;
    __ubuf__ float *src_ub0_2 = input_x_local_UB + RowSize;
    __ubuf__ float *src_ub0_3 = input_x_local_UB + RowSize + RowSize / 2;
    __ubuf__ float *src_ub1_0 = src_ub0_0 + ReduceSize * RowSize;
    __ubuf__ float *src_ub1_1 = src_ub0_0 + ReduceSize * RowSize + RowSize / 2;
    __ubuf__ float *src_ub1_2 = src_ub0_0 + ReduceSize * RowSize + RowSize;
    __ubuf__ float *src_ub1_3 = src_ub0_0 + ReduceSize * RowSize + RowSize + RowSize / 2;
    __ubuf__ float *src_ub2_0 = src_ub0_0 + ReduceSize * RowSize * 2;
    __ubuf__ float *src_ub2_1 = src_ub0_0 + ReduceSize * RowSize * 2 + RowSize / 2;
    __ubuf__ float *src_ub2_2 = src_ub0_0 + ReduceSize * RowSize * 2 + RowSize;
    __ubuf__ float *src_ub2_3 = src_ub0_0 + ReduceSize * RowSize * 2 + RowSize + RowSize / 2;
    __ubuf__ float *src_ub3_0 = src_ub0_0 + ReduceSize * RowSize * 3;
    __ubuf__ float *src_ub3_1 = src_ub0_0 + ReduceSize * RowSize * 3 + RowSize / 2;
    __ubuf__ float *src_ub3_2 = src_ub0_0 + ReduceSize * RowSize * 3 + RowSize;
    __ubuf__ float *src_ub3_3 = src_ub0_0 + ReduceSize * RowSize * 3 + RowSize + RowSize / 2;

    __ubuf__ float *x_softmax_0_0 = x_softmax;
    __ubuf__ float *x_softmax_0_1 = x_softmax + RowSize / 2;
    __ubuf__ float *x_softmax_1_0 = x_softmax + (ReduceSize * RowSize);
    __ubuf__ float *x_softmax_1_1 = x_softmax + (ReduceSize * RowSize) + RowSize / 2;
    __ubuf__ float *x_softmax_2_0 = x_softmax + (ReduceSize * RowSize * 2);
    __ubuf__ float *x_softmax_2_1 = x_softmax + (ReduceSize * RowSize * 2) + RowSize / 2;
    __ubuf__ float *x_softmax_3_0 = x_softmax + (ReduceSize * RowSize * 3);
    __ubuf__ float *x_softmax_3_1 = x_softmax + (ReduceSize * RowSize * 3) + RowSize / 2;

    for (uint16_t iter_sc = 0; iter_sc < uint16_t(vScRealSize / 4); ++iter_sc) {
        Duplicate(max0_0, minValue);
        Duplicate(max1_0, minValue);
        Duplicate(max2_0, minValue);
        Duplicate(max3_0, minValue);
        Duplicate(max0_1, minValue);
        Duplicate(max1_1, minValue);
        Duplicate(max2_1, minValue);
        Duplicate(max3_1, minValue);
        Duplicate(max0_2, minValue);
        Duplicate(max1_2, minValue);
        Duplicate(max2_2, minValue);
        Duplicate(max3_2, minValue);
        Duplicate(max0_3, minValue);
        Duplicate(max1_3, minValue);
        Duplicate(max2_3, minValue);
        Duplicate(max3_3, minValue);

        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0_0, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1_0, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2_0, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3_0, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0_1, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1_1, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2_1, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3_1, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0_2, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1_2, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2_2, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3_2, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0_3, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1_3, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2_3, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3_3, 0, preg_all);
        for (uint16_t iter_m = 0; iter_m < uint16_t(ReduceSize / 2); ++iter_m) {
            LoadAlign(src0_0, src_ub0_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(src0_1, src_ub0_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(src0_2, src_ub0_2 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(src0_3, src_ub0_3 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);

            LoadAlign(src1_0, src_ub1_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(src1_1, src_ub1_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(src1_2, src_ub1_2 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(src1_3, src_ub1_3 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);

            LoadAlign(src2_0, src_ub2_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(src2_1, src_ub2_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(src2_2, src_ub2_2 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(src2_3, src_ub2_3 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);

            LoadAlign(src3_0, src_ub3_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(src3_1, src_ub3_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(src3_2, src_ub3_2 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(src3_3, src_ub3_3 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);

            Max(max0_0, max0_0, src0_0, preg_all);
            Max(max0_1, max0_1, src0_1, preg_all);
            Max(max0_2, max0_2, src0_2, preg_all);
            Max(max0_3, max0_3, src0_3, preg_all);
            Max(max1_0, max1_0, src1_0, preg_all);
            Max(max1_1, max1_1, src1_1, preg_all);
            Max(max1_2, max1_2, src1_2, preg_all);
            Max(max1_3, max1_3, src1_3, preg_all);
            Max(max2_0, max2_0, src2_0, preg_all);
            Max(max2_1, max2_1, src2_1, preg_all);
            Max(max2_2, max2_2, src2_2, preg_all);
            Max(max2_3, max2_3, src2_3, preg_all);
            Max(max3_0, max3_0, src3_0, preg_all);
            Max(max3_1, max3_1, src3_1, preg_all);
            Max(max3_2, max3_2, src3_2, preg_all);
            Max(max3_3, max3_3, src3_3, preg_all);
        }
        Max(max0_0, max0_0, max0_2, preg_all);
        Max(max0_1, max0_1, max0_3, preg_all);
        Max(max1_0, max1_0, max1_2, preg_all);
        Max(max1_1, max1_1, max1_3, preg_all);
        Max(max2_0, max2_0, max2_2, preg_all);
        Max(max2_1, max2_1, max2_3, preg_all);
        Max(max3_0, max3_0, max3_2, preg_all);
        Max(max3_1, max3_1, max3_3, preg_all);

        for (uint16_t iter_m = 0; iter_m < uint16_t(ReduceSize / 2); ++iter_m) {
            LoadAlign(vreg_x_f32_0_0, src_ub0_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_f32_0_1, src_ub0_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_f32_0_2, src_ub0_2 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_f32_0_3, src_ub0_3 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);

            LoadAlign(vreg_x_f32_1_0, src_ub1_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_f32_1_1, src_ub1_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_f32_1_2, src_ub1_2 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_f32_1_3, src_ub1_3 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);

            LoadAlign(vreg_x_f32_2_0, src_ub2_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_f32_2_1, src_ub2_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_f32_2_2, src_ub2_2 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_f32_2_3, src_ub2_3 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);

            LoadAlign(vreg_x_f32_3_0, src_ub3_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_f32_3_1, src_ub3_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_f32_3_2, src_ub3_2 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_f32_3_3, src_ub3_3 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            
            FusedExpSub(vreg_x_exp_0_0, vreg_x_f32_0_0, max0_0, preg_all);
            FusedExpSub(vreg_x_exp_0_1, vreg_x_f32_0_1, max0_1, preg_all);
            FusedExpSub(vreg_x_exp_0_2, vreg_x_f32_0_2, max0_0, preg_all);
            FusedExpSub(vreg_x_exp_0_3, vreg_x_f32_0_3, max0_1, preg_all);
            FusedExpSub(vreg_x_exp_1_0, vreg_x_f32_1_0, max1_0, preg_all);
            FusedExpSub(vreg_x_exp_1_1, vreg_x_f32_1_1, max1_1, preg_all);
            FusedExpSub(vreg_x_exp_1_2, vreg_x_f32_1_2, max1_0, preg_all);
            FusedExpSub(vreg_x_exp_1_3, vreg_x_f32_1_3, max1_1, preg_all);
            FusedExpSub(vreg_x_exp_2_0, vreg_x_f32_2_0, max2_0, preg_all);
            FusedExpSub(vreg_x_exp_2_1, vreg_x_f32_2_1, max2_1, preg_all);
            FusedExpSub(vreg_x_exp_2_2, vreg_x_f32_2_2, max2_0, preg_all);
            FusedExpSub(vreg_x_exp_2_3, vreg_x_f32_2_3, max2_1, preg_all);
            FusedExpSub(vreg_x_exp_3_0, vreg_x_f32_3_0, max3_0, preg_all);
            FusedExpSub(vreg_x_exp_3_1, vreg_x_f32_3_1, max3_1, preg_all);
            FusedExpSub(vreg_x_exp_3_2, vreg_x_f32_3_2, max3_0, preg_all);
            FusedExpSub(vreg_x_exp_3_3, vreg_x_f32_3_3, max3_1, preg_all);
            
            Add(vreg_x_sum_0_0, vreg_x_exp_0_0, vreg_x_sum_0_0, preg_all);
            Add(vreg_x_sum_0_1, vreg_x_exp_0_1, vreg_x_sum_0_1, preg_all);
            Add(vreg_x_sum_0_2, vreg_x_exp_0_2, vreg_x_sum_0_2, preg_all);
            Add(vreg_x_sum_0_3, vreg_x_exp_0_3, vreg_x_sum_0_3, preg_all);
            Add(vreg_x_sum_1_0, vreg_x_exp_1_0, vreg_x_sum_1_0, preg_all);
            Add(vreg_x_sum_1_1, vreg_x_exp_1_1, vreg_x_sum_1_1, preg_all);
            Add(vreg_x_sum_1_2, vreg_x_exp_1_2, vreg_x_sum_1_2, preg_all);
            Add(vreg_x_sum_1_3, vreg_x_exp_1_3, vreg_x_sum_1_3, preg_all);
            Add(vreg_x_sum_2_0, vreg_x_exp_2_0, vreg_x_sum_2_0, preg_all);
            Add(vreg_x_sum_2_1, vreg_x_exp_2_1, vreg_x_sum_2_1, preg_all);
            Add(vreg_x_sum_2_2, vreg_x_exp_2_2, vreg_x_sum_2_2, preg_all);
            Add(vreg_x_sum_2_3, vreg_x_exp_2_3, vreg_x_sum_2_3, preg_all);
            Add(vreg_x_sum_3_0, vreg_x_exp_3_0, vreg_x_sum_3_0, preg_all);
            Add(vreg_x_sum_3_1, vreg_x_exp_3_1, vreg_x_sum_3_1, preg_all);
            Add(vreg_x_sum_3_2, vreg_x_exp_3_2, vreg_x_sum_3_2, preg_all);
            Add(vreg_x_sum_3_3, vreg_x_exp_3_3, vreg_x_sum_3_3, preg_all);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub0_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_0_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub0_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_0_1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub0_2 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_0_2, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub0_3 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_0_3, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub1_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_1_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub1_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_1_1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub1_2 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_1_2, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub1_3 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_1_3, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub2_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_2_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub2_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_2_1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub2_2 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_2_2, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub2_3 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_2_3, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub3_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_3_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub3_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_3_1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub3_2 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_3_2, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub3_3 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_3_3, preg_all);
        }
        Add(vreg_x_sum_0_0, vreg_x_sum_0_0, vreg_x_sum_0_2, preg_all);
        Add(vreg_x_sum_0_1, vreg_x_sum_0_1, vreg_x_sum_0_3, preg_all);
        Add(vreg_x_sum_1_0, vreg_x_sum_1_0, vreg_x_sum_1_2, preg_all);
        Add(vreg_x_sum_1_1, vreg_x_sum_1_1, vreg_x_sum_1_3, preg_all);
        Add(vreg_x_sum_2_0, vreg_x_sum_2_0, vreg_x_sum_2_2, preg_all);
        Add(vreg_x_sum_2_1, vreg_x_sum_2_1, vreg_x_sum_2_3, preg_all);
        Add(vreg_x_sum_3_0, vreg_x_sum_3_0, vreg_x_sum_3_2, preg_all);
        Add(vreg_x_sum_3_1, vreg_x_sum_3_1, vreg_x_sum_3_3, preg_all);

        LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t iter_m = 0; iter_m < ReduceSize; ++iter_m) {
            LoadAlign(vreg_x_exp_0_0, src_ub0_0 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_exp_0_1, src_ub0_1 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_exp_1_0, src_ub1_0 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_exp_1_1, src_ub1_1 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_exp_2_0, src_ub2_0 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_exp_2_1, src_ub2_1 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_exp_3_0, src_ub3_0 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_exp_3_1, src_ub3_1 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4);

            Div(vreg_x_softmax_0_0, vreg_x_exp_0_0, vreg_x_sum_0_0, preg_all);
            Div(vreg_x_softmax_0_1, vreg_x_exp_0_1, vreg_x_sum_0_1, preg_all);
            Div(vreg_x_softmax_1_0, vreg_x_exp_1_0, vreg_x_sum_1_0, preg_all);
            Div(vreg_x_softmax_1_1, vreg_x_exp_1_1, vreg_x_sum_1_1, preg_all);
            Div(vreg_x_softmax_2_0, vreg_x_exp_2_0, vreg_x_sum_2_0, preg_all);
            Div(vreg_x_softmax_2_1, vreg_x_exp_2_1, vreg_x_sum_2_1, preg_all);
            Div(vreg_x_softmax_3_0, vreg_x_exp_3_0, vreg_x_sum_3_0, preg_all);
            Div(vreg_x_softmax_3_1, vreg_x_exp_3_1, vreg_x_sum_3_1, preg_all); 

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)x_softmax_0_0 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_softmax_0_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)x_softmax_0_1 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_softmax_0_1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)x_softmax_1_0 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_softmax_1_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)x_softmax_1_1 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_softmax_1_1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)x_softmax_2_0 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_softmax_2_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)x_softmax_2_1 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_softmax_2_1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)x_softmax_3_0 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_softmax_3_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)x_softmax_3_1 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_softmax_3_1, preg_all);
        }
    }
    // 尾块处理
    for (uint16_t iter_sc = 0; iter_sc < uint16_t(vScRealSize % 4); ++iter_sc) {
        Duplicate(max0_0, minValue);
        Duplicate(max0_1, minValue);
        Duplicate(max0_2, minValue);
        Duplicate(max0_3, minValue);

        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0_0, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0_1, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0_2, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0_3, 0, preg_all);
        for (uint16_t iter_m = 0; iter_m < uint16_t(ReduceSize / 2); ++iter_m) {
            LoadAlign(src0_0, src_ub0_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4));
            LoadAlign(src0_1, src_ub0_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4));
            LoadAlign(src0_2, src_ub0_2 + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4));
            LoadAlign(src0_3, src_ub0_3 + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4));

            Max(max0_0, max0_0, src0_0, preg_all);
            Max(max0_1, max0_1, src0_1, preg_all);
            Max(max0_2, max0_2, src0_2, preg_all);
            Max(max0_3, max0_3, src0_3, preg_all);
        }
        Max(max0_0, max0_0, max0_2, preg_all);
        Max(max0_1, max0_1, max0_3, preg_all);

        for (uint16_t iter_m = 0; iter_m < ReduceSize / 2; ++iter_m) {
            LoadAlign(vreg_x_f32_0_0, src_ub0_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4));
            LoadAlign(vreg_x_f32_0_1, src_ub0_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4));
            LoadAlign(vreg_x_f32_0_2, src_ub0_2 + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4));
            LoadAlign(vreg_x_f32_0_3, src_ub0_3 + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4));
            
            FusedExpSub(vreg_x_exp_0_0, vreg_x_f32_0_0, max0_0, preg_all);
            FusedExpSub(vreg_x_exp_0_1, vreg_x_f32_0_1, max0_1, preg_all);
            FusedExpSub(vreg_x_exp_0_2, vreg_x_f32_0_2, max0_0, preg_all);
            FusedExpSub(vreg_x_exp_0_3, vreg_x_f32_0_3, max0_1, preg_all);
            
            Add(vreg_x_sum_0_0, vreg_x_exp_0_0, vreg_x_sum_0_0, preg_all);
            Add(vreg_x_sum_0_1, vreg_x_exp_0_1, vreg_x_sum_0_1, preg_all);
            Add(vreg_x_sum_0_2, vreg_x_exp_0_2, vreg_x_sum_0_2, preg_all);
            Add(vreg_x_sum_0_3, vreg_x_exp_0_3, vreg_x_sum_0_3, preg_all);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub0_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4)),
                vreg_x_exp_0_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub0_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4)),
                vreg_x_exp_0_1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub0_2 + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4)),
                vreg_x_exp_0_2, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub0_3 + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4)),
                vreg_x_exp_0_3, preg_all);
        }
        Add(vreg_x_sum_0_0, vreg_x_sum_0_0, vreg_x_sum_0_2, preg_all);
        Add(vreg_x_sum_0_1, vreg_x_sum_0_1, vreg_x_sum_0_3, preg_all);

        LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t iter_m = 0; iter_m < ReduceSize; ++iter_m) {
            LoadAlign(vreg_x_exp_0_0, src_ub0_0 + iter_m * RowSize + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4));
            LoadAlign(vreg_x_exp_0_1, src_ub0_1 + iter_m * RowSize + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4));

            Div(vreg_x_softmax_0_0, vreg_x_exp_0_0, vreg_x_sum_0_0, preg_all);
            Div(vreg_x_softmax_0_1, vreg_x_exp_0_1, vreg_x_sum_0_1, preg_all);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)x_softmax_0_0 + iter_m * RowSize + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4)),
                vreg_x_softmax_0_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)x_softmax_0_1 + iter_m * RowSize + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4)),
                vreg_x_softmax_0_1, preg_all);
        }
    }
}

template <typename T>
__simd_vf__ inline void SoftmaxDndBase64(__ubuf__ T *x_softmax, __ubuf__ float *input_x_local_UB,
    const uint32_t RowSize, const uint32_t ReduceSize, const uint32_t vScRealSize,
    const T minValue)
{
    RegTensor<float> vreg_x_sum_0_0;
    RegTensor<float> vreg_x_sum_1_0;
    RegTensor<float> vreg_x_sum_2_0;
    RegTensor<float> vreg_x_sum_3_0;
    RegTensor<float> vreg_x_sum_0_1;
    RegTensor<float> vreg_x_sum_1_1;
    RegTensor<float> vreg_x_sum_2_1;
    RegTensor<float> vreg_x_sum_3_1;

    RegTensor<float> vreg_x_exp_0_0;
    RegTensor<float> vreg_x_exp_1_0;
    RegTensor<float> vreg_x_exp_2_0;
    RegTensor<float> vreg_x_exp_3_0;
    RegTensor<float> vreg_x_exp_0_1;
    RegTensor<float> vreg_x_exp_1_1;
    RegTensor<float> vreg_x_exp_2_1;
    RegTensor<float> vreg_x_exp_3_1;

    RegTensor<float> vreg_x_f32_0_0;
    RegTensor<float> vreg_x_f32_1_0;
    RegTensor<float> vreg_x_f32_2_0;
    RegTensor<float> vreg_x_f32_3_0;
    RegTensor<float> vreg_x_f32_0_1;
    RegTensor<float> vreg_x_f32_1_1;
    RegTensor<float> vreg_x_f32_2_1;
    RegTensor<float> vreg_x_f32_3_1;

    RegTensor<float> vreg_x_softmax_0;
    RegTensor<float> vreg_x_softmax_1;
    RegTensor<float> vreg_x_softmax_2;
    RegTensor<float> vreg_x_softmax_3;
    MaskReg preg_all;
    MaskReg preg_136;
    preg_all = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = static_cast<uint32_t>(128ULL);
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0_0, src1_0, src2_0, src3_0, src0_1, src1_1, src2_1, src3_1;
    RegTensor<float> max0_0, max1_0, max2_0, max3_0, max0_1, max1_1, max2_1, max3_1;
    RegTensor<float> vreg_min;

    __ubuf__ float *src_ub0_0 = input_x_local_UB;
    __ubuf__ float *src_ub0_1 = input_x_local_UB + RowSize;
    __ubuf__ float *src_ub1_0 = src_ub0_0 + ReduceSize * RowSize;
    __ubuf__ float *src_ub1_1 = src_ub0_0 + ReduceSize * RowSize + RowSize;
    __ubuf__ float *src_ub2_0 = src_ub0_0 + ReduceSize * RowSize * 2;
    __ubuf__ float *src_ub2_1 = src_ub0_0 + ReduceSize * RowSize * 2 + RowSize;
    __ubuf__ float *src_ub3_0 = src_ub0_0 + ReduceSize * RowSize * 3;
    __ubuf__ float *src_ub3_1 = src_ub0_0 + ReduceSize * RowSize * 3 + RowSize;

    __ubuf__ float *x_softmax_0 = x_softmax;
    __ubuf__ float *x_softmax_1 = x_softmax + (ReduceSize * RowSize);
    __ubuf__ float *x_softmax_2 = x_softmax + (ReduceSize * RowSize * 2);
    __ubuf__ float *x_softmax_3 = x_softmax + (ReduceSize * RowSize * 3);

    for (uint16_t iter_sc = 0; iter_sc < uint16_t(vScRealSize / 4); ++iter_sc) {
        Duplicate(max0_0, minValue);
        Duplicate(max1_0, minValue);
        Duplicate(max2_0, minValue);
        Duplicate(max3_0, minValue);
        Duplicate(max0_1, minValue);
        Duplicate(max1_1, minValue);
        Duplicate(max2_1, minValue);
        Duplicate(max3_1, minValue);

        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0_0, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1_0, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2_0, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3_0, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0_1, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1_1, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2_1, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3_1, 0, preg_all);
        for (uint16_t iter_m = 0; iter_m < uint16_t(ReduceSize / 2); ++iter_m) {
            LoadAlign(src0_0, src_ub0_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(src0_1, src_ub0_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);

            LoadAlign(src1_0, src_ub1_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(src1_1, src_ub1_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);

            LoadAlign(src2_0, src_ub2_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(src2_1, src_ub2_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);

            LoadAlign(src3_0, src_ub3_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(src3_1, src_ub3_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);

            Max(max0_0, max0_0, src0_0, preg_all);
            Max(max0_1, max0_1, src0_1, preg_all);
            Max(max1_0, max1_0, src1_0, preg_all);
            Max(max1_1, max1_1, src1_1, preg_all);
            Max(max2_0, max2_0, src2_0, preg_all);
            Max(max2_1, max2_1, src2_1, preg_all);
            Max(max3_0, max3_0, src3_0, preg_all);
            Max(max3_1, max3_1, src3_1, preg_all);
        }
        Max(max0_0, max0_0, max0_1, preg_all);
        Max(max1_0, max1_0, max1_1, preg_all);
        Max(max2_0, max2_0, max2_1, preg_all);
        Max(max3_0, max3_0, max3_1, preg_all);

        for (uint16_t iter_m = 0; iter_m < uint16_t(ReduceSize / 2); ++iter_m) {
            LoadAlign(vreg_x_f32_0_0, src_ub0_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_f32_0_1, src_ub0_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);

            LoadAlign(vreg_x_f32_1_0, src_ub1_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_f32_1_1, src_ub1_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);

            LoadAlign(vreg_x_f32_2_0, src_ub2_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_f32_2_1, src_ub2_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);

            LoadAlign(vreg_x_f32_3_0, src_ub3_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_f32_3_1, src_ub3_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            
            FusedExpSub(vreg_x_exp_0_0, vreg_x_f32_0_0, max0_0, preg_all);
            FusedExpSub(vreg_x_exp_0_1, vreg_x_f32_0_1, max0_0, preg_all);
            FusedExpSub(vreg_x_exp_1_0, vreg_x_f32_1_0, max1_0, preg_all);
            FusedExpSub(vreg_x_exp_1_1, vreg_x_f32_1_1, max1_0, preg_all);
            FusedExpSub(vreg_x_exp_2_0, vreg_x_f32_2_0, max2_0, preg_all);
            FusedExpSub(vreg_x_exp_2_1, vreg_x_f32_2_1, max2_0, preg_all);
            FusedExpSub(vreg_x_exp_3_0, vreg_x_f32_3_0, max3_0, preg_all);
            FusedExpSub(vreg_x_exp_3_1, vreg_x_f32_3_1, max3_0, preg_all);
            
            Add(vreg_x_sum_0_0, vreg_x_exp_0_0, vreg_x_sum_0_0, preg_all);
            Add(vreg_x_sum_0_1, vreg_x_exp_0_1, vreg_x_sum_0_1, preg_all);
            Add(vreg_x_sum_1_0, vreg_x_exp_1_0, vreg_x_sum_1_0, preg_all);
            Add(vreg_x_sum_1_1, vreg_x_exp_1_1, vreg_x_sum_1_1, preg_all);
            Add(vreg_x_sum_2_0, vreg_x_exp_2_0, vreg_x_sum_2_0, preg_all);
            Add(vreg_x_sum_2_1, vreg_x_exp_2_1, vreg_x_sum_2_1, preg_all);
            Add(vreg_x_sum_3_0, vreg_x_exp_3_0, vreg_x_sum_3_0, preg_all);
            Add(vreg_x_sum_3_1, vreg_x_exp_3_1, vreg_x_sum_3_1, preg_all);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub0_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_0_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub0_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_0_1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub1_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_1_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub1_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_1_1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub2_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_2_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub2_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_2_1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub3_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_3_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub3_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_3_1, preg_all);
        }
        Add(vreg_x_sum_0_0, vreg_x_sum_0_0, vreg_x_sum_0_1, preg_all);
        Add(vreg_x_sum_1_0, vreg_x_sum_1_0, vreg_x_sum_1_1, preg_all);
        Add(vreg_x_sum_2_0, vreg_x_sum_2_0, vreg_x_sum_2_1, preg_all);
        Add(vreg_x_sum_3_0, vreg_x_sum_3_0, vreg_x_sum_3_1, preg_all);

        LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t iter_m = 0; iter_m < ReduceSize; ++iter_m) {
            LoadAlign(vreg_x_exp_0_0, src_ub0_0 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_exp_1_0, src_ub1_0 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_exp_2_0, src_ub2_0 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_exp_3_0, src_ub3_0 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4);

            Div(vreg_x_softmax_0, vreg_x_exp_0_0, vreg_x_sum_0_0, preg_all);
            Div(vreg_x_softmax_1, vreg_x_exp_1_0, vreg_x_sum_1_0, preg_all);
            Div(vreg_x_softmax_2, vreg_x_exp_2_0, vreg_x_sum_2_0, preg_all);
            Div(vreg_x_softmax_3, vreg_x_exp_3_0, vreg_x_sum_3_0, preg_all); 

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)x_softmax_0 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_softmax_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)x_softmax_1 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_softmax_1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)x_softmax_2 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_softmax_2, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)x_softmax_3 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_softmax_3, preg_all);
        }
    }
    // 尾块处理
    for (uint16_t iter_sc = 0; iter_sc < uint16_t(vScRealSize % 4); ++iter_sc) {
        Duplicate(max0_0, minValue);
        Duplicate(max0_1, minValue);

        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0_0, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0_1, 0, preg_all);
        for (uint16_t iter_m = 0; iter_m < uint16_t(ReduceSize / 2); ++iter_m) {
            LoadAlign(src0_0, src_ub0_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4));
            LoadAlign(src0_1, src_ub0_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4));

            Max(max0_0, max0_0, src0_0, preg_all);
            Max(max0_1, max0_1, src0_1, preg_all);
        }
        Max(max0_0, max0_0, max0_1, preg_all);

        for (uint16_t iter_m = 0; iter_m < ReduceSize / 2; ++iter_m) {
            LoadAlign(vreg_x_f32_0_0, src_ub0_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4));
            LoadAlign(vreg_x_f32_0_1, src_ub0_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4));
            
            FusedExpSub(vreg_x_exp_0_0, vreg_x_f32_0_0, max0_0, preg_all);
            FusedExpSub(vreg_x_exp_0_1, vreg_x_f32_0_1, max0_0, preg_all);
            
            Add(vreg_x_sum_0_0, vreg_x_exp_0_0, vreg_x_sum_0_0, preg_all);
            Add(vreg_x_sum_0_1, vreg_x_exp_0_1, vreg_x_sum_0_1, preg_all);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub0_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4)),
                vreg_x_exp_0_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub0_1 + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4)),
                vreg_x_exp_0_1, preg_all);
        }
        Add(vreg_x_sum_0_0, vreg_x_sum_0_0, vreg_x_sum_0_1, preg_all);

        LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t iter_m = 0; iter_m < ReduceSize; ++iter_m) {
            LoadAlign(vreg_x_exp_0_0, src_ub0_0 + iter_m * RowSize + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4));
            Div(vreg_x_softmax_0, vreg_x_exp_0_0, vreg_x_sum_0_0, preg_all);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)x_softmax_0 + iter_m * RowSize + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4)),
                vreg_x_softmax_0, preg_all);
        }
    }
}

template <typename T>
__simd_vf__ inline void SoftmaxDndBase32(__ubuf__ T *x_softmax, __ubuf__ float *input_x_local_UB,
    const uint32_t RowSize, const uint32_t ReduceSize, const uint32_t vScRealSize,
    const T minValue)
{
    RegTensor<float> vreg_x_sum_0_0;
    RegTensor<float> vreg_x_sum_1_0;
    RegTensor<float> vreg_x_sum_2_0;
    RegTensor<float> vreg_x_sum_3_0;
    RegTensor<float> vreg_x_sum_0_1;
    RegTensor<float> vreg_x_sum_1_1;
    RegTensor<float> vreg_x_sum_2_1;
    RegTensor<float> vreg_x_sum_3_1;

    RegTensor<float> vreg_x_exp_0_0;
    RegTensor<float> vreg_x_exp_1_0;
    RegTensor<float> vreg_x_exp_2_0;
    RegTensor<float> vreg_x_exp_3_0;
    RegTensor<float> vreg_x_exp_0_1;
    RegTensor<float> vreg_x_exp_1_1;
    RegTensor<float> vreg_x_exp_2_1;
    RegTensor<float> vreg_x_exp_3_1;

    RegTensor<float> vreg_x_f32_0_0;
    RegTensor<float> vreg_x_f32_1_0;
    RegTensor<float> vreg_x_f32_2_0;
    RegTensor<float> vreg_x_f32_3_0;
    RegTensor<float> vreg_x_f32_0_1;
    RegTensor<float> vreg_x_f32_1_1;
    RegTensor<float> vreg_x_f32_2_1;
    RegTensor<float> vreg_x_f32_3_1;

    RegTensor<float> vreg_x_softmax_0;
    RegTensor<float> vreg_x_softmax_1;
    RegTensor<float> vreg_x_softmax_2;
    RegTensor<float> vreg_x_softmax_3;

    MaskReg preg_LHalf;
    MaskReg preg_HHalf;
    MaskReg preg_all;
    MaskReg preg_136;
    preg_all = CreateMask<T, MaskPattern::ALL>();
    preg_LHalf = CreateMask<T, MaskPattern::VL32>();
    Not(preg_HHalf, preg_LHalf, preg_all);
    uint32_t sreg_92 = static_cast<uint32_t>(128ULL);
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> max0, max1, max2, max3;
    RegTensor<float> src0_0, src1_0, src2_0, src3_0, src0_1, src1_1, src2_1, src3_1;
    RegTensor<float> max0_0, max1_0, max2_0, max3_0, max0_1, max1_1, max2_1, max3_1;
    RegTensor<float> vreg_min;

    __ubuf__ float *src_ub0_0 = input_x_local_UB;
    __ubuf__ float *src_ub0_1 = input_x_local_UB + RowSize * 2;
    __ubuf__ float *src_ub1_0 = src_ub0_0 + ReduceSize * RowSize;
    __ubuf__ float *src_ub1_1 = src_ub0_0 + ReduceSize * RowSize + RowSize * 2;
    __ubuf__ float *src_ub2_0 = src_ub0_0 + ReduceSize * RowSize * 2;
    __ubuf__ float *src_ub2_1 = src_ub0_0 + ReduceSize * RowSize * 2 + RowSize * 2;
    __ubuf__ float *src_ub3_0 = src_ub0_0 + ReduceSize * RowSize * 3;
    __ubuf__ float *src_ub3_1 = src_ub0_0 + ReduceSize * RowSize * 3 + RowSize * 2;

    __ubuf__ float *x_softmax_0 = x_softmax;
    __ubuf__ float *x_softmax_1 = x_softmax + (ReduceSize * RowSize);
    __ubuf__ float *x_softmax_2 = x_softmax + (ReduceSize * RowSize * 2);
    __ubuf__ float *x_softmax_3 = x_softmax + (ReduceSize * RowSize * 3);

    for (uint16_t iter_sc = 0; iter_sc < uint16_t(vScRealSize / 4); ++iter_sc) {
        Duplicate(max0, minValue);
        Duplicate(max1, minValue);
        Duplicate(max2, minValue);
        Duplicate(max3, minValue);
        Duplicate(max0_0, minValue);
        Duplicate(max1_0, minValue);
        Duplicate(max2_0, minValue);
        Duplicate(max3_0, minValue);
        Duplicate(max0_1, minValue);
        Duplicate(max1_1, minValue);
        Duplicate(max2_1, minValue);
        Duplicate(max3_1, minValue);

        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0_0, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1_0, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2_0, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3_0, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0_1, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1_1, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2_1, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3_1, 0, preg_all);
        for (uint16_t iter_m = 0; iter_m < uint16_t(ReduceSize / 4); ++iter_m) {
            LoadAlign(src0_0, src_ub0_0 + iter_m * RowSize * 4 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(src0_1, src_ub0_1 + iter_m * RowSize * 4 + ReduceSize * RowSize * iter_sc * 4);

            LoadAlign(src1_0, src_ub1_0 + iter_m * RowSize * 4 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(src1_1, src_ub1_1 + iter_m * RowSize * 4 + ReduceSize * RowSize * iter_sc * 4);

            LoadAlign(src2_0, src_ub2_0 + iter_m * RowSize * 4 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(src2_1, src_ub2_1 + iter_m * RowSize * 4 + ReduceSize * RowSize * iter_sc * 4);

            LoadAlign(src3_0, src_ub3_0 + iter_m * RowSize * 4 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(src3_1, src_ub3_1 + iter_m * RowSize * 4 + ReduceSize * RowSize * iter_sc * 4);

            Max(max0_0, max0_0, src0_0, preg_all);
            Max(max0_1, max0_1, src0_1, preg_all);
            Max(max1_0, max1_0, src1_0, preg_all);
            Max(max1_1, max1_1, src1_1, preg_all);
            Max(max2_0, max2_0, src2_0, preg_all);
            Max(max2_1, max2_1, src2_1, preg_all);
            Max(max3_0, max3_0, src3_0, preg_all);
            Max(max3_1, max3_1, src3_1, preg_all);
        }
        Max(max0, max0_0, max0_1, preg_all);
        Max(max1, max1_0, max1_1, preg_all);
        Max(max2, max2_0, max2_1, preg_all);
        Max(max3, max3_0, max3_1, preg_all);

        Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(max0_0, max0, preg_LHalf);
        Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(max0_1, max0, preg_HHalf);
        Max(max0, max0_0, max0_1, preg_LHalf);

        Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(max1_0, max1, preg_LHalf);
        Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(max1_1, max1, preg_HHalf);
        Max(max1, max1_0, max1_1, preg_LHalf);

        Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(max2_0, max2, preg_LHalf);
        Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(max2_1, max2, preg_HHalf);
        Max(max2, max2_0, max2_1, preg_LHalf);

        Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(max3_0, max3, preg_LHalf);
        Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(max3_1, max3, preg_HHalf);
        Max(max3, max3_0, max3_1, preg_LHalf);

        for (uint16_t iter_m = 0; iter_m < uint16_t(ReduceSize / 2); ++iter_m) {
            LoadAlign(vreg_x_f32_0_0, src_ub0_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_f32_0_1, (src_ub0_0 + RowSize) + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);

            LoadAlign(vreg_x_f32_1_0, src_ub1_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_f32_1_1, (src_ub1_0 + RowSize) + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);

            LoadAlign(vreg_x_f32_2_0, src_ub2_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_f32_2_1, (src_ub2_0 + RowSize) + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);

            LoadAlign(vreg_x_f32_3_0, src_ub3_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_f32_3_1, (src_ub3_0 + RowSize) + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4);
            
            FusedExpSub(vreg_x_exp_0_0, vreg_x_f32_0_0, max0, preg_all);
            FusedExpSub(vreg_x_exp_0_1, vreg_x_f32_0_1, max0, preg_all);
            FusedExpSub(vreg_x_exp_1_0, vreg_x_f32_1_0, max1, preg_all);
            FusedExpSub(vreg_x_exp_1_1, vreg_x_f32_1_1, max1, preg_all);
            FusedExpSub(vreg_x_exp_2_0, vreg_x_f32_2_0, max2, preg_all);
            FusedExpSub(vreg_x_exp_2_1, vreg_x_f32_2_1, max2, preg_all);
            FusedExpSub(vreg_x_exp_3_0, vreg_x_f32_3_0, max3, preg_all);
            FusedExpSub(vreg_x_exp_3_1, vreg_x_f32_3_1, max3, preg_all);
            
            Add(vreg_x_sum_0_0, vreg_x_exp_0_0, vreg_x_sum_0_0, preg_all);
            Add(vreg_x_sum_0_1, vreg_x_exp_0_1, vreg_x_sum_0_1, preg_all);
            Add(vreg_x_sum_1_0, vreg_x_exp_1_0, vreg_x_sum_1_0, preg_all);
            Add(vreg_x_sum_1_1, vreg_x_exp_1_1, vreg_x_sum_1_1, preg_all);
            Add(vreg_x_sum_2_0, vreg_x_exp_2_0, vreg_x_sum_2_0, preg_all);
            Add(vreg_x_sum_2_1, vreg_x_exp_2_1, vreg_x_sum_2_1, preg_all);
            Add(vreg_x_sum_3_0, vreg_x_exp_3_0, vreg_x_sum_3_0, preg_all);
            Add(vreg_x_sum_3_1, vreg_x_exp_3_1, vreg_x_sum_3_1, preg_all);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub0_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_0_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub0_0 + RowSize + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_0_1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub1_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_1_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub1_0 + RowSize + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_1_1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub2_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_2_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub2_0 + RowSize + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_2_1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub3_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_3_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub3_0 + RowSize + iter_m * RowSize * 2 + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_exp_3_1, preg_all);
        }
        Add(vreg_x_sum_0_0, vreg_x_sum_0_0, vreg_x_sum_0_1, preg_all);
        Add(vreg_x_sum_1_0, vreg_x_sum_1_0, vreg_x_sum_1_1, preg_all);
        Add(vreg_x_sum_2_0, vreg_x_sum_2_0, vreg_x_sum_2_1, preg_all);
        Add(vreg_x_sum_3_0, vreg_x_sum_3_0, vreg_x_sum_3_1, preg_all);

        LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t iter_m = 0; iter_m < ReduceSize; ++iter_m) {
            LoadAlign(vreg_x_exp_0_0, src_ub0_0 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_exp_1_0, src_ub1_0 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_exp_2_0, src_ub2_0 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4);
            LoadAlign(vreg_x_exp_3_0, src_ub3_0 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4);

            Div(vreg_x_softmax_0, vreg_x_exp_0_0, vreg_x_sum_0_0, preg_all);
            Div(vreg_x_softmax_1, vreg_x_exp_1_0, vreg_x_sum_1_0, preg_all);
            Div(vreg_x_softmax_2, vreg_x_exp_2_0, vreg_x_sum_2_0, preg_all);
            Div(vreg_x_softmax_3, vreg_x_exp_3_0, vreg_x_sum_3_0, preg_all); 

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)x_softmax_0 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_softmax_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)x_softmax_1 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_softmax_1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)x_softmax_2 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_softmax_2, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)x_softmax_3 + iter_m * RowSize + ReduceSize * RowSize * iter_sc * 4),
                vreg_x_softmax_3, preg_all);
        }
    }
    // 尾块处理
    for (uint16_t iter_sc = 0; iter_sc < uint16_t(vScRealSize % 4); ++iter_sc) {
        Duplicate(max0, minValue);
        Duplicate(max0_0, minValue);
        Duplicate(max0_1, minValue);

        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0_0, 0, preg_all);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0_1, 0, preg_all);
        for (uint16_t iter_m = 0; iter_m < uint16_t(ReduceSize / 4); ++iter_m) {
            LoadAlign(src0_0, src_ub0_0 + iter_m * RowSize * 4 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4));
            LoadAlign(src0_1, src_ub0_1 + iter_m * RowSize * 4 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4));

            Max(max0_0, max0_0, src0_0, preg_all);
            Max(max0_1, max0_1, src0_1, preg_all);
        }
        Max(max0, max0_0, max0_1, preg_all);

        Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(max0_0, max0, preg_LHalf);
        Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(max0_1, max0, preg_HHalf);
        Max(max0, max0_0, max0_1, preg_LHalf);

        for (uint16_t iter_m = 0; iter_m < uint16_t(ReduceSize / 2); ++iter_m) {
            LoadAlign(vreg_x_f32_0_0, src_ub0_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4));
            LoadAlign(vreg_x_f32_0_1, (src_ub0_0 + RowSize) + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4));
            
            FusedExpSub(vreg_x_exp_0_0, vreg_x_f32_0_0, max0, preg_all);
            FusedExpSub(vreg_x_exp_0_1, vreg_x_f32_0_1, max0, preg_all);
            
            Add(vreg_x_sum_0_0, vreg_x_exp_0_0, vreg_x_sum_0_0, preg_all);
            Add(vreg_x_sum_0_1, vreg_x_exp_0_1, vreg_x_sum_0_1, preg_all);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub0_0 + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4)),
                vreg_x_exp_0_0, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)src_ub0_0 + RowSize + iter_m * RowSize * 2 + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4)),
                vreg_x_exp_0_1, preg_all);
        }
        Add(vreg_x_sum_0_0, vreg_x_sum_0_0, vreg_x_sum_0_1, preg_all);

        LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t iter_m = 0; iter_m < ReduceSize; ++iter_m) {
            LoadAlign(vreg_x_exp_0_0, src_ub0_0 + iter_m * RowSize + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4));

            Div(vreg_x_softmax_0, vreg_x_exp_0_0, vreg_x_sum_0_0, preg_all);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(((__ubuf__ T *&)x_softmax_0 + iter_m * RowSize + ReduceSize * RowSize * (iter_sc + vScRealSize / 4 * 4)),
                vreg_x_softmax_0, preg_LHalf);
        }
    }
}

/*
 * @ingroup ProcessVec1Vf
 * @brief compute max = reducemax, exp(x-max)/sum(exp(x-max))
 * @param [out] dstTensor, output LocalTensor
 * @param [in] srcTensor, input LocalTensor
 * @param [in] RowSize, input rows
 * @param [in] vScBaseSize, input colums, should be 256 bytes aligned, the value is originN aligned to 64
 * @param [in] vScRealSize, input origin colums, support range: 0 < originN <= 128
 * @param [in] scale, scale value
 * @param [in] minValue, minimum value
 */

template <typename T>
__aicore__ inline void SoftmaxDnVF(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                   const uint32_t RowSize, const uint32_t ReduceSize, const uint32_t vScRealSize,
                                   const T minValue, const uint32_t dDealSize)
{
    __ubuf__ T *x_softmax = (__ubuf__ T*) dstTensor.GetPhyAddr();
    __ubuf__ T *input_x_local_UB = (__ubuf__ T*) srcTensor.GetPhyAddr();
    if (dDealSize == 64) {
        SoftmaxDndBase64<T>(x_softmax, input_x_local_UB, RowSize,
            ReduceSize, vScRealSize, minValue);
    } else if (dDealSize == 32) {
        SoftmaxDndBase32<T>(x_softmax, input_x_local_UB, RowSize,
            ReduceSize, vScRealSize, minValue);
    } else {
        SoftmaxDndBase128<T>(x_softmax, input_x_local_UB, RowSize,
            ReduceSize, vScRealSize, minValue);
    }
}
}
#endif // VF_SOFTMAX_H