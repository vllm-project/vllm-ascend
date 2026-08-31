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
 * \file vf_basic_block_fullquant_unaligned512_update.h
 * \brief
 */
#ifndef VF_BASIC_BLOCK_FULLQUANT_UNALIGNED512_UPDATE_H
#define VF_BASIC_BLOCK_FULLQUANT_UNALIGNED512_UPDATE_H

#include "vf_basic_block_utils.h"

using namespace regbaseutil;

namespace FaVectorApi {
template <typename T, typename T2, typename OUTPUT_T, uint32_t s1BaseSize = 16, uint32_t s2BaseSize = 512,
          bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false>
__simd_vf__ void ProcessVec1UpdateGeneralImpl512GqaFullquantVF(
    __ubuf__ T2 *expUb1, __ubuf__ T2 *expUb2, __ubuf__ T *srcUb1, __ubuf__ T *srcUb2, __ubuf__ T *maxUb,
    __ubuf__ T *srcMaxUb, __ubuf__ T *tmpExpSumUb, __ubuf__ T *tmpMaxUb, __ubuf__ T *tmpMaxUb2,
    __ubuf__ uint8_t *indexesUb, const uint16_t m, const uint32_t n, const T minValue, const float pScale)
{
    RegTensor<half> vreg_min;
    RegTensor<half> vreg_p_scale;
    RegTensor<half> vreg_ln_p_scale;
    RegTensor<half> vreg_input_x_1;
    RegTensor<half> vreg_input_x_unroll_1;
    RegTensor<half> vreg_input_x_2;
    RegTensor<half> vreg_input_x_unroll_2;
    RegTensor<half> vreg_max_tmp;        // 一个分形前8行的max
    RegTensor<half> vreg_max_tmp_unroll; // 一个分形后8行
    RegTensor<half> vreg_max_tmp_2;
    RegTensor<half> vreg_max_tmp_unroll_2;
    RegTensor<half> vreg_in_max;
    RegTensor<half> vreg_input_max;
    RegTensor<half> vreg_max_new;
    RegTensor<half> vreg_max;
    RegTensor<half> vreg_max_2;
    RegTensor<float> vreg_exp_sum_1; // 前8行的 和
    RegTensor<float> vreg_exp_sum_2; // 接着的后8行
    RegTensor<float> vreg_exp_0_2;
    RegTensor<float> vreg_exp_1_2;
    RegTensor<float> vreg_exp_2_2;
    RegTensor<float> vreg_exp_3_2;
    RegTensor<float> vreg_exp_0_1;
    RegTensor<float> vreg_exp_1_1;
    RegTensor<float> vreg_exp_2_1;
    RegTensor<float> vreg_exp_3_1;
    // f8
    RegTensor<T2> vreg_exp_0_f8_1;
    RegTensor<T2> vreg_exp_2_f8_1;
    RegTensor<T2> vreg_exp_1_f8_1;
    RegTensor<T2> vreg_exp_3_f8_1;
    RegTensor<T2> vreg_exp_0_f8_2;
    RegTensor<T2> vreg_exp_2_f8_2;
    RegTensor<T2> vreg_exp_1_f8_2;
    RegTensor<T2> vreg_exp_3_f8_2;
    RegTensor<T2> vreg_exp_merge_tmp_f8_1_1; // 前8行
    RegTensor<T2> vreg_exp_merge_tmp_f8_1_2;
    RegTensor<T2> vreg_exp_merge_f8_1;
    RegTensor<T2> vreg_exp_merge_tmp_f8_2_1; // 后8行
    RegTensor<T2> vreg_exp_merge_tmp_f8_2_2;
    RegTensor<T2> vreg_exp_merge_f8_2;

    RegTensor<uint8_t> vreg_exp_merge_f8_idxs;
    UnalignRegForStore ureg_max;
    UnalignReg ureg_exp_sum;
    MaskReg preg_all = CreateMask<half, MaskPattern::ALL>();
    MaskReg preg_all_b8 = CreateMask<uint8_t, MaskPattern::ALL>();

    Duplicate(vreg_min, minValue);
    Duplicate(vreg_p_scale, static_cast<half>(pScale));
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_all);
    for (uint16_t i = 0; i < 4; ++i) {
        // 一次循环处理 [32, 16, 16]
        LoadAlign(vreg_input_x_1, srcUb1 + i * 16 * 16);                 // 第一个256中第一个[64, 16]的前8行
        LoadAlign(vreg_input_x_unroll_1, srcUb1 + i * 16 * 16 + 8 * 16); // 第一个[64, 16]接下来的8行
        LoadAlign(vreg_input_x_2, srcUb2 + i * 16 * 16);                 // 第二个256
        LoadAlign(vreg_input_x_unroll_2, srcUb2 + i * 16 * 16 + 8 * 16);

        Max(vreg_max_tmp, vreg_input_x_1, vreg_input_x_2, preg_all); // 第 1/2 个256的第一个[64, 16]前8行的最大值
        Max(vreg_max_tmp_unroll, vreg_input_x_unroll_1, vreg_input_x_unroll_2, preg_all);

        for (uint16_t j = 1; j < n / 16; ++j) { // 每个256内部，第j个[64, 16]，n / 16为NZ shape的第一个维度
            LoadAlign(vreg_input_x_1, srcUb1 + i * 16 * 16 + j * 64 * 16); // 第一个256，搬入第j个[64, 16]的两个8行
            LoadAlign(vreg_input_x_unroll_1, srcUb1 + i * 16 * 16 + j * 64 * 16 + 8 * 16);
            LoadAlign(vreg_input_x_2, srcUb2 + i * 16 * 16 + j * 64 * 16); // 第二个256
            LoadAlign(vreg_input_x_unroll_2, srcUb2 + i * 16 * 16 + j * 64 * 16 + 8 * 16);

            Max(vreg_max_tmp, vreg_max_tmp, vreg_input_x_1,
                preg_all); // 读入第j个 [64, 16]，和第1个256块已经读入的前j-1个[64, 16]的max再取max
            Max(vreg_max_tmp, vreg_max_tmp, vreg_input_x_2,
                preg_all); // 读入第j个 [64, 16]，和第2个256块已经读入的前j-1个[64, 16]的max再取max
            Max(vreg_max_tmp_unroll, vreg_max_tmp_unroll, vreg_input_x_unroll_1, preg_all);
            Max(vreg_max_tmp_unroll, vreg_max_tmp_unroll, vreg_input_x_unroll_2, preg_all);
        }
        ReduceDataBlock<AscendC::MicroAPI::ReduceType::MAX>(vreg_max_tmp, vreg_max_tmp, preg_all); // 只剩8行的8个max值
        ReduceDataBlock<AscendC::MicroAPI::ReduceType::MAX>(vreg_max_tmp_unroll, vreg_max_tmp_unroll, preg_all);
        Sub(vreg_max_tmp, vreg_max_tmp, vreg_ln_p_scale, preg_all);
        Sub(vreg_max_tmp_unroll, vreg_max_tmp_unroll, vreg_ln_p_scale, preg_all);
        StoreUnAlign<half, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ half *&)tmpMaxUb), vreg_max_tmp,
                                                                    ureg_max, 8); // 存入这16行的 max 到 ub
        StoreUnAlign<half, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ half *&)tmpMaxUb), vreg_max_tmp_unroll,
                                                                    ureg_max, 8);
    }
    StoreUnAlignPost<half, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ half *&)tmpMaxUb), ureg_max, 0);

    LoadAlign(vreg_in_max, srcMaxUb);                         // 之前基本块的最大值
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();     // 等写完UB再读入
    LoadAlign(vreg_input_max, tmpMaxUb2);                     // 当前基本块的Max
    Max(vreg_max_new, vreg_input_max, vreg_in_max, preg_all); // 新max
    StoreAlign<half, MicroAPI::StoreDist::DIST_NORM_B16>(((__ubuf__ half *&)tmpMaxUb2), vreg_max_new,
                                                         preg_all); // 64个max，128B
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < 4; ++i) {
        LoadAlign<half, MicroAPI::LoadDist::DIST_E2B_B16>(vreg_max, tmpMaxUb2 + i * 16); // 读8行的max 并广播
        LoadAlign<half, MicroAPI::LoadDist::DIST_E2B_B16>(vreg_max_2, tmpMaxUb2 + i * 16 + 8);

        Duplicate(vreg_exp_sum_1, 0, preg_all); // sum清零
        Duplicate(vreg_exp_sum_2, 0, preg_all);
        // 第一个256
        for (uint16_t j = 0; j < n / 32; ++j) {
            // n / 32 是: 两个fp16的分形（一行16个元素）合一个fp8的分形（一行32个元素），第j个[64, 32]
            // 把前面的两个[64,16]拼起来（横向分形cast成一个），第j个[64, 32]
            LoadAlign<half, MicroAPI::LoadDist::DIST_NORM>(vreg_input_x_1,
                                                           srcUb1 + i * 16 * 16 + j * 64 * 32); // 第一个分形的前8行
            LoadAlign<half, MicroAPI::LoadDist::DIST_NORM>(vreg_input_x_unroll_1,
                                                           srcUb1 + i * 16 * 16 + j * 64 * 32 +
                                                               64 * 16); // 第二个分形的前8行（第二个[64,16]）
            // 上两个cast成一个分形, 下两个cast一个
            LoadAlign<half, MicroAPI::LoadDist::DIST_NORM>(vreg_input_x_2, srcUb1 + i * 16 * 16 + j * 64 * 32 +
                                                                               128); // 第一个分形的后8行
            LoadAlign<half, MicroAPI::LoadDist::DIST_NORM>(vreg_input_x_unroll_2,
                                                           srcUb1 + i * 16 * 16 + j * 64 * 32 + 128 + 64 * 16);
            // ExpSub
            ExpSub<float, half, RegLayout::ZERO>(vreg_exp_0_1, vreg_input_x_1, vreg_max, preg_all);
            ExpSub<float, half, RegLayout::ONE>(vreg_exp_2_1, vreg_input_x_1, vreg_max, preg_all);
            ExpSub<float, half, RegLayout::ZERO>(vreg_exp_1_1, vreg_input_x_unroll_1, vreg_max, preg_all);
            ExpSub<float, half, RegLayout::ONE>(vreg_exp_3_1, vreg_input_x_unroll_1, vreg_max, preg_all);

            ExpSub<float, half, RegLayout::ZERO>(vreg_exp_0_2, vreg_input_x_2, vreg_max_2, preg_all);
            ExpSub<float, half, RegLayout::ONE>(vreg_exp_2_2, vreg_input_x_2, vreg_max_2, preg_all);
            ExpSub<float, half, RegLayout::ZERO>(vreg_exp_1_2, vreg_input_x_unroll_2, vreg_max_2, preg_all);
            ExpSub<float, half, RegLayout::ONE>(vreg_exp_3_2, vreg_input_x_unroll_2, vreg_max_2, preg_all);
            Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_0_1, preg_all);
            Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_2_1, preg_all);
            Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_1_1, preg_all);
            Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_3_1, preg_all);
            Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_0_2, preg_all);
            Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_2_2, preg_all);
            Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_1_2, preg_all);
            Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_3_2, preg_all);
            Cast<T2, float, castTraitZero>(vreg_exp_0_f8_1, vreg_exp_0_1, preg_all);
            Cast<T2, float, castTraitTwo>(vreg_exp_2_f8_1, vreg_exp_2_1, preg_all);
            Cast<T2, float, castTraitOne>(vreg_exp_1_f8_1, vreg_exp_1_1, preg_all);
            Cast<T2, float, castTraitThree>(vreg_exp_3_f8_1, vreg_exp_3_1, preg_all);
            // Or
            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_1, (RegTensor<uint8_t> &)vreg_exp_0_f8_1,
               (RegTensor<uint8_t> &)vreg_exp_2_f8_1, preg_all_b8);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_2, (RegTensor<uint8_t> &)vreg_exp_1_f8_1,
               (RegTensor<uint8_t> &)vreg_exp_3_f8_1, preg_all_b8);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_f8_1, (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_1,
               (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_2, preg_all_b8);

            // 原512没传 indexesUb
            LoadAlign(vreg_exp_merge_f8_idxs, indexesUb);
            Gather(vreg_exp_merge_f8_1, vreg_exp_merge_f8_1, vreg_exp_merge_f8_idxs);
            StoreAlign(expUb1 + i * 16 * 32 + j * 64 * 32, vreg_exp_merge_f8_1, preg_all_b8);

            // 16行中的 后8行
            Cast<T2, float, castTraitZero>(vreg_exp_0_f8_2, vreg_exp_0_2, preg_all);
            Cast<T2, float, castTraitTwo>(vreg_exp_2_f8_2, vreg_exp_2_2, preg_all);
            Cast<T2, float, castTraitOne>(vreg_exp_1_f8_2, vreg_exp_1_2, preg_all);
            Cast<T2, float, castTraitThree>(vreg_exp_3_f8_2, vreg_exp_3_2, preg_all);

            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_2_1, (RegTensor<uint8_t> &)vreg_exp_0_f8_2,
               (RegTensor<uint8_t> &)vreg_exp_2_f8_2, preg_all_b8);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_2_2, (RegTensor<uint8_t> &)vreg_exp_1_f8_2,
               (RegTensor<uint8_t> &)vreg_exp_3_f8_2, preg_all_b8);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_f8_2, (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_2_1,
               (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_2_2, preg_all_b8);

            Gather(vreg_exp_merge_f8_2, vreg_exp_merge_f8_2, vreg_exp_merge_f8_idxs);
            StoreAlign(expUb1 + i * 16 * 32 + j * 64 * 32 + 256, vreg_exp_merge_f8_2, preg_all_b8);
        }
        // 第二个256
        for (uint16_t j = 0; j < n / 32; ++j) {
            LoadAlign<half, MicroAPI::LoadDist::DIST_NORM>(vreg_input_x_1,
                                                           srcUb2 + i * 16 * 16 + j * 64 * 32); // 第一个分形的前8行
            LoadAlign<half, MicroAPI::LoadDist::DIST_NORM>(vreg_input_x_unroll_1,
                                                           srcUb2 + i * 16 * 16 + j * 64 * 32 +
                                                               64 * 16); // 第二个分形的前8行（第二个 [64,16] ）
            LoadAlign<half, MicroAPI::LoadDist::DIST_NORM>(vreg_input_x_2, srcUb2 + i * 16 * 16 + j * 64 * 32 +
                                                                               128); // 第一个分形的后8行
            LoadAlign<half, MicroAPI::LoadDist::DIST_NORM>(vreg_input_x_unroll_2,
                                                           srcUb2 + i * 16 * 16 + j * 64 * 32 + 128 + 64 * 16);
            // expSub
            ExpSub<float, half, RegLayout::ZERO>(vreg_exp_0_1, vreg_input_x_1, vreg_max, preg_all);
            ExpSub<float, half, RegLayout::ONE>(vreg_exp_2_1, vreg_input_x_1, vreg_max, preg_all);
            ExpSub<float, half, RegLayout::ZERO>(vreg_exp_1_1, vreg_input_x_unroll_1, vreg_max, preg_all);
            ExpSub<float, half, RegLayout::ONE>(vreg_exp_3_1, vreg_input_x_unroll_1, vreg_max, preg_all);

            ExpSub<float, half, RegLayout::ZERO>(vreg_exp_0_2, vreg_input_x_2, vreg_max_2, preg_all);
            ExpSub<float, half, RegLayout::ONE>(vreg_exp_2_2, vreg_input_x_2, vreg_max_2, preg_all);
            ExpSub<float, half, RegLayout::ZERO>(vreg_exp_1_2, vreg_input_x_unroll_2, vreg_max_2, preg_all);
            ExpSub<float, half, RegLayout::ONE>(vreg_exp_3_2, vreg_input_x_unroll_2, vreg_max_2, preg_all);
            // add
            Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_0_1, preg_all);
            Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_2_1, preg_all);
            Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_1_1, preg_all);
            Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_3_1, preg_all);
            Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_0_2, preg_all);
            Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_2_2, preg_all);
            Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_1_2, preg_all);
            Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_3_2, preg_all);
            // cast
            Cast<T2, float, castTraitZero>(vreg_exp_0_f8_1, vreg_exp_0_1, preg_all);
            Cast<T2, float, castTraitTwo>(vreg_exp_2_f8_1, vreg_exp_2_1, preg_all);
            Cast<T2, float, castTraitOne>(vreg_exp_1_f8_1, vreg_exp_1_1, preg_all);
            Cast<T2, float, castTraitThree>(vreg_exp_3_f8_1, vreg_exp_3_1, preg_all);
            // or
            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_1, (RegTensor<uint8_t> &)vreg_exp_0_f8_1,
               (RegTensor<uint8_t> &)vreg_exp_2_f8_1, preg_all_b8);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_2, (RegTensor<uint8_t> &)vreg_exp_1_f8_1,
               (RegTensor<uint8_t> &)vreg_exp_3_f8_1, preg_all_b8);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_f8_1, (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_1,
               (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_2, preg_all_b8);

            LoadAlign(vreg_exp_merge_f8_idxs, indexesUb);
            Gather(vreg_exp_merge_f8_1, vreg_exp_merge_f8_1, vreg_exp_merge_f8_idxs);
            StoreAlign(expUb2 + i * 16 * 32 + j * 64 * 32, vreg_exp_merge_f8_1, preg_all_b8);

            Cast<T2, float, castTraitZero>(vreg_exp_0_f8_2, vreg_exp_0_2, preg_all);
            Cast<T2, float, castTraitTwo>(vreg_exp_2_f8_2, vreg_exp_2_2, preg_all);
            Cast<T2, float, castTraitOne>(vreg_exp_1_f8_2, vreg_exp_1_2, preg_all);
            Cast<T2, float, castTraitThree>(vreg_exp_3_f8_2, vreg_exp_3_2, preg_all);

            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_2_1, (RegTensor<uint8_t> &)vreg_exp_0_f8_2,
               (RegTensor<uint8_t> &)vreg_exp_2_f8_2, preg_all_b8);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_2_2, (RegTensor<uint8_t> &)vreg_exp_1_f8_2,
               (RegTensor<uint8_t> &)vreg_exp_3_f8_2, preg_all_b8);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_f8_2, (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_2_1,
               (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_2_2, preg_all_b8);

            Gather(vreg_exp_merge_f8_2, vreg_exp_merge_f8_2, vreg_exp_merge_f8_idxs);
            StoreAlign(expUb2 + i * 16 * 32 + j * 64 * 32 + 256, vreg_exp_merge_f8_2, preg_all_b8);
        }
        ReduceDataBlock<AscendC::MicroAPI::ReduceType::SUM>(vreg_exp_sum_1, vreg_exp_sum_1, preg_all);
        ReduceDataBlock<AscendC::MicroAPI::ReduceType::SUM>(vreg_exp_sum_2, vreg_exp_sum_2, preg_all);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)tmpExpSumUb), vreg_exp_sum_1,
                                                                     ureg_exp_sum, 8);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)tmpExpSumUb), vreg_exp_sum_2,
                                                                     ureg_exp_sum, 8);
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)tmpExpSumUb), ureg_max, 0);
}

template <typename T, typename T2, typename OUTPUT_T, uint32_t s1BaseSize = 16, uint32_t s2BaseSize = 512,
          bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false>
__aicore__ inline void ProcessVec1UpdateGeneralImpl512GqaFullquant(
    const LocalTensor<T2> &dstTensor, const LocalTensor<T> &srcTensor, const LocalTensor<T> &maxTensor,
    const LocalTensor<T> &inMaxTensor, const LocalTensor<uint8_t> &sharedTmpBuffer,
    const LocalTensor<uint8_t> &indexesTensor, const uint16_t m, const uint32_t originN, const T scale,
    const float dScaleQK, const T minValue, const float pScale)
{
    uint32_t n = originN >> 1;

    __ubuf__ T *srcUb1 = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ T *srcUb2 = (__ubuf__ T *)srcTensor.GetPhyAddr() + 64 * n;
    __ubuf__ T2 *expUb1 = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ T2 *expUb2 = (__ubuf__ T2 *)dstTensor.GetPhyAddr() + 64 * n;
    __ubuf__ T *maxUb = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *srcMaxUb = (__ubuf__ T *)inMaxTensor.GetPhyAddr();

    __ubuf__ T *tmpExpSumUb = (__ubuf__ T *)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T *tmpMaxUb = (__ubuf__ T *)sharedTmpBuffer.GetPhyAddr() + halfRepSize;
    __ubuf__ T *tmpMaxUb2 = (__ubuf__ T *)sharedTmpBuffer.GetPhyAddr() + halfRepSize;

    __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)indexesTensor.GetPhyAddr();

    ProcessVec1UpdateGeneralImpl512GqaFullquantVF<half, T2, OUTPUT_T, s1BaseSize, s2BaseSize, hasAtten, pseMode,
                                                  hasDrop, isMlaSgd>(expUb1, expUb2, srcUb1, srcUb2, maxUb, srcMaxUb,
                                                                     tmpExpSumUb, tmpMaxUb, tmpMaxUb2, indexesUb, m, n,
                                                                     minValue, pScale);
}
} // namespace FaVectorApi

#endif // VF_BASIC_BLOCK_FULLQUANT_UNALIGNED512_UPDATE_H
