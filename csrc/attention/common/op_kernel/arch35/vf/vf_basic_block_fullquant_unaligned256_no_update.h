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
#ifndef VF_BASIC_BLOCK_FULLQUANT_UNALIGNED256_NO_UPDATE_H
#define VF_BASIC_BLOCK_FULLQUANT_UNALIGNED256_NO_UPDATE_H

#include "vf_basic_block_utils.h"

using namespace regbaseutil;

namespace FaVectorApi {
template <typename T, typename T2, typename OUTPUT_T, uint32_t s1BaseSize = 16, uint32_t s2BaseSize = 512,
          bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__simd_vf__ void ProcessVec1NoUpdateGeneralImpl256GqaFullquantVF(
    __ubuf__ T2 *expUb, __ubuf__ T *srcUb, __ubuf__ T *maxUb, __ubuf__ T *tmpExpSumUb, __ubuf__ T *tmpMaxUb,
    __ubuf__ T *tmpMaxUb2, __ubuf__ uint8_t *indexesUb, const uint16_t m, const uint32_t n, const T minValue,
    const uint32_t tailN, const int64_t tailNOffset, const float pScale)
{
    RegTensor<half> vreg_p_scale;
    RegTensor<half> vreg_ln_p_scale;
    RegTensor<half> vreg_input_x_1;
    RegTensor<half> vreg_input_x_unroll_1;
    RegTensor<half> vreg_input_x_2;
    RegTensor<half> vreg_input_x_unroll_2;
    RegTensor<half> vreg_min;

    RegTensor<half> vreg_max_tmp;        // 一个分形前8行的max
    RegTensor<half> vreg_max_tmp_unroll; // 一个分形后8行，一共一次处理16行

    RegTensor<float> vreg_exp_sum_1; // 前8行的和
    RegTensor<float> vreg_exp_sum_2; // 接着的后8行，一共一次处理16行

    RegTensor<half> vreg_in_max;
    RegTensor<half> vreg_input_max;
    RegTensor<half> vreg_max_new;
    RegTensor<half> vreg_max_2;
    RegTensor<half> vreg_max;

    RegTensor<float> vreg_exp_0_2;
    RegTensor<float> vreg_exp_1_2;
    RegTensor<float> vreg_exp_2_2;
    RegTensor<float> vreg_exp_3_2;
    RegTensor<float> vreg_exp_0_1;
    RegTensor<float> vreg_exp_1_1;
    RegTensor<float> vreg_exp_2_1;
    RegTensor<float> vreg_exp_3_1;

    RegTensor<T2> vreg_exp_merge_tmp_f8_1_1; // 前8行
    RegTensor<T2> vreg_exp_merge_tmp_f8_1_2;
    RegTensor<T2> vreg_exp_merge_f8_1;
    RegTensor<T2> vreg_exp_merge_tmp_f8_2_1; // 后8行
    RegTensor<T2> vreg_exp_merge_tmp_f8_2_2;
    RegTensor<T2> vreg_exp_merge_f8_2;
    RegTensor<T2> vreg_exp_0_f8_1;
    RegTensor<T2> vreg_exp_2_f8_1;
    RegTensor<T2> vreg_exp_1_f8_1;
    RegTensor<T2> vreg_exp_3_f8_1;
    RegTensor<T2> vreg_exp_0_f8_2;
    RegTensor<T2> vreg_exp_2_f8_2;
    RegTensor<T2> vreg_exp_1_f8_2;
    RegTensor<T2> vreg_exp_3_f8_2;
    RegTensor<uint8_t> vreg_exp_merge_f8_indexes;
    MaskReg preg_all = CreateMask<half, MaskPattern::ALL>();
    MaskReg preg_all_b8 = CreateMask<uint8_t, MaskPattern::ALL>();
    UnalignRegForStore ureg_max;
    UnalignReg ureg_exp_sum;
    Duplicate(vreg_min, minValue);
    Duplicate(vreg_p_scale, static_cast<half>(pScale));
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_all);
    for (uint16_t i = 0; i < m / 8; ++i) {
        StoreAlign<half, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ half *&)srcUb + tailNOffset + i * 128, vreg_min,
                                                             preg_all);
    }
    // LS同步
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    for (uint16_t i = 0; i < 4; ++i) {
        LoadAlign(vreg_input_x_1, srcUb + i * 16 * 16); // 第一个[64, 16]的前8行
        LoadAlign(vreg_input_x_unroll_1, srcUb + i * 16 * 16 + 8 * 16);

        Max(vreg_max_tmp, vreg_input_x_1, vreg_input_x_1, preg_all); // 最大值
        Max(vreg_max_tmp_unroll, vreg_input_x_unroll_1, vreg_input_x_unroll_1, preg_all);

        for (uint16_t j = 1; j < n / 16; ++j) {
            LoadAlign(vreg_input_x_1, srcUb + i * 16 * 16 + j * 64 * 16); // 第一个256，搬入第j个[64, 16]的两个8行
            LoadAlign(vreg_input_x_unroll_1, srcUb + i * 16 * 16 + j * 64 * 16 + 8 * 16);

            Max(vreg_max_tmp, vreg_max_tmp, vreg_input_x_1,
                preg_all); // 读入第j个[64, 16]，和已经读入的 [64, 16]的max再取max
            Max(vreg_max_tmp_unroll, vreg_max_tmp_unroll, vreg_input_x_unroll_1, preg_all);
        }
        ReduceDataBlock<AscendC::MicroAPI::ReduceType::MAX>(vreg_max_tmp, vreg_max_tmp, preg_all);
        ReduceDataBlock<AscendC::MicroAPI::ReduceType::MAX>(vreg_max_tmp_unroll, vreg_max_tmp_unroll, preg_all);
        Sub(vreg_max_tmp, vreg_max_tmp, vreg_ln_p_scale, preg_all);
        Sub(vreg_max_tmp_unroll, vreg_max_tmp_unroll, vreg_ln_p_scale, preg_all);
        StoreUnAlign<half, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ half *&)tmpMaxUb), vreg_max_tmp,
                                                                    ureg_max, 8);
        StoreUnAlign<half, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ half *&)tmpMaxUb), vreg_max_tmp_unroll,
                                                                    ureg_max, 8);
    }
    StoreUnAlignPost<half, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ half *&)tmpMaxUb), ureg_max, 0);

    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < 4; ++i) {
        LoadAlign<half, MicroAPI::LoadDist::DIST_E2B_B16>(vreg_max, tmpMaxUb2 + i * 16);
        LoadAlign<half, MicroAPI::LoadDist::DIST_E2B_B16>(vreg_max_2, tmpMaxUb2 + i * 16 + 8);
        // sum 清零
        Duplicate(vreg_exp_sum_1, 0, preg_all);
        Duplicate(vreg_exp_sum_2, 0, preg_all);

        for (uint16_t j = 0; j < n / 32; ++j) {
            // n / 32 ，两个fp16的分形（一行16个元素）合一个fp8的分形（一行32个元素），第j个[64, 32]
            LoadAlign<half, MicroAPI::LoadDist::DIST_NORM>(vreg_input_x_1, srcUb + i * 16 * 16 + j * 64 * 32);
            LoadAlign<half, MicroAPI::LoadDist::DIST_NORM>(vreg_input_x_unroll_1,
                                                           srcUb + i * 16 * 16 + j * 64 * 32 + 64 * 16);

            LoadAlign<half, MicroAPI::LoadDist::DIST_NORM>(vreg_input_x_2, srcUb + i * 16 * 16 + j * 64 * 32 + 128);
            LoadAlign<half, MicroAPI::LoadDist::DIST_NORM>(vreg_input_x_unroll_2,
                                                           srcUb + i * 16 * 16 + j * 64 * 32 + 128 + 64 * 16);
            ExpSub<float, half, RegLayout::ZERO>(vreg_exp_0_1, vreg_input_x_1, vreg_max, preg_all);
            ExpSub<float, half, RegLayout::ONE>(vreg_exp_2_1, vreg_input_x_1, vreg_max, preg_all);
            ExpSub<float, half, RegLayout::ZERO>(vreg_exp_1_1, vreg_input_x_unroll_1, vreg_max, preg_all);
            ExpSub<float, half, RegLayout::ONE>(vreg_exp_3_1, vreg_input_x_unroll_1, vreg_max, preg_all);
            ExpSub<float, half, RegLayout::ZERO>(vreg_exp_0_2, vreg_input_x_2, vreg_max_2, preg_all);
            ExpSub<float, half, RegLayout::ONE>(vreg_exp_2_2, vreg_input_x_2, vreg_max_2, preg_all);
            ExpSub<float, half, RegLayout::ZERO>(vreg_exp_1_2, vreg_input_x_unroll_2, vreg_max_2, preg_all);
            ExpSub<float, half, RegLayout::ONE>(vreg_exp_3_2, vreg_input_x_unroll_2, vreg_max_2, preg_all);
            Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_0_1, preg_all); // ADD计算
            Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_2_1, preg_all);
            Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_1_1, preg_all);
            Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_3_1, preg_all);
            Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_0_2, preg_all);
            Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_2_2, preg_all);
            Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_1_2, preg_all);
            Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_3_2, preg_all);
            Cast<T2, float, castTraitZero>(vreg_exp_0_f8_1, vreg_exp_0_1, preg_all); // cast计算
            Cast<T2, float, castTraitTwo>(vreg_exp_2_f8_1, vreg_exp_2_1, preg_all);
            Cast<T2, float, castTraitOne>(vreg_exp_1_f8_1, vreg_exp_1_1, preg_all);
            Cast<T2, float, castTraitThree>(vreg_exp_3_f8_1, vreg_exp_3_1, preg_all);

            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_1, (RegTensor<uint8_t> &)vreg_exp_0_f8_1,
               (RegTensor<uint8_t> &)vreg_exp_2_f8_1, preg_all_b8); // or计算
            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_2, (RegTensor<uint8_t> &)vreg_exp_1_f8_1,
               (RegTensor<uint8_t> &)vreg_exp_3_f8_1, preg_all_b8);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_f8_1, (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_1,
               (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_2, preg_all_b8);

            LoadAlign(vreg_exp_merge_f8_indexes, indexesUb);
            Gather(vreg_exp_merge_f8_1, vreg_exp_merge_f8_1, vreg_exp_merge_f8_indexes);
            StoreAlign(expUb + i * 16 * 32 + j * 64 * 32, vreg_exp_merge_f8_1, preg_all_b8);

            // 16 行中的后 8 行
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

            Gather(vreg_exp_merge_f8_2, vreg_exp_merge_f8_2, vreg_exp_merge_f8_indexes);
            StoreAlign(expUb + i * 16 * 32 + j * 64 * 32 + 256, vreg_exp_merge_f8_2, preg_all_b8);
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
          bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__aicore__ inline void ProcessVec1NoUpdateGeneralImpl256GqaFullquant(
    const LocalTensor<T2> &dstTensor, const LocalTensor<T> &srcTensor, const LocalTensor<T> &maxTensor,
    const LocalTensor<T> &inMaxTensor, const LocalTensor<T> &expSumTensor, const LocalTensor<uint8_t> &indexesTensor,
    const uint16_t m, const uint32_t originN, const T scale, const float dScaleQK, const T minValue, const float pScale)
{
    const uint32_t n = (originN + 31) >> 5 << 5;
    const uint32_t tailN = n - originN;
    const int64_t tailNOffset = 64 * ((originN + 15) >> 4 << 4);
    __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)indexesTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ T2 *expUb = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ T *maxUb = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *tmpExpSumUb = (__ubuf__ T *)expSumTensor.GetPhyAddr();
    __ubuf__ T *tmpMaxUb = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *tmpMaxUb2 = (__ubuf__ T *)maxTensor.GetPhyAddr();
    ProcessVec1NoUpdateGeneralImpl256GqaFullquantVF<T, T2, OUTPUT_T, s1BaseSize, s2BaseSize, hasAtten, pseMode,
                                                    hasDrop>(expUb, srcUb, maxUb, tmpExpSumUb, tmpMaxUb, tmpMaxUb2,
                                                             indexesUb, m, n, minValue, tailN, tailNOffset, pScale);
}
} // namespace FaVectorApi

#endif // VF_BASIC_BLOCK_FULLQUANT_UNALIGNED256_NO_UPDATE_H
