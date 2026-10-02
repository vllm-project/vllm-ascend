/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_MLA_C8_VEC128_PACKED_P_H
#define FLASH_MLA_C8_VEC128_PACKED_P_H

#include "../vf/vf_basic_block_utils.h"

namespace FaVectorApi {

// C8-only specialization of the original aligned128 update, preserving its
// two-pass computation and sharedTmp contract. Compute448/pScale once for
// the live32 rows, instead of two per-row FP32 multiplies and divides.
__simd_vf__ inline void C8ProcessVec1Update128PackedPVf(
    __ubuf__ fp8_e4m3fn_t *dst, __ubuf__ uint8_t *indexes, __ubuf__ float *src,
    __ubuf__ float *inMax, __ubuf__ float *tmpSum, __ubuf__ float *tmpMax,
    __ubuf__ float *pScale, __ubuf__ float *qScale, const uint16_t m,
    const float scale, const float kvScale)
{
    RegTensor<float> x0, x1, qkScale, maximum, tileMax, oldMax, newMax;
    RegTensor<float> e0, e1, expSum, dynamicPScale, numerator, inversePScale;
    RegTensor<fp8_e4m3fn_t> castEven, castOdd, castPair, packed;
    RegTensor<uint8_t> gatherIndexes;
    UnalignRegForStore maxStore, sumStore;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    MaskReg allBytes = CreateMask<uint8_t, MaskPattern::ALL>();
    uint32_t liveRows = m;
    MaskReg rows = UpdateMask<float>(liveRows);
    uint32_t byteCount = 128;
    MaskReg bytes128 = UpdateMask<uint8_t>(byteCount);
    __ubuf__ float *maxWrite = tmpMax;
    __ubuf__ float *sumWrite = tmpSum;
    __ubuf__ fp8_e4m3fn_t *dstWrite = dst;

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign(x0, src + i * 128);
        LoadAlign(x1, src + i * 128 + 64);
        LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(qkScale, qScale + i);
        Muls(qkScale, qkScale, scale, all);
        Muls(qkScale, qkScale, kvScale, all);
        Mul(x0, x0, qkScale, all);
        Mul(x1, x1, qkScale, all);
        StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(src + i * 128, x0, all);
        StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(src + i * 128 + 64, x1, all);
        Max(maximum, x0, x1, all);
        Reduce<Reg::ReduceType::MAX, float, float, Reg::MaskMergeMode::ZEROING>(tileMax, maximum, all);
        StoreUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(maxWrite, tileMax, maxStore, 1);
    }
    StoreUnAlignPost<float, Reg::PostLiteral::POST_MODE_UPDATE>(maxWrite, maxStore, 0);
    LoadAlign(oldMax, inMax);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LoadAlign(tileMax, tmpMax);
    Max(newMax, tileMax, oldMax, rows);
    StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(tmpMax, newMax, rows);
    ExpSub(dynamicPScale, tileMax, newMax, rows);
    Adds(dynamicPScale, dynamicPScale, floatEps, rows);
    // pScale has64 FP32 slots but this AIV owns at most32 rows. Preserve
    // the first32 for PV, and use the otherwise unused tail for448/pScale.
    StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(pScale, dynamicPScale, rows);
    Duplicate(numerator, fp8e4m3MaxValue);
    Div(inversePScale, numerator, dynamicPScale, rows);
    StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(pScale + 32, inversePScale, rows);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    LoadAlign(gatherIndexes, indexes);
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(maximum, tmpMax + i);
        LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(x0, x1, src + i * 128);
        ExpSub(e0, x0, maximum, all);
        ExpSub(e1, x1, maximum, all);
        Add(expSum, e0, e1, all);
        Reduce<Reg::ReduceType::SUM, float, float, Reg::MaskMergeMode::ZEROING>(expSum, expSum, all);
        StoreUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(sumWrite, expSum, sumStore, 1);
        LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(inversePScale, pScale + 32 + i);
        Mul(e0, e0, inversePScale, all);
        Mul(e1, e1, inversePScale, all);
        Cast<fp8_e4m3fn_t, float, castTraitRintZero>(castEven, e0, all);
        Cast<fp8_e4m3fn_t, float, castTraitRintTwo>(castOdd, e1, all);
        Or((RegTensor<uint8_t> &)castPair, (RegTensor<uint8_t> &)castEven,
           (RegTensor<uint8_t> &)castOdd, allBytes);
        Gather(packed, castPair, gatherIndexes);
        StoreAlign<fp8_e4m3fn_t, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            dstWrite, packed, 33, 1, bytes128);
    }
    StoreUnAlignPost<float, Reg::PostLiteral::POST_MODE_UPDATE>(sumWrite, sumStore, 0);
}

__aicore__ inline void C8ProcessVec1Update128PackedP(
    const LocalTensor<fp8_e4m3fn_t> &dst, const LocalTensor<uint8_t> &indexes,
    const LocalTensor<float> &src, const LocalTensor<float> &inMax,
    const LocalTensor<uint8_t> &sharedTmp, const LocalTensor<float> &pScale,
    const uint16_t m, const float scale, const LocalTensor<float> &qScale, const float kvScale)
{
    // Update tiles only, full128 columns, M64 split over two AIVs (m<=32),
    // no mask/PSE/dropout/sink. The outer caller retains all other fallbacks.
    // The original UpdateExpSumAndExpMax must still run after this helper.
    C8ProcessVec1Update128PackedPVf(
        (__ubuf__ fp8_e4m3fn_t *)dst.GetPhyAddr(), (__ubuf__ uint8_t *)indexes.GetPhyAddr(),
        (__ubuf__ float *)src.GetPhyAddr(), (__ubuf__ float *)inMax.GetPhyAddr(),
        (__ubuf__ float *)sharedTmp.GetPhyAddr(), (__ubuf__ float *)sharedTmp.GetPhyAddr() + 64,
        (__ubuf__ float *)pScale.GetPhyAddr(), (__ubuf__ float *)qScale.GetPhyAddr(), m, scale, kvScale);
}

} // namespace FaVectorApi
#endif // FLASH_MLA_C8_VEC128_PACKED_P_H
