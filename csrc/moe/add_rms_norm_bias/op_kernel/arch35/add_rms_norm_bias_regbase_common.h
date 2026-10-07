/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Derived from cann/ops-nn v9.2.0-beta.2, commit
// 30ef7dd563c8a4b74c3161835c8e47d1d96f87b6.
// Source: norm/add_rms_norm/op_kernel/arch35/add_rms_norm_regbase_common.h


/* !
 * \file add_rms_norm_bias_regbase_common.h
 * \brief AddRmsNorm regbase common
 */
#ifndef ADD_RMS_NORM_BIAS_REGBASE_COMMON_H
#define ADD_RMS_NORM_BIAS_REGBASE_COMMON_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../rms_norm_base.h"
#include "deps/rms_norm_regbase_common.h"
#include "deps/reduce_common_regbase.h"
namespace AddRmsNormBiasA5 {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using namespace NormCommon;
using namespace NormCommon::NormCommonRegbase;
using AscendC::MicroAPI::Add;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;
using NormCommon::V_LENGTH;
using NormCommon::NormCommonRegbase::LoadRegForDtype;
using NormCommon::NormCommonRegbase::StoreRegForDtype;
using RmsNorm::BUFFER_NUM;
using RmsNorm::castTraitB162B32;
using RmsNorm::castTraitB322B16;
using RmsNorm::CeilDiv;
using RmsNorm::DOUBLE_BUFFER_NUM;
using RmsNorm::is_same;
using RmsNorm::ONCE_VECTOR_SIZE;

template <typename T>
__aicore__ inline void LoadForHandleRemainV1(__ubuf__ T* mainAddr, __ubuf__ T* tailAddr, uint16_t offset1,
                                             uint16_t offset2, RegTensor<float>& mainA, RegTensor<float>& mainB,
                                             RegTensor<float>& tailA, RegTensor<float>& tailB, MaskReg& pregLoop,
                                             __ubuf__ float* xFp32MainAddr, __ubuf__ float* xFp32TailAddr,
                                             __ubuf__ T* mainAddr2, __ubuf__ T* tailAddr2)
{
    RegTensor<float> mainA2, mainB2, tailA2, tailB2;
    if constexpr (IsSameType<T, half>::value) {
        // x1 load and cast
        RegTensor<half> xFp16MainA, xFp16MainB, xFp16TailA, xFp16TailB;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16MainA, mainAddr + offset1);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16MainB, mainAddr + offset2);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16TailA, tailAddr + offset1);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16TailB, tailAddr + offset2);
        Cast<float, half, castTraitB162B32>(mainA, xFp16MainA, pregLoop);
        Cast<float, half, castTraitB162B32>(mainB, xFp16MainB, pregLoop);
        Cast<float, half, castTraitB162B32>(tailA, xFp16TailA, pregLoop);
        Cast<float, half, castTraitB162B32>(tailB, xFp16TailB, pregLoop);
        // x2 load and cast
        RegTensor<half> xFp16MainA2, xFp16MainB2, xFp16TailA2, xFp16TailB2;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16MainA2, mainAddr2 + offset1);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16MainB2, mainAddr2 + offset2);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16TailA2, tailAddr2 + offset1);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16TailB2, tailAddr2 + offset2);
        Cast<float, half, castTraitB162B32>(mainA2, xFp16MainA2, pregLoop);
        Cast<float, half, castTraitB162B32>(mainB2, xFp16MainB2, pregLoop);
        Cast<float, half, castTraitB162B32>(tailA2, xFp16TailA2, pregLoop);
        Cast<float, half, castTraitB162B32>(tailB2, xFp16TailB2, pregLoop);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        // x1 load and cast
        RegTensor<bfloat16_t> xBFp16MainA, xBFp16MainB, xBFp16TailA, xBFp16TailB;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16MainA, mainAddr + offset1);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16MainB, mainAddr + offset2);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16TailA, tailAddr + offset1);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16TailB, tailAddr + offset2);
        Cast<float, bfloat16_t, castTraitB162B32>(mainA, xBFp16MainA, pregLoop);
        Cast<float, bfloat16_t, castTraitB162B32>(mainB, xBFp16MainB, pregLoop);
        Cast<float, bfloat16_t, castTraitB162B32>(tailA, xBFp16TailA, pregLoop);
        Cast<float, bfloat16_t, castTraitB162B32>(tailB, xBFp16TailB, pregLoop);
        // x2 load and cast
        RegTensor<bfloat16_t> xBFp16MainA2, xBFp16MainB2, xBFp16TailA2, xBFp16TailB2;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16MainA2, mainAddr2 + offset1);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16MainB2, mainAddr2 + offset2);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16TailA2, tailAddr2 + offset1);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16TailB2, tailAddr2 + offset2);
        // x2 cast
        Cast<float, bfloat16_t, castTraitB162B32>(mainA2, xBFp16MainA2, pregLoop);
        Cast<float, bfloat16_t, castTraitB162B32>(mainB2, xBFp16MainB2, pregLoop);
        Cast<float, bfloat16_t, castTraitB162B32>(tailA2, xBFp16TailA2, pregLoop);
        Cast<float, bfloat16_t, castTraitB162B32>(tailB2, xBFp16TailB2, pregLoop);
    } else {
        LoadAlign(mainA, mainAddr + offset1);
        LoadAlign(mainB, mainAddr + offset2);
        LoadAlign(tailA, tailAddr + offset1);
        LoadAlign(tailB, tailAddr + offset2);
        // load x2
        LoadAlign(mainA2, mainAddr2 + offset1);
        LoadAlign(mainB2, mainAddr2 + offset2);
        LoadAlign(tailA2, tailAddr2 + offset1);
        LoadAlign(tailB2, tailAddr2 + offset2);
    }
    // add x1 + x2
    Add(mainA, mainA, mainA2, pregLoop);
    Add(mainB, mainB, mainB2, pregLoop);
    Add(tailA, tailA, tailA2, pregLoop);
    Add(tailB, tailB, tailB2, pregLoop);
    StoreAlign(xFp32MainAddr + offset1, mainA, pregLoop);
    StoreAlign(xFp32MainAddr + offset2, mainB, pregLoop);
    StoreAlign(xFp32TailAddr + offset1, tailA, pregLoop);
    StoreAlign(xFp32TailAddr + offset2, tailB, pregLoop);
    Mul(mainA, mainA, mainA, pregLoop);
    Mul(mainB, mainB, mainB, pregLoop);
    Mul(tailA, tailA, tailA, pregLoop);
    Mul(tailB, tailB, tailB, pregLoop);
}

template <typename T>
__aicore__ inline void LoadForHandleMasterV1(__ubuf__ T* masterAddr, uint16_t offset1, uint16_t offset2,
                                             RegTensor<float>& mainA, RegTensor<float>& mainB, MaskReg& pregLoop,
                                             __ubuf__ float* xFp32MasterAddr, __ubuf__ T* masterAddr2)
{
    if constexpr (IsSameType<T, half>::value) {
        // x1/x2 load and cast
        RegTensor<half> xFp16MainA, xFp16MainB;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16MainA, masterAddr + offset1);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16MainB, masterAddr + offset2);
        Cast<float, half, castTraitB162B32>(mainA, xFp16MainA, pregLoop);
        Cast<float, half, castTraitB162B32>(mainB, xFp16MainB, pregLoop);
        RegTensor<half> xFp16MainA2, xFp16MainB2;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16MainA2, masterAddr2 + offset1);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16MainB2, masterAddr2 + offset2);
        RegTensor<float> mainA2, mainB2;
        Cast<float, half, castTraitB162B32>(mainA2, xFp16MainA2, pregLoop);
        Cast<float, half, castTraitB162B32>(mainB2, xFp16MainB2, pregLoop);
        // add x1 + x2
        Add(mainA, mainA, mainA2, pregLoop);
        Add(mainB, mainB, mainB2, pregLoop);
        StoreAlign(xFp32MasterAddr + offset1, mainA, pregLoop);
        StoreAlign(xFp32MasterAddr + offset2, mainB, pregLoop);
        Mul(mainA, mainA, mainA, pregLoop);
        Mul(mainB, mainB, mainB, pregLoop);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        // x1/x2 load and cast
        RegTensor<bfloat16_t> xBFp16MainA, xBFp16MainB;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16MainA, masterAddr + offset1);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16MainB, masterAddr + offset2);
        Cast<float, bfloat16_t, castTraitB162B32>(mainA, xBFp16MainA, pregLoop);
        Cast<float, bfloat16_t, castTraitB162B32>(mainB, xBFp16MainB, pregLoop);
        RegTensor<bfloat16_t> xBFp16MainA2, xBFp16MainB2;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16MainA2, masterAddr2 + offset1);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16MainB2, masterAddr2 + offset2);
        RegTensor<float> mainA2, mainB2;
        Cast<float, bfloat16_t, castTraitB162B32>(mainA2, xBFp16MainA2, pregLoop);
        Cast<float, bfloat16_t, castTraitB162B32>(mainB2, xBFp16MainB2, pregLoop);
        // add x1 + x2
        Add(mainA, mainA, mainA2, pregLoop);
        Add(mainB, mainB, mainB2, pregLoop);
        StoreAlign(xFp32MasterAddr + offset1, mainA, pregLoop);
        StoreAlign(xFp32MasterAddr + offset2, mainB, pregLoop);
        Mul(mainA, mainA, mainA, pregLoop);
        Mul(mainB, mainB, mainB, pregLoop);
    } else {
        LoadAlign(mainA, masterAddr + offset1);
        LoadAlign(mainB, masterAddr + offset2);
        RegTensor<float> mainA2, mainB2;
        LoadAlign(mainA2, masterAddr2 + offset1);
        LoadAlign(mainB2, masterAddr2 + offset2);
        // add x1 + x2
        Add(mainA, mainA, mainA2, pregLoop);
        Add(mainB, mainB, mainB2, pregLoop);
        StoreAlign(xFp32MasterAddr + offset1, mainA, pregLoop);
        StoreAlign(xFp32MasterAddr + offset2, mainB, pregLoop);
        // x * x
        Mul(mainA, mainA, mainA, pregLoop);
        Mul(mainB, mainB, mainB, pregLoop);
    }
}

template <typename T>
__aicore__ inline void LoadForHandleRemainV2(__ubuf__ T* mainAddr, __ubuf__ T* tailAddr, uint16_t offset1,
                                             uint16_t offset2, RegTensor<float>& mainA, RegTensor<float>& mainB,
                                             RegTensor<float>& tailA, RegTensor<float>& tailB, MaskReg& pregMask,
                                             __ubuf__ T* mainAddr2, __ubuf__ T* tailAddr2)
{
    if constexpr (IsSameType<T, half>::value) {
        // x2 load and cast
        RegTensor<half> xFp16MainA, xFp16MainB, xFp16TailA, xFp16TailB;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16MainA, mainAddr + offset1);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16MainB, mainAddr + offset2);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16TailA, tailAddr + offset1);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16TailB, tailAddr + offset2);
        Cast<float, half, castTraitB162B32>(mainA, xFp16MainA, pregMask);
        Cast<float, half, castTraitB162B32>(mainB, xFp16MainB, pregMask);
        Cast<float, half, castTraitB162B32>(tailA, xFp16TailA, pregMask);
        Cast<float, half, castTraitB162B32>(tailB, xFp16TailB, pregMask);
        RegTensor<half> xFp16MainA2, xFp16MainB2, xFp16TailA2, xFp16TailB2;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16MainA2, mainAddr2 + offset1);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16MainB2, mainAddr2 + offset2);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16TailA2, tailAddr2 + offset1);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16TailB2, tailAddr2 + offset2);
        RegTensor<float> mainA2, mainB2, tailA2, tailB2;
        Cast<float, half, castTraitB162B32>(mainA2, xFp16MainA2, pregMask);
        Cast<float, half, castTraitB162B32>(mainB2, xFp16MainB2, pregMask);
        Cast<float, half, castTraitB162B32>(tailA2, xFp16TailA2, pregMask);
        Cast<float, half, castTraitB162B32>(tailB2, xFp16TailB2, pregMask);
        // add x1 + x2
        Add(mainA, mainA, mainA2, pregMask);
        Add(mainB, mainB, mainB2, pregMask);
        Add(tailA, tailA, tailA2, pregMask);
        Add(tailB, tailB, tailB2, pregMask);
        Mul(mainA, mainA, mainA, pregMask);
        Mul(mainB, mainB, mainB, pregMask);
        Mul(tailA, tailA, tailA, pregMask);
        Mul(tailB, tailB, tailB, pregMask);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        // x2 load and cast
        RegTensor<bfloat16_t> xBFp16MainA, xBFp16MainB, xBFp16TailA, xBFp16TailB;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16MainA, mainAddr + offset1);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16MainB, mainAddr + offset2);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16TailA, tailAddr + offset1);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16TailB, tailAddr + offset2);
        Cast<float, bfloat16_t, castTraitB162B32>(mainA, xBFp16MainA, pregMask);
        Cast<float, bfloat16_t, castTraitB162B32>(mainB, xBFp16MainB, pregMask);
        Cast<float, bfloat16_t, castTraitB162B32>(tailA, xBFp16TailA, pregMask);
        Cast<float, bfloat16_t, castTraitB162B32>(tailB, xBFp16TailB, pregMask);
        RegTensor<bfloat16_t> xBFp16MainA2, xBFp16MainB2, xBFp16TailA2, xBFp16TailB2;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16MainA2, mainAddr2 + offset1);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16MainB2, mainAddr2 + offset2);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16TailA2, tailAddr2 + offset1);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16TailB2, tailAddr2 + offset2);
        RegTensor<float> mainA2, mainB2, tailA2, tailB2;
        Cast<float, bfloat16_t, castTraitB162B32>(mainA2, xBFp16MainA2, pregMask);
        Cast<float, bfloat16_t, castTraitB162B32>(mainB2, xBFp16MainB2, pregMask);
        Cast<float, bfloat16_t, castTraitB162B32>(tailA2, xBFp16TailA2, pregMask);
        Cast<float, bfloat16_t, castTraitB162B32>(tailB2, xBFp16TailB2, pregMask);
        // add x1 + x2
        Add(mainA, mainA, mainA2, pregMask);
        Add(mainB, mainB, mainB2, pregMask);
        Add(tailA, tailA, tailA2, pregMask);
        Add(tailB, tailB, tailB2, pregMask);
        Mul(mainA, mainA, mainA, pregMask);
        Mul(mainB, mainB, mainB, pregMask);
        Mul(tailA, tailA, tailA, pregMask);
        Mul(tailB, tailB, tailB, pregMask);
    } else {
        LoadAlign(mainA, mainAddr + offset1);
        LoadAlign(mainB, mainAddr + offset2);
        LoadAlign(tailA, tailAddr + offset1);
        LoadAlign(tailB, tailAddr + offset2);
        // load x2
        RegTensor<float> mainA2, mainB2, tailA2, tailB2;
        LoadAlign(mainA2, mainAddr2 + offset1);
        LoadAlign(mainB2, mainAddr2 + offset2);
        LoadAlign(tailA2, tailAddr2 + offset1);
        LoadAlign(tailB2, tailAddr2 + offset2);
        // add x1 + x2
        Add(mainA, mainA, mainA2, pregMask);
        Add(mainB, mainB, mainB2, pregMask);
        Add(tailA, tailA, tailA2, pregMask);
        Add(tailB, tailB, tailB2, pregMask);
        // x * x
        Mul(mainA, mainA, mainA, pregMask);
        Mul(mainB, mainB, mainB, pregMask);
        Mul(tailA, tailA, tailA, pregMask);
        Mul(tailB, tailB, tailB, pregMask);
    }
}

template <typename T>
__aicore__ inline void LoadForHandleMasterV2(__ubuf__ T* masterAddr, uint16_t offset1, uint16_t offset2,
                                             RegTensor<float>& mainA, RegTensor<float>& mainB, MaskReg& pregMask,
                                             __ubuf__ T* masterAddr2)
{
    if constexpr (IsSameType<T, half>::value) {
        // x1/x2 load and cast
        RegTensor<half> xFp16MainA, xFp16MainB;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16MainA, masterAddr + offset1);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16MainB, masterAddr + offset2);
        Cast<float, half, castTraitB162B32>(mainA, xFp16MainA, pregMask);
        Cast<float, half, castTraitB162B32>(mainB, xFp16MainB, pregMask);
        RegTensor<half> xFp16MainA2, xFp16MainB2;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16MainA2, masterAddr2 + offset1);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16MainB2, masterAddr2 + offset2);
        RegTensor<float> mainA2, mainB2;
        Cast<float, half, castTraitB162B32>(mainA2, xFp16MainA2, pregMask);
        Cast<float, half, castTraitB162B32>(mainB2, xFp16MainB2, pregMask);
        // add x1 + x2
        Add(mainA, mainA, mainA2, pregMask);
        Add(mainB, mainB, mainB2, pregMask);

        Mul(mainA, mainA, mainA, pregMask);
        Mul(mainB, mainB, mainB, pregMask);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        // x1/x2 load and cast
        RegTensor<bfloat16_t> xBFp16MainA, xBFp16MainB;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16MainA, masterAddr + offset1);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16MainB, masterAddr + offset2);
        Cast<float, bfloat16_t, castTraitB162B32>(mainA, xBFp16MainA, pregMask);
        Cast<float, bfloat16_t, castTraitB162B32>(mainB, xBFp16MainB, pregMask);
        RegTensor<bfloat16_t> xBFp16MainA2, xBFp16MainB2;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16MainA2, masterAddr2 + offset1);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16MainB2, masterAddr2 + offset2);
        RegTensor<float> mainA2, mainB2;
        Cast<float, bfloat16_t, castTraitB162B32>(mainA2, xBFp16MainA2, pregMask);
        Cast<float, bfloat16_t, castTraitB162B32>(mainB2, xBFp16MainB2, pregMask);
        // add x1 + x2
        Add(mainA, mainA, mainA2, pregMask);
        Add(mainB, mainB, mainB2, pregMask);
        Mul(mainA, mainA, mainA, pregMask);
        Mul(mainB, mainB, mainB, pregMask);
    } else {
        LoadAlign(mainA, masterAddr + offset1);
        LoadAlign(mainB, masterAddr + offset2);
        // x2 load
        RegTensor<float> mainA2, mainB2;
        LoadAlign(mainA2, masterAddr2 + offset1);
        LoadAlign(mainB2, masterAddr2 + offset2);
        // add x1 + x2
        Add(mainA, mainA, mainA2, pregMask);
        Add(mainB, mainB, mainB2, pregMask);
        Mul(mainA, mainA, mainA, pregMask);
        Mul(mainB, mainB, mainB, pregMask);
    }
}

template <typename T>
__aicore__ inline void ComputeFormerImplV2(LocalTensor<float>& dstLocal, LocalTensor<T>& xLocal1,
                                           LocalTensor<T>& xLocal2, LocalTensor<float>& workLocal, uint32_t offset,
                                           uint32_t count, uint32_t powerSplit)
{
    uint32_t remainTile = count - powerSplit;
    uint32_t masterTile = powerSplit - remainTile;
    uint32_t remainSreg = remainTile;
    uint16_t remainRepeats = remainTile / (2 * V_LENGTH);
    uint32_t masterSreg = masterTile;
    uint16_t masterRepeats = masterTile / (2 * V_LENGTH);

    uint32_t mergeTile = powerSplit / (2 * V_LENGTH);
    uint32_t mergeSreg = mergeTile;
    uint16_t mergeRepeats = mergeTile / (2 * V_LENGTH);

    uint32_t meanTile = mergeRepeats == 0 ? mergeTile : mergeRepeats;
    uint32_t meanSreg = meanTile;

    __ubuf__ T* mainAddr = (__ubuf__ T*)xLocal1.GetPhyAddr();
    __ubuf__ T* tailAddr = (__ubuf__ T*)xLocal1.GetPhyAddr() + int64_t(powerSplit);
    __ubuf__ T* masterAddr = (__ubuf__ T*)xLocal1.GetPhyAddr() + int64_t(remainTile);
    __ubuf__ T* mainAddr2 = (__ubuf__ T*)xLocal2.GetPhyAddr();
    __ubuf__ T* tailAddr2 = (__ubuf__ T*)xLocal2.GetPhyAddr() + int64_t(powerSplit);
    __ubuf__ T* masterAddr2 = (__ubuf__ T*)xLocal2.GetPhyAddr() + int64_t(remainTile);

    __ubuf__ float* workAddr = (__ubuf__ float*)workLocal.GetPhyAddr();
    __ubuf__ float* dstAddr = (__ubuf__ float*)dstLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> mainA, mainB, tailA, tailB, vMean, vDupReg, rstdReg;
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregMask;

        for (uint16_t i = 0; i < (uint16_t)remainRepeats; ++i) {
            pregMask = UpdateMask<float>(remainSreg);
            LoadForHandleRemainV2(mainAddr, tailAddr, (i * 2 + 0) * V_LENGTH, (i * 2 + 1) * V_LENGTH, mainA, mainB,
                                  tailA, tailB, pregMask, mainAddr2, tailAddr2);
            Add(mainA, mainA, tailA, pregMask);
            Add(mainB, mainB, tailB, pregMask);
            Add(mainA, mainA, mainB, pregMask);
            Reduce<ReduceType::SUM>(vMean, mainA, pregMask);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + i, vMean, pregMerge);
        }
        for (uint16_t i = 0; i < (uint16_t)masterRepeats; ++i) {
            pregMask = UpdateMask<float>(masterSreg);
            LoadForHandleMasterV2(masterAddr, (i * 2 + 0) * V_LENGTH, (i * 2 + 1) * V_LENGTH, mainA, mainB, pregMask,
                                  masterAddr2);
            Add(mainA, mainA, mainB, pregMask);
            Reduce<ReduceType::SUM>(vMean, mainA, pregMask);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + remainRepeats + i, vMean, pregMerge);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < (uint16_t)mergeRepeats; ++i) {
            pregMask = UpdateMask<float>(mergeSreg);
            LoadAlign(mainA, workAddr + (i * 2 + 0) * V_LENGTH);
            LoadAlign(mainB, workAddr + (i * 2 + 1) * V_LENGTH);
            Add(mainA, mainA, mainB, pregMask);
            Reduce<ReduceType::SUM>(vMean, mainA, pregMask);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + i, vMean, pregMerge);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        {
            pregMask = UpdateMask<float>(meanSreg);
            LoadAlign(mainA, workAddr + 0);
            Reduce<ReduceType::SUM>(vMean, mainA, pregMask);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr + offset, vMean, pregMerge);
        }
    }
}

template <typename T>
__aicore__ inline void ComputeFormerImplV1MultiN(LocalTensor<T>& xLocal1, LocalTensor<T>& xLocal2,
                                                 LocalTensor<float>& xFp32, LocalTensor<float>& workLocal,
                                                 LocalTensor<float>& rstdLocal, float avgFactor, float epsilon,
                                                 uint32_t offset, uint32_t count, uint32_t powerSplit, uint32_t curRows)
{
    uint32_t remainTile = count - powerSplit;
    uint16_t remainRepeats = remainTile / (2 * V_LENGTH);

    uint32_t masterTile = powerSplit - remainTile;
    uint16_t masterRepeats = masterTile / (2 * V_LENGTH);

    uint32_t mergeTile = powerSplit / (2 * V_LENGTH);
    uint16_t mergeRepeats = mergeTile / (2 * V_LENGTH);

    uint32_t meanTile = mergeRepeats == 0 ? mergeTile : mergeRepeats;

    __ubuf__ T* mainAddr = (__ubuf__ T*)xLocal1.GetPhyAddr();
    __ubuf__ T* tailAddr = (__ubuf__ T*)xLocal1.GetPhyAddr() + int64_t(powerSplit);
    __ubuf__ T* masterAddr = (__ubuf__ T*)xLocal1.GetPhyAddr() + int64_t(remainTile);
    __ubuf__ T* mainAddr2 = (__ubuf__ T*)xLocal2.GetPhyAddr();
    __ubuf__ T* tailAddr2 = (__ubuf__ T*)xLocal2.GetPhyAddr() + int64_t(powerSplit);
    __ubuf__ T* masterAddr2 = (__ubuf__ T*)xLocal2.GetPhyAddr() + int64_t(remainTile);
    __ubuf__ float *xFp32MainAddr, *xFp32TailAddr, *xFp32MasterAddr;

    xFp32MainAddr = (__ubuf__ float*)xFp32.GetPhyAddr();
    xFp32TailAddr = (__ubuf__ float*)xFp32.GetPhyAddr() + int64_t(powerSplit);
    xFp32MasterAddr = (__ubuf__ float*)xFp32.GetPhyAddr() + int64_t(remainTile);

    __ubuf__ float* workAddr = (__ubuf__ float*)workLocal.GetPhyAddr();
    __ubuf__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();

    uint32_t curRowsAlign = CeilDiv((int32_t)curRows, 2);
    int64_t unrollOffset = (curRows / 2) * count;
    bool isWithTail = curRowsAlign - (curRows / 2);
    uint32_t tailOffset = offset + curRows / 2;

    __ubuf__ T* mainAddr3 = (__ubuf__ T*)xLocal1.GetPhyAddr() + unrollOffset;
    __ubuf__ T* tailAddr3 = (__ubuf__ T*)xLocal1.GetPhyAddr() + int64_t(powerSplit) + unrollOffset;
    __ubuf__ T* masterAddr3 = (__ubuf__ T*)xLocal1.GetPhyAddr() + int64_t(remainTile) + unrollOffset;
    __ubuf__ T* mainAddr4 = (__ubuf__ T*)xLocal2.GetPhyAddr() + unrollOffset;
    __ubuf__ T* tailAddr4 = (__ubuf__ T*)xLocal2.GetPhyAddr() + int64_t(powerSplit) + unrollOffset;
    __ubuf__ T* masterAddr4 = (__ubuf__ T*)xLocal2.GetPhyAddr() + int64_t(remainTile) + unrollOffset;
    __ubuf__ float *xFp32MainAddr3, *xFp32TailAddr3, *xFp32MasterAddr3;

    xFp32MainAddr3 = (__ubuf__ float*)xFp32.GetPhyAddr() + unrollOffset;
    xFp32TailAddr3 = (__ubuf__ float*)xFp32.GetPhyAddr() + int64_t(powerSplit) + unrollOffset;
    xFp32MasterAddr3 = (__ubuf__ float*)xFp32.GetPhyAddr() + int64_t(remainTile) + unrollOffset;

    __ubuf__ float* workAddr1 = (__ubuf__ float*)workLocal.GetPhyAddr() + ONCE_VECTOR_SIZE;
    __ubuf__ float* rstdAddr1 = (__ubuf__ float*)rstdLocal.GetPhyAddr() + curRows / 2;

    __VEC_SCOPE__
    {
        for (uint16_t row = 0; row < static_cast<uint16_t>(curRows / 2); row++) {
            uint32_t remainSreg = remainTile;
            uint32_t masterSreg = masterTile;
            uint32_t meanSreg = meanTile;
            uint32_t mergeSreg = mergeTile;
            RegTensor<float> mainA, mainB, tailA, tailB, vMean, vDupReg, rstdReg;
            MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregMask;

            uint32_t remainSreg1 = remainTile;
            uint32_t masterSreg1 = masterTile;
            uint32_t mergeSreg1 = mergeTile;
            uint32_t meanSreg1 = meanTile;
            RegTensor<float> mainA3, mainB3, tailA3, tailB3, vMean3, vDupReg3, rstdReg3;
            MaskReg pregMain1 = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregMerge1 = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregMask1;

            for (uint16_t i = 0; i < (uint16_t)remainRepeats; ++i) {
                pregMask = UpdateMask<float>(remainSreg);
                LoadForHandleRemainV1(mainAddr, tailAddr, (i * 2 + 0) * V_LENGTH, (i * 2 + 1) * V_LENGTH, mainA, mainB,
                                      tailA, tailB, pregMask, xFp32MainAddr, xFp32TailAddr, mainAddr2, tailAddr2);
                Add(mainA, mainA, tailA, pregMask);
                Add(mainB, mainB, tailB, pregMask);
                Add(mainA, mainA, mainB, pregMask);
                Reduce<ReduceType::SUM>(vMean, mainA, pregMask);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + i, vMean, pregMerge);
            }
            for (uint16_t i = 0; i < (uint16_t)masterRepeats; ++i) {
                pregMask = UpdateMask<float>(masterSreg);
                LoadForHandleMasterV1(masterAddr, (i * 2 + 0) * V_LENGTH, (i * 2 + 1) * V_LENGTH, mainA, mainB,
                                      pregMask, xFp32MasterAddr, masterAddr2);
                Add(mainA, mainA, mainB, pregMask);
                Reduce<ReduceType::SUM>(vMean, mainA, pregMask);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + remainRepeats + i, vMean, pregMerge);
            }
            // unroll part
            for (uint16_t i = 0; i < (uint16_t)remainRepeats; ++i) {
                pregMask1 = UpdateMask<float>(remainSreg1);
                LoadForHandleRemainV1(mainAddr3, tailAddr3, (i * 2 + 0) * V_LENGTH, (i * 2 + 1) * V_LENGTH, mainA3,
                                      mainB3, tailA3, tailB3, pregMask1, xFp32MainAddr3, xFp32TailAddr3, mainAddr4,
                                      tailAddr4);
                Add(mainA3, mainA3, tailA3, pregMask1);
                Add(mainB3, mainB3, tailB3, pregMask1);
                Add(mainA3, mainA3, mainB3, pregMask1);
                Reduce<ReduceType::SUM>(vMean3, mainA3, pregMask1);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr1 + i, vMean3, pregMerge1);
            }
            for (uint16_t i = 0; i < (uint16_t)masterRepeats; ++i) {
                pregMask1 = UpdateMask<float>(masterSreg1);
                LoadForHandleMasterV1(masterAddr3, (i * 2 + 0) * V_LENGTH, (i * 2 + 1) * V_LENGTH, mainA3, mainB3,
                                      pregMask1, xFp32MasterAddr3, masterAddr4);
                Add(mainA3, mainA3, mainB3, pregMask1);
                Reduce<ReduceType::SUM>(vMean3, mainA3, pregMask1);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr1 + remainRepeats + i, vMean3, pregMerge1);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            for (uint16_t i = 0; i < (uint16_t)mergeRepeats; ++i) {
                pregMask = UpdateMask<float>(mergeSreg);
                LoadAlign(mainA, workAddr + (i * 2 + 0) * V_LENGTH);
                LoadAlign(mainB, workAddr + (i * 2 + 1) * V_LENGTH);
                Add(mainA, mainA, mainB, pregMask);
                Reduce<ReduceType::SUM>(vMean, mainA, pregMask);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + i, vMean, pregMerge);
            }
            // unroll part
            for (uint16_t i = 0; i < (uint16_t)mergeRepeats; ++i) {
                pregMask1 = UpdateMask<float>(mergeSreg1);
                LoadAlign(mainA3, workAddr1 + (i * 2 + 0) * V_LENGTH);
                LoadAlign(mainB3, workAddr1 + (i * 2 + 1) * V_LENGTH);
                Add(mainA3, mainA3, mainB3, pregMask1);
                Reduce<ReduceType::SUM>(vMean3, mainA3, pregMask1);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr1 + i, vMean3, pregMerge1);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            {
                pregMask = UpdateMask<float>(meanSreg);
                LoadAlign(mainA, workAddr + 0);
                Reduce<ReduceType::SUM>(vMean, mainA, pregMask);
                Muls(vMean, vMean, avgFactor, pregMerge);
                Adds(vMean, vMean, epsilon, pregMerge);
                Sqrt(vMean, vMean, pregMerge);
                Duplicate(vDupReg, float(1.0), pregMerge);
                Div(rstdReg, vDupReg, vMean, pregMerge);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdAddr + offset, rstdReg, pregMerge);
            }
            // unroll part
            {
                pregMask1 = UpdateMask<float>(meanSreg1);
                LoadAlign(mainA3, workAddr1 + 0);
                Reduce<ReduceType::SUM>(vMean3, mainA3, pregMask1);
                Muls(vMean3, vMean3, avgFactor, pregMerge1);
                Adds(vMean3, vMean3, epsilon, pregMerge1);
                Sqrt(vMean3, vMean3, pregMerge1);
                Duplicate(vDupReg3, float(1.0), pregMerge1);
                Div(rstdReg3, vDupReg3, vMean3, pregMerge1);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdAddr1 + offset, rstdReg3, pregMerge1);
            }
            offset += 1;
            mainAddr += int64_t(count);
            tailAddr += int64_t(count);
            masterAddr += int64_t(count);
            mainAddr2 += int64_t(count);
            tailAddr2 += int64_t(count);
            masterAddr2 += int64_t(count);
            xFp32MainAddr += int64_t(count);
            xFp32TailAddr += int64_t(count);
            xFp32MasterAddr += int64_t(count);

            mainAddr3 += int64_t(count);
            tailAddr3 += int64_t(count);
            masterAddr3 += int64_t(count);
            mainAddr4 += int64_t(count);
            tailAddr4 += int64_t(count);
            masterAddr4 += int64_t(count);
            xFp32MainAddr3 += int64_t(count);
            xFp32TailAddr3 += int64_t(count);
            xFp32MasterAddr3 += int64_t(count);
        }
    }
    uint32_t tailDataOffset = unrollOffset + (curRows / 2) * count;
    __ubuf__ T* mainAddr5 = (__ubuf__ T*)xLocal1.GetPhyAddr() + tailDataOffset;
    __ubuf__ T* tailAddr5 = (__ubuf__ T*)xLocal1.GetPhyAddr() + int64_t(powerSplit) + tailDataOffset;
    __ubuf__ T* masterAddr5 = (__ubuf__ T*)xLocal1.GetPhyAddr() + int64_t(remainTile) + tailDataOffset;
    __ubuf__ T* mainAddr6 = (__ubuf__ T*)xLocal2.GetPhyAddr() + tailDataOffset;
    __ubuf__ T* tailAddr6 = (__ubuf__ T*)xLocal2.GetPhyAddr() + int64_t(powerSplit) + tailDataOffset;
    __ubuf__ T* masterAddr6 = (__ubuf__ T*)xLocal2.GetPhyAddr() + int64_t(remainTile) + tailDataOffset;
    __ubuf__ float *xFp32MainAddr5, *xFp32TailAddr5, *xFp32MasterAddr5;

    xFp32MainAddr5 = (__ubuf__ float*)xFp32.GetPhyAddr() + tailDataOffset;
    xFp32TailAddr5 = (__ubuf__ float*)xFp32.GetPhyAddr() + int64_t(powerSplit) + tailDataOffset;
    xFp32MasterAddr5 = (__ubuf__ float*)xFp32.GetPhyAddr() + int64_t(remainTile) + tailDataOffset;

    if (isWithTail) {
        __VEC_SCOPE__
        {
            uint32_t remainSreg1 = remainTile;
            uint32_t masterSreg1 = masterTile;
            uint32_t meanSreg1 = meanTile;
            uint32_t mergeSreg1 = mergeTile;
            RegTensor<float> mainA1, mainB1, tailA1, tailB1, vMean1, vDupReg1, rstdReg1;
            MaskReg pregMerge1 = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregMain1 = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregMask1;

            for (uint16_t i = 0; i < (uint16_t)remainRepeats; ++i) {
                pregMask1 = UpdateMask<float>(remainSreg1);
                LoadForHandleRemainV1(mainAddr5, tailAddr5, (i * 2 + 0) * V_LENGTH, (i * 2 + 1) * V_LENGTH, mainA1,
                                      mainB1, tailA1, tailB1, pregMask1, xFp32MainAddr5, xFp32TailAddr5, mainAddr6,
                                      tailAddr6);
                Add(mainA1, mainA1, tailA1, pregMask1);
                Add(mainB1, mainB1, tailB1, pregMask1);
                Add(mainA1, mainA1, mainB1, pregMask1);
                Reduce<ReduceType::SUM>(vMean1, mainA1, pregMask1);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr1 + i, vMean1, pregMerge1);
            }
            for (uint16_t i = 0; i < (uint16_t)masterRepeats; ++i) {
                pregMask1 = UpdateMask<float>(masterSreg1);
                LoadForHandleMasterV1(masterAddr5, (i * 2 + 0) * V_LENGTH, (i * 2 + 1) * V_LENGTH, mainA1, mainB1,
                                      pregMask1, xFp32MasterAddr5, masterAddr6);
                Add(mainA1, mainA1, mainB1, pregMask1);
                Reduce<ReduceType::SUM>(vMean1, mainA1, pregMask1);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr1 + remainRepeats + i, vMean1, pregMerge1);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            for (uint16_t i = 0; i < (uint16_t)mergeRepeats; ++i) {
                pregMask1 = UpdateMask<float>(mergeSreg1);
                LoadAlign(mainA1, workAddr1 + (i * 2 + 0) * V_LENGTH);
                LoadAlign(mainB1, workAddr1 + (i * 2 + 1) * V_LENGTH);
                Add(mainA1, mainA1, mainB1, pregMask1);
                Reduce<ReduceType::SUM>(vMean1, mainA1, pregMask1);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr1 + i, vMean1, pregMerge1);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            {
                pregMask1 = UpdateMask<float>(meanSreg1);
                LoadAlign(mainA1, workAddr1 + 0);
                Reduce<ReduceType::SUM>(vMean1, mainA1, pregMask1);
                Muls(vMean1, vMean1, avgFactor, pregMerge1);
                Adds(vMean1, vMean1, epsilon, pregMerge1);
                Sqrt(vMean1, vMean1, pregMerge1);
                Duplicate(vDupReg1, float(1.0), pregMerge1);
                Div(rstdReg1, vDupReg1, vMean1, pregMerge1);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdAddr1 + tailOffset, rstdReg1, pregMerge1);
            }
        }
    }
}

template <typename T, bool HAS_BETA>
__aicore__ inline void ComputeLatterY(LocalTensor<float>& xFp32, LocalTensor<T>& gammaLocal,
                                      LocalTensor<T>& betaLocal, LocalTensor<T>& yLocal,
                                      LocalTensor<float>& rstdLocal, uint32_t offset, uint32_t count,
                                      LocalTensor<T> xOutLocal)
{
    uint32_t calCount = count / 2;
    uint32_t sreg = (uint32_t)calCount;
    uint16_t repeatTimes = CeilDivision(calCount, V_LENGTH);

    __ubuf__ float* xAddr1 = (__ubuf__ float*)xFp32.GetPhyAddr();
    __ubuf__ float* xAddr2 = (__ubuf__ float*)xFp32.GetPhyAddr() + calCount;
    __ubuf__ T* gammaAddr1 = (__ubuf__ T*)gammaLocal.GetPhyAddr();
    __ubuf__ T* gammaAddr2 = (__ubuf__ T*)gammaLocal.GetPhyAddr() + calCount;
    __ubuf__ float* srcAddr2 = (__ubuf__ float*)rstdLocal.GetPhyAddr();
    __ubuf__ T* yAddr1 = (__ubuf__ T*)yLocal.GetPhyAddr();
    __ubuf__ T* yAddr2 = (__ubuf__ T*)yLocal.GetPhyAddr() + calCount;
    __ubuf__ T* xOutAddr1 = (__ubuf__ T*)xOutLocal.GetPhyAddr();
    __ubuf__ T* xOutAddr2 = (__ubuf__ T*)xOutLocal.GetPhyAddr() + calCount;
    __ubuf__ T* betaAddr1 = nullptr;
    __ubuf__ T* betaAddr2 = nullptr;
    if constexpr (HAS_BETA) {
        betaAddr1 = (__ubuf__ T*)betaLocal.GetPhyAddr();
        betaAddr2 = betaAddr1 + calCount;
    }

    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        __VEC_SCOPE__
        {
            RegTensor<T> yB16Reg1, yB16Reg2;
            RegTensor<float> xB32Reg1, xB32Reg2;
            RegTensor<T> gammaReg1, gammaReg2;
            RegTensor<float> rstdReg;
            RegTensor<float> dst1Reg, gammaFp32Reg1, yReg1;
            RegTensor<float> dst2Reg, gammaFp32Reg2, yReg2;
            RegTensor<T> xout16Reg1, xout16Reg2;
            RegTensor<float> betaFp32Reg1, betaFp32Reg2;
            MaskReg maskReg;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, srcAddr2 + offset);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                maskReg = UpdateMask<float>(sreg);
                LoadAlign(xB32Reg1, xAddr1 + i * V_LENGTH);
                LoadAlign(xB32Reg2, xAddr2 + i * V_LENGTH);
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(gammaReg1, gammaAddr1 + i * V_LENGTH);
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(gammaReg2, gammaAddr2 + i * V_LENGTH);
                Cast<float, T, castTraitB162B32>(gammaFp32Reg1, gammaReg1, maskReg);
                Cast<float, T, castTraitB162B32>(gammaFp32Reg2, gammaReg2, maskReg);
                Mul(dst1Reg, xB32Reg1, rstdReg, maskReg);
                Mul(dst2Reg, xB32Reg2, rstdReg, maskReg);
                Mul(yReg1, dst1Reg, gammaFp32Reg1, maskReg);
                Mul(yReg2, dst2Reg, gammaFp32Reg2, maskReg);
                if constexpr (HAS_BETA) {
                    LoadRegForDtype<T>(betaAddr1, betaFp32Reg1, maskReg, i * V_LENGTH);
                    LoadRegForDtype<T>(betaAddr2, betaFp32Reg2, maskReg, i * V_LENGTH);
                    Add(yReg1, yReg1, betaFp32Reg1, maskReg);
                    Add(yReg2, yReg2, betaFp32Reg2, maskReg);
                }
                Cast<T, float, castTraitB322B16>(yB16Reg1, yReg1, maskReg);
                Cast<T, float, castTraitB322B16>(yB16Reg2, yReg2, maskReg);
                StoreAlign<T, StoreDist::DIST_PACK_B32>(yAddr1 + i * V_LENGTH, yB16Reg1, maskReg);
                StoreAlign<T, StoreDist::DIST_PACK_B32>(yAddr2 + i * V_LENGTH, yB16Reg2, maskReg);
                // outX
                Cast<T, float, castTraitB322B16>(xout16Reg1, xB32Reg1, maskReg);
                Cast<T, float, castTraitB322B16>(xout16Reg2, xB32Reg2, maskReg);
                StoreAlign<T, StoreDist::DIST_PACK_B32>(xOutAddr1 + i * V_LENGTH, xout16Reg1, maskReg);
                StoreAlign<T, StoreDist::DIST_PACK_B32>(xOutAddr2 + i * V_LENGTH, xout16Reg2, maskReg);
            }
        }
    } else if constexpr (IsSameType<T, float>::value) {
        __VEC_SCOPE__
        {
            RegTensor<float> rstdReg;
            RegTensor<float> xReg1, gammaReg1, yReg1, vRegTmp1;
            RegTensor<float> xReg2, gammaReg2, yReg2, vRegTmp2;
            RegTensor<float> betaReg1, betaReg2;
            MaskReg maskReg;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, srcAddr2 + offset);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                maskReg = UpdateMask<float>(sreg);
                LoadAlign(xReg1, xAddr1 + i * V_LENGTH);
                LoadAlign(xReg2, xAddr2 + i * V_LENGTH);
                LoadAlign(gammaReg1, gammaAddr1 + i * V_LENGTH);
                LoadAlign(gammaReg2, gammaAddr2 + i * V_LENGTH);
                Mul(vRegTmp1, xReg1, rstdReg, maskReg);
                Mul(vRegTmp2, xReg2, rstdReg, maskReg);
                Mul(yReg1, vRegTmp1, gammaReg1, maskReg);
                Mul(yReg2, vRegTmp2, gammaReg2, maskReg);
                if constexpr (HAS_BETA) {
                    LoadRegForDtype<float>(betaAddr1, betaReg1, maskReg, i * V_LENGTH);
                    LoadRegForDtype<float>(betaAddr2, betaReg2, maskReg, i * V_LENGTH);
                    Add(yReg1, yReg1, betaReg1, maskReg);
                    Add(yReg2, yReg2, betaReg2, maskReg);
                }
                StoreAlign(yAddr1 + i * V_LENGTH, yReg1, maskReg);
                StoreAlign(yAddr2 + i * V_LENGTH, yReg2, maskReg);
                // outX
                StoreAlign(xOutAddr1 + i * V_LENGTH, xReg1, maskReg);
                StoreAlign(xOutAddr2 + i * V_LENGTH, xReg2, maskReg);
            }
        }
    }
}
} // namespace AddRmsNormBiasA5
#endif // ADD_RMS_NORM_BIAS_REGBASE_COMMON_H
