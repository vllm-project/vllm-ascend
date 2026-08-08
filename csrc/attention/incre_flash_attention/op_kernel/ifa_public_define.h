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
 * \file ifa_public_define.h
 * \brief
 */
#ifndef IFA_PUBLIC_DEFINE_H
#define IFA_PUBLIC_DEFINE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;
using AscendC::AIC;
using AscendC::AIV;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::SetFlag;
using AscendC::ShapeInfo;
using AscendC::SoftmaxConfig;
using AscendC::WaitFlag;
using matmul::Matmul;
using matmul::MatmulType;

constexpr float FLT_MAX = 3.402823466e+38F;
constexpr int32_t VMLA_ONE_REPEATE_ROW_COUNT = 4;
constexpr int32_t VMLA_ONE_REPEATE_COLUMN_COUNT = 16;
constexpr int32_t FP16_ONE_BLOCK_SIZE = 16;
constexpr int32_t FP32_ONE_BLOCK_SIZE = 8;
constexpr int32_t ALIGN_BLOCK_SIZE = 16;

constexpr MatmulConfig CFG_NORM_EXCEED = GetNormalConfig(true);
constexpr MatmulConfig CFG_MDL_EXCEED = GetMDLConfig(true);

// CFG_NORM_EXCEED_INIT: doNorm, enable intrinsicsCheck and Init
constexpr MatmulConfig CFG_NORM_EXCEED_INIT{.doNorm = true,
                                            .doBasicBlock = false,
                                            .doMultiDataLoad = false,
                                            .basicM = 0,
                                            .basicN = 0,
                                            .basicK = 0,
                                            .intrinsicsCheck = true,
                                            .isNBatch = false,
                                            .enVecND2NZ = false,
                                            .doSpecialBasicBlock = false,
                                            .doMTE2Preload = false,
                                            .singleCoreM = 0,
                                            .singleCoreN = 0,
                                            .singleCoreK = 0,
                                            .stepM = 0,
                                            .stepN = 0,
                                            .baseMN = 0,
                                            .singleCoreMN = 0,
                                            .enUnitFlag = true,
                                            .isPerTensor = false,
                                            .hasAntiQuantOffset = false,
                                            .doIBShareNorm = false,
                                            .doSpecialMDL = false,
                                            .enableInit = false,
                                            .batchMode = BatchMode::NONE,
                                            .enableEnd = false,
                                            .enableGetTensorC = false,
                                            .enableSetOrgShape = true,
                                            .enableSetBias = false,
                                            .enableSetTail = true,
                                            .enableQuantVector = false,
                                            .enableSetDefineData = true,
                                            .iterateMode = IterateMode::ITERATE_MODE_ALL};

// CFG_MDL_EXCEED_INIT: enable MDL, intrinsicsCheck and Init
constexpr MatmulConfig CFG_MDL_EXCEED_INIT{.doNorm = false,
                                           .doBasicBlock = false,
                                           .doMultiDataLoad = true,
                                           .basicM = 0,
                                           .basicN = 0,
                                           .basicK = 0,
                                           .intrinsicsCheck = true,
                                           .isNBatch = false,
                                           .enVecND2NZ = false,
                                           .doSpecialBasicBlock = false,
                                           .doMTE2Preload = false,
                                           .singleCoreM = 0,
                                           .singleCoreN = 0,
                                           .singleCoreK = 0,
                                           .stepM = 0,
                                           .stepN = 0,
                                           .baseMN = 0,
                                           .singleCoreMN = 0,
                                           .enUnitFlag = false,
                                           .isPerTensor = false,
                                           .hasAntiQuantOffset = false,
                                           .doIBShareNorm = false,
                                           .doSpecialMDL = false,
                                           .enableInit = false,
                                           .batchMode = BatchMode::NONE,
                                           .enableEnd = false,
                                           .enableGetTensorC = false,
                                           .enableSetOrgShape = true,
                                           .enableSetBias = false,
                                           .enableSetTail = true,
                                           .enableQuantVector = false,
                                           .enableSetDefineData = true,
                                           .iterateMode = IterateMode::ITERATE_MODE_ALL};

// CFG_MDL_EXCEED_INIT_CALLBACK: enable MDL, intrinsicsCheck and Init, enable CALLBACK, enable unitflag
constexpr MatmulConfig CFG_MDL_EXCEED_INIT_CALLBACK{.doNorm = false,
                                                    .doBasicBlock = false,
                                                    .doMultiDataLoad = true,
                                                    .basicM = 0,
                                                    .basicN = 0,
                                                    .basicK = 0,
                                                    .intrinsicsCheck = true,
                                                    .isNBatch = false,
                                                    .enVecND2NZ = false,
                                                    .doSpecialBasicBlock = false,
                                                    .doMTE2Preload = false,
                                                    .singleCoreM = 0,
                                                    .singleCoreN = 0,
                                                    .singleCoreK = 0,
                                                    .stepM = 0,
                                                    .stepN = 0,
                                                    .baseMN = 0,
                                                    .singleCoreMN = 0,
                                                    .enUnitFlag = true, // enable unitflag
                                                    .isPerTensor = false,
                                                    .hasAntiQuantOffset = false,
                                                    .doIBShareNorm = false,
                                                    .doSpecialMDL = false,
                                                    .enableInit = false,
                                                    .batchMode = BatchMode::NONE,
                                                    .enableEnd = false,
                                                    .enableGetTensorC = false,
                                                    .enableSetOrgShape = true,
                                                    .enableSetBias = false,
                                                    .enableSetTail = true,
                                                    .enableQuantVector = false,
                                                    .enableSetDefineData = true,
                                                    .iterateMode = IterateMode::ITERATE_MODE_ALL,
                                                    .enableReuse = false};

constexpr SoftmaxConfig IFA_SOFTMAX_FLASHV2_CFG = {false}; // 将isCheckTiling设置为false

#define IFA_SOFTMAX_WITHOUT_BRC
constexpr SoftmaxConfig IFA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC}; // 将isCheckTiling设置为false, 输入输出的max&sum&exp的shape为(m, 1)

constexpr float FLOAT_ZERO = 0;
constexpr float FLOAT_MAX = FLT_MAX;
constexpr float FLOAT_EPS = 2.220446049250313e-16;

constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;

constexpr uint32_t MAX_UINT16 = 65535;
constexpr uint64_t BYTE_BLOCK = 32UL;
constexpr uint32_t REPEAT_BLOCK_BYTE = 256;
constexpr uint32_t IFA_MAX_REPEAT_TIMES = 256;

constexpr int32_t FLOAT_VECTOR_SIZE_I = 64;
constexpr int32_t VECTOR_SIZE_I = 128;
constexpr int32_t BLOCK_SIZE_I = 16;
constexpr uint32_t L0AB_HALF_BUF_SIZE_I = 16384;
constexpr uint32_t CUBE_MATRIX_SIZE_I = 256;
constexpr uint32_t STRIDE_UPPER_BOUND = 65535;

constexpr uint32_t L1_UINT8_BLOCK_SIZE = 131072;
constexpr int32_t UB_UINT8_BLOCK_SIZE_I = 32768;
constexpr int32_t UB_UINT8_LINE_SIZE_I = 1024;
constexpr int32_t UB_FLOAT_LINE_SIZE_I = 256;
constexpr int32_t UB_HALF_LINE_SIZE_I = 512;
constexpr uint32_t MAX_LEN_64_BYTES = 64;
constexpr uint32_t DEC_UB_UINT8_BLOCK_SIZE = 8192;
constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256; 
// repeat stride不能超过256

enum class CalcMode
{
    CALC_MODE_DEFAULT = 0, 
    CALC_MODE_PREFILL = 1
};

enum class LAYOUT {
    BSH = 0,
    BSND = 0,
    BNSD = 1,
    NZ = 2,
    TND = 3,
    NBSD = 4,
    NTD = 5
};

enum class AMLAMODE {
    NORMAL = 0,
    AMLA = 1,
    AMLA_3BUF = 2
};

struct TNDFDSplitInfo {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t s1Idx;
    uint32_t kvSplitNum; // [B, N2, S1]确定后，kv被切分的份数
    uint64_t combineTaskPrefixSum = 0;
};


template <typename Q_T, typename KV_T, typename OUT_T, typename ORIGIN_T, const bool PAGE_ATTENTION = false,
          const bool FLASH_DECODE = false, LAYOUT LAYOUT_T = LAYOUT::BSH, const uint8_t ANTIQUANT_MODE = 0,
          const bool SHARED_PREFIX = false, LAYOUT KV_LAYOUT_T = LAYOUT::BSH, const AMLAMODE AMLA = AMLAMODE::NORMAL,
          const bool BALANCE = false, const bool ENABLE_TREE = false,
          typename TILING_T = IncreFlashAttentionTilingDataV2, typename... Args>
struct IFAType {
    using queryType = Q_T;
    using kvType = KV_T;
    using outputType = OUT_T;
    using orginalType = ORIGIN_T;
	using TilingType = TILING_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr bool flashDecode = FLASH_DECODE;
    static constexpr LAYOUT layout = LAYOUT_T;
    static constexpr uint8_t antiquantMode = ANTIQUANT_MODE;
    static constexpr bool sharedPrefix = SHARED_PREFIX;
    static constexpr LAYOUT kvLayout = KV_LAYOUT_T;
    static constexpr AMLAMODE isAMla = AMLA;
    static constexpr bool isBalance = BALANCE;
    static constexpr bool enableTree = ENABLE_TREE;
};

constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);

__aicore__ inline void VecMulMat(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                 uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // vec mul by row
    // dstUb[i, j] = src0Ub[j] * src1Ub[i, j],
    // src0Ub:[1, columnCount] src1Ub:[dealRowCount, actualColumnCount] dstUb:[dealRowCount, columnCount]
    if (columnCount < REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) { // dstRepStride为0~255,columnCount需要小于2048
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        repeatParams.src0RepStride = 0;
        repeatParams.src1RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        uint32_t mask = FP32_REPEAT_ELEMENT_NUM;
        uint32_t loopCount = actualColumnCount / mask;
        uint32_t remainCount = actualColumnCount % mask;
        uint32_t offset = 0;
        for (int i = 0; i < loopCount; i++) {
            // offset = i * mask
            Mul(dstUb[offset], src0Ub[offset], src1Ub[offset], mask, dealRowCount, repeatParams);
            offset += mask;
        }
        if (remainCount > 0) {
            // offset = loopCount * mask
            Mul(dstUb[offset], src0Ub[offset], src1Ub[offset], remainCount, dealRowCount, repeatParams);
        }
    } else {
        uint32_t offset = 0;
        for (int i = 0; i < dealRowCount; i++) {
            Mul(dstUb[offset], src0Ub, src1Ub[offset], actualColumnCount);
            offset += columnCount;
        }
    }
}

__aicore__ inline void MatDivVec(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                 uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // vec mul by row
    // dstUb[i, j] = src1Ub[i, j] / src0Ub[j],
    // src0Ub:[dealRowCount, actualColumnCount] src1Ub:[1, columnCount] dstUb:[dealRowCount, columnCount]
    // restraint: dealRowCount < 256
    if (columnCount < REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) { // dstRepStride为0~255,columnCount需要小于2048
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        repeatParams.src1RepStride = 0;
        repeatParams.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        uint32_t mask = FP32_REPEAT_ELEMENT_NUM;
        uint32_t loopCount = actualColumnCount / mask;
        uint32_t remainCount = actualColumnCount % mask;
        uint32_t offset = 0;
        for (int i = 0; i < loopCount; i++) {
            // offset = i * mask
            Div(dstUb[offset], src0Ub[offset], src1Ub[offset], mask, dealRowCount, repeatParams);
            offset += mask;
        }
        if (remainCount > 0) {
            // offset = loopCount * mask
            Div(dstUb[offset], src0Ub[offset], src1Ub[offset], remainCount, dealRowCount, repeatParams);
        }
    } else {
        uint32_t offset = 0;
        for (int i = 0; i < dealRowCount; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub, actualColumnCount);
            offset += columnCount;
        }
    }
}

__aicore__ inline void VecMulBlkMat(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                 uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // vec mul by row
    // src0Ub:[1, columnCount] src1Ub:[dealRowCount, 8] dstUb:[dealRowCount, columnCount]
    constexpr uint32_t VECTOR_BLOCK_SIZE = 8;
    BinaryRepeatParams repeatParams;
    uint32_t mask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t loopCount = actualColumnCount / mask;
    uint32_t remainCount = actualColumnCount % mask;
    if (columnCount < REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) { // dstRepStride为0~255,columnCount需要小于2048
        // [1, columnCount] * [dealRowCount, 8]
        repeatParams.src0BlkStride = 1;
        repeatParams.src0RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.src1RepStride = 1;
        repeatParams.dstBlkStride = 1;
        repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < loopCount; i++) {
            Mul(dstUb[offset], src0Ub[offset], src1Ub, mask, dealRowCount, repeatParams);
            offset += mask;
        }
        if (remainCount > 0) {
            Mul(dstUb[offset], src0Ub[offset], src1Ub, remainCount, dealRowCount, repeatParams);
        }
    } else {
        // [1, columnCount] * [1, 8]
        repeatParams.src0BlkStride = 1;
        repeatParams.src0RepStride = VECTOR_BLOCK_SIZE;
        repeatParams.src1BlkStride = 0;
        repeatParams.src1RepStride = 0;
        repeatParams.dstBlkStride = 1;
        repeatParams.dstRepStride = VECTOR_BLOCK_SIZE;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Mul(dstUb[i * columnCount], src0Ub, src1Ub[i * FP32_BLOCK_ELEMENT_NUM], mask, loopCount, repeatParams);
            if (remainCount > 0) {
                Mul(dstUb[i * columnCount + loopCount * mask], src0Ub[loopCount * mask],
                    src1Ub[i * FP32_BLOCK_ELEMENT_NUM], remainCount, 1, repeatParams);
            }
        }
    }
}

__aicore__ inline void VecAddMat(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                 uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // vec add by row
    // dstUb[i, j] = src0Ub[j] + src1Ub[i, j],
    // src0Ub:[1, columnCount] src1Ub:[dealRowCount, columnCount] dstUb:[dealRowCount, columnCount]
    if (columnCount < REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) { // dstRepStride为0~255,columnCount需要小于2048
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        repeatParams.src0RepStride = 0;
        repeatParams.src1RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        uint32_t mask = FP32_REPEAT_ELEMENT_NUM;
        uint32_t loopCount = actualColumnCount / mask;
        uint32_t remainCount = actualColumnCount % mask;

        uint64_t offset = 0;
        for (int i = 0; i < loopCount; i++) {
            Add(dstUb[offset], src0Ub[offset], src1Ub[offset], mask, dealRowCount, repeatParams);
            offset += mask;
        }
        if (remainCount > 0) {
            Add(dstUb[offset], src0Ub[offset], src1Ub[offset], remainCount, dealRowCount, repeatParams);
        }
    } else {
        uint32_t offset = 0;
        for (int i = 0; i < dealRowCount; i++) {
            Add(dstUb[offset], src0Ub, src1Ub[offset], actualColumnCount);
            offset += columnCount;
        }
    }
}

__aicore__ inline void RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // divs by row, 每行的元素除以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 0;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 1;
    repeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    if (columnRepeatCount <= dealRowCount) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub, dtypeMask, dealRowCount, repeatParamsDiv);
            offset += dtypeMask;
        }
    } else {
        BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub[i * FP32_BLOCK_ELEMENT_NUM], dtypeMask, columnRepeatCount,
                columnRepeatParams);
            offset += columnCount;
        }
    }
    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub, dRemain, dealRowCount, repeatParamsDiv);
    }
}

template <typename T>
__aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // muls by row, 每行的元素乘以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount] src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    // dealRowCount is repeat times, must be less 256
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2; // 32/4 * 2 = 16
    }

    // 每次只能连续读取256B的数据进行计算，故每次只能处理256B/sizeof(dType)=
    // 列方向分dLoop次，每次处理8列数据
    uint32_t dLoop = actualColumnCount / repeatElementNum;
    uint32_t dRemain = actualColumnCount % repeatElementNum;
    // REPEATE_STRIDE_UP_BOUND=256， 此限制由于src0RepStride数据类型为uint8之多256个datablock间距
    if (columnCount < REPEATE_STRIDE_UP_BOUND * blockElementNum) {
        BinaryRepeatParams repeatParams;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0RepStride = columnCount / blockElementNum;
        repeatParams.src1RepStride = 1;
        repeatParams.dstRepStride = columnCount / blockElementNum;

        // 如果以列为repeat所处理的次数小于行处理次数，则以列方式处理。反之则以行进行repeat处理
        if (dLoop <= dealRowCount) {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, dealRowCount, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            BinaryRepeatParams columnRepeatParams;
            columnRepeatParams.src0BlkStride = 1;
            columnRepeatParams.src1BlkStride = 0;
            columnRepeatParams.dstBlkStride = 1;
            columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            columnRepeatParams.src1RepStride = 0;
            columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum,
                    dLoop, columnRepeatParams);
            }
        }

        // 最后一次完成[dealRowCount, dRemain] * [dealRowCount, blockElementNum] 只计算有效部分
        if (dRemain > 0) {
            Mul(dstUb[dLoop * repeatElementNum], src0Ub[dLoop * repeatElementNum], src1Ub, dRemain, dealRowCount,
                repeatParams);
        }
    } else {
        BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = 8; // 每个repeat为256B数据，正好8个datablock
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = 8;
        repeatParams.dstBlkStride = 1;
        // 每次计算一行，共计算dealRowCount行
        for (uint32_t i = 0; i < dealRowCount; i++) {
            // 计算一行中的dLoop个repeat, 每个repeat计算256/block_size 个data_block
            Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum],
                repeatElementNum, dLoop, repeatParams);
            //  计算一行中的尾块
            if (dRemain > 0) {
                Mul(dstUb[i * columnCount + dLoop * repeatElementNum], src0Ub[i * columnCount + dLoop * repeatElementNum],
                    src1Ub[i * blockElementNum], dRemain, 1, repeatParams);
            }
        }
    }
}

__aicore__ inline void RowSum(LocalTensor<float> &dstUb, LocalTensor<float> srcUb, uint32_t dealRowCount,
                              uint32_t columnCount, uint32_t actualColumnCount)
{
    // sum by row, 按行求和
    // dstUb[i] = sum(srcUb[i, :])
    // src0Ub:[dealRowCount, columnCount] dstUb:[1, dealRowCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockCount = actualColumnCount / dtypeMask;
    uint32_t remain = actualColumnCount % dtypeMask;
    constexpr int32_t BLOCK_EXPAND_ROUNDS = 2;

    BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0RepStride = columnCount / (BYTE_BLOCK / sizeof(float));
    repeatParamsMax.src1RepStride = columnCount / (BYTE_BLOCK / sizeof(float));
    repeatParamsMax.dstRepStride = columnCount / (BYTE_BLOCK / sizeof(float));
    if (blockCount > 0 && remain > 0) {
        Add(srcUb, srcUb, srcUb[blockCount * dtypeMask], remain, dealRowCount, repeatParamsMax);
        PipeBarrier<PIPE_V>();
    }

    for (uint32_t loopCount = blockCount / BLOCK_EXPAND_ROUNDS; loopCount > 0; loopCount = blockCount / BLOCK_EXPAND_ROUNDS) {
        blockCount = (blockCount + 1) / BLOCK_EXPAND_ROUNDS;
        for (uint32_t j = 0; j < loopCount; j++) {
            Add(srcUb[j * dtypeMask], srcUb[j * dtypeMask], srcUb[(j + blockCount) * dtypeMask], dtypeMask,
                dealRowCount, repeatParamsMax);
        }
        PipeBarrier<PIPE_V>();
    }

    WholeReduceSum(dstUb, srcUb, (actualColumnCount < dtypeMask) ? actualColumnCount : dtypeMask, dealRowCount, 1, 1,
                   columnCount / (BYTE_BLOCK / sizeof(float)));
}

__aicore__ inline uint32_t GetMinPowerTwo(uint32_t cap)
{
    uint32_t i = 1;
    while (i < cap) {
        i = i << 1;
    }
    return i;
}

__aicore__ inline void RowSumForLongColumnCount(LocalTensor<float> &dstUb, LocalTensor<float> srcUb, uint32_t dealRowCount,
                              uint32_t columnCount, uint32_t actualColumnCount)
{
    // sum by row, 按行求和
    // dstUb[i] = sum(srcUb[i, :])
    // src0Ub:[dealRowCount, columnCount] dstUb:[1, dealRowCount]
    // columnCount要求32元素对齐
    uint32_t newColumnCount = columnCount;
    uint32_t newActualColumnCount = actualColumnCount;
    if (columnCount >= REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) {
        uint32_t split = GetMinPowerTwo(actualColumnCount);
        split = split >> 1;

        // deal tail
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Add(srcUb[offset], srcUb[offset], srcUb[offset + split], actualColumnCount - split);
            offset += columnCount;
        }
        PipeBarrier<PIPE_V>();

        uint32_t validLen = split;
        while (validLen > BUFFER_SIZE_BYTE_1K) {
            uint32_t copyLen = validLen / 2;

            offset = 0;
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Add(srcUb[offset], srcUb[offset], srcUb[offset + copyLen], copyLen);
                offset += columnCount;
            }
            PipeBarrier<PIPE_V>();

            validLen = copyLen;
        }

        for (uint32_t i = 0; i < dealRowCount; i++) {
            DataCopy(srcUb[i * validLen], srcUb[i * columnCount], validLen);
            PipeBarrier<PIPE_V>();
        }

        newColumnCount = validLen;
        newActualColumnCount = validLen;
    }

    RowSum(dstUb, srcUb, dealRowCount, newColumnCount, newActualColumnCount);
}

__aicore__ inline void RowMax(LocalTensor<float> &dstUb, LocalTensor<float> &srcUb, uint32_t dealRowCount,
                              uint32_t columnCount, uint32_t actualColumnCount)
{
    // max by row, 按行求最大值
    // dstUb[i] = max(srcUb[i, :])
    // src0Ub:[dealRowCount, columnCount] dstUb:[1, dealRowCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockCount = actualColumnCount / dtypeMask;
    uint32_t remain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.src1RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    if (blockCount > 0 && remain > 0) {
        Max(srcUb, srcUb, srcUb[blockCount * dtypeMask], remain, dealRowCount, repeatParamsMax);
        PipeBarrier<PIPE_V>();
    }

    for (uint32_t loopCount = blockCount / 2; loopCount > 0; loopCount = blockCount / 2) {  // 2: 二分
        blockCount = (blockCount + 1) / 2;  // 2: 除2向上取整
        for (uint32_t j = 0; j < loopCount; j++) {
            Max(srcUb[j * dtypeMask], srcUb[j * dtypeMask], srcUb[(j + blockCount) * dtypeMask], dtypeMask,
                dealRowCount, repeatParamsMax);
        }
        PipeBarrier<PIPE_V>();
    }

    WholeReduceMax(dstUb, srcUb, (actualColumnCount < dtypeMask) ? actualColumnCount : dtypeMask, dealRowCount, 1, 1,
                   columnCount / FP32_BLOCK_ELEMENT_NUM, ReduceOrder::ORDER_ONLY_VALUE);
}

__aicore__ inline void RowMaxForLongColumnCount(LocalTensor<float> &dstUb, LocalTensor<float> srcUb, uint32_t dealRowCount,
                              uint32_t columnCount, uint32_t actualColumnCount)
{
    // max by row, 按行求最大值
    // dstUb[i] = max(srcUb[i, :])
    // src0Ub:[dealRowCount, columnCount] dstUb:[1, dealRowCount]
    uint32_t newColumnCount = columnCount;
    uint32_t newActualColumnCount = actualColumnCount;
    if (columnCount >= REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) {
        uint32_t split = GetMinPowerTwo(actualColumnCount);
        split = split >> 1;

        // deal tail
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Max(srcUb[offset], srcUb[offset], srcUb[offset + split], actualColumnCount - split);
            offset += columnCount;
        }
        PipeBarrier<PIPE_V>();

        uint32_t validLen = split;
        while (validLen > BUFFER_SIZE_BYTE_1K) {
            uint32_t copyLen = validLen / 2;

            offset = 0;
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Max(srcUb[offset], srcUb[offset], srcUb[offset + copyLen], copyLen);
                offset += columnCount;
            }
            PipeBarrier<PIPE_V>();

            validLen = copyLen;
        }

        for (uint32_t i = 0; i < dealRowCount; i++) {
            DataCopy(srcUb[i * validLen], srcUb[i * columnCount], validLen);
            PipeBarrier<PIPE_V>();
        }

        newColumnCount = validLen;
        newActualColumnCount = validLen;
    }

    RowMax(dstUb, srcUb, dealRowCount, newColumnCount, newActualColumnCount);
}

#ifdef ENABLE_DUMP_DATA
#define DO_DUMP_DATA(srcTensor, id, len) AscendC::DumpTensor(srcTensor, id, len)
#else
#define DO_DUMP_DATA(srcTensor, id, len)
#endif

template <TPosition pos, uint8_t BUFFER_NUM>
class BufferManager {
public:
    __aicore__ inline BufferManager() {}

    __aicore__ inline void Init() {
        for (uint8_t i = 0; i < BUFFER_NUM; i++) {
            if constexpr (pos == TPosition::A1 || pos == TPosition::B1) {
                srcEventIds_[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
                dstEventIds_[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
                SetFlag<HardEvent::MTE1_MTE2>(dstEventIds_[i]);
            } else if constexpr (pos == TPosition::A2 || pos == TPosition::B2) {
                srcEventIds_[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>();
                dstEventIds_[i] = GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>();
                SetFlag<HardEvent::M_MTE1>(dstEventIds_[i]);
            } else if constexpr (pos == TPosition::CO1) {
                srcEventIds_[i] = GetTPipePtr()->AllocEventID<HardEvent::M_FIX>();
                dstEventIds_[i] = GetTPipePtr()->AllocEventID<HardEvent::FIX_M>();
                SetFlag<HardEvent::FIX_M>(dstEventIds_[i]);
            }
        }
    }

    __aicore__ inline uint8_t GetBuffer() {
        if constexpr (pos == TPosition::A1 || pos == TPosition::B1) {
            WaitFlag<HardEvent::MTE1_MTE2>(dstEventIds_[bufferId_]);
        } else if constexpr (pos == TPosition::A2 || pos == TPosition::B2) {
            WaitFlag<HardEvent::M_MTE1>(dstEventIds_[bufferId_]);
        } else if constexpr (pos == TPosition::CO1) {
            WaitFlag<HardEvent::FIX_M>(dstEventIds_[bufferId_]);
        }
        return bufferId_;
    }

    __aicore__ inline void EnQue() {
        if constexpr (pos == TPosition::A1 || pos == TPosition::B1) {
            SetFlag<HardEvent::MTE2_MTE1>(srcEventIds_[bufferId_]);
        } else if constexpr (pos == TPosition::A2 || pos == TPosition::B2) {
            SetFlag<HardEvent::MTE1_M>(srcEventIds_[bufferId_]);
        } else if constexpr (pos == TPosition::CO1) {
            SetFlag<HardEvent::M_FIX>(srcEventIds_[bufferId_]);
        }
    }

    __aicore__ inline uint8_t DeQue() {
        if constexpr (pos == TPosition::A1 || pos == TPosition::B1) {
            WaitFlag<HardEvent::MTE2_MTE1>(srcEventIds_[bufferId_]);
        } else if constexpr (pos == TPosition::A2 || pos == TPosition::B2) {
            WaitFlag<HardEvent::MTE1_M>(srcEventIds_[bufferId_]);
        } else if constexpr (pos == TPosition::CO1) {
            WaitFlag<HardEvent::M_FIX>(srcEventIds_[bufferId_]);
        }
        return bufferId_;
    }

    __aicore__ inline void ReleaseBuffer() {
        if constexpr (pos == TPosition::A1 || pos == TPosition::B1) {
            SetFlag<HardEvent::MTE1_MTE2>(dstEventIds_[bufferId_]);
        } else if constexpr (pos == TPosition::A2 || pos == TPosition::B2) {
            SetFlag<HardEvent::M_MTE1>(dstEventIds_[bufferId_]);
        } else if constexpr (pos == TPosition::CO1) {
            SetFlag<HardEvent::FIX_M>(dstEventIds_[bufferId_]);
        }
        if constexpr (BUFFER_NUM == 2) {
            bufferId_ = bufferId_ ^ 1U;
        }
    }

    __aicore__ inline void Destory() {
        for (uint8_t i = 0; i < BUFFER_NUM; i++) {
            if constexpr (pos == TPosition::A1 || pos == TPosition::B1) {
                WaitFlag<HardEvent::MTE1_MTE2>(dstEventIds_[i]);
                GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(srcEventIds_[i]);
                GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(dstEventIds_[i]);
            } else if constexpr (pos == TPosition::A2 || pos == TPosition::B2) {
                WaitFlag<HardEvent::M_MTE1>(dstEventIds_[i]);
                GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_M>(srcEventIds_[i]);
                GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(dstEventIds_[i]);
            } else if constexpr (pos == TPosition::CO1) {
                WaitFlag<HardEvent::FIX_M>(dstEventIds_[i]);
                GetTPipePtr()->ReleaseEventID<HardEvent::M_FIX>(srcEventIds_[i]);
                GetTPipePtr()->ReleaseEventID<HardEvent::FIX_M>(dstEventIds_[i]);
            }
        }
    }

private:
    TEventID srcEventIds_[BUFFER_NUM];
    TEventID dstEventIds_[BUFFER_NUM];
    uint8_t bufferId_ = 0;
};

#define PRE_LOAD_NUM_MLA 2
#define IFA_PRELOAD_TASK_CACHE_SIZE 3

struct ExtraInfoMla {
    uint32_t loop;
    uint32_t bIdx;
    uint32_t gIdx;
    uint32_t s1Idx;
    uint32_t s2Idx;
    uint32_t bn2IdxInCurCore;
    uint32_t curSInnerLoopTimes;
    uint64_t tndBIdxOffset;
    uint64_t tensorAOffset;
    uint64_t tensorBOffset;
    uint64_t tensorARopeOffset;
    uint64_t tensorBRopeOffset;
    uint64_t attenOutOffset;
    uint64_t attenMaskOffset;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;
    bool isFirstSInnerLoop;
    bool isChangeBatch;
    uint32_t s2BatchOffset;
    uint32_t gSize;
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t mSize;
    uint32_t mSizeV;
    uint32_t mSizeVStart;
    uint32_t tndIsS2SplitCore;
    uint32_t tndCoreStartKVSplitPos;
    bool isBmm2Output;
    bool isValid = false;

    uint64_t antiqKeyParamOffset;
    uint64_t antiqKeyRopeParamOffset;
    uint64_t antiqValueParamOffset;
    static constexpr uint32_t n2Idx = 0;
    uint64_t actS1Size = 1;
    bool isLastTask = false;
};

#define QUANT_MM2_FP16
#define MM1_RES_ADD
#define MM1_DATO
#define SOFTMAX_NEW

#endif // IFA_PUBLIC_DEFINE_H
