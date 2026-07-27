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
 * \file matmul.h
 * \brief
 */
#ifndef MATMUL_H
#define MATMUL_H
#include "buffers_policy.h"
using namespace AscendC;

namespace fa_base_matmul {

constexpr uint32_t UNITFLAG_DISABLE = 0;
constexpr uint32_t UNITFLAG_ENABLE = 2;
constexpr uint32_t UNITFLAG_EN_OUTER_LAST = 3;
static constexpr uint32_t FP16_ONE_FRACTAL_ELEMENT = 16; // 一个分形512B,16*16个fp16
static constexpr uint32_t INT4_ONE_FRACTAL_ELEMENT = 64; // 一个分形512B,16*64个fp16
static constexpr uint32_t ONE_FRACTAL_H_ELEMENT = 16; // 一个分形512B,height方向为16个element
static constexpr uint32_t ONE_FRACTAL_W_BYTE = 32; // 一个分形512B,weight方向为32B
static constexpr uint32_t LOAD3D_L1W_SIZE = 16;
static constexpr uint32_t MMAD_MN_SIZE_10 = 10;
static constexpr uint8_t LOAD3D_STRIDE_W = 1;
static constexpr uint8_t LOAD3D_STRIDE_H = 1;
static constexpr uint8_t LOAD3D_FILTER_W = 1;
static constexpr uint8_t LOAD3D_FILTER_H = 1;
static constexpr uint8_t LOAD3D_DILA_FILTER_W = 1;
static constexpr uint8_t LOAD3D_DILA_FILTER_H = 1;
static constexpr uint32_t K_STEP_ALIGN_BASE = 2;
static constexpr uint32_t M_STEP_ALIGN_BASE = 2;
static constexpr uint32_t MX_FP4_PTG_PCG_SCALE_PARAM = 32; // mxfp4场景随路伪量化中32个元素共用一个scale
static constexpr uint32_t HI_FP4_PTG_PCG_SCALE_PARAM = 64; // hifp4场景随路伪量化中64个元素共用一个scale

struct MMParam {
    uint32_t singleM;
    uint32_t singleN;
    uint32_t singleK;
    bool isLeftTranspose;
    bool isRightTranspose;
    bool cmatrixInitVal = true;
    bool isOutKFisrt = true; // 默认值为true， true：在L1切K轴的场景中，表示首轮K
    uint32_t unitFlag = 0;  // 0：disable: 不配置unitFlag
                            // 2：enable: 行为在切K接口中（MatmulK），会将mmadParams.unitFlag设置为 0b10
                            // 3：enable: 行为在切K接口中（MatmulK），在k的最后一轮循环，会将mmadParams.unitFlag设置为 0b11
                            // 外部使用时，在外层k循环的最后一轮将该参数配置为3
    uint32_t realM = 0; // bmm2以s1realsize为M轴，不赋值时不影响现有代码逻辑
};

__aicore__ inline MMParam MakeMMParam(uint32_t singleM, uint32_t singleN, uint32_t singleK, bool isLeftTranspose,
                                      bool isRightTranspose, bool cmatrixInitVal = true, bool isOutKFisrt = true,
                                      uint32_t unitFlag = 0, uint32_t realM = 0)
{
    return {.singleM = singleM,
            .singleN = singleN,
            .singleK = singleK,
            .isLeftTranspose = isLeftTranspose,
            .isRightTranspose = isRightTranspose,
            .cmatrixInitVal = cmatrixInitVal,
            .isOutKFisrt = isOutKFisrt,
            .unitFlag = unitFlag,
            .realM = realM
            };
}

enum class ABLayout {
    MK = 0,
    KM = 1,
    KN = 2,
    NK = 3,
};

template <typename T>
__aicore__ inline T AlignUp(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

#if ((__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102))
template <typename T>
__aicore__ inline uint32_t GetBlockNum(uint32_t size) {
    if constexpr (IsSameType<T, float>::value) {
        return ((size + 7) >> 3 << 3) >> 3;
    } else if constexpr ((IsSameType<T, fp8_e5m2_t>::value ||
                          IsSameType<T, fp8_e4m3fn_t>::value ||
                          IsSameType<T, hifloat8_t>::value ||
                          IsSameType<T, int8_t>::value)) {
        return ((size + 31) >> 5 << 5) >> 5;
    } else {
        return ((size + 15) >> 4 << 4) >> 4;
    }
}
// L1->L0A + 切k/切M/全载
template <typename T>
__aicore__ inline void LoadDataToL0A(LocalTensor<T>& aL0Tensor, const LocalTensor<T>& aL1Tensor,
                                    const MMParam& mmParam, uint64_t L1Aoffset, uint32_t kSplitSize, uint32_t mSplitSize)
{
    LoadData2DParamsV2 loadData2DParamsA; // 基础API LoadData的参数结构体
    loadData2DParamsA.mStartPosition = 0; // 以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16 element
    loadData2DParamsA.kStartPosition = 0; // 以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B
    loadData2DParamsA.ifTranspose = mmParam.isLeftTranspose; // 是否启用转置功能，对每个分型矩阵进行转置
    if (loadData2DParamsA.ifTranspose) {
        loadData2DParamsA.mStep = ((kSplitSize + 15) >> 4 << 4) >> 4; // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16 element,取值范围：mStep属于[0,255]
        if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, hifloat8_t>::value || IsSameType<T, int8_t>::value) {
            loadData2DParamsA.mStep = (loadData2DParamsA.mStep + 1) >> 1 << 1;
        }
        loadData2DParamsA.kStep = GetBlockNum<T>(mSplitSize); // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
    } else {
        loadData2DParamsA.mStep = ((mSplitSize + 15) >> 4 << 4) >> 4; // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16 element,取值范围：mStep属于[0,255]
        loadData2DParamsA.kStep = GetBlockNum<T>(kSplitSize); // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
    }
    if constexpr (IsSameType<T, float>::value) {
        if (loadData2DParamsA.ifTranspose) {
            loadData2DParamsA.kStep = CeilAlign(loadData2DParamsA.kStep, K_STEP_ALIGN_BASE);
        }
    }
    if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, hifloat8_t>::value || IsSameType<T, int8_t>::value) {
        // 配合ub->L1使用 256 * 32 / 256
        // 64搬运对齐
        loadData2DParamsA.srcStride = loadData2DParamsA.ifTranspose ? ((kSplitSize + 63) >> 6 << 6) >> 4 : ((mSplitSize + 31) >> 5 << 5) >> 4; // 以M*K矩阵为例，源矩阵K方向前一个分形起始地址与后一个分形起始地址的间隔，单位：512B
    } else {
        loadData2DParamsA.srcStride = loadData2DParamsA.ifTranspose ? ((mmParam.singleK + 15) >> 4 << 4) >> 4 : loadData2DParamsA.mStep;
    }
    if (mmParam.realM != 0) {
        loadData2DParamsA.mStep = ((mmParam.realM + 15) >> 4 << 4) >> 4;
    }
    loadData2DParamsA.dstStride = loadData2DParamsA.ifTranspose ? (mSplitSize + 15) >> 4 : loadData2DParamsA.mStep;
    if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, hifloat8_t>::value || IsSameType<T, int8_t>::value) {
        if (loadData2DParamsA.ifTranspose && (loadData2DParamsA.dstStride & 1)) {
            uint32_t l0bLoop = (loadData2DParamsA.mStep + 1) >> 1;
            loadData2DParamsA.mStep = M_STEP_ALIGN_BASE;
            uint64_t dstOffset = 0;
            uint64_t dstAddrStride = (mSplitSize + 15) / 16 * 16 * 32;
            uint16_t oriMStep = loadData2DParamsA.mStartPosition;
            for (uint32_t idx = 0; idx < l0bLoop; ++idx) {
                loadData2DParamsA.mStartPosition = oriMStep + M_STEP_ALIGN_BASE * idx;
                LoadData(aL0Tensor[dstOffset], aL1Tensor[L1Aoffset], loadData2DParamsA);
                dstOffset += dstAddrStride;
            }
        } else {
            LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData2DParamsA);
        }
    } else {
        LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData2DParamsA);
    }
}

// L1->L0A + 切k/切M/全载
template <typename T, typename U = T>
__aicore__ inline void LoadDataToL0AMx(LocalTensor<U>& aL0Tensor, const LocalTensor<T>& aL1Tensor, const LocalTensor<fp8_e8m0_t>& aScaleL1Tensor,
                                    const MMParam& mmParam, uint64_t L1Aoffset, uint32_t kSplitSize, uint32_t mSplitSize)
{
    LoadData2DParamsV2 loadData2DParamsA; // 基础API LoadData的参数结构体
    loadData2DParamsA.mStartPosition = 0; // 以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16 element
    loadData2DParamsA.kStartPosition = 0; // 以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B
    loadData2DParamsA.ifTranspose = mmParam.isLeftTranspose; // 是否启用转置功能，对每个分型矩阵进行转置
    if (loadData2DParamsA.ifTranspose) {
        loadData2DParamsA.mStep = ((kSplitSize + 15) >> 4 << 4) >> 4; // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16 element,取值范围：mStep属于[0,255]
        if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, hifloat8_t>::value) {
            loadData2DParamsA.mStep = (loadData2DParamsA.mStep + 1) >> 1 << 1;
        }
        loadData2DParamsA.kStep = GetBlockNum<T>(mSplitSize); // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
    } else {
        loadData2DParamsA.mStep = ((mSplitSize + 15) >> 4 << 4) >> 4; // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16 element,取值范围：mStep属于[0,255]
        loadData2DParamsA.kStep = GetBlockNum<T>(kSplitSize); // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
    }
    if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, hifloat8_t>::value) {
        // 配合ub->L1使用 256 * 32 / 256
        // 64搬运对齐
        loadData2DParamsA.srcStride = loadData2DParamsA.ifTranspose ? 256 >> 4 : ((mSplitSize + 31) >> 5 << 5) >> 4; // 以M*K矩阵为例，源矩阵K方向前一个分形起始地址与后一个分形起始地址的间隔，单位：512B
    } else {
        loadData2DParamsA.srcStride = loadData2DParamsA.ifTranspose ? ((mmParam.singleK + 15) >> 4 << 4) >> 4 : loadData2DParamsA.mStep;
    }
    LoadData2DMxParams loadData2DMxParamsA;
    loadData2DMxParamsA.xStartPosition = 0;
    loadData2DMxParamsA.yStartPosition = 0;
    loadData2DMxParamsA.xStep = ((mSplitSize + 15) >> 4 << 4) >> 4;
    loadData2DMxParamsA.yStep = (kSplitSize + 63) >> 5 >> 1;
    loadData2DMxParamsA.srcStride = loadData2DMxParamsA.yStep;
    loadData2DMxParamsA.dstStride = loadData2DMxParamsA.yStep;
    if (mmParam.realM != 0) {
        loadData2DParamsA.mStep = ((mmParam.realM + 15) >> 4 << 4) >> 4;
        loadData2DMxParamsA.xStep = ((mmParam.realM + 15) >> 4 << 4) >> 4;
    }
    loadData2DParamsA.dstStride = loadData2DParamsA.ifTranspose ? (mSplitSize + 15) >> 4 : loadData2DParamsA.mStep;
    if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, hifloat8_t>::value) {
        if (loadData2DParamsA.ifTranspose && (loadData2DParamsA.dstStride & 1)) {
            uint32_t l0bLoop = (loadData2DParamsA.mStep + 1) >> 1;
            loadData2DParamsA.mStep = M_STEP_ALIGN_BASE;
            loadData2DMxParamsA.xStep = loadData2DParamsA.mStep ;
            uint64_t dstOffset = 0;
            uint64_t dstAddrStride = (mSplitSize + 15) / 16 * 16 * 32;
            uint16_t oriMStep = loadData2DParamsA.mStartPosition;
            uint16_t oriMScaleStep = loadData2DMxParamsA.xStartPosition;
            for (uint32_t idx = 0; idx < l0bLoop; ++idx) {
                loadData2DParamsA.mStartPosition = oriMStep + M_STEP_ALIGN_BASE * idx;
                loadData2DMxParamsA.xStartPosition = oriMScaleStep + M_STEP_ALIGN_BASE * idx;
                LoadData(aL0Tensor[dstOffset], aL1Tensor[L1Aoffset], aScaleL1Tensor[L1Aoffset >> 5], loadData2DParamsA,
                         loadData2DMxParamsA);
                dstOffset += dstAddrStride;
            }
        } else {
            LoadData(aL0Tensor, aL1Tensor[L1Aoffset], aScaleL1Tensor[L1Aoffset >> 5], loadData2DParamsA,
                     loadData2DMxParamsA);
        }
    } else {
        LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData2DParamsA);
    }
}

// L1->L0B + 切k/切M/全载
template <typename T>
__aicore__ inline void LoadDataToL0B(LocalTensor<T>& bL0Tensor, const LocalTensor<T>& bL1Tensor,
                                    const MMParam& mmParam, uint64_t L1Boffset, uint32_t kSplitSize, uint32_t nSplitSize, int nLoops = 1)
{
    LoadData2DParamsV2 loadData2DParamsB; // 基础API LoadData的参数结构体
    loadData2DParamsB.mStartPosition = 0; // 以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16 element
    loadData2DParamsB.kStartPosition = 0; // 以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B
    loadData2DParamsB.ifTranspose = !mmParam.isRightTranspose; // 是否启用转置功能，对每个分型矩阵进行转置
    if (loadData2DParamsB.ifTranspose) {
        loadData2DParamsB.mStep = ((kSplitSize + 15) >> 4 << 4) >> 4; // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16 element,取值范围：mStep属于[0,255]
        if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, hifloat8_t>::value || IsSameType<T, int8_t>::value) {
            loadData2DParamsB.mStep = (loadData2DParamsB.mStep + 1) >> 1 << 1;
        }
        loadData2DParamsB.kStep = GetBlockNum<T>(nSplitSize); // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
    } else {
        loadData2DParamsB.mStep = ((nSplitSize + 15) >> 4 << 4) >> 4; // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16 element,取值范围：mStep属于[0,255]
        loadData2DParamsB.kStep = GetBlockNum<T>(kSplitSize); // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
    }
    if constexpr (IsSameType<T, float>::value) {
        if (loadData2DParamsB.ifTranspose) {
            loadData2DParamsB.kStep = CeilAlign(loadData2DParamsB.kStep, K_STEP_ALIGN_BASE);
        }
    }
    if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, hifloat8_t>::value || IsSameType<T, int8_t>::value) {
        if (loadData2DParamsB.ifTranspose) {
            loadData2DParamsB.srcStride = ((kSplitSize + 31) >> 5 << 5) >> 4;
        } else {
            loadData2DParamsB.srcStride = ((nSplitSize + 31) >> 5 << 5) >> 4;
        }
    } else {
        loadData2DParamsB.srcStride = loadData2DParamsB.ifTranspose ? (((mmParam.singleK + 15) >> 4 << 4) >> 4) : (((mmParam.singleN + 15 ) >> 4 << 4) >> 4); // 以M*K矩阵为例，源矩阵K方向前一个分形起始地址与后一个分形起始地址的间隔，单位：512B
    }
    loadData2DParamsB.dstStride = loadData2DParamsB.ifTranspose ? (nSplitSize + 15) >> 4 : loadData2DParamsB.mStep; // 以M*K矩阵为例，目标矩阵K方向前一个分形起始地址与后一个分形起始地址的间隔，单位：512B
    if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, hifloat8_t>::value || IsSameType<T, int8_t>::value) {
        if (loadData2DParamsB.ifTranspose && (loadData2DParamsB.dstStride & 1)) {
            uint32_t l0bLoop = (loadData2DParamsB.mStep + 1) >> 1;
            loadData2DParamsB.mStep = M_STEP_ALIGN_BASE;
            uint64_t dstOffset = 0;
            uint64_t dstAddrStride = (nSplitSize + 15) / 16 * 16 * 32;
            uint16_t oriMStep = loadData2DParamsB.mStartPosition;
            for (uint32_t idx = 0; idx < l0bLoop; ++idx) {
                loadData2DParamsB.mStartPosition = oriMStep + M_STEP_ALIGN_BASE * idx;
                LoadData(bL0Tensor[dstOffset], bL1Tensor[L1Boffset], loadData2DParamsB);
                dstOffset += dstAddrStride;
            }
        } else {
            LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData2DParamsB);
        }
    } else {
        LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData2DParamsB);
    }
}

template <typename T, typename U = T, typename Scale_T>
__aicore__ inline void LoadDataToL0BMx(LocalTensor<U>& bL0Tensor, const LocalTensor<T>& bL1Tensor, const LocalTensor<Scale_T>& bScaleL1Tensor,
                                    const MMParam& mmParam, uint64_t L1Boffset, uint32_t kSplitSize, uint32_t nSplitSize, int nLoops = 1)
{
    LoadData2DParamsV2 loadData2DParamsB; // 基础API LoadData的参数结构体
    loadData2DParamsB.mStartPosition = 0; // 以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16 element
    loadData2DParamsB.kStartPosition = 0; // 以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B
    loadData2DParamsB.ifTranspose = !mmParam.isRightTranspose; // 是否启用转置功能，对每个分型矩阵进行转置
    if (loadData2DParamsB.ifTranspose) {
        if constexpr (IsSameType<T, fp4x2_e2m1_t>::value || IsSameType<T, hifloat4x2_t>::value) {
            loadData2DParamsB.mStep = ((mmParam.singleK + 63) >> 6 << 6) / 16;
            loadData2DParamsB.kStep = kSplitSize / 64;
            loadData2DParamsB.srcStride = ((mmParam.singleK + 15) >> 4 << 4) / 16;
            loadData2DParamsB.dstStride = kSplitSize / 16;
        } else if constexpr (IsSameType<T, fp8_e4m3fn_t>::value) {
            loadData2DParamsB.mStep = ((kSplitSize + 15) >> 4 << 4) >> 4; // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16 element,取值范围：mStep属于[0,255]
            loadData2DParamsB.kStep = GetBlockNum<T>(nSplitSize); // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
            loadData2DParamsB.srcStride = ((kSplitSize + 31) >> 5 << 5) >> 4;
            loadData2DParamsB.dstStride = (nSplitSize + 15) >> 4;
        }
    } else {
        if constexpr (IsSameType<T, fp4x2_e2m1_t>::value || IsSameType<T, hifloat4x2_t>::value) {
            loadData2DParamsB.mStep = ((mmParam.singleN + 15) >> 4 << 4) / 16;
            loadData2DParamsB.kStep = kSplitSize / 64;
            loadData2DParamsB.srcStride = loadData2DParamsB.mStep;
            loadData2DParamsB.dstStride = loadData2DParamsB.mStep;
        } else if constexpr (IsSameType<T, fp8_e4m3fn_t>::value) {
            loadData2DParamsB.mStep = ((nSplitSize + 15) >> 4 << 4) >> 4; // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16 element,取值范围：mStep属于[0,255]
            loadData2DParamsB.kStep = GetBlockNum<T>(kSplitSize); // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
            loadData2DParamsB.srcStride = ((nSplitSize + 31) >> 5 << 5) >> 4;
            loadData2DParamsB.dstStride = loadData2DParamsB.mStep;
        }
    }

    LoadData2DMxParams loadData2DMxParamsB;
    if constexpr (IsSameType<T, fp8_e4m3fn_t>::value) {
        loadData2DMxParamsB.xStartPosition = 0;
        loadData2DMxParamsB.yStartPosition = 0;
        loadData2DMxParamsB.xStep = ((nSplitSize + 15) >> 4 << 4) >> 4;
        loadData2DMxParamsB.yStep = (kSplitSize + 63)  >> 5 >> 1;
        loadData2DMxParamsB.srcStride = loadData2DMxParamsB.yStep;
        loadData2DMxParamsB.dstStride = loadData2DMxParamsB.yStep;
        if (loadData2DParamsB.ifTranspose && (loadData2DParamsB.dstStride & 1)) {
            uint32_t l0bLoop = (loadData2DParamsB.mStep + 1) >> 1;
            loadData2DParamsB.mStep = M_STEP_ALIGN_BASE;
            loadData2DMxParamsB.xStep = loadData2DParamsB.mStep;
            uint64_t dstOffset = 0;
            uint64_t dstAddrStride = (nSplitSize + 15) / 16 * 16 * 32;
            uint16_t oriMStep = loadData2DParamsB.mStartPosition;
            uint16_t oriMScaleStep = loadData2DMxParamsB.xStartPosition;
            for (uint32_t idx = 0; idx < l0bLoop; ++idx) {
                loadData2DParamsB.mStartPosition = oriMStep + M_STEP_ALIGN_BASE * idx;
                loadData2DMxParamsB.xStartPosition = oriMScaleStep + M_STEP_ALIGN_BASE * idx;
                LoadData(bL0Tensor[dstOffset], bL1Tensor[L1Boffset], bScaleL1Tensor[L1Boffset >> 5], loadData2DParamsB,
                         loadData2DMxParamsB);
                dstOffset += dstAddrStride;
            }
        } else {
            LoadData(bL0Tensor, bL1Tensor[L1Boffset], bScaleL1Tensor[L1Boffset >> 5], loadData2DParamsB,
                     loadData2DMxParamsB);
        }
    } else if constexpr (IsSameType<T, fp4x2_e2m1_t>::value) {
        if (loadData2DParamsB.ifTranspose) {
            loadData2DMxParamsB.xStartPosition= 0; // 矩阵X轴方向的起始位置，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。
            loadData2DMxParamsB.yStartPosition = mmParam.singleN / MX_FP4_PTG_PCG_SCALE_PARAM / 2; // 源矩阵Y轴方向的起始位置，即K维度方向，单位为32B。
            loadData2DMxParamsB.xStep = (kSplitSize + 15) / 16; // 源矩阵X轴方向搬运长度，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。取值范围：xStep∈[0, 255]。
            loadData2DMxParamsB.yStep = (mmParam.singleK + 63) / MX_FP4_PTG_PCG_SCALE_PARAM / 2; // 源矩阵Y轴方向搬运长度，即K维度方向，单位为32B。取值范围：yStep∈[0, 255]。
            // TODO: mmParam.singleM is ???
            loadData2DMxParamsB.srcStride = mmParam.singleM / MX_FP4_PTG_PCG_SCALE_PARAM / 2; // 源矩阵X方向前一个分形起始地址与后一个分形起始地址的间隔，单位为32B。
            loadData2DMxParamsB.dstStride = loadData2DMxParamsB.yStep; // 目标矩阵X方向前一个分形起始地址与后一个分形起始地址的间隔，单位为32B。
            LoadData(bL0Tensor, bL1Tensor, bScaleL1Tensor, loadData2DParamsB,loadData2DMxParamsB);
        } else {
            loadData2DMxParamsB.xStartPosition = mmParam.singleK / 16; // 矩阵X轴方向的起始位置，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。
            loadData2DMxParamsB.yStartPosition = 0;
            loadData2DMxParamsB.xStep = (mmParam.singleN + 15) / 16; // 源矩阵X轴方向搬运长度，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。取值范围：xStep∈[0, 255]。
            loadData2DMxParamsB.yStep = (kSplitSize + 63) / MX_FP4_PTG_PCG_SCALE_PARAM / 2; // 源矩阵Y轴方向搬运长度，即K维度方向，单位为32B。取值范围：yStep∈[0, 255]。
            loadData2DMxParamsB.srcStride = loadData2DMxParamsB.yStep;
            loadData2DMxParamsB.dstStride = loadData2DMxParamsB.yStep; // 目标矩阵X方向前一个分形起始地址与后一个分形起始地址的间隔，单位为32B。
            LoadData(bL0Tensor, bL1Tensor,bScaleL1Tensor, loadData2DParamsB,loadData2DMxParamsB);
        }
    } else if constexpr (IsSameType<T, hifloat4x2_t>::value) {
        if (loadData2DParamsB.ifTranspose) {
            loadData2DMxParamsB.xStartPosition= 0; // 矩阵X轴方向的起始位置，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。
            loadData2DMxParamsB.yStartPosition = mmParam.singleN * 2 / HI_FP4_PTG_PCG_SCALE_PARAM; // 源矩阵Y轴方向的起始位置，即K维度方向，单位为32B。
            loadData2DMxParamsB.xStep = (kSplitSize + 15) / 16; // 源矩阵X轴方向搬运长度，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。取值范围：xStep∈[0, 255]。
            loadData2DMxParamsB.yStep = (mmParam.singleK + 31) * 2 / HI_FP4_PTG_PCG_SCALE_PARAM; // 1个fp32伪装为2个bf16，故最小分型数量翻倍。源矩阵Y轴方向搬运长度，即K维度方向，单位为32B。取值范围：yStep∈[0, 255]。
            // TODO: mmParam.singleM is ???
            loadData2DMxParamsB.srcStride = mmParam.singleM * 2 / HI_FP4_PTG_PCG_SCALE_PARAM; // 源矩阵X方向前一个分形起始地址与后一个分形起始地址的间隔，单位为32B。
            loadData2DMxParamsB.dstStride = loadData2DMxParamsB.yStep; // 目标矩阵X方向前一个分形起始地址与后一个分形起始地址的间隔，单位为32B。
            LoadData(bL0Tensor, bL1Tensor, bScaleL1Tensor, loadData2DParamsB,loadData2DMxParamsB);
        } else {
            loadData2DMxParamsB.xStartPosition= mmParam.singleK / 16; // 矩阵X轴方向的起始位置，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。
            loadData2DMxParamsB.yStartPosition = 0; // 源矩阵Y轴方向的起始位置，即K维度方向，单位为32B。
            loadData2DMxParamsB.xStep = (mmParam.singleN + 15) / 16; // 源矩阵X轴方向搬运长度，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。取值范围：xStep∈[0, 255]。
            loadData2DMxParamsB.yStep = (kSplitSize + 31) * 2 / HI_FP4_PTG_PCG_SCALE_PARAM; // 1个fp32伪装为2个bf16，故最小分型数量翻倍。源矩阵Y轴方向搬运长度，即K维度方向，单位为32B。取值范围：yStep∈[0, 255]。
            loadData2DMxParamsB.srcStride = loadData2DMxParamsB.yStep; // 源矩阵X方向前一个分形起始地址与后一个分形起始地址的间隔，单位为32B。
            loadData2DMxParamsB.dstStride = loadData2DMxParamsB.yStep; // 目标矩阵X方向前一个分形起始地址与后一个分形起始地址的间隔，单位为32B。
            LoadData(bL0Tensor, bL1Tensor,bScaleL1Tensor, loadData2DParamsB,loadData2DMxParamsB);
        }
    }
}

// 全载
// 外部L1切入K时，需要传入cmatrixInitVal的标记
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL, typename L0AType, typename L0BType, typename AScaleType = float, typename BScaleType = float, typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MatmulFullMX(const LocalTensor<A> &aL1Tensor,
                              const LocalTensor<B> &bL1Tensor,
                              const LocalTensor<AScaleType> &aScaleL1Tensor,
                              const LocalTensor<BScaleType> &bScaleL1Tensor,
                              L0AType &aL0BuffsDb,
                              L0BType &bL0BuffsDb,
                              const LocalTensor<C> &cL0Tensor,
                              const MMParam &param)
{
    Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
    l0aBuffer.Wait<HardEvent::M_MTE1>();
    LocalTensor<L0ADType> L0ATensor = l0aBuffer.GetTensor<L0ADType>();
    if constexpr (IsSameType<L0ADType, mx_fp8_e4m3_t>::value) {
        LoadDataToL0AMx<A, L0ADType>(L0ATensor, aL1Tensor, aScaleL1Tensor, param, 0, param.singleK, param.singleM);
    } else if constexpr (IsSameType<L0ADType, fp8_e4m3fn_t>::value) {
        LoadDataToL0A(L0ATensor, aL1Tensor, param, 0, param.singleK, param.singleM);
    }
    l0aBuffer.Set<HardEvent::MTE1_M>();

    Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
    l0bBuffer.Wait<HardEvent::M_MTE1>();
    LocalTensor<L0BDType> L0BTensor = l0bBuffer.GetTensor<L0BDType>();
    if constexpr (IsSameType<L0BDType, mx_fp8_e4m3_t>::value) {
        LoadDataToL0BMx<B, L0BDType>(L0BTensor, bL1Tensor, bScaleL1Tensor, param, 0, param.singleK, param.singleN);
    } else if constexpr (IsSameType<L0BDType, fp8_e4m3fn_t>::value) {
        LoadDataToL0B(L0BTensor, bL1Tensor, param, 0, param.singleK, param.singleN);
    }
    l0bBuffer.Set<HardEvent::MTE1_M>();

    l0aBuffer.Wait<HardEvent::MTE1_M>();
    l0bBuffer.Wait<HardEvent::MTE1_M>();

    MmadParams mmadParams;
    mmadParams.m = param.singleM;
    if (param.realM != 0) {
        mmadParams.m = param.realM;
    }
    mmadParams.n = param.singleN;
    mmadParams.k = param.singleK;
    mmadParams.cmatrixInitVal = param.isOutKFisrt;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = param.unitFlag;
    if (mmadParams.m == 1) {
        mmadParams.m = 16;
    }

    Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

    l0aBuffer.Set<HardEvent::M_MTE1>();
    l0bBuffer.Set<HardEvent::M_MTE1>();
}

// 切K mx
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL, typename L0AType, typename L0BType, typename AScaleType = float, typename BScaleType = float, typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MatmulKMx(const LocalTensor<A> &aL1Tensor,
                              const LocalTensor<B> &bL1Tensor,
                              const LocalTensor<AScaleType> &aScaleL1Tensor,
                              const LocalTensor<BScaleType> &bScaleL1Tensor,
                              L0AType &aL0BuffsDb,
                              L0BType &bL0BuffsDb,
                              const LocalTensor<C> &cL0Tensor,
                              const MMParam &param)
{
    uint32_t kLoops = (param.singleK + baseK - 1) / baseK;
    uint32_t tailSize = param.singleK % baseK;
    uint32_t tailK = tailSize ? tailSize : baseK;
    uint64_t L1Aoffset = param.isLeftTranspose ? baseK << 4 : ((param.singleM + 15) >> 4 << 4) * baseK;
    uint64_t L1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 4;
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<A, fp8_e5m2_t>::value || IsSameType<A, fp8_e4m3fn_t>::value || IsSameType<A, hifloat8_t>::value) {
        L1Aoffset = ((param.singleM + 31) >> 5 << 5) * baseK;
        L1Boffset = ((param.singleN + 31) >> 5 << 5) * baseK;
    }
    if constexpr (IsSameType<A, float>::value) {
        L1Aoffset = param.isLeftTranspose ? baseK << 3 : ((param.singleM + 15) >> 4 << 4) * baseK;
        L1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 3;
    }
#endif
    for (uint32_t k = 0; k < kLoops; k++) {
        uint32_t tileK = (k == (kLoops - 1)) ? tailK : baseK;
        Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
        l0aBuffer.Wait<HardEvent::M_MTE1>(); // mte1等Matmul：上一轮matmul完成后才能搬运新数据到L0A
        LocalTensor<L0ADType> L0ATensor = l0aBuffer.GetTensor<L0ADType>();
        if constexpr (IsSameType<L0ADType, mx_fp8_e4m3_t>::value) {
            LoadDataToL0AMx<A, L0ADType>(L0ATensor, aL1Tensor, aScaleL1Tensor, param, k * L1Aoffset, tileK, param.singleM); // s2,
        } else if constexpr (IsSameType<L0ADType, fp8_e4m3fn_t>::value) {
            LoadDataToL0A(L0ATensor, aL1Tensor, param, k * L1Aoffset, tileK, param.singleM);
        }
        l0aBuffer.Set<HardEvent::MTE1_M>(); // mte1搬运完后，通知可以开始matmul

        Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
        l0bBuffer.Wait<HardEvent::M_MTE1>(); // mte1等Matmul：上一轮matmul完成后才能搬运新数据到L0B
        LocalTensor<L0BDType> L0BTensor = l0bBuffer.GetTensor<L0BDType>();
        uint64_t loopNum = param.isRightTranspose ? 1 : kLoops;
        if constexpr (IsSameType<L0BDType, mx_fp8_e4m3_t>::value) {
            LoadDataToL0BMx<B, L0BDType>(L0BTensor, bL1Tensor, bScaleL1Tensor, param, k * L1Boffset, tileK, param.singleN, loopNum); // tileK.D
        } else if constexpr (IsSameType<L0BDType, fp8_e4m3fn_t>::value) {
            LoadDataToL0B(L0BTensor, bL1Tensor, param, k * L1Boffset, tileK, param.singleN, loopNum);
        }
        l0bBuffer.Set<HardEvent::MTE1_M>(); // mte1搬运完后，通知可以开始matmul

        l0aBuffer.Wait<HardEvent::MTE1_M>(); // matmul等mte1：L0A数据搬运完成后才能开始matmul
        l0bBuffer.Wait<HardEvent::MTE1_M>(); // matmul等mte1：L0B数据搬运完成后才能开始matmul

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) {
            mmadParams.m = param.realM;
        }
        mmadParams.n = param.singleN;
        mmadParams.k = tileK;
        if (mmadParams.m == 1) {  // m等于1或默认开GEMV模式，文档上没有写怎么关闭GEMV，所以规避当做矩阵运算
            mmadParams.m = 16;
        }
        mmadParams.cmatrixInitVal = param.isOutKFisrt && (k == 0);
        mmadParams.cmatrixSource = false;
        if (param.unitFlag != 0) {
            mmadParams.unitFlag = (param.unitFlag == UNITFLAG_EN_OUTER_LAST) && (k == kLoops - 1) ?
                                  UNITFLAG_EN_OUTER_LAST : UNITFLAG_ENABLE;
        }

        Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

        l0aBuffer.Set<HardEvent::M_MTE1>(); // matmul完成后，通知mte1可以开始搬运新数据到L0A
        l0bBuffer.Set<HardEvent::M_MTE1>(); // matmul完成后，通知mte1可以开始搬运新数据到L0B
    }
}

template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL, typename L0AType, typename L0BType, typename AScaleType = float, typename BScaleType = float, typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MatmulMMx(const LocalTensor<A> &aL1Tensor,
                              const LocalTensor<B> &bL1Tensor,
                              const LocalTensor<AScaleType> &aScaleL1Tensor,
                              const LocalTensor<BScaleType> &bScaleL1Tensor,
                              L0AType &aL0BuffsDb,
                              L0BType &bL0BuffsDb,
                              const LocalTensor<C> &cL0Tensor,
                              const MMParam &param)
{
    uint32_t mLoops = (param.singleM + baseM - 1) / baseM;
    uint32_t tailSize = param.singleM % baseM;
    uint32_t tailM = tailSize ? tailSize : baseM;
    uint64_t L1Aoffset = param.isLeftTranspose ? baseM << 4 : ((param.singleK + 15) >> 4 << 4) * baseM; // 要对齐
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<A, fp8_e5m2_t>::value || IsSameType<A, fp8_e4m3fn_t>::value || IsSameType<A, hifloat8_t>::value) {
        L1Aoffset = ((param.singleK + 31) >> 5 << 5) * baseM;
    }
#endif

    uint64_t L0Coffset = ((param.singleN + 31) >> 5 << 5) * baseM;
    Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
    l0bBuffer.Wait<HardEvent::M_MTE1>(); // mte1等Matmul：上一轮matmul完成后才能搬运新数据到L0A
    LocalTensor<L0BDType> L0BTensor = l0bBuffer.GetTensor<L0BDType>();
    if constexpr (IsSameType<L0BDType, mx_fp8_e4m3_t>::value) {
        LoadDataToL0BMx<B, L0BDType>(L0BTensor, bL1Tensor, bScaleL1Tensor, param, 0, param.singleK, param.singleN);
    } else if constexpr (IsSameType<L0BDType, fp8_e4m3fn_t>::value) {
        LoadDataToL0B(L0BTensor, bL1Tensor, param, 0, param.singleK, param.singleN);
    }
    l0bBuffer.Set<HardEvent::MTE1_M>(); // mte1搬运完后，通知可以matmul
    l0bBuffer.Wait<HardEvent::MTE1_M>(); //  matmul等mte1：L0A数据搬运完成后才能开始matmul

    for (uint32_t m = 0; m < mLoops; m++) {
        uint32_t tileM = (m == (mLoops - 1)) ? tailM : baseM;
        Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
        l0aBuffer.Wait<HardEvent::M_MTE1>(); // mte1等Matmul：上一轮matmul完成后才能搬运新数据到L0A
        LocalTensor<L0ADType> L0ATensor = l0aBuffer.GetTensor<L0ADType>();
        uint64_t loopNum = param.isRightTranspose ? mLoops : 1;
        if constexpr (IsSameType<L0ADType, mx_fp8_e4m3_t>::value) {
            LoadDataToL0AMx<A, L0ADType>(L0ATensor, aL1Tensor, aScaleL1Tensor, param, m * L1Aoffset, tileM, param.singleK);
        }
        l0aBuffer.Set<HardEvent::MTE1_M>(); // mte1搬运完后，通知可以开始matmul
        l0aBuffer.Wait<HardEvent::MTE1_M>(); // matmul等mte1：L0A数据搬运完成后才能开始matmul

        MmadParams mmadParams;
        mmadParams.m = tileM;
        mmadParams.n = param.singleN;
        mmadParams.k = param.singleK;
        if (mmadParams.m == 1) {  // m等于1或默认开GEMV模式，文档上没有写怎么关闭GEMV，所以规避当做矩阵运算
            mmadParams.m = 16;
        }
        mmadParams.cmatrixInitVal = param.isOutKFisrt && (m == 0);
        mmadParams.cmatrixSource = false;
        if (param.unitFlag != 0) {
            mmadParams.unitFlag = (param.unitFlag == UNITFLAG_EN_OUTER_LAST) && (m == mLoops - 1) ?
                                  UNITFLAG_EN_OUTER_LAST : UNITFLAG_ENABLE;
        }
        Mmad(cL0Tensor[m * L0Coffset], L0ATensor, L0BTensor, mmadParams);
        l0aBuffer.Set<HardEvent::M_MTE1>(); // matmul完成后，通知mte1可以开始搬运新数据到L0B
    }
    l0bBuffer.Set<HardEvent::M_MTE1>(); // matmul完成后，通知mte1可以开始搬运新数据到L0A
}
#else
static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix isSetPadding;
template <typename T, ABLayout AL>
__aicore__ inline void LoadDataToL0A(LocalTensor<T>& aL0Tensor, const LocalTensor<T>& aL1Tensor,
                                     const MMParam& mmParam, uint64_t L1Aoffset, uint32_t kSplitSize,
                                     uint32_t mSplitSize)
{
    if constexpr (AL == ABLayout::MK) {
        LoadData3DParamsV2<T> loadData3DParams;
        loadData3DParams.l1H = mSplitSize / LOAD3D_L1W_SIZE; // 源操作数height
        loadData3DParams.l1W = LOAD3D_L1W_SIZE; // 源操作数weight
        loadData3DParams.padList[0] = 0;
        loadData3DParams.padList[1] = 0;
        loadData3DParams.padList[2] = 0;
        loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果

        loadData3DParams.mExtension = mSplitSize; // 在目的操作数height维度的传输长度
        loadData3DParams.kExtension = kSplitSize; // 在目的操作数width维度的传输长度
        loadData3DParams.mStartPt = 0; // 卷积核在目的操作数width维度的起点
        loadData3DParams.kStartPt = 0; // 卷积核在目的操作数height维度的起点
        loadData3DParams.strideW = 1; // 卷积核在源操作数width维度滑动的步长
        loadData3DParams.strideH = 1; // 卷积核在源操作数height维度滑动的步长
        loadData3DParams.filterW = 1; // 卷积核width
        loadData3DParams.filterSizeW = false; // 是否在filterW的基础上将卷积核width增加256个元素
        loadData3DParams.filterH = 1; // 卷积核height
        loadData3DParams.filterSizeH = false; // 是否在filterH的基础上将卷积核height增加256个元素
        loadData3DParams.dilationFilterW = 1; // 卷积核width膨胀系数
        loadData3DParams.dilationFilterH = 1; // 卷积核height膨胀系数
        loadData3DParams.enTranspose = 0; // 是否启用转置功能，对整个目标矩阵进行转置
        loadData3DParams.fMatrixCtrl = 0;
        loadData3DParams.channelSize = kSplitSize; // 源操作数的通道数。膨胀系数为1时，目的weight为filterW*filterH*channelSize
        LoadData<T, LOAD3DV2_CONFIG>(aL0Tensor, aL1Tensor[L1Aoffset], loadData3DParams);
    } else if constexpr (AL == ABLayout::KM) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0; // 分型矩阵ID，表明搬运起始位置为源操作数中第0个分型
        loadData2DParams.repeatTimes = (kSplitSize / ONE_FRACTAL_H_ELEMENT) * (mmParam.singleM /
                                       (ONE_FRACTAL_W_BYTE / sizeof(T))); // 迭代次数，每个迭代可以处理512B数据
        loadData2DParams.srcStride = 1; // 相邻迭代间，源操作数前一个分型和后一个分型起始地址的间隔（单位512B）
        loadData2DParams.dstGap = 0; // 相邻迭代间，目的操作数前一个分型的结束地址和后一个分型起始地址的间隔（单位512B）
        loadData2DParams.ifTranspose = true;
        LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData2DParams);
    }
}

// L1→L0B + 切K/切N/全载
template <typename T, ABLayout BL>
__aicore__ inline void LoadDataToL0B(LocalTensor<T>& bL0Tensor, const LocalTensor<T>& bL1Tensor,
                                     const MMParam& mmParam, uint64_t L1Boffset, uint32_t kSplitSize,
                                     uint32_t nSplitSize)
{
    if constexpr (BL == ABLayout::KN) {
        LoadData3DParamsV2<T> loadData3DParams;
        loadData3DParams.l1H = kSplitSize / LOAD3D_L1W_SIZE; // 源操作数height
        loadData3DParams.l1W = LOAD3D_L1W_SIZE; // 源操作数weight=16，目的height=l1H*L1W
        loadData3DParams.padList[0] = 0;
        loadData3DParams.padList[1] = 0;
        loadData3DParams.padList[2] = 0;
        loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果

        loadData3DParams.mExtension = kSplitSize; // 在目的操作数height维度的传输长度
        loadData3DParams.kExtension = nSplitSize; // 在目的操作数width维度的传输长度
        loadData3DParams.mStartPt = 0; // 卷积核在目的操作数width维度的起点
        loadData3DParams.kStartPt = 0; // 卷积核在目的操作数height维度的起点
        loadData3DParams.strideW = LOAD3D_STRIDE_W;
        loadData3DParams.strideH = LOAD3D_STRIDE_H;
        loadData3DParams.filterW = LOAD3D_FILTER_W;
        loadData3DParams.filterSizeW = false; // 是否在filterW的基础上将卷积核width增加256个元素
        loadData3DParams.filterH = LOAD3D_FILTER_H;
        loadData3DParams.filterSizeH = false; // 是否在filterH的基础上将卷积核height增加256个元素
        loadData3DParams.dilationFilterW = LOAD3D_DILA_FILTER_W; // 卷积核width膨胀系数
        loadData3DParams.dilationFilterH = LOAD3D_DILA_FILTER_H; // 卷积核height膨胀系数
        loadData3DParams.enTranspose = 1; // 是否启用转置功能
        loadData3DParams.fMatrixCtrl = 0; // 使用FMATRIX_LEFT还是使用FMATRIX_RIGHT，=0使用FMATRIX_LEFT，=1使用FMATRIX_RIGHT 1
        loadData3DParams.channelSize = nSplitSize; // 源操作数的通道数。膨胀系数为1时，目的weight为filterW*filterH*channelSize
        LoadData<T, LOAD3DV2_CONFIG>(bL0Tensor, bL1Tensor[L1Boffset], loadData3DParams);
    } else if constexpr (BL == ABLayout::NK) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0;
        loadData2DParams.repeatTimes = (nSplitSize + (ONE_FRACTAL_H_ELEMENT - 1)) / ONE_FRACTAL_H_ELEMENT *
                                       (kSplitSize / (ONE_FRACTAL_W_BYTE / sizeof(T))); // 迭代次数，每个迭代可以处理512B数据
        loadData2DParams.srcStride = 1;
        loadData2DParams.dstGap = 0;
        loadData2DParams.ifTranspose = false;
        LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData2DParams);
    }
}

#endif
// 全载
// 外部L1切入K时，需要传入cmatrixInitVal的标记
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType, typename AScaleType = fp8_e8m0_t, typename BScaleType = fp8_e8m0_t,
          typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MatmulFull(const LocalTensor<A> &aL1Tensor,
                                  const LocalTensor<B> &bL1Tensor,
                                  L0AType &aL0BuffsDb,
                                  L0BType &bL0BuffsDb,
                                  const LocalTensor<C> &cL0Tensor,
                                  struct MMParam &param,
                                  const LocalTensor<AScaleType> &aScaleL1Tensor = LocalTensor<AScaleType>(),
                                  const LocalTensor<BScaleType> &bScaleL1Tensor = LocalTensor<AScaleType>())
{
    Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
    l0aBuffer.Wait<HardEvent::M_MTE1>();
    LocalTensor<L0ADType> L0ATensor = l0aBuffer.GetTensor<L0ADType>();
#if ((__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102))
    if constexpr (IsSameType<L0ADType, mx_fp8_e4m3_t>::value) {
        LoadDataToL0AMx<A, L0ADType>(L0ATensor, aL1Tensor, aScaleL1Tensor, param, 0, param.singleK, param.singleM); // d,s2
    } else
#endif
    {
        LoadDataToL0A(L0ATensor, aL1Tensor, param, 0, param.singleK, param.singleM); // s2*d,d,s2
    }
    l0aBuffer.Set<HardEvent::MTE1_M>();

    Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
    l0bBuffer.Wait<HardEvent::M_MTE1>();
    LocalTensor<L0BDType> L0BTensor = l0bBuffer.GetTensor<L0BDType>();
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<L0BDType, mx_fp8_e4m3_t>::value) {
        LoadDataToL0BMx<B, L0BDType>(L0BTensor, bL1Tensor, bScaleL1Tensor, param, 0, param.singleK, param.singleN);
    } else
#endif
    {
        LoadDataToL0B(L0BTensor, bL1Tensor, param, 0, param.singleK, param.singleN);
    }
    l0bBuffer.Set<HardEvent::MTE1_M>();

    l0aBuffer.Wait<HardEvent::MTE1_M>();
    l0bBuffer.Wait<HardEvent::MTE1_M>();

    MmadParams mmadParams;
    mmadParams.m = param.singleM;
    if (param.realM != 0) {
        mmadParams.m = param.realM;
    }
    mmadParams.n = param.singleN;
    mmadParams.k = param.singleK;
    mmadParams.cmatrixInitVal = param.isOutKFisrt;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = param.unitFlag;
    if (mmadParams.m == 1) {
        mmadParams.m = 16;
    }

    Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

    l0aBuffer.Set<HardEvent::M_MTE1>();
    l0bBuffer.Set<HardEvent::M_MTE1>();
}

// 切K
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType, typename AScaleType = fp8_e8m0_t, typename BScaleType = fp8_e8m0_t,
          typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MatmulK(const LocalTensor<A> &aL1Tensor,
                              const LocalTensor<B> &bL1Tensor,
                              L0AType &aL0BuffsDb,
                              L0BType &bL0BuffsDb,
                              const LocalTensor<C> &cL0Tensor,
                              const MMParam &param,
                              const LocalTensor<AScaleType> &aScaleL1Tensor = LocalTensor<AScaleType>(),
                              const LocalTensor<BScaleType> &bScaleL1Tensor = LocalTensor<AScaleType>())
{
    uint32_t kLoops = (param.singleK + baseK - 1) / baseK;
    uint32_t tailSize = param.singleK % baseK;
    uint32_t tailK = tailSize ? tailSize : baseK;
    uint64_t L1Aoffset = param.isLeftTranspose ? baseK << 4 : ((param.singleM + 15) >> 4 << 4) * baseK;
    uint64_t L1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 4;
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<A, fp8_e5m2_t>::value || IsSameType<A, fp8_e4m3fn_t>::value ||
                  IsSameType<A, hifloat8_t>::value || IsSameType<A, int8_t>::value) {
        L1Aoffset = ((param.singleM + 31) >> 5 << 5) * baseK;
        L1Boffset = ((param.singleN + 31) >> 5 << 5) * baseK;
    }
    if constexpr (IsSameType<A, float>::value) {
        L1Aoffset = param.isLeftTranspose ? baseK << 3 : ((param.singleM + 15) >> 4 << 4) * baseK;
        L1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 3;
    }
#endif

    for (uint32_t k = 0; k < kLoops; k++) {
        uint32_t tileK = (k == (kLoops - 1)) ? tailK : baseK;
        Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
        l0aBuffer.Wait<HardEvent::M_MTE1>(); // mte1等Matmul：上一轮matmul完成后才能搬运新数据到L0A
        LocalTensor<L0ADType> L0ATensor = l0aBuffer.GetTensor<L0ADType>();
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
        if constexpr (IsSameType<L0ADType, mx_fp8_e4m3_t>::value) {
            LoadDataToL0AMx<A, L0ADType>(L0ATensor, aL1Tensor, aScaleL1Tensor, param, k * L1Aoffset, tileK, param.singleM); // s2,
        } else
#endif
        {
            LoadDataToL0A(L0ATensor, aL1Tensor, param, k * L1Aoffset, tileK, param.singleM); // s2*d,d,s2
        }

        Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
        l0bBuffer.Wait<HardEvent::M_MTE1>(); // mte1等Matmul：上一轮matmul完成后才能搬运新数据到L0B
        LocalTensor<L0BDType> L0BTensor = l0bBuffer.GetTensor<L0BDType>();
        uint64_t loopNum = param.isRightTranspose ? 1 : kLoops;
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
        if constexpr (IsSameType<L0BDType, mx_fp8_e4m3_t>::value) {
            LoadDataToL0BMx<B, L0BDType>(L0BTensor, bL1Tensor, bScaleL1Tensor, param, k * L1Boffset, tileK, param.singleN, loopNum); // tileK.D
        } else
#endif
        {
            LoadDataToL0B(L0BTensor, bL1Tensor, param, k * L1Boffset, tileK, param.singleN, loopNum);
        }
        l0bBuffer.Set<HardEvent::MTE1_M>(); // mte1搬运完后，通知可以开始matmul
        // l0aBuffer和l0bBuffer共用MTE1_M，在D=512场景减少同步指令数量，提升性能
        l0bBuffer.Wait<HardEvent::MTE1_M>(); // matmul等mte1：L0B数据搬运完成后才能开始matmul

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) {
            mmadParams.m = param.realM;
        }
        mmadParams.n = param.singleN;
        mmadParams.k = tileK;
        if (mmadParams.m == 1) {  // m等于1或默认开GEMV模式，文档上没有写怎么关闭GEMV，所以规避当做矩阵运算
            mmadParams.m = 16;
        }
        mmadParams.cmatrixInitVal = param.isOutKFisrt && (k == 0);
        mmadParams.cmatrixSource = false;
        if (param.unitFlag != 0) {
            mmadParams.unitFlag = (param.unitFlag == UNITFLAG_EN_OUTER_LAST) && (k == kLoops - 1) ?
                                  UNITFLAG_EN_OUTER_LAST : UNITFLAG_ENABLE;
        }
        Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

        l0aBuffer.Set<HardEvent::M_MTE1>(); // matmul完成后，通知mte1可以开始搬运新数据到L0A
        l0bBuffer.Set<HardEvent::M_MTE1>(); // matmul完成后，通知mte1可以开始搬运新数据到L0B
    }
}

// 切K---int8带偏置的实现
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL, typename L0AType, typename L0BType>
__aicore__ inline void MatmulKbias(const LocalTensor<A> &aL1Tensor,
                              const LocalTensor<B> &bL1Tensor,
                              L0AType &aL0BuffsDb,
                              L0BType &bL0BuffsDb,
                              const LocalTensor<int32_t> &cL0Tensor,
                              const LocalTensor<int32_t> &biasTensor,
                              const MMParam &param)
{
    uint32_t kLoops = (param.singleK + baseK - 1) / baseK; // 尾块处理，如果dSize不可以被256整除的时候，也就是存在尾块的时候，循环次数需要+1，这里的尾块始终是针对一个基本块计算的视角来看的
    uint32_t tailSize = param.singleK % baseK; // 针对dsize维度来说，需要按照256切分，如果不可以整除的话，求出最后一个尾块的dsize，命名为tailSize
    uint32_t tailK = tailSize ? tailSize : baseK; // 沿着K轴切分的尾块的K
    uint64_t L1Aoffset = param.isLeftTranspose ? baseK << 4 : ((param.singleM + 15) >> 4 << 4) * baseK; // 给传入的s1realsize对齐到16的倍数
    uint64_t L1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 4; // 给传入的s2realsize对齐到16的倍数
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<A, fp8_e5m2_t>::value || IsSameType<A, fp8_e4m3fn_t>::value || IsSameType<A, hifloat8_t>::value || IsSameType<A, int8_t>::value) {
        L1Aoffset = ((param.singleM + 31) >> 5 << 5) * baseK; // 给传入的s1realsize对齐到32的倍数
        L1Boffset = ((param.singleN + 31) >> 5 << 5) * baseK; // 给传入的s2realsize对齐到32的倍数
    }
    if constexpr (IsSameType<A, float>::value) {
        L1Aoffset = param.isLeftTranspose ? baseK << 3 : ((param.singleM + 15) >> 4 << 4) * baseK;
        L1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 3;
    }
#endif

    for (uint32_t k = 0; k < kLoops; k++) {
        uint32_t tileK = (k == (kLoops - 1)) ? tailK : baseK;
        Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
        l0aBuffer.Wait<HardEvent::M_MTE1>(); // mte1等Matmul：上一轮matmul完成后才能搬运新数据到L0A
        LocalTensor<A> L0ATensor = l0aBuffer.GetTensor<A>();
        LoadDataToL0A(L0ATensor, aL1Tensor, param, k * L1Aoffset, tileK, param.singleM);
        l0aBuffer.Set<HardEvent::MTE1_M>(); // mte1搬运完后，通知可以开始matmul

        Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
        l0bBuffer.Wait<HardEvent::M_MTE1>(); // mte1等Matmul：上一轮matmul完成后才能搬运新数据到L0B
        LocalTensor<B> L0BTensor = l0bBuffer.GetTensor<B>();
        uint64_t loopNum = param.isRightTranspose ? 1 : kLoops;
        LoadDataToL0B(L0BTensor, bL1Tensor, param, k * L1Boffset, tileK, param.singleN, loopNum);
        l0bBuffer.Set<HardEvent::MTE1_M>(); // mte1搬运完后，通知可以开始matmul

        l0aBuffer.Wait<HardEvent::MTE1_M>(); // matmul等mte1：L0A数据搬运完成后才能开始matmul
        l0bBuffer.Wait<HardEvent::MTE1_M>(); // matmul等mte1：L0B数据搬运完成后才能开始matmul

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) {
            mmadParams.m = param.realM;
        }
        mmadParams.n = param.singleN;
        mmadParams.k = tileK;
        if (mmadParams.m == 1) {  // m等于1或默认开GEMV模式，文档上没有写怎么关闭GEMV，所以规避当做矩阵运算
            mmadParams.m = 16;
        }
        mmadParams.cmatrixInitVal = false;
        mmadParams.cmatrixSource = (k == 0);
        if (param.unitFlag != 0) {
            mmadParams.unitFlag = (param.unitFlag == UNITFLAG_EN_OUTER_LAST) && (k == kLoops - 1) ?
                                  UNITFLAG_EN_OUTER_LAST : UNITFLAG_ENABLE;
        }

        if (k == 0) {
            Mmad(cL0Tensor, L0ATensor, L0BTensor, biasTensor, mmadParams);
        } else {
            Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);
        }

        l0aBuffer.Set<HardEvent::M_MTE1>(); // matmul完成后，通知mte1可以开始搬运新数据到L0A
        l0bBuffer.Set<HardEvent::M_MTE1>(); // matmul完成后，通知mte1可以开始搬运新数据到L0B
    }
}

// 切N
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType, typename AScaleType = fp8_e8m0_t, typename BScaleType = fp8_e8m0_t,
          typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MatmulN(const LocalTensor<A> &aL1Tensor,
                              const LocalTensor<B> &bL1Tensor,
                              L0AType &aL0BuffsDb,
                              L0BType &bL0BuffsDb,
                              const LocalTensor<C> &cL0Tensor,
                              const MMParam &param,
                              const LocalTensor<AScaleType> &aScaleL1Tensor = LocalTensor<AScaleType>(),
                              const LocalTensor<BScaleType> &bScaleL1Tensor = LocalTensor<AScaleType>())
{
    uint32_t nLoops = (param.singleN + baseN - 1) / baseN; // 尾块处理
    uint32_t tailSize = param.singleN % baseN;
    uint32_t tailN = tailSize ? tailSize : baseN;
    uint64_t L1Boffset = param.isRightTranspose ? (baseN << 4) : ((param.singleK + 15) >> 4 << 4) * baseN;
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<A, fp8_e5m2_t>::value || IsSameType<A, fp8_e4m3fn_t>::value ||
                  IsSameType<A, hifloat8_t>::value || IsSameType<A, int8_t>::value) {
        L1Boffset = ((param.singleK + 31) >> 5 << 5) * baseN;
    }
#endif
    uint64_t L0Coffset = ((param.singleM + 15) >> 4 << 4) * baseN;
    if (param.realM != 0) {
        L0Coffset = ((param.realM + 15) >> 4 << 4) * baseN;
    }

    Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
    l0aBuffer.Wait<HardEvent::M_MTE1>(); // mte1等Matmul：上一轮matmul完成后才能搬运新数据到L0A
    LocalTensor<L0ADType> L0ATensor = l0aBuffer.GetTensor<L0ADType>();
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<L0ADType, mx_fp8_e4m3_t>::value) {
        LoadDataToL0AMx<A, L0ADType>(L0ATensor, aL1Tensor, aScaleL1Tensor, param, 0, param.singleK, param.singleM); // d,s2
    } else
#endif
    {
        LoadDataToL0A(L0ATensor, aL1Tensor, param, 0, param.singleK, param.singleM); // s2*d,d,s2
    }
    for (uint32_t n = 0; n < nLoops; n++) {
        uint32_t tileN = (n == (nLoops - 1)) ? tailN : baseN;

        Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
        l0bBuffer.Wait<HardEvent::M_MTE1>(); // mte1等Matmul：上一轮matmul完成后才能搬运新数据到L0B
        LocalTensor<L0BDType> L0BTensor = l0bBuffer.GetTensor<L0BDType>();
        uint64_t loopNum = param.isRightTranspose ? nLoops : 1;
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
        if constexpr (IsSameType<L0BDType, mx_fp8_e4m3_t>::value) {
            LoadDataToL0BMx<B, L0BDType>(L0BTensor, bL1Tensor, bScaleL1Tensor, param, n * L1Boffset, param.singleK, tileN, loopNum); // tileK.D
        } else
#endif
        {
            LoadDataToL0B(L0BTensor, bL1Tensor, param, n * L1Boffset, param.singleK, tileN, loopNum);
        }
        l0bBuffer.Set<HardEvent::MTE1_M>(); // mte1搬运完后，通知可以开始matmul
        // l0aBuffer和l0bBuffer共用MTE1_M，在D=512场景减少同步指令数量，提升性能
        l0bBuffer.Wait<HardEvent::MTE1_M>(); // matmul等mte1：L0B数据搬运完成后才能开始matmul

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) {
            mmadParams.m = param.realM;
        }
        mmadParams.n = tileN;
        mmadParams.k = param.singleK;
        if (mmadParams.m == 1) {
            mmadParams.m = FP16_ONE_FRACTAL_ELEMENT;
        }
        mmadParams.cmatrixInitVal = param.isOutKFisrt;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = param.unitFlag;
        Mmad(cL0Tensor[n * L0Coffset], L0ATensor, L0BTensor, mmadParams);

        l0bBuffer.Set<HardEvent::M_MTE1>(); // matmul完成后，通知mte1可以开始搬运新数据到L0B
    }
    l0aBuffer.Set<HardEvent::M_MTE1>(); // matmul完成后，通知mte1可以开始搬运新数据到L0A
}

// 切M
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL>
__aicore__ inline void MatmulKM(const LocalTensor<A> &aL1Tensor,
                              const LocalTensor<B> &bL1Tensor,
                              BuffersPolicyDB<BufferType::L0A> &aL0BuffsDb,
                              BuffersPolicyDB<BufferType::L0B> &bL0BuffsDb,
                              const LocalTensor<C> &cL0Tensor,
                              struct MMParam &param)
{
    uint32_t mLoops = (param.singleM + baseM - 1) / baseM; // 尾块处理
    uint32_t kLoops = (param.singleK + baseK - 1) / baseK; // 尾块处理
    uint32_t mSplitSize = (mLoops == 1) ? param.singleM : baseM;
    uint32_t mplitTailSize = (param.singleM % baseM) ? (param.singleM % baseM) : mSplitSize;
    uint32_t kSplitSize = (kLoops == 1) ? param.singleK : baseK;
    uint32_t kSplitTailSize = (param.singleK % baseK) ? (param.singleK % baseK) : kSplitSize;
    uint64_t L1Boffset = kSplitSize * param.singleN;
    uint64_t L0Coffset = mSplitSize * param.singleN;

    for (uint32_t k = 0; k < kLoops; k++) {
        kSplitSize = (k == (kLoops - 1)) ? kSplitTailSize : kSplitSize;
        for (uint32_t m = 0; m < mLoops; m++){
            mSplitSize = (m == (mLoops - 1)) ? mplitTailSize : mSplitSize;
            Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
            l0aBuffer.Wait<HardEvent::M_MTE1>(); // 占用
            LocalTensor<A> L0ATensor = l0aBuffer.GetTensor<A>();
            LoadDataToL0A(L0ATensor, aL1Tensor, param,
                k * param.singleM * kSplitSize + m * kSplitSize * mSplitSize,
                kSplitSize, mSplitSize);
            l0aBuffer.Set<HardEvent::MTE1_M>(); // 通知
            l0aBuffer.Wait<HardEvent::MTE1_M>(); // 等待L0A

            Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
            l0bBuffer.Wait<HardEvent::M_MTE1>(); // 占用
            LocalTensor<B> L0BTensor = l0bBuffer.GetTensor<B>();
            LoadDataToL0B(L0BTensor, bL1Tensor, param, k * L1Boffset, kSplitSize, param.singleN);
            l0bBuffer.Set<HardEvent::MTE1_M>(); // 通知
            l0bBuffer.Wait<HardEvent::MTE1_M>(); // 等待L0B

            MmadParams mmadParams;
            mmadParams.m = mSplitSize;
            mmadParams.n = param.singleN;
            mmadParams.k = kSplitSize;
            mmadParams.cmatrixInitVal = param.isOutKFisrt && (k == 0); // 配置C矩阵初始值是否为0。默认值ture
            mmadParams.cmatrixSource = false; //  来源于CO1
            Mmad(cL0Tensor[m * L0Coffset], L0ATensor, L0BTensor, mmadParams);

            l0aBuffer.Set<HardEvent::M_MTE1>(); // 释放L0A
            l0bBuffer.Set<HardEvent::M_MTE1>(); // 释放L0B
        }
    }
}

template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL, typename L0AType, typename L0BType>
__aicore__ inline void MatmulBase(const LocalTensor<A> &aL1Tensor,
                                  const LocalTensor<B> &bL1Tensor,
                                  L0AType &aL0BuffsDb,
                                  L0BType &bL0BuffsDb,
                                  const LocalTensor<C> &cL0Tensor,
                                  struct MMParam &param)
{
    if ((param.singleK + baseK - 1) / baseK > 1) {
        MatmulK<A, B, C, baseM, baseN, baseK, AL, BL>(aL1Tensor, bL1Tensor, aL0BuffsDb, bL0BuffsDb, cL0Tensor, param);
    } else if ((param.singleN + baseN - 1) / baseN > 1) {
        MatmulN<A, B, C, baseM, baseN, baseK, AL, BL>(aL1Tensor, bL1Tensor, aL0BuffsDb, bL0BuffsDb, cL0Tensor, param);
    } else {
        MatmulFull<A, B, C, baseM, baseN, baseK, AL, BL>(aL1Tensor, bL1Tensor, aL0BuffsDb, bL0BuffsDb, cL0Tensor, param);
    }
}

template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL>
__aicore__ inline void MatmulKPP(const LocalTensor<A> &aL1Tensor,
                                 const LocalTensor<B> &bL1Tensor,
                                 BuffersPolicyDB<BufferType::L0A> &aL0BuffsDb,
                                 BuffersPolicyDB<BufferType::L0B> &bL0BuffsDb,
                                 const LocalTensor<C> &cL0Tensor,
                                 const MMParam &param)
{
    uint32_t kLoops = (param.singleK + baseK - 1) / baseK;
    uint32_t kSplitSize = (kLoops == 1) ? param.singleK : baseK;
    uint32_t kSplitSizeAlign = AlignUp(kSplitSize, FP16_ONE_FRACTAL_ELEMENT);
    uint64_t L1Aoffset = AlignUp(param.singleM, FP16_ONE_FRACTAL_ELEMENT) * kSplitSize;
    uint64_t L1Boffset = AlignUp(param.singleN, FP16_ONE_FRACTAL_ELEMENT) * kSplitSize;
    for (uint32_t k = 0; k < kLoops; k++) {
        if (k == kLoops - 1) {
            kSplitSize = (param.singleK % baseK) ? (param.singleK % baseK) : kSplitSize;
            kSplitSizeAlign = AlignUp(kSplitSize, FP16_ONE_FRACTAL_ELEMENT);
        }
        Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
        l0aBuffer.Wait<HardEvent::M_MTE1>(); // mte1等Matmul:上一轮matmul完成后才能搬运新数据到L0A
        LocalTensor<A> L0ATensor = l0aBuffer.GetTensor<A>();
        LoadDataToL0A<A, AL>(L0ATensor, aL1Tensor, param, k * L1Aoffset, kSplitSizeAlign, param.singleM);
        Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
        LocalTensor<B> L0BTensor = l0bBuffer.GetTensor<B>();
        LoadDataToL0B<B, BL>(L0BTensor, bL1Tensor, param, k * L1Boffset, kSplitSizeAlign, param.singleN);

        l0aBuffer.Set<HardEvent::MTE1_M>(); // mte1搬运完后，通知可以开始matmul
        l0aBuffer.Wait<HardEvent::MTE1_M>(); // matmul等mte1：L0A数据搬运完成后才能开始matmul

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        mmadParams.n = param.singleN;
        mmadParams.k = kSplitSize;
        if (mmadParams.m == 1) { //m等于1会默认开GEMV模式，且不可关闭GEMV，所以规避当作矩阵计算
            mmadParams.m = FP16_ONE_FRACTAL_ELEMENT;
        }
        mmadParams.cmatrixInitVal = (param.isOutKFisrt == true) && (k == 0);
        mmadParams.cmatrixSource = false;
        if (param.unitFlag != 0) {
            mmadParams.unitFlag = (param.unitFlag == UNITFLAG_EN_OUTER_LAST) && (k == kLoops - 1) ?
                                  UNITFLAG_EN_OUTER_LAST : UNITFLAG_ENABLE;
        }

        Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);
    #if (__CCE_AICORE__ != 310) && (!(defined __DAV_310R6__))
        if ((mmadParams.m / FP16_ONE_FRACTAL_ELEMENT) * (mmadParams.n / FP16_ONE_FRACTAL_ELEMENT) < MMAD_MN_SIZE_10) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    #endif
        l0aBuffer.Set<HardEvent::M_MTE1>(); // matmul完成后，通知mte1可以开始搬运新数据到L0A
    }
}
template <typename T, ABLayout AL>
__aicore__ inline void LoadDataToL0A(LocalTensor<T>& aL0Tensor, const LocalTensor<T>& aL1Tensor,
                                    uint32_t rowSize, uint32_t kSplitSize, uint32_t mSplitSize)
{
    uint32_t blockElementCnt = ONE_FRACTAL_W_BYTE / sizeof(T);
    if constexpr (IsSameType<T, int4b_t>::value) {
        blockElementCnt = INT4_ONE_FRACTAL_ELEMENT;
    }
    if constexpr (AL == ABLayout::MK) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0; // 分型矩阵ID，表明搬运起始位置为源操作数中第0个分型
        loadData2DParams.srcStride = 1; // 相邻迭代间，源操作数前一个分型和后一个分型起始地址的间隔（单位512B）
        loadData2DParams.dstGap = kSplitSize / blockElementCnt - 1; // 相邻迭代间，目的操作数前一个分型的结束地址和后一个分型起始地址的间隔（单位512B）
        loadData2DParams.repeatTimes = mSplitSize / ONE_FRACTAL_H_ELEMENT; // 迭代次数，每个迭代可以处理512B数据
        loadData2DParams.ifTranspose = false;
        uint32_t loopTimes = kSplitSize / blockElementCnt;
        uint64_t l1Offset = rowSize * blockElementCnt;
        uint64_t l0Offset = ONE_FRACTAL_H_ELEMENT * blockElementCnt;
        for(uint32_t loop = 0; loop < loopTimes; loop++) {
            LoadData(aL0Tensor[loop * l0Offset], aL1Tensor[loop * l1Offset], loadData2DParams);
        }
    } else if constexpr (AL == ABLayout::KM) {
        LoadData2dTransposeParams loadData2dTransposeParams;
        loadData2dTransposeParams.startIndex = 0;
        loadData2dTransposeParams.srcStride = 1;
        loadData2dTransposeParams.dstFracGap = (kSplitSize + blockElementCnt -1) / blockElementCnt;
        loadData2dTransposeParams.dstGap = mSplitSize / ONE_FRACTAL_H_ELEMENT - 1;
        if(rowSize == kSplitSize) {
            loadData2dTransposeParams.repeatTimes = (kSplitSize + blockElementCnt - 1) / blockElementCnt;
            uint32_t loopTimes = mSplitSize / blockElementCnt;
            uint64_t l1Offset = rowSize * blockElementCnt;
            uint64_t l0Offset = kSplitSize * blockElementCnt;
            for(uint32_t loop = 0; loop < loopTimes; loop++) {
                LoadDataWithTranspose(aL0Tensor[loop * l0Offset], aL1Tensor[loop * l1Offset], loadData2dTransposeParams);
            }
        } else {
            loadData2dTransposeParams.repeatTimes = ((kSplitSize + blockElementCnt - 1) / blockElementCnt) * (mSplitSize / blockElementCnt);
            LoadDataWithTranspose(aL0Tensor, aL1Tensor, loadData2dTransposeParams);
        }
    }
}

template <typename T, ABLayout BL>
__aicore__ inline void LoadDataToL0B(LocalTensor<T>& bL0Tensor, const LocalTensor<T>& bL1Tensor,
                                     uint32_t rowSize, uint32_t kSplitSize, uint32_t nSplitSize)
{
    uint32_t blockElementCnt = ONE_FRACTAL_W_BYTE / sizeof(T);
    if constexpr (IsSameType<T, int4b_t>::value) {
        blockElementCnt = INT4_ONE_FRACTAL_ELEMENT;
    }
    if constexpr (BL == ABLayout::KN) {
        LoadData2dTransposeParams loadData2dTransposeParams;

        loadData2dTransposeParams.startIndex = 0;
        loadData2dTransposeParams.srcStride = 1;
        loadData2dTransposeParams.dstFracGap = 0;
        loadData2dTransposeParams.dstGap = nSplitSize / ONE_FRACTAL_H_ELEMENT - 1;
        loadData2dTransposeParams.repeatTimes = (kSplitSize + blockElementCnt - 1) / blockElementCnt;

        uint32_t loopTimes = nSplitSize / blockElementCnt;
        uint64_t l1Offset = rowSize * blockElementCnt;
        uint64_t l0Offset = blockElementCnt * blockElementCnt;
        for(uint32_t loop = 0; loop < loopTimes; loop++) {
            LoadDataWithTranspose(bL0Tensor[loop * l0Offset], bL1Tensor[loop * l1Offset], loadData2dTransposeParams);
        }
    } else if constexpr (BL == ABLayout::NK) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0; // 分型矩阵ID，表明搬运起始位置为源操作数中第0个分型
        loadData2DParams.srcStride = 1; // 相邻迭代间，源操作数前一个分型和后一个分型起始地址的间隔（单位512B）
        loadData2DParams.dstGap = 0; // 相邻迭代间，目的操作数前一个分型的结束地址和后一个分型起始地址的间隔（单位512B）
        loadData2DParams.ifTranspose = false;
        if(rowSize == kSplitSize) {
            loadData2DParams.repeatTimes = ((nSplitSize + ONE_FRACTAL_H_ELEMENT - 1) / ONE_FRACTAL_H_ELEMENT) * (kSplitSize / blockElementCnt);// 迭代次数，每个迭代可以处理512B数据
            LoadData(bL0Tensor, bL1Tensor, loadData2DParams);
        } else {
            loadData2DParams.repeatTimes = (nSplitSize + ONE_FRACTAL_H_ELEMENT - 1) / ONE_FRACTAL_H_ELEMENT;// 迭代次数，每个迭代可以处理512B数据
            uint32_t loopTimes = kSplitSize / blockElementCnt;
            uint64_t l1Offset = nSplitSize * blockElementCnt;
            uint64_t l0Offset = rowSize * blockElementCnt;
            for (uint32_t loop = 0; loop < loopTimes; loop++) {
                LoadData(bL0Tensor[loop * l0Offset], bL1Tensor[loop * l1Offset], loadData2DParams);
            }
        }
    }
}
}
#endif