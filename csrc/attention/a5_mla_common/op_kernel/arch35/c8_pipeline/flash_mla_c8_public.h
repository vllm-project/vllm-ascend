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
 * \file flash_mla_c8_public.h
 * \brief
 */
#ifndef FIA_PUBLIC_DEFINE_ARCH35_H
#define FIA_PUBLIC_DEFINE_ARCH35_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "util.h"
#include "../../vector_common.h"

enum class FlashMlaC8Layout { None=0, LAYOUT_BSH=1, LAYOUT_SBH=2, LAYOUT_BNSD=3, LAYOUT_TND=4, LAYOUT_NTD_TND=5, LAYOUT_NTD=6, LAYOUT_NTD_DCP=7 };

namespace AttentionCommon {

enum class FIA_LAYOUT : uint32_t {
    BSH = 0,
    BSND = 0,
    BNSD = 1,
    NZ = 2,
    TND = 3,
    NBSD = 4,
    NTD = 5
};

enum class FiaKernelType : uint8_t {
    NO_QUANT = 0,
    ANTI_QUANT,
    FULL_QUANT
};

struct FDparamsX {
    uint32_t fdCoreEnable;
    uint32_t fdBN2Idx;
    uint32_t fdMIdx;
    uint32_t fdS2SplitNum;
    uint32_t mStart;
    uint32_t mLen;
    uint32_t fdWorkspaceIdx;
};

struct RunInfoX {
    uint32_t loop = 0;
    uint32_t mloop = 0;
    bool isValid = false;
    bool isChangeBatch = false;
    bool isFirstS2Loop = false;
    bool isLastS2Loop = false;

    uint32_t bIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx = 0;
    uint32_t gIdx = 0;
    uint32_t s1Idx = 0;
    uint32_t s2Idx = 0;
    uint32_t s2LocalIdx = 0; // 每核本地 S2 索引，从0开始累加，用于判断本核内 S2 的第几个 base 块
    uint32_t realN2Idx = 0;   // GS1合轴时为n2Idx，不合轴时为n1Idx
    uint64_t actS1Size = 1;   // 当前处理head的S1轴实际大小
    uint64_t actS2Size = 1;   // 当前处理head的S2轴实际大小
    uint32_t actMSize = 0;    // GS1方向上的长度
    uint32_t actMSizeAlign32; // GS1 方向上长度对齐
    uint32_t actVecMSize;     // VEC 视角, 基本块GS1方向长度
    uint32_t vecMbaseIdx;     // VEC 对应的M 轴起始位置,V0 为0， V1 为 V0的actVecMSize

    uint32_t actSingleLoopS2Size = 0; // S2方向长度
    uint32_t actSingleLoopS2SizeAlign;

    bool isS2SplitCore = false;
    uint32_t faTmpOutWsPos = 0; // FA阶段，S2外切，需要写到workspace时，写出到第几块M*D的GM块

    int64_t preTokensLeftUp = 0;
    int64_t nextTokensLeftUp = 0;

    uint64_t qPaddingBeginOffset = 0;
    uint64_t kvPaddingBeginOffset = 0;
};

struct StridesConstInfo {
    uint64_t bnStride = 0;
    uint64_t n2Stride = 0;
};

struct CommonConstInfo {
    /* 轴长度 */
    uint32_t bSize;
    uint64_t t1Size;
    uint64_t t2Size;
    uint32_t dSize;
    uint32_t dSizeV;
    uint32_t dBasicBlock;
    uint32_t dSizeRope;
    uint32_t gSize; /* g轴的大小 */
    uint32_t n2Size;
    uint32_t realGSize;
    uint32_t realN2Size;
    uint64_t s1Size;             /* s1总大小 */
    uint64_t s2Size;             /* s2总大小 */
    uint64_t actualSeqLenSize;   /* 用户输入的actualseq的长度 */
    uint64_t actualSeqLenKVSize; /* 用户输入的actualseq_kv的长度 */

    /* strides */
    StridesConstInfo keyStrides;
    StridesConstInfo valueStrides;
    StridesConstInfo kRopeStrides;
    StridesConstInfo kScaleStrides;
    StridesConstInfo vScaleStrides;

    /* FA kernel meta */
    uint32_t bN2Start;
    uint32_t bN2End;
    uint32_t gS1OStart;
    uint32_t gS1OEnd;
    uint32_t s2OStart;
    uint32_t s2OEnd;
    uint32_t coreFirstTmpOutWsPos;

    /* mask */
    uint32_t sparseMode; // sparse
    uint32_t attenMaskBatch;
    uint32_t attenMaskS1Size;
    uint32_t attenMaskS2Size;
    int64_t preTokens;
    int64_t nextTokens;
    bool isRowInvalidOpen;
    bool isExistRowInvalid;
    float scaleValue;

    /* 核信息 */
    uint32_t aicIdx;
    uint32_t aivIdx;
    uint8_t subBlockIdx;
    uint32_t coreNum;

    /* S1外切 */
    bool enableS1OutSplit;
    uint32_t totalSize;

    /* FA中间结果写出workspace信息 */
    uint32_t accumOutSize;
    uint32_t logSumExpSize;

    /* 输出shape */
    FIA_LAYOUT outputLayout;
};

/* 高阶特性 */
struct PAConstInfo {
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
    uint32_t paLayoutType;
};

struct LseConstInfo {
    bool isSoftmaxLseEnable;
};

struct SinkConstInfo {
    bool learnableSinkFlag;
};

struct PseConstInfo {
    uint32_t pseShiftByBatch;
    int64_t pseS1Size;
    int64_t pseS2Size;
    uint32_t pseStride;
};

struct TensorListConstInfo {
    bool isKvContinuous; /* 是否为tensorlist */
};

struct PostQuantConstInfo {
    bool isPostQuantPerChnl;
    bool isPostQuantBF16;
    bool isPostQuantOffsetExist;
    float postQuantScaleValue;
    float postQuantOffsetValue;
};

struct LeftPaddingConstInfo {
    bool isQHasLeftPadding;
    bool isKVHasLeftPadding;
    int64_t queryRightPaddingSize;
    int64_t kvRightPaddingSize;
};

struct SysPrefixConstInfo {
    bool isActualSharedPrefixLenNull;
    int64_t actualKVPrefixSize; /* 保存prefix实际长度 */
    int64_t kvPrefixSize;       /* 保存prefix shape完整长度 */
    int64_t prefixLoopCount;    /* 保存prefix参与的S2方向循环次数 */
};

template <FiaKernelType>
struct ConstInfo_t;

template <>
struct ConstInfo_t<FiaKernelType::NO_QUANT> : CommonConstInfo, PAConstInfo, LseConstInfo {
    bool needInit =
        false; /* 启动时是否需要预清输出(attentionOut写0/lse写3e+99), 由host下发fiaEmptyTensorParams.needInit */
    uint8_t l2CacheOffFlag = 0; /* gSize*s1Size<=64 场景关闭K/V的L2 Cache, 由host下发fiaBaseParams.l2CacheOffFlag */
};

template <>
struct ConstInfo_t<FiaKernelType::FULL_QUANT> : CommonConstInfo, PAConstInfo, LseConstInfo, TensorListConstInfo { bool needInit = false; };

__aicore__ inline int64_t ClipSInnerToken(int64_t sInnerToken, int64_t minValue, int64_t maxValue)
{
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}

template <FlashMlaC8Layout LAYOUT>
__aicore__ inline constexpr fa_base_vector::UbInputFormat GeInputUbFormat()
{
    static_assert((LAYOUT == FlashMlaC8Layout::LAYOUT_BSH) || (LAYOUT == FlashMlaC8Layout::LAYOUT_BNSD) ||
                      (LAYOUT == FlashMlaC8Layout::LAYOUT_TND) || (LAYOUT == FlashMlaC8Layout::LAYOUT_NTD),
                  "Get Query GmFormat fail, LAYOUT_T is incorrect");
    if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_BSH || LAYOUT == FlashMlaC8Layout::LAYOUT_TND) {
        return fa_base_vector::UbInputFormat::S1G;
    } else if constexpr (LAYOUT == FlashMlaC8Layout::LAYOUT_BNSD || LAYOUT == FlashMlaC8Layout::LAYOUT_NTD) {
        return fa_base_vector::UbInputFormat::GS1;
    }
}
} // namespace AttentionCommon

#endif // FIA_PUBLIC_DEFINE_ARCH35_H
