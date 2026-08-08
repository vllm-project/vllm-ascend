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
 * \file util_regbase.h
 * \brief
 */

#ifndef FLASH_ATTENTION_UTIL_REGBASE_H
#define FLASH_ATTENTION_UTIL_REGBASE_H

#include "util.h"

using AscendC::TQue;
using AscendC::QuePosition;

namespace regbaseutil {
constexpr uint16_t regBytes = 256;
constexpr int64_t MAX_PRE_NEXT_TOKENS = 0x7FFFFFFF;
enum class VselrIndexEnum {GT_64_AND_LTE_128_INDEX = 0, GT_0_AND_LTE_64_INDEX = 1, DN_INDEX = 2};
enum class DTemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned48 = 48,
    Aligned64 = 64,
    Aligned80 = 80,
    Aligned96 = 96,
    Aligned128 = 128,
    Aligned160 = 160,
    Aligned192 = 192,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned576 = 576,
    Aligned768 = 768,
    NotAligned,
};

enum class S1TemplateType {
    Aligned16 = 16,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    NotAligned,
};

enum class S2TemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned1024 = 1024,
    NotAligned,
};
template<bool isInfer = false>
struct RunParamStr;

#define COMMON_RUN_PARAM \
    int64_t boIdx; \
    int64_t s1oIdx; \
    int64_t n2oIdx; \
    int64_t goIdx; \
    int32_t s2LoopEndIdx;          /* S2方向的循环控制信息 souter层确定 */ \
    int64_t s2LineStartIdx = 0;    /* S2方向按行的起始位置 */ \
    int64_t s2LineEndIdx;          /* S2方向按行的结束位置 */ \
    /* cube视角的sOuter，在SAMEAB场景中cubeSOuterSize为两倍的 halfS1RealSize souter层确定 */ \
    uint32_t s1RealSize; \
    uint32_t s1RealSizeAlign32;    /* dn场景使用 */ \
    uint32_t halfS1RealSize; \
    uint32_t firstHalfS1RealSize; \
    int64_t tensorQOffset;         /* query的offset souter层确定 */ \
    int64_t attentionOutOffset;    /* attentionOut的offset souter层确定 */ \
    int64_t actualS1Size;      /* Q的actualSeqLength */ \
    int64_t actualS2Size;    /* KV的actualSeqLength */ \
    uint64_t b1SSOffset; \
    uint64_t b1SSAttenMaskOffset; \
    uint64_t b1SSOffsetAlign16; \
    int64_t qRopeNBGOffset;             /* QueryRope 的 offset */ \
    int64_t kRopeNBGOffset;             /* G方向上,不同g的KeyRope的offset */ 

template<>
struct RunParamStr<false> {  // 分核与切块需要使用到参数
    COMMON_RUN_PARAM;
};

template<>
struct RunParamStr<true> {  // 分核与切块需要使用到参数
    COMMON_RUN_PARAM;
    /* 推理新增 */
    int64_t s1LoopTimes;
    // BN循环生产的数据
    int64_t s2InCurrentBatch;                 // Tensorlist场景，不同batch的S2长度，后续用计算KvStride
    int64_t preTokensPerBatch = MAX_PRE_NEXT_TOKENS; // 左上顶点的pretoken
    int64_t nextTokensPerBatch = MAX_PRE_NEXT_TOKENS; // 左上顶点的nexttoken

    // NBS1循环生产的数据
    int64_t sOuterOffset;               // 单个S内 souter的 souterIdx * halfS1RealSize souter层确定
    int64_t cubeSOuterOffset;           // 单个S内 souter的 souterIdx * halfS1RealSize souter层确定
    int64_t keyCoreOffset;              // BN方向上，不同BN的Key的offset batch层确定
    int64_t valueCoreOffset;            // BN方向上，不同BN的value的offset batch层确定
    uint64_t pseShiftCoreOffset;        // Souter方向上，不同souter的pseShift的offset
    int64_t keyOffset;              // mm1 Key 的offset,后续更名为KFinalOffset

    // q k v attenMask不同轴的offset
    // B轴offset 
    int64_t qBOffset;             // bIdx * seqSize * multiHeadQ，后续更名为qBOffset batch层确定
    int64_t qRopeBOffset;         // batch层确定 actualSeqLengths相关

    // 左padding
    int64_t queryLeftPaddingSize;       // batch层确定
    int64_t kvLeftPaddingSize;          // batch层确定

    // lse 输出offset
    int64_t softmaxLseOffset;       // souter层确定

    // IFA_MLA
    int64_t actualSeqLengthOfMlaPerBatch = 0; // 在mla场景下Q的actualSeqLength
    int64_t nextTokensOfMlaPerBatch = 0;   // 在mla场景下左上顶点的nexttoken，用于计算BNSD的行无效
};

#define COMMON_RUN_INFO \
    int64_t s2StartIdx; /* s2的起始位置，sparse场景下可能不是0 */ \
    int64_t s2EndIdx; \
    int64_t s2LoopCount; /* s2循环当前的循环index */ \
    int64_t s2LoopLimit; \
    int64_t s1oIdx = 0; /* s1轴的index */ \
    int64_t boIdx = 0; /* b轴的index */ \
    int64_t n2oIdx = 0; /* n2轴的index */ \
    int64_t goIdx = 0; /* g轴的index */ \
    int32_t s1RealSize; \
    int32_t s1RealSizeAlign32; \
    int32_t halfS1RealSize; /* vector侧实际的s1基本块大小，如果Cube基本块=128，那么halfS1RealSize=64 */ \
    int32_t firstHalfS1RealSize; /* 当s1RealSize不是2的整数倍时，v0比v1少计算一行，计算subblock偏移的时候需要使用v0的s1 size */ \
    int32_t s2RealSize; /* s2方向基本块的真实长度 */ \
    int64_t s2AlignedSize; /* s2方向基本块对齐到16之后的长度 */ \
    int32_t vec2S1BaseSize; /* vector2侧开循环之后，经过切分的S1大小，例如把64切分成两份32 */ \
    int32_t vec2S1RealSize; /* vector2侧开循环之后，经过切分的S1的尾块大小，例如把63切分成两份32和31，第二份的实际大小是31 */ \
    int64_t vecCoreOffset; /* vec核基于cube核起始处s1方向偏移 */ \
    int64_t queryOffset; /* mm1 Query的offset*/\
    int64_t keyOffset; /* mm1 Key的offset */ \
    int64_t valueOffset; /* mm2 Value的offset*/ \
    int64_t qRopeOffset; \
    int64_t kRopeOffset; \
    \
    int64_t taskId; \
    int64_t multiCoreInnerIdx = 0; \
    \
    int64_t attentionOutOffset; \
    int64_t s1SizeAcc; /* 对于非TND场景 = boIdx * pseInfo.s2Size; TND场景等于前面boIdx个batch的s2之和（每个batch的s2不同）*/ \
    int64_t s2SizeAcc; /* 对于非TND场景 = boIdx * pseInfo.s2Size; TND场景等于前面boIdx个batch的s2之和（每个batch的s2不同）*/ \
    int64_t actualS1Size; /* 非TND场景=总s1Size, Tnd场景下当前batch对应的s1 */ \
    int64_t actualS2Size; /* 非TND场景=总s2Size, Tnd场景下当前batch对应的s2 */ \
    int64_t preTokensPerBatch; /* vector2 左上顶点的pretoken */ \
    int64_t nextTokensPerBatch; /* vector2 左上顶点的nexttoken */ \
    int64_t b1SSOffset; /* 非TND = boIdx * s1 * s2; TND = 前面boIdx个batch的s1*S2之和 */ \
    /* 训练场景下等于b1SSOffset，推理场景下非TND mask支持大于s1 * s2；TND场景推理的mask是补过pad的 */ \
    int64_t b1SSAttenMaskOffset; \
    int64_t b1SSOffsetAlign; /* TND场景s2 16对齐之后，前面batch的s1*s2之和 */ \
    int64_t deScaleKvOffset; /* KV的反量化scale内容在Gm中的偏移 原始shape为 [B, N2, 1, Ceil(S2, 128), 1] */ \
    int64_t nextTokensOfMlaPerBatch = 0; /* 在mla场景下左上顶点的nexttoken，用于计算BNSD的行无效 */ \
    uint8_t taskIdMod2; \
    uint8_t taskIdMod3; \
    uint8_t multiCoreIdxMod2 = 0; \
    uint8_t multiCoreIdxMod3 = 0; \
    int64_t sOuterOffset

template<bool isInfer = false>
struct RunInfo;

template <>
struct RunInfo<true> {
    COMMON_RUN_INFO;
    // 推理新增
    uint64_t pseShiftOffset;              // vector1 pse 的 offset
    int64_t queryLeftPaddingSize;
    int64_t kvLeftPaddingSize;
    // lse 输出offset
    int64_t softmaxLseOffset;

    // FD相关
    int64_t flashDecodeS2Idx;

    // tensorlist相关
    int64_t s2InCurrentBatch;
};

template<>
struct RunInfo<false> {
    COMMON_RUN_INFO;
};

#define COMMON_CONST_INFO \
    /* 全局的基本块信息 */ \
    uint32_t s1BaseSize; \
    uint32_t s2BaseSize; \
    int64_t dSize; \
    int64_t dSizeV; \
    int64_t dBasicBlock; \
    int64_t dSizeRope; \
    int64_t gSize; /* g轴的大小 */ \
    int64_t n2Size; \
    int64_t s1Size; /* s1总大小 */ \
    int64_t s2Size; /* s2总大小 */ \
    /* 轴的乘积 */ \
    int64_t s1D; \
    int64_t gS1D; \
    int64_t n2GS1D; \
    int64_t s2D; \
    int64_t n2S2D; \
    int64_t s1Dv; \
    int64_t gS1Dv; \
    int64_t n2GS1Dv; \
    int64_t s2Dv; \
    int64_t n2S2Dv; \
    int64_t s1S2; \
    int64_t gS1; \
    int64_t gD; \
    int64_t n2D; \
    int64_t bN2D; \
    int64_t gDv; \
    int64_t n2Dv; \
    int64_t bN2Dv; \
    int64_t n2G; \
    int64_t n2GD; \
    int64_t bN2GD; \
    int64_t n2GDv; \
    int64_t bN2GDv; \
    int64_t gS2; \
    int64_t s1Dr; \
    int64_t gS1Dr; \
    int64_t n2GS1Dr; \
    int64_t s2Dr; \
    int64_t n2S2Dr; \
    int64_t gDr; \
    int64_t n2Dr; \
    int64_t bN2Dr; \
    int64_t n2GDr; \
    int64_t bN2GDr; \
    int32_t s2BaseN2D; \
    int32_t s1BaseN2GD; \
    int64_t s2BaseBN2D; \
    int64_t s1BaseBN2GD; \
    int32_t s1BaseD; \
    int32_t s2BaseD; \
    int64_t s2BaseN2Dv; \
    int64_t s2BaseBN2Dv; \
    int64_t s1BaseN2GDv; \
    int64_t s1BaseBN2GDv; \
    int32_t s1BaseDv; \
    int32_t s2BaseDv; \
    int64_t s1OuterSize; \
    /* matmul跳读参数 */ \
    int64_t mm1Ka; \
    int64_t mm1Kb; \
    int64_t mm2Kb; \
    /* dq 或者attentionOut的Stride */ \
    int64_t attentionOutStride; \
    uint32_t aivIdx; \
    uint8_t layoutType; \
    uint8_t subBlockIdx;\
    bool softMaxCheckRes; \
    float keepProb; \
    float scaleValue; \
    int64_t matmulMSize     /* 在matmul运算中，左矩阵的M轴大小需要区分GS1合轴与不合轴的情况 */


#define ROPE_INFO \
    /* rope参数 */ \
    int64_t s1DR; \
    int64_t gS1DR; \
    int64_t n2GS1DR; \
    int64_t s2DR; \
    int64_t n2S2DR; \
    int64_t gDR; \
    int64_t n2DR; \
    int64_t bN2DR; \
    int64_t n2GDR; \
    int64_t bN2GDR; \
    int64_t s2BaseN2DR; \
    int64_t s2BaseBN2DR; \
    int64_t s1BaseN2GDR; \
    int64_t s1BaseBN2GDR; \
    int64_t s1BaseDR; \
    int64_t s2BaseDR; \
    int64_t mm1RopeKa; /* rope matmul 跳读参数*/ \
    int64_t mm1RopeKb

#define INFER_CONST_INFO \
    /* 推理新增 */ \
    bool isRowInvalid; /* 是否使能行无效 */ \
    bool isActualLenDimsNull; /* 判断是否有actualseq */ \
    bool isActualLenDimsKVNull; /* 判断是否有actualseq_kv */ \
    bool isGqa; \
    \
    uint32_t actualSeqLenSize; /* 用户输入的actualseq的长度 */ \
    uint32_t actualSeqLenKVSize; /* 用户输入的actualseq_kv的长度 */ \
    uint32_t isKvContinuous; /* 是否为tensorlist */ \
    /* service mm1 mm2 pageAttention */ \
    uint32_t blockTableDim2; \
    uint32_t blockSize; \
    uint32_t paLayoutType; \
    uint32_t paBlockNumSum; \
    /* LAYOUT是否为BNSD_BSND */ \
    uint32_t isBSNDOut; \
    /* GS1合轴场景，外层循环是B、N2，内层循环G、S1，headNumRatio = 1 */ \
    /* GS1不合轴场景，外层循环是B、N2、G，内层循环S1，headNumRatio = gSize */ \
    uint32_t headNumRatio; \
    bool rsvd1; \
    bool isSoftmaxLseEnable; \
    /* 左padding */ \
    bool isQHasLeftPadding; \
    bool isKVHasLeftPadding; \
    int64_t queryRightPaddingSize; \
    int64_t kvRightPaddingSize; \
    /* FD */ \
    int64_t sInnerLoopSize; /* FD s2总大小 */ \
    int64_t actualCombineLoopSize; /* 实际规约块数 */ \
    int64_t splitKVNum; \
    /* 后量化 */ \
    bool isPostQuantPerChnl; \
    bool isPostQuantBF16; \
    bool isPostQuantOffsetExist; \
    float postQuantScaleValue; \
    float postQuantOffsetValue; \
    /* 全量化 */ \
    uint64_t deqScaleQKValue; \
    float quantScalePValue; \
    uint64_t deqScaleVValue

#define CV_SHARED_PARAMS \
    /* base params */ \
    uint32_t bSize;  \
    uint32_t n2Size;  \
    uint32_t gSize;  \
    uint32_t s1Size;  \
    uint32_t s2Size;  \
    uint32_t dSize : 16;  \
    uint32_t dSizeV : 16;  \
    /* special params */  \
    int64_t preTokens;  \
    int64_t nextTokens;  \
    uint32_t attenMaskS1Size;  \
    uint32_t attenMaskS2Size;  \
    int64_t s1SparseValidSize;  \
    int64_t s2SparseValidSize;  \
    /* core params */ \
    volatile int64_t multiCoreInnerOffset;  /* 二次赋值的变量需要volatile修饰 */ \
    volatile int64_t multiCoreInnerLimit;  /* 二次赋值的变量需要volatile修饰 */ \
    uint32_t s1OuterSize;  \
    uint32_t bandIndex;  \
    uint32_t compressMode : 4;  \
    uint32_t implMode : 4;  \
    uint32_t layoutType : 4;  \
    uint32_t sparseType : 8;  \
    uint32_t dSizeRope : 12; \
    uint32_t coreNum;

#define FAG_CV_SHARED_PARAMS \
    /* base params */ \
    float qScaleDs
 
struct FagCVSharedParams {
    FAG_CV_SHARED_PARAMS;
};
 
template<bool isInfer = false, bool hasRope = false>
struct ConstInfo;

template<>
struct ConstInfo<true, true> {
    COMMON_CONST_INFO;
    INFER_CONST_INFO;
    ROPE_INFO;
};

template <>
struct ConstInfo<true, false> {
    COMMON_CONST_INFO;
    INFER_CONST_INFO;
};

template <>
struct ConstInfo<false, true> {
    COMMON_CONST_INFO;
    ROPE_INFO;
    int64_t n2GS1o; // 训练特有
    int64_t gS1o;
};

template <>
struct ConstInfo<false, false> {
    COMMON_CONST_INFO;
    int64_t n2GS1o; // 训练特有
    int64_t gS1o;
};

/* only support b32 or b64 */
template <bool isInfer = false, bool isPa = false>
struct CVSharedParams;

template<>
struct CVSharedParams<false, false> {
    CV_SHARED_PARAMS;
};

/* CVSharedParams需要小于等于CacheLine的大小：128Bytes */
template<>
struct CVSharedParams<true, false> {
    CV_SHARED_PARAMS;
    uint32_t fromFused : 1;
    uint32_t isGqa : 1;
    uint32_t isKvContinuous : 1;
    uint32_t isRowInvalid : 1;
    uint32_t isActualSeqLengthsNull : 1;
    uint32_t isActualSeqLengthsKVNull : 1;
    uint32_t isQHasLeftPadding : 1;
    uint32_t isKVHasLeftPadding : 1;
    uint32_t isBSNDOut : 1;
    uint32_t needInit : 1;
    uint32_t isPostQuantPerChnl : 1;
    uint32_t isPostQuantBF16 : 1;
    uint32_t headNumRatio : 20;

    uint32_t actualSeqLengthsSize;
    uint32_t actualSeqLengthsKVSize;
    uint32_t splitKVNum;

    uint32_t bnStartIdx;
    uint32_t bnEndIdx;

    uint32_t queryRightPaddingSize;
    uint32_t kvRightPaddingSize;
};

template<>
struct CVSharedParams<true, true> {
    CV_SHARED_PARAMS;
    uint32_t fromFused : 1;
    uint32_t isGqa : 1;
    uint32_t isKvContinuous : 1;
    uint32_t isRowInvalid : 1;
    uint32_t isActualSeqLengthsNull : 1;
    uint32_t isActualSeqLengthsKVNull : 1;
    uint32_t isQHasLeftPadding : 1;
    uint32_t isKVHasLeftPadding : 1;
    uint32_t isBSNDOut : 1;
    uint32_t needInit : 1;
    uint32_t isPostQuantPerChnl : 1;
    uint32_t isPostQuantBF16 : 1;
    uint32_t headNumRatio : 20;

    uint32_t actualSeqLengthsSize;
    uint32_t actualSeqLengthsKVSize;
    uint32_t splitKVNum;

    uint32_t bnStartIdx;
    uint32_t bnEndIdx;

    uint32_t queryRightPaddingSize;
    uint32_t kvRightPaddingSize;

    int32_t blockSize;
    int32_t blockTableDim2;
    int32_t paBlockNumSum;
    uint32_t paLayoutType;
};
}

#endif // FLASH_ATTENTION_UTIL_REGBASE_H
