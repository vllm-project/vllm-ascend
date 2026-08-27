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
 * \file util_regbase.h
 * \brief
 */

#ifndef KV_QUANT_SAS_UTIL_REGBASE_H
#define KV_QUANT_SAS_UTIL_REGBASE_H

#include "util.h"

using AscendC::TQue;
using AscendC::QuePosition;

namespace regbaseutil {
constexpr int64_t MAX_PRE_NEXT_TOKENS = 0x7FFFFFFF;

#define COMMON_RUN_PARAM \
    int64_t boIdx; \
    int64_t s1oIdx; \
    int64_t n2oIdx; \
    int64_t goIdx; \
    int64_t s2LoopEndIdx;          /* S2方向的循环控制信息 souter层确定 */ \
    int64_t s2LineStartIdx = 0;    /* S2方向按行的起始位置 */ \
    int64_t s2LineEndIdx;          /* S2方向按行的结束位置 */ \
    int64_t s2CmpLineEndIdx; \
    /* cube视角的sOuter，在SAMEAB场景中cubeSOuterSize为两倍的 halfS1RealSize souter层确定 */ \
    uint32_t s1RealSize; \
    uint32_t halfS1RealSize; \
    uint32_t firstHalfS1RealSize; \
    uint32_t mRealSize; \
    uint32_t halfMRealSize; \
    uint32_t firstHalfMRealSize; \
    int64_t attentionOutOffset;    /* attentionOut的offset souter层确定 */ \
    int32_t actualS1Size;      /* Q的actualSeqLength */ \
    int32_t actualS2Size    /* KV的actualSeqLength */ \

struct RunParamStr {  // 分核与切块需要使用到参数
    COMMON_RUN_PARAM;
    /* 推理新增 */
    int64_t gs1LoopStartIdx;
    int64_t gs1LoopEndIdx;
    // BN循环生产的数据
    int64_t preTokensPerBatch = MAX_PRE_NEXT_TOKENS; // 左上顶点的pretoken
    int64_t nextTokensPerBatch = MAX_PRE_NEXT_TOKENS; // 左上顶点的nexttoken

    // NBS1循环生产的数据
    int64_t sOuterOffset;               // 单个S内 souter的 souterIdx * halfS1RealSize souter层确定
    int64_t cubeSOuterOffset;           // 单个S内 souter的 souterIdx * halfS1RealSize souter层确定
    int64_t mOuterOffset;
    int64_t cubeMOuterOffset;

    // lse 输出offset
    int64_t softmaxLseOffset;       // souter层确定

    int64_t qSNumInOneBlock;
    int64_t oriKvLoopEndIdx;
    int64_t cmpKvLoopEndIdx;
};

#define COMMON_RUN_INFO \
    int64_t s2StartIdx; /* s2的起始位置，sparse场景下可能不是0 */ \
    int64_t s2EndIdx; \
    int64_t s2LoopCount; /* s2循环当前的循环index */ \
    int64_t s2LoopLimit; \
    int64_t s1oIdx = 0; /* s1轴的index */ \
    int64_t loop = 0; /* for v0 perload loop */ \
    int64_t boIdx = 0; /* b轴的index */ \
    int64_t n2oIdx = 0; /* n2轴的index */ \
    int64_t goIdx = 0; /* g轴的index */ \
    int32_t s1RealSize; \
    int32_t halfS1RealSize; /* vector侧实际的s1基本块大小，如果Cube基本块=128，那么halfS1RealSize=64 */ \
    int32_t firstHalfS1RealSize; /* 当s1RealSize不是2的整数倍时，v0比v1少计算一行，计算subblock偏移的时候需要使用v0的s1 size */ \
    int32_t mRealSize; \
    int32_t halfMRealSize; \
    int32_t firstHalfMRealSize; \
    int32_t s2RealSize; /* s2方向基本块的真实长度 */ \
    int64_t s2AlignedSize; /* s2方向基本块对齐到16之后的长度 */ \
    int32_t vec2S1BaseSize; /* vector2侧开循环之后，经过切分的S1大小，例如把64切分成两份32 */ \
    int32_t vec2S1RealSize; /* vector2侧开循环之后，经过切分的S1的尾块大小，例如把63切分成两份32和31，第二份的实际大小是31 */ \
    int32_t vec2MBaseSize; \
    int32_t vec2MRealSize; \
    int64_t taskId; \
    int64_t multiCoreInnerIdx = 0; \
    int64_t attentionOutOffset; \
    int32_t actualS1Size; /* 非TND场景=总s1Size, Tnd场景下当前batch对应的s1 */ \
    int32_t actualS2Size; /* 非TND场景=总s2Size, Tnd场景下当前batch对应的s2 */ \
    int64_t preTokensPerBatch; /* vector2 左上顶点的pretoken */ \
    int64_t nextTokensPerBatch; /* vector2 左上顶点的nexttoken */ \
    uint8_t taskIdMod2; \
    uint8_t taskIdMod3; \
    uint8_t multiCoreIdxMod2 = 0; \
    uint8_t multiCoreIdxMod3 = 0; \
    int64_t sOuterOffset; \
    int64_t mOuterOffset; \
    bool    isCmp; \

struct RunInfo {
    COMMON_RUN_INFO;
    // 推理新增
    // lse 输出offset
    int64_t softmaxLseOffset;

    int64_t qSNumInOneBlock;
    int64_t oriKvLoopEndIdx;
    int64_t cmpKvLoopEndIdx;
};

#define COMMON_CONST_INFO \
    /* 全局的基本块信息 */ \
    uint32_t bSize; \
    uint32_t needInit; \
    uint32_t s1BaseSize; \
    uint32_t s2BaseSize; \
    int64_t dSize; /* query d 512 */ \
    int64_t dSizeV; /* key d 512 */ \
    int64_t dSizeVInput; /* key inpue d 640 = rope + nope + scale + pad */ \
    int64_t dSizeNope; /* key nope d 448 */ \
    int64_t dSizeRope; /* key rope d 64 */ \
    int64_t tileSize; /* 64 */ \
    int64_t sparseMode = 3; \
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
    /* matmul跳读参数 */ \
    int64_t mm1Ka; \
    /* dq 或者attentionOut的Stride */ \
    int64_t attentionOutStride; \
    uint32_t aivIdx; \
    uint8_t subBlockIdx;\

#define INFER_CONST_INFO \
    /* 推理 */ \
    bool isActualLenDimsNull; /* 判断是否有actualseq */ \
    bool isActualLenDimsKVNull; /* 判断是否有actualseq_kv */ \
    bool isSoftmaxLseEnable; \
    bool rsvd1; \
    uint32_t sparseBlockCount; \
    uint32_t actualSeqLenSize; /* 用户输入的actualseq的长度 */ \
    uint32_t actualSeqLenKVSize; /* 用户输入的actualseq_kv的长度 */ \
    /* service mm1 mm2 pageAttention */ \
    uint32_t oriBlockSize; \
    uint32_t cmpBlockSize; \
    uint32_t paLayoutType; \
    uint32_t oriMaxBlockNumPerBatch; \
    uint32_t cmpMaxBlockNumPerBatch; \
    int32_t oriWinLeft; \
    int32_t oriWinRight; \
    uint32_t sparseBlockSize; \
    uint32_t cmpRatio; \
    float softmaxScale; \
    uint32_t oriKvStride; \
    uint32_t cmpKvStride; \

#define CV_SHARED_PARAMS \
    /* base params */ \
    uint32_t s1BaseSize; \
    uint32_t s2BaseSize; \
    uint32_t bSize;  \
    uint32_t n2Size;  \
    uint32_t gSize;  \
    uint32_t s1Size;  \
    uint32_t s2Size;  \
    uint32_t dSize : 10;  \
    uint32_t dSizeVInput : 12;  \
    uint32_t needInit : 4; \
    uint32_t layoutType : 4;  \
    uint32_t isActualSeqLengthsNull : 1; \
    uint32_t isActualSeqLengthsKVNull : 1; \
    uint32_t sparseBlockCount; \
    float softmaxScale; \
    uint32_t cmpRatio : 9; \
    uint32_t dSizeRope : 11; \
    uint32_t oriMaskMode : 6;\
    uint32_t cmpMaskMode : 6; \
    int32_t oriWinLeft; \
    int32_t oriWinRight; \
    uint32_t tileSize : 8; \
    /* pa params */  \
    uint32_t oriBlockSize : 12; \
    uint32_t cmpBlockSize : 12; \
    uint32_t oriMaxBlockNumPerBatch; \
    uint32_t cmpMaxBlockNumPerBatch; \
    uint32_t oriKvStride; \
    uint32_t cmpKvStride; \


struct ConstInfo{
    COMMON_CONST_INFO;
    INFER_CONST_INFO;
};

/* only support b32 or b64 */
struct CVSharedParams {
    CV_SHARED_PARAMS;
};
}

#endif // KV_QUANT_SAS_UTIL_REGBASE_H
