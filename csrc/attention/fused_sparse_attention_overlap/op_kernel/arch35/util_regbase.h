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

#ifndef UTIL_REGBASE_H
#define UTIL_REGBASE_H

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
    int64_t s2LoopEndIdx;          /* S2-loop control information determined by the sOuter layer */ \
    int64_t s2LineStartIdx = 0;    /* Starting row position along S2 */ \
    int64_t s2LineEndIdx;          /* Ending row position along S2 */ \
    int64_t s2CmpLineEndIdx; \
    /* sOuter from the Cube perspective; for SAMEAB, cubeSOuterSize is twice halfS1RealSize and is determined by the sOuter layer */ \
    uint32_t s1RealSize; \
    uint32_t halfS1RealSize; \
    uint32_t firstHalfS1RealSize; \
    uint32_t mRealSize; \
    uint32_t halfMRealSize; \
    uint32_t firstHalfMRealSize; \
    int64_t attentionOutOffset;    /* attentionOut offset determined by the sOuter layer */ \
    int32_t actualS1Size;      /* actualSeqLength of Q */ \
    int32_t actualS2Size;    /* actualSeqLength of KV */ \
    int64_t tensorQOffset; \
    int64_t tensorQRopeOffset; \
    int64_t qBOffset; \
    int64_t qRopeBOffset;

struct RunParamStr {  // Parameters required for core partitioning and block partitioning
    COMMON_RUN_PARAM;
    /* Added for inference */
    int64_t gs1LoopStartIdx;
    int64_t gs1LoopEndIdx;
    // Data produced by the BN loop
    int64_t preTokensPerBatch = MAX_PRE_NEXT_TOKENS; // preTokens at the upper-left vertex
    int64_t nextTokensPerBatch = MAX_PRE_NEXT_TOKENS; // nextTokens at the upper-left vertex

    // Data produced by the NBS1 loop
    int64_t sOuterOffset;               // sOuterIdx * halfS1RealSize within one S, determined by the sOuter layer
    int64_t cubeSOuterOffset;           // Cube sOuterIdx * halfS1RealSize within one S, determined by the sOuter layer
    int64_t mOuterOffset;
    int64_t cubeMOuterOffset;

    // LSE output offset
    int64_t softmaxLseOffset;       // Determined by the sOuter layer

    int64_t qSNumInOneBlock;
    int64_t kvLoopEndIdx;
};

#define COMMON_RUN_INFO \
    int64_t s2StartIdx; /* Starting S2 position, which may be nonzero for sparse attention */ \
    int64_t s2EndIdx; \
    int64_t s2LoopCount; /* Current S2-loop index */ \
    int64_t s2LoopLimit; \
    int64_t s1oIdx = 0; /* S1-axis index */ \
    int64_t loop = 0; /* for v0 perload loop */ \
    int64_t boIdx = 0; /* B-axis index */ \
    int64_t n2oIdx = 0; /* N2-axis index */ \
    int64_t goIdx = 0; /* G-axis index */ \
    int32_t s1RealSize; \
    int32_t halfS1RealSize; /* Actual S1 base-block size on Vector; if the Cube base block is 128, halfS1RealSize is 64 */ \
    int32_t firstHalfS1RealSize; /* When s1RealSize is odd, v0 computes one fewer row than v1; use v0's S1 size for the subblock offset */ \
    int32_t mRealSize; \
    int32_t halfMRealSize; \
    int32_t firstHalfMRealSize; \
    int32_t s2RealSize; /* Actual length of the S2 base block */ \
    int64_t s2AlignedSize; /* S2 base-block length after alignment to 16 */ \
    int32_t vec2S1BaseSize; /* S1 size after Vector2 loop partitioning, for example splitting 64 into two blocks of 32 */ \
    int32_t vec2S1RealSize; /* Tail S1 size after Vector2 loop partitioning; splitting 63 into 32 and 31 gives a tail size of 31 */ \
    int32_t vec2MBaseSize; \
    int32_t vec2MRealSize; \
    int64_t taskId; \
    int64_t multiCoreInnerIdx = 0; \
    int64_t attentionOutOffset; \
    int32_t actualS1Size; /* Total s1Size for non-TND; current batch S1 for TND */ \
    int32_t actualS2Size; /* Total s2Size for non-TND; current batch S2 for TND */ \
    int64_t preTokensPerBatch; /* Vector2 preTokens at the upper-left vertex */ \
    int64_t nextTokensPerBatch; /* Vector2 nextTokens at the upper-left vertex */ \
    uint8_t taskIdMod2; \
    uint8_t taskIdMod3; \
    uint8_t multiCoreIdxMod2 = 0; \
    uint8_t multiCoreIdxMod3 = 0; \
    int64_t sOuterOffset; \
    int64_t mOuterOffset; \
    int64_t queryOffset; \
    int64_t queryRopeOffset

struct RunInfo {
    COMMON_RUN_INFO;
    // Added for inference
    // LSE output offset
    int64_t softmaxLseOffset;

    int64_t qSNumInOneBlock;
    int64_t kvLoopEndIdx;
};

#define COMMON_CONST_INFO \
    /* Global base-block information */ \
    uint32_t bSize; \
    uint32_t needInit; \
    uint32_t s1BaseSize; \
    uint32_t s2BaseSize; \
    int64_t dSize; /* query d 512 */ \
    int64_t dSizeV; /* key d 512 */ \
    int64_t dSizeVInput; /* key inpue d 656 = rope + nope + scale + pad */ \
    int64_t dSizeNope; /* key nope d 448 */ \
    int64_t dSizeRope; /* key rope d 64 */ \
    int64_t tileSize; /* 64 */ \
    int64_t sparseMode; \
    int64_t gSize; /* G-axis size */ \
    int64_t n2Size; \
    int64_t s1Size; /* Total S1 size */ \
    int64_t s2Size; /* Total S2 size */ \
    /* Axis products */ \
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
    bool returnSoftmaxLse; \
    /* Matmul strided-read parameters */ \
    int64_t mm1Ka; \
    /* Stride of dQ or attentionOut */ \
    int64_t attentionOutStride; \
    uint32_t aivIdx; \
    uint8_t layoutType; \
    uint8_t subBlockIdx;\
    /* Core-partition information */ \
    uint32_t s2Start; \
    uint32_t s2End; \
    uint32_t bN2Start; \
    uint32_t bN2End; \
    uint32_t gS1Start; \
    uint32_t gS1End

#define INFER_CONST_INFO \
    /* Inference */ \
    bool isActualLenDimsNull; /* Whether actualseq is provided */ \
    bool isActualLenDimsKVNull; /* Whether actualseq_kv is provided */ \
    bool isSoftmaxLseEnable; \
    bool rsvd1; \
    uint32_t sparseBlockCount; \
    uint32_t actualSeqLenSize; /* Length of the user-provided actualseq */ \
    uint32_t actualSeqLenKVSize; /* Length of the user-provided actualseq_kv */ \
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
    float softmaxScale

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
    int64_t dSizeVInput : 12;  \
    uint32_t needInit : 4; \
    uint32_t layoutType : 4;  \
    uint32_t isActualSeqLengthsNull : 1; \
    uint32_t isActualSeqLengthsKVNull : 1; \
    uint32_t sparseBlockCount; \
    float softmaxScale; \
    uint32_t cmpRatio : 9; \
    uint32_t dSizeRope : 11; \
    uint32_t oriMaskMode : 6; \
    uint32_t cmpMaskMode : 6; \
    int32_t oriWinLeft; \
    int32_t oriWinRight; \
    uint32_t tileSize : 8; \
    /* pa params */  \
    uint32_t oriBlockSize : 12; \
    uint32_t cmpBlockSize : 12; \
    uint32_t oriMaxBlockNumPerBatch; \
    uint32_t cmpMaxBlockNumPerBatch; \
    uint32_t usedCoreNum; \
    bool returnSoftmaxLse

struct ConstInfo {
    COMMON_CONST_INFO;
    INFER_CONST_INFO;
};

/* only support b32 or b64 */
struct CVSharedParams {
    CV_SHARED_PARAMS;
};
}

#endif // UTIL_REGBASE_H
