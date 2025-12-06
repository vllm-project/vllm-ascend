// Adapted from
//   https://gitee.com/ascend/ascend-transformer-boost.git
//   https://gitee.com/ascend/op-plugin.git
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//

#include <fstream>
#include <iostream>
#include <math.h>
#include <stdexcept>
#include "acl/acl.h"
// #include "defines.h"
// #include "torch_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/mla_preprocess_tiling.h"

// #include "aclrtlaunch_mla_preprocess.h"

// namespace sglang {
namespace mlapo {

constexpr uint32_t DIM_2 = 2;

constexpr uint32_t AXES_ALIGN_SIZE = 512;
constexpr uint32_t BASE_BLOCK_STEP = 2;
constexpr uint32_t CONST_16 = 16;
constexpr uint32_t CONST_32 = 32;
constexpr uint32_t CONST_128 = 128;
constexpr uint32_t CONST_256 = 256;
constexpr uint32_t CONST_512 = 512;
constexpr uint32_t L1_BUFFER_SIZE = 524288;
constexpr uint32_t L1_PINGPONG_BUFFER_LEN = 262144;
constexpr uint32_t L0AB_PINGPONG_BUFFER_LEN = 131072;
constexpr uint32_t L1_SCALE_SIZE = 4096;
constexpr uint32_t L1_BIAS_SIZE = 2048;
constexpr uint32_t L0C_SIZE = 128 * 1024;
constexpr uint32_t CONCAT_SIZE = 512;

constexpr uint32_t HIDDEN_STRATE = 7168;
constexpr uint32_t HIDDEN_STRATE_ROPE = 192;
constexpr uint32_t HIDDEN_STRATE_MM = 2112;
constexpr uint32_t HIDDEN_STRATE_RMS = 1536;
constexpr uint32_t UB_SIZE = 196352;
constexpr uint32_t HEADDIM = 64;
constexpr uint32_t FP32_REPEAT_MASK = 64;
constexpr uint32_t FP16_REPEAT_MASK = 128;

constexpr int32_t NUM1 = 1;
constexpr int32_t NUM2 = 2;
constexpr int32_t NUM3 = 3;
constexpr int32_t NUM4 = 4;
constexpr int32_t NUM8 = 8;
constexpr uint32_t INDEX_WDQKV = 5;
constexpr uint32_t INDEX_WUQ = 18;
constexpr uint32_t INDEX_WUK = 20;

constexpr uint32_t MAX_SUPPORT_TOKEN_NUMS = 1024;

inline uint32_t CeilDiv(const uint32_t dividend, const uint32_t divisor)
{
    if (divisor == 0) {
        return UINT32_MAX;
    }
    return (dividend + divisor - 1) / divisor;
}

inline uint32_t RoundUp(const uint32_t val, const uint32_t align = 16)
{
    if (align == 0) {
        return 0;
    }
    return (val + align - 1) / align * align;
}

inline uint32_t RoundDown(const uint32_t val, const uint32_t align = 16)
{
    if (align == 0) {
        return 0;
    }
    return val / align * align;
}

template <typename T = uint32_t>
inline T Max(const T a, const T b)
{
    return a > b ? a : b;
}

template <typename T = uint32_t>
inline T Min(const T a, const T b)
{
    return a < b ? a : b;
}

struct MlaPreprocess {
    enum class QuantMode : int32_t {
        PER_TENSOR_ASYMM_QUANT = 0,
        PER_TOKEN_SYMM_QUANT,
        PER_TOKEN_ASYMM_QUANT,
        NO_QUANT
    };
};
using QuantMode = MlaPreprocess::QuantMode;

struct PlatformInfo {
    uint32_t coreNum;
    uint32_t coreNumAic;
    uint32_t coreNumAiv;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l2Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
};

struct OpParam {
    uint32_t N;
    uint32_t headNum;
    int32_t cacheMode;
    QuantMode quantMode;
    caffe2::TypeMeta inDtype;
    bool enableInnerOut;
};

class PpMatmulTilingApi
{
public:
    PpMatmulTilingApi(struct PlatformInfo &platformInfo, uint32_t numBatch, uint32_t m, uint32_t k, uint32_t n,
                      bool transA, bool transB, bool enDequant, bool deqOnTheFly)
        : platformInfo_(platformInfo),
          numBatch_(numBatch),
          m_(m),
          k_(k),
          n_(n),
          transA_(transA),
          transB_(transB),
          enDequant_(enDequant),
          deqOnTheFly_(deqOnTheFly)
    {
        inDataSize_ = enDequant ? sizeof(uint8_t) : sizeof(uint16_t);
    }
    void GetTilingData(PpMatmulTilingData &tiling);

private:
    void GetTileSize();
    float GetCost(const uint32_t m0, const uint32_t n0);
    void UpdateTileSize(const uint32_t m0, const uint32_t n0);
    void Swizzle();
    uint32_t ComputeL1AbSize();
    uint32_t ComputeK0ForABpingpong(uint32_t l1AbSize);
    bool IsLoadAllAmat(uint32_t l1AbSize);
    uint32_t ComputeK0ForOnlyBpingpong(uint32_t l1AbSize);

private:
    uint32_t numBatch_{0};
    uint32_t m_{0};
    uint32_t k_{0};
    uint32_t n_{0};
    uint32_t m0_{0};
    uint32_t k0_{0};
    uint32_t n0_{0};
    uint32_t mLoop_{0};
    uint32_t kLoop_{0};
    uint32_t nLoop_{0};
    uint32_t coreLoop_{0};
    uint32_t swizzleCount_{0};
    uint32_t blockDim_{0};
    uint32_t swizzleDirect_{0};
    uint32_t inDataSize_{0};
    uint32_t b0matPingPongBufferLen_{L1_PINGPONG_BUFFER_LEN};
    bool transA_{false};
    bool transB_{false};
    bool enDequant_{false};
    bool enShuffleK_{false};
    bool enLoadAllAmat_{false};
    bool deqOnTheFly_{false};

    struct PlatformInfo platformInfo_;
};

void PpMatmulTilingApi::GetTilingData(PpMatmulTilingData &tiling)
{
    GetTileSize();
    tiling.numBatch = numBatch_;
    tiling.m = m_;
    tiling.k = k_;
    tiling.n = n_;
    tiling.m0 = m0_;
    tiling.k0 = k0_;
    tiling.n0 = n0_;
    tiling.mLoop = mLoop_;
    tiling.kLoop = kLoop_;
    tiling.nLoop = nLoop_;
    tiling.coreLoop = coreLoop_;
    tiling.swizzleCount = swizzleCount_;
    tiling.swizzleDirect = swizzleDirect_;
    tiling.enShuffleK = static_cast<uint32_t>(enShuffleK_);
    tiling.blockDim = blockDim_;
    tiling.enLoadAllAmat = static_cast<uint32_t>(enLoadAllAmat_);
    tiling.b0matPingPongBufferLen = b0matPingPongBufferLen_;
}

void PpMatmulTilingApi::GetTileSize()
{
    bool priFlag = !(m_ < n_);
    uint32_t roundBase = pow(2, ceil(log(CeilDiv(priFlag ? n_ : m_, CONST_16)))) * CONST_16;
    uint32_t priAxes = RoundUp(priFlag ? m_ : n_, CONST_16);
    uint32_t subAxes = RoundUp(priFlag ? n_ : m_, roundBase);
    float minCost = __FLT_MAX__;
    uint32_t maxAxes0 = AXES_ALIGN_SIZE;
    uint32_t maxPriAxes0 = Min(maxAxes0, priAxes);
    uint32_t maxSubAxes0 = Min(maxAxes0, subAxes);
    for (uint32_t priAxes0 = CONST_16; priAxes0 <= maxPriAxes0; priAxes0 *= BASE_BLOCK_STEP) {
        for (uint32_t subAxes0 = CONST_16; subAxes0 <= maxSubAxes0; subAxes0 *= BASE_BLOCK_STEP) {
            if (priAxes0 * subAxes0 * sizeof(float) > platformInfo_.l0cSize) {
                continue;
            }
            uint32_t newM0 = priFlag ? priAxes0 : subAxes0;
            uint32_t newN0 = priFlag ? subAxes0 : priAxes0;
            if (newN0 > CONST_256 && enDequant_) {
                continue;
            }
            float cost = GetCost(newM0, newN0);
            if (cost < minCost) {
                minCost = cost;
                UpdateTileSize(newM0, newN0);
            }
        }
    }

    Swizzle();

    uint32_t l1AbSize = ComputeL1AbSize();
    k0_ = ComputeK0ForABpingpong(l1AbSize);
    kLoop_ = CeilDiv(k_, k0_);
}

uint32_t PpMatmulTilingApi::ComputeK0ForOnlyBpingpong(uint32_t l1AbSize)
{
    enLoadAllAmat_ = true;
    b0matPingPongBufferLen_ = static_cast<uint32_t>(
        static_cast<float>((l1AbSize - RoundUp(m_, CONST_16) * RoundUp(k_, CONST_32) * inDataSize_) / DIM_2));
    uint32_t k0MaxB0 =
        static_cast<uint32_t>(static_cast<float>(b0matPingPongBufferLen_ / (RoundUp(n0_, CONST_16) * inDataSize_)));
    uint32_t k0B0 = k0MaxB0 < CONST_512 ? RoundDown(k0MaxB0, CONST_32) : RoundDown(k0MaxB0, CONST_512);
    return k0B0 > CONST_512 ? RoundDown(k0B0, CONST_512) : k0B0;
}

bool PpMatmulTilingApi::IsLoadAllAmat(uint32_t l1AbSize)
{
    return (coreLoop_ > blockDim_) && enDequant_ && (kLoop_ > 1) &&
           (l1AbSize > RoundUp(m_, CONST_16) * RoundUp(k_, CONST_32) * inDataSize_) && (mLoop_ == 1);
}

uint32_t PpMatmulTilingApi::ComputeK0ForABpingpong(uint32_t l1AbSize)
{
    uint32_t k0Max = static_cast<uint32_t>(static_cast<float>(l1AbSize / DIM_2) / ((m0_ + n0_) * inDataSize_));
    uint32_t tmpK0;
    if (enDequant_) {
        tmpK0 = k0Max < CONST_512 ? RoundDown(k0Max, CONST_32) : RoundDown(k0Max, CONST_512);
    } else {
        tmpK0 = k0Max < CONST_256 ? RoundDown(k0Max, CONST_16) : RoundDown(k0Max, CONST_256);
    }
    if (tmpK0 > CONST_512) {
        tmpK0 = RoundDown(tmpK0, CONST_512);
    }
    return tmpK0;
}

uint32_t PpMatmulTilingApi::ComputeL1AbSize()
{
    if (enDequant_ && deqOnTheFly_) {
        return L1_BUFFER_SIZE;
    }
    return enDequant_ ? (L1_BUFFER_SIZE - L1_BIAS_SIZE - L1_SCALE_SIZE) : L1_BUFFER_SIZE;
}

float PpMatmulTilingApi::GetCost(const uint32_t m0, const uint32_t n0)
{
    float aCoef = 1.0;
    float bCoef = 1.0;
    float bwCoef = 5.0;
    uint32_t mLoop = CeilDiv(m_, m0);
    uint32_t nLoop = CeilDiv(n_, n0);
    if (mLoop == 0 || nLoop == 0) {
        return __FLT_MAX__;
    }
    uint32_t rqdNumCore = numBatch_ * mLoop * nLoop;
    uint32_t blockDim = Min(rqdNumCore, platformInfo_.coreNumAic);
    uint32_t mOnce = blockDim < nLoop ? m0 : blockDim / nLoop * m0;
    uint32_t nOnce = blockDim < nLoop ? platformInfo_.coreNumAic * n0 : n_;
    if (mOnce * k_ * sizeof(uint16_t) > platformInfo_.l2Size) {
        aCoef = bwCoef;
    }
    if (nOnce * k_ * sizeof(uint16_t) > platformInfo_.l2Size) {
        bCoef = bwCoef;
    }
    if (transA_ && m0 % CONST_256 == 0) {
        aCoef *= NUM2;
    }
    if (!transB_ && n0 % CONST_256 == 0) {
        bCoef *= NUM2;
    }
    return 1 / (aCoef * static_cast<float>(n0)) + 1 / (bCoef * static_cast<float>(m0));
}

void PpMatmulTilingApi::UpdateTileSize(const uint32_t m0, const uint32_t n0)
{
    m0_ = m0;
    n0_ = n0;
    mLoop_ = CeilDiv(m_, m0_);
    nLoop_ = CeilDiv(n_, n0_);
    coreLoop_ = numBatch_ * mLoop_ * nLoop_;
    const uint32_t maxNumCubeCore = platformInfo_.coreNumAic;
    if (mLoop_ == 1 && transB_ && coreLoop_ % maxNumCubeCore < maxNumCubeCore / NUM4 * NUM3) {
        uint32_t tmpM0 = RoundUp(m_, CONST_16);
        uint32_t maxN0 = L0C_SIZE / (tmpM0 * sizeof(float));
        if (enDequant_) {
            maxN0 = maxN0 < CONST_256 ? maxN0 : CONST_256;
        }
        uint32_t x = CeilDiv(n_, maxNumCubeCore);
        uint32_t y = CeilDiv(x, maxN0);
        uint32_t tmpN0 = RoundUp(CeilDiv(x, y), CONST_16);
        uint32_t rqdL0cSize = tmpM0 * tmpN0 * sizeof(float);
        if (rqdL0cSize < L0C_SIZE && (tmpM0 + tmpN0) * CONST_256 * inDataSize_ < L1_BUFFER_SIZE) {
            m0_ = tmpM0;
            n0_ = tmpN0;
            nLoop_ = CeilDiv(n_, n0_);
            coreLoop_ = numBatch_ * nLoop_;
        }
    }
    blockDim_ = Min(coreLoop_, maxNumCubeCore);
}

void PpMatmulTilingApi::Swizzle()
{
    float minCost = m_ * k_ + k_ * n_;
    for (uint32_t i = 1; i <= blockDim_; ++i) {
        int c = static_cast<int32_t>((blockDim_ + i - 1) / i);
        float cost;
        // B0 + A < A0 + B
        if (i * n0_ + m_ < m0_ * c + n_) {
            swizzleDirect_ = 1;  // Nz
            cost = n0_ * i + m0_ * c;
            if (cost <= minCost) {
                minCost = cost;
                swizzleCount_ = i;
            }
        } else {
            swizzleDirect_ = 0;  // Zn
            cost = m0_ * i + n0_ * c;
            if (cost < minCost) {
                minCost = cost;
                swizzleCount_ = i;
            }
        }
    }
}

class MlaPreprocessTiling
{
public:
    MlaPreprocessTiling(struct PlatformInfo &platformInfo, struct OpParam &opParam, MlaTilingData *tilingData)
    {
        this->tilingData = tilingData;
        this->platformInfo = platformInfo;
        this->opParam = opParam;
    }
    void Init();

    void RmsNormQuantTiling();
    void RopeConcatTiling();
    void EinSumQuantTiling();

    void SetTilingKey();
    void SetMlapoWorkSpace();

private:
    MlaTilingData *tilingData;
    struct PlatformInfo platformInfo;
    struct OpParam opParam;
};

void MlaPreprocessTiling::RmsNormQuantTiling()
{
    tilingData->rmsNumCore1 = platformInfo.coreNumAiv;
    tilingData->rmsNumCol1 = HIDDEN_STRATE;
    tilingData->rmsNumRow1 = opParam.N;
    tilingData->rmsQuantMin1 = -CONST_128;
    tilingData->rmsNumCore2 = platformInfo.coreNumAiv;
    tilingData->rmsNumCol2 = HIDDEN_STRATE_MM;
    tilingData->rmsNumRow2 = opParam.N;
    tilingData->rmsQuantMin2 = -CONST_128;
}

void MlaPreprocessTiling::RopeConcatTiling()
{
    uint32_t ntokens = opParam.N;
    uint32_t hiddenSizeQ = HEADDIM * opParam.headNum;
    uint32_t headDim = HEADDIM;
    uint32_t headNumQ = hiddenSizeQ / headDim;
    uint32_t concatSize = CONCAT_SIZE;
    uint32_t maxCore = platformInfo.coreNumAiv;
    uint32_t maxUbSize = platformInfo.ubSize;

    uint32_t allHeadNum = ntokens * headNumQ;

    uint32_t tempCore = (allHeadNum + maxCore - 1) / maxCore;
    uint32_t realCore = (allHeadNum + tempCore - 1) / tempCore;   // Actual number of the core for operation
    uint32_t nlCoreRun = (allHeadNum + realCore - 1) / realCore;  // The number of heads in the front core
    uint32_t lCoreRun = allHeadNum - (realCore - 1) * nlCoreRun;  // The number of heads in the tail core

    uint32_t dataTypeSize = 2;

    // Calculate how many lines can be moved at a time. q 4+2、reverseq 4、neg 4、sin 4+2、cos 4+2  + concat 2
    uint32_t allSize =
        headDim * (3 * (4 + dataTypeSize) + 2 * 4) + concatSize * dataTypeSize;  // lift precision calculation of ROPE
    uint32_t maxNPerLoopForUb = maxUbSize / allSize;  // the maximum number of rows at a time for UB
    uint32_t preCoreLoopTime = (nlCoreRun + maxNPerLoopForUb - 1) / maxNPerLoopForUb;  // Number of cycles of front core
    uint32_t preCoreLoopNLast =
        nlCoreRun -
        (preCoreLoopTime - 1) * maxNPerLoopForUb;  // rows of data processed in the last batch of the front core
    uint32_t lastCoreLoopTime = (lCoreRun + maxNPerLoopForUb - 1) / maxNPerLoopForUb;  // Number of cycles of tail core
    uint32_t lastCoreLoopNLast =
        lCoreRun -
        (lastCoreLoopTime - 1) * maxNPerLoopForUb;  // rows of data processed in the last batch of the tail core

    tilingData->hiddenSizeQ = hiddenSizeQ;
    tilingData->headNumQ = headNumQ;
    tilingData->headDim = headDim;
    tilingData->concatSize = concatSize;
    tilingData->rotaryCoeff = NUM2;
    tilingData->ntokens = ntokens;
    tilingData->realCore = realCore;
    tilingData->nlCoreRun = nlCoreRun;
    tilingData->lCoreRun = nlCoreRun;
    tilingData->maxNPerLoopForUb = maxNPerLoopForUb;
    tilingData->preCoreLoopTime = preCoreLoopTime;
    tilingData->preCoreLoopNLast = preCoreLoopNLast;
    tilingData->lastCoreLoopTime = lastCoreLoopTime;
    tilingData->lastCoreLoopNLast = lastCoreLoopNLast;
}

void MlaPreprocessTiling::EinSumQuantTiling()
{
    uint32_t aivCore = platformInfo.coreNumAiv;
    uint32_t ubSize = UB_SIZE - 1024;

    // input shape
    uint32_t esqBatch = opParam.N;          // tokenNum
    uint32_t esqHeadNum = opParam.headNum;  // headNum
    uint32_t esqColNum = AXES_ALIGN_SIZE;   // 512

    // split core
    uint32_t esqFrontCore = esqBatch % aivCore;
    uint32_t esqTailCore = aivCore - esqFrontCore;
    uint32_t esqFrontCoreBatch = CeilDiv(esqBatch, aivCore);
    uint32_t esqTailCoreBatch = esqBatch / aivCore;

    // split ub --> calc H' <-- The number of rows handled in a UB cycle.
    uint32_t splitFactor = 0;
    uint32_t esqHeadPerLoop = 0;  // The number of head rows per UB calculation
    uint32_t repeatMask = 0;

    if (opParam.inDtype == at::kBFloat16 || opParam.quantMode == QuantMode::PER_TOKEN_SYMM_QUANT) {
        // Move scales in at once, broadcast, and cache them all H * 32bytes
        uint32_t scaleUb = RoundUp(esqHeadNum) * CONST_32;
        // bf16 input [H', colNum](f16 + fp32 + int8), ub reuse
        splitFactor = esqColNum * (sizeof(uint16_t) + sizeof(float) + sizeof(uint8_t));
        splitFactor *= NUM2;
        esqHeadPerLoop = (ubSize - scaleUb) / splitFactor;  // 26
        repeatMask = FP32_REPEAT_MASK;
    } else {
        // fp16 input [H', cloNum](fp16*2 + int8) + [H', 1](fp16) + [H', 16](fp16)
        splitFactor =
            esqColNum * (NUM2 * sizeof(uint16_t) + sizeof(uint8_t)) + sizeof(uint16_t) + (CONST_16 * sizeof(uint16_t));
        esqHeadPerLoop = ubSize / splitFactor;
        repeatMask = FP16_REPEAT_MASK;
        esqHeadPerLoop = RoundDown(esqHeadPerLoop);
    }
    uint32_t esqUbHeadLoop = esqHeadNum / esqHeadPerLoop;  // UB complete cycles
    uint32_t esqHeadTail = esqHeadNum % esqHeadPerLoop;    // The number of rows that UB last processed the head.
    uint32_t esqColLoop = esqColNum / repeatMask;  // Each row counts the number of times to cycle through columns.
    uint32_t esqColTail =
        esqColNum % repeatMask;  // colNum is not 64/128 aligned, the number of columns is calculated last.

    tilingData->esqFrontCore = esqFrontCore;
    tilingData->esqTailCore = esqTailCore;
    tilingData->esqFrontCoreBatch = esqFrontCoreBatch;
    tilingData->esqTailCoreBatch = esqTailCoreBatch;
    tilingData->esqHeadNum = esqHeadNum;
    tilingData->esqColNum = esqColNum;
    tilingData->esqUbHeadLoop = esqUbHeadLoop;
    tilingData->esqHeadPerLoop = esqHeadPerLoop;
    tilingData->esqHeadTail = esqHeadTail;
    tilingData->esqColLoop = esqColLoop;
    tilingData->esqColTail = esqColTail;
}

void MlaPreprocessTiling::SetMlapoWorkSpace()
{
    uint64_t s1wsFactor =
        static_cast<uint64_t>(opParam.cacheMode == 2 ? std::max(HIDDEN_STRATE * sizeof(int8_t),
                                                                opParam.headNum * AXES_ALIGN_SIZE * sizeof(uint16_t))
                                                     : HIDDEN_STRATE * sizeof(int8_t));
    uint64_t workSizeS1 = s1wsFactor;
    uint64_t workSizeS2 = opParam.headNum * HIDDEN_STRATE_ROPE * sizeof(uint16_t);
    uint64_t workSizeS3 = HIDDEN_STRATE_MM * sizeof(uint16_t);
    uint64_t workSizeS4 = std::max(opParam.headNum * HIDDEN_STRATE_ROPE, HIDDEN_STRATE_MM) * sizeof(uint32_t);

    uint64_t maxWorkspaceSize = workSizeS1;
    maxWorkspaceSize = std::max(maxWorkspaceSize, workSizeS2);
    maxWorkspaceSize = std::max(maxWorkspaceSize, workSizeS3);
    maxWorkspaceSize = std::max(maxWorkspaceSize, workSizeS4);
    maxWorkspaceSize *= static_cast<uint64_t>(opParam.N);

    uint64_t pertokenWorkspace = static_cast<uint64_t>(opParam.N) * sizeof(float) * 2;

    uint64_t userWorkspaceSize;
    if (opParam.inDtype == at::kBFloat16 || opParam.quantMode == QuantMode::PER_TOKEN_SYMM_QUANT) {
        userWorkspaceSize = 4 * maxWorkspaceSize + pertokenWorkspace;
    } else {
        userWorkspaceSize = 3 * maxWorkspaceSize;
    }

    tilingData->userWorkspaceSize = userWorkspaceSize;
    tilingData->s1Offset = 0;
    tilingData->s2Offset = tilingData->s1Offset + maxWorkspaceSize;
    tilingData->s3Offset = tilingData->s2Offset + maxWorkspaceSize;
    tilingData->s4Offset = tilingData->s3Offset + maxWorkspaceSize;
    tilingData->s5Offset = tilingData->s4Offset + maxWorkspaceSize;
}

void MlaPreprocessTiling::SetTilingKey()
{
    uint64_t tilingKey = (static_cast<uint64_t>(opParam.enableInnerOut)) << 9;
    tilingKey |= (static_cast<uint64_t>(opParam.inDtype == at::kBFloat16)) << 8;

    tilingKey |= static_cast<uint64_t>(opParam.cacheMode);
    tilingKey |= (static_cast<uint64_t>(opParam.quantMode) << 3);

    tilingData->tilingKey = tilingKey;
}

void MlaPreprocessTiling::Init()
{
    tilingData->numCore = platformInfo.coreNumAic;
    tilingData->n = opParam.N;

    bool deqOnTheFly = false;
    if (opParam.inDtype == at::kBFloat16 || opParam.quantMode == QuantMode::PER_TOKEN_SYMM_QUANT) {
        deqOnTheFly = true;
    }

    PpMatmulTilingApi mm1TilingApi(platformInfo,
                                   1,                 // numBatch
                                   opParam.N,         // m
                                   HIDDEN_STRATE,     // k
                                   HIDDEN_STRATE_MM,  // n
                                   false,             // transA
                                   true,              // transB
                                   true,              // enDequant
                                   deqOnTheFly);      // in bf16.cce?
    mm1TilingApi.GetTilingData(tilingData->mm1);

    PpMatmulTilingApi mm2TilingApi(platformInfo,
                                   1,                                     // numBatch
                                   opParam.N,                             // m
                                   HIDDEN_STRATE_RMS,                     // k
                                   opParam.headNum * HIDDEN_STRATE_ROPE,  // n
                                   false,                                 // transA
                                   true,                                  // transB
                                   true,                                  // enDequant
                                   deqOnTheFly);                          // in bf16.cce?
    mm2TilingApi.GetTilingData(tilingData->mm2);

    PpMatmulTilingApi mm3TilingApi(platformInfo,
                                   opParam.headNum,  // numBatch
                                   opParam.N,        // m
                                   CONST_128,        // k
                                   CONCAT_SIZE,      // n
                                   false,            // transA
                                   false,            // transB
                                   false,            // enDequant
                                   deqOnTheFly);     // in bf16.cce?
    mm3TilingApi.GetTilingData(tilingData->mm3);

    RmsNormQuantTiling();
    RopeConcatTiling();
    EinSumQuantTiling();

    SetMlapoWorkSpace();
    SetTilingKey();

    return;
}

std::unordered_map<c10::string_view, uint16_t> cache_mode_map = {
    {"krope_ctkv", 1}, {"int8_nzcache", 2}, {"nzcache", 3}};

std::unordered_map<c10::string_view, uint16_t> quant_mode_map = {
    {"per_tensor_quant_asymm", 0},
    {"per_token_quant_symm", 1},
};

template <typename MapType>
inline int get_op_mode(const MapType &mode_map, c10::optional<c10::string_view> mode_opt, c10::string_view default_mode,
                       const char *mode_name)
{
    c10::string_view mode_str = mode_opt.value_or(default_mode);
    auto it = mode_map.find(mode_str);
    TORCH_CHECK(it != mode_map.end(), "Unsupported ", mode_name, " value: '", mode_str, "'");
    return it->second;
}

std::tuple<at::Tensor, at::Tensor, uint32_t> mla_preprocess_tiling(
    const at::Tensor &hiddenState,
    const at::Tensor &wuk,
    c10::optional<c10::string_view> cache_mode,
    c10::optional<c10::string_view> quant_mode,
    bool enable_inner_out
)
{
    auto cacheMode = get_op_mode(cache_mode_map, cache_mode, "krope_ctkv", "cache_mode");
    auto quantMode = get_op_mode(quant_mode_map, quant_mode, "per_token_quant_symm", "quant_mode");

    platform_ascendc::PlatformAscendC *platformAscendC = platform_ascendc::PlatformAscendCManager::GetInstance();

    struct PlatformInfo platformInfo;
    platformInfo.coreNum = platformAscendC->GetCoreNum();
    platformInfo.coreNumAic = platformAscendC->GetCoreNumAic();
    platformInfo.coreNumAiv = platformAscendC->GetCoreNumAiv();
    platformAscendC->GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformInfo.ubSize);
    platformAscendC->GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo.l1Size);
    platformAscendC->GetCoreMemSize(platform_ascendc::CoreMemType::L2, platformInfo.l2Size);
    platformAscendC->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo.l0aSize);
    platformAscendC->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo.l0bSize);
    platformAscendC->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo.l0cSize);

    int32_t N = hiddenState.sizes()[0];
    int32_t headNum = wuk.sizes()[0];

    OpParam opParam;
    opParam.N = N;
    opParam.headNum = headNum;
    opParam.cacheMode = static_cast<int32_t>(cacheMode);
    opParam.quantMode = static_cast<QuantMode>(quantMode);
    opParam.inDtype = hiddenState.options().dtype();
    opParam.enableInnerOut = enable_inner_out;

    MlaTilingData tilingData;
    MlaPreprocessTiling mlaTiling(platformInfo, opParam, &tilingData);

    mlaTiling.Init();
    uint32_t blockDim = platformInfo.coreNumAic;

    // workspace
    uint64_t system_workspace_size = static_cast<uint64_t>(platformAscendC->GetLibApiWorkSpaceSize());
    uint64_t workspace_size = system_workspace_size + tilingData.userWorkspaceSize;
    auto options = at::TensorOptions().dtype(at::kByte).device(hiddenState.options().device());
    auto workspace_tensor = at::empty({static_cast<int64_t>(workspace_size)}, options);

    // tiling
    int32_t bIndex = N - 1;
    uint32_t tilingSize = sizeof(MlaTilingData);
    static auto global_tiling_data =
        at::empty({tilingSize * MAX_SUPPORT_TOKEN_NUMS},
                  at::TensorOptions().dtype(at::kByte).device(hiddenState.options().device()));
    if (bIndex >= 0 && bIndex < MAX_SUPPORT_TOKEN_NUMS) {
        aclrtMemcpy(global_tiling_data.data_ptr<uint8_t>() + (tilingSize * bIndex), tilingSize, &tilingData, tilingSize,
                    ACL_MEMCPY_HOST_TO_DEVICE);
    } else {
        // Handle the case where bIndex is out of range
        TORCH_CHECK(false, "bIndex is out of range: ", bIndex);
    }
    at::Tensor tiling = at::from_blob(
        global_tiling_data.data_ptr<uint8_t>() + (tilingSize * bIndex),
        tilingSize,
        at::kByte);

    return std::make_tuple(workspace_tensor, tiling, blockDim);
}

}  // namespace npu_kernel
