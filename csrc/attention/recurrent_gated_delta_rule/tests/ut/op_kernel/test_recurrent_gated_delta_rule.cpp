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
 * \file test_recurrent_gated_delta_rule.cpp
 *
 */
#include <array>
#include <vector>
#include <gtest/gtest.h>
#include <cstring>
#include <iostream>
#include <string>
#include <cstdint>
#include <unistd.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#endif

using namespace std;

struct alignas(8) RecurrentGatedDeltaRuleTilingData {
    uint32_t vectorCoreNum;
    uint32_t ubCalSize;
    uint32_t ubRestBytes;
    uint32_t t;
    uint32_t nk;
    uint32_t dk;
    uint32_t nv;
    uint32_t dv;
    uint32_t sBlockNum;
    uint32_t b;
    uint32_t vStep;
    uint32_t stateOutBufferNum;
    uint32_t attnOutBufferNum;
    float scale;
    uint32_t hasGama;
    uint32_t hasGamaK;
    uint32_t hasAcceptedTokens;
};

extern "C" __global__ __aicore__ void
recurrent_gated_delta_rule(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR state, GM_ADDR cuSeqlens,
                           GM_ADDR ssmStateIndices, GM_ADDR g, GM_ADDR gk, GM_ADDR numAcceptedTokens, GM_ADDR out,
                           GM_ADDR stateOut, GM_ADDR workspaceGM, GM_ADDR tilingGM);

template <typename T>
T* GmAllocWrapper(size_t size) {
    T *ptr = reinterpret_cast<T *>(AscendC::GmAlloc(size));
    assert(ptr != nullptr && "GM allocation failed");
    return ptr; 
}

void InitTilingData(RecurrentGatedDeltaRuleTilingData* tilingData, uint32_t b, uint32_t t, uint32_t nk, uint32_t nv,
                    uint32_t dk, uint32_t dv, uint32_t hasGama, uint32_t hasGamaK, uint32_t hasAcceptedTokens,
                    uint32_t vStep = 32, uint32_t stateOutBufferNum = 1, uint32_t attnOutBufferNum = 1) {
    tilingData->vectorCoreNum = 8;
    tilingData->ubCalSize = 192 * 1024;
    tilingData->ubRestBytes = 96 * 1024;
    tilingData->t = t;
    tilingData->nk = nk;
    tilingData->dk = dk;
    tilingData->nv = nv;
    tilingData->dv = dv;
    tilingData->sBlockNum = t * nv;
    tilingData->b = b;
    tilingData->vStep = vStep;
    tilingData->stateOutBufferNum = stateOutBufferNum;
    tilingData->attnOutBufferNum = attnOutBufferNum;
    tilingData->scale = 1.0f;
    tilingData->hasGama = hasGama;
    tilingData->hasGamaK = hasGamaK;
    tilingData->hasAcceptedTokens = hasAcceptedTokens;
}


void InitInputData(uint8_t* stateGm, size_t shapeState,
                   uint8_t* queryGm, size_t shapeQ,
                   uint8_t* keyGm, size_t shapeK,
                   uint8_t* valueGm, size_t shapeV,
                   uint8_t* betaGm, size_t shapeBeta,
                   uint8_t* gamaGm, size_t shapeGama,
                   uint8_t* actSeqLenGM, size_t b,
                   uint8_t* ssmStaIdGm, size_t t,
                   uint8_t* numAccTokGm, size_t b_size,
                   uint8_t* attnOutGM, size_t shapeAttnOut) {
    memset(stateGm, 0, shapeState);
    memset(queryGm, 0, shapeQ);
    memset(keyGm, 0, shapeK);
    memset(valueGm, 0, shapeV);
    memset(betaGm, 0, shapeBeta);
    memset(gamaGm, 0, shapeGama);
    memset(attnOutGM, 0, shapeAttnOut);

    int32_t* actSeqLen = reinterpret_cast<int32_t*>(actSeqLenGM);
    for (size_t i = 0; i < b; ++i) {
        actSeqLen[i] = 2;
    }
    int32_t* ssmStaId = reinterpret_cast<int32_t*>(ssmStaIdGm);
    for (size_t i = 0; i < t; ++i) {
        ssmStaId[i] = static_cast<int32_t>(i);
    }
    int32_t* numAccTok = reinterpret_cast<int32_t*>(numAccTokGm);
    for (size_t i = 0; i < b_size; ++i) {
        numAccTok[i] = 1;
    }
}

struct RGDRTestParams {
    uint32_t hasGama;
    uint32_t hasGamaK;
    uint32_t hasAcceptedTokens;
    uint32_t dv;
    uint32_t vStep;
    uint32_t stateOutBufferNum;
    uint32_t attnOutBufferNum;
};

class RecurrentGatedDeltaRuleTest : public testing::TestWithParam<RGDRTestParams> {
protected:
    uint32_t b = 4;
    uint32_t mtp = 2;
    uint32_t t = b * mtp;       // 8
    uint32_t nk = 4;
    uint32_t nv = 4;
    uint32_t dk = 32;
    uint32_t dv = 32;

    size_t shapeState = 0;
    size_t shapeQ = 0;
    size_t shapeK = 0;
    size_t shapeV = 0;
    size_t shapeBeta = 0;
    size_t shapeGama = 0;
    size_t shapeActSeqLen = 0;
    size_t shapeSsmStaId = 0;
    size_t shapeNumAccTok = 0;
    size_t shapeAttnOut = 0;
    size_t allWorkspaceSize = 196608;
    size_t tilingSize = sizeof(RecurrentGatedDeltaRuleTilingData);

    uint8_t* stateGm = nullptr;
    uint8_t* queryGm = nullptr;
    uint8_t* keyGm = nullptr;
    uint8_t* valueGm = nullptr;
    uint8_t* betaGm = nullptr;
    uint8_t* gamaGm = nullptr;
    uint8_t* actSeqLenGM = nullptr;
    uint8_t* ssmStaIdGm = nullptr;
    uint8_t* numAccTokGm = nullptr;
    uint8_t* attnOutGM = nullptr;
    uint8_t* workspace = nullptr;
    uint8_t* tiling = nullptr;

    void SetUp() override {
        auto params = GetParam();
        dv = params.dv;
        shapeState = t * nv * dv * dk * sizeof(bfloat16_t);
        shapeQ = t * nk * dk * sizeof(bfloat16_t);
        shapeK = t * nk * dk * sizeof(bfloat16_t);
        shapeV = t * nv * dv * sizeof(bfloat16_t);
        shapeBeta = t * nv * sizeof(bfloat16_t);
        shapeGama = t * nv * sizeof(float);
        shapeActSeqLen = b * sizeof(int32_t);
        shapeSsmStaId = t * sizeof(int32_t);
        shapeNumAccTok = b * sizeof(int32_t);
        shapeAttnOut = t * nv * dv * sizeof(bfloat16_t);
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        stateGm = GmAllocWrapper<uint8_t>(shapeState);
        queryGm = GmAllocWrapper<uint8_t>(shapeQ);
        keyGm = GmAllocWrapper<uint8_t>(shapeK);
        valueGm = GmAllocWrapper<uint8_t>(shapeV);
        betaGm = GmAllocWrapper<uint8_t>(shapeBeta);
        gamaGm = GmAllocWrapper<uint8_t>(shapeGama);
        actSeqLenGM = GmAllocWrapper<uint8_t>(shapeActSeqLen);
        ssmStaIdGm = GmAllocWrapper<uint8_t>(shapeSsmStaId);
        numAccTokGm = GmAllocWrapper<uint8_t>(shapeNumAccTok);
        attnOutGM = GmAllocWrapper<uint8_t>(shapeAttnOut);
        workspace = GmAllocWrapper<uint8_t>(allWorkspaceSize);
        tiling = GmAllocWrapper<uint8_t>(tilingSize);

        InitInputData(stateGm, shapeState,
                      queryGm, shapeQ,
                      keyGm, shapeK,
                      valueGm, shapeV,
                      betaGm, shapeBeta,
                      gamaGm, shapeGama,
                      actSeqLenGM, b,
                      ssmStaIdGm, t,
                      numAccTokGm, b,
                      attnOutGM, shapeAttnOut);

        RecurrentGatedDeltaRuleTilingData* tilingData = reinterpret_cast<RecurrentGatedDeltaRuleTilingData*>(tiling);
        InitTilingData(tilingData, b, t, nk, nv, dk, dv,
                       params.hasGama,
                       params.hasGamaK,
                       params.hasAcceptedTokens,
                       params.vStep,
                       params.stateOutBufferNum,
                       params.attnOutBufferNum);
    }

    void TearDown() override {
        AscendC::GmFree(stateGm);
        AscendC::GmFree(queryGm);
        AscendC::GmFree(keyGm);
        AscendC::GmFree(valueGm);
        AscendC::GmFree(betaGm);
        AscendC::GmFree(gamaGm);
        AscendC::GmFree(actSeqLenGM);
        AscendC::GmFree(ssmStaIdGm);
        AscendC::GmFree(numAccTokGm);
        AscendC::GmFree(attnOutGM);
        AscendC::GmFree(workspace);
        AscendC::GmFree(tiling);
    }
};


INSTANTIATE_TEST_SUITE_P(
    GeneralTests, 
    RecurrentGatedDeltaRuleTest, 
    testing::Values(
        RGDRTestParams{1, 0, 1, 32, 32, 1, 1}, // general_test_01: has g, no gk, with AcceptedTokens
        RGDRTestParams{0, 0, 1, 32, 32, 1, 1}, // general_test_02: no g, no gk, with AcceptedTokens
        RGDRTestParams{1, 0, 0, 32, 32, 1, 1}, // general_test_03: has g, no gk, without AcceptedTokens
        RGDRTestParams{1, 0, 1, 64, 32, 1, 1}  // multi_v_tile_prefetch: dv > vStep covers split-phase input path
    )
);

TEST_P(RecurrentGatedDeltaRuleTest, RunTest) {
    auto params = GetParam();
    std::cout << "tets config: hasGama=" << params.hasGama
              << ", hasGamaK=" << params.hasGamaK
              << ", hasAcceptedTokens=" << params.hasAcceptedTokens << std::endl;

    uint32_t blockDim = 8;
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(recurrent_gated_delta_rule, blockDim,
                queryGm, keyGm, valueGm, betaGm, stateGm, actSeqLenGM, ssmStaIdGm,
                gamaGm, nullptr, numAccTokGm, attnOutGM, stateGm, workspace, tiling);

}

// FP32 test fixture for 310P
class RecurrentGatedDeltaRuleFP32Test : public testing::TestWithParam<RGDRTestParams> {
protected:
    uint32_t b = 4;
    uint32_t mtp = 2;
    uint32_t t = b * mtp;
    uint32_t nk = 4;
    uint32_t nv = 4;
    uint32_t dk = 32;
    uint32_t dv = 32;

    size_t shapeState = 0;
    size_t shapeQ = 0;
    size_t shapeK = 0;
    size_t shapeV = 0;
    size_t shapeBeta = 0;
    size_t shapeGama = 0;
    size_t shapeActSeqLen = 0;
    size_t shapeSsmStaId = 0;
    size_t shapeNumAccTok = 0;
    size_t shapeAttnOut = 0;
    size_t allWorkspaceSize = 196608;
    size_t tilingSize = sizeof(RecurrentGatedDeltaRuleTilingData);

    uint8_t* stateGm = nullptr;
    uint8_t* queryGm = nullptr;
    uint8_t* keyGm = nullptr;
    uint8_t* valueGm = nullptr;
    uint8_t* betaGm = nullptr;
    uint8_t* gamaGm = nullptr;
    uint8_t* actSeqLenGM = nullptr;
    uint8_t* ssmStaIdGm = nullptr;
    uint8_t* numAccTokGm = nullptr;
    uint8_t* attnOutGM = nullptr;
    uint8_t* workspace = nullptr;
    uint8_t* tiling = nullptr;

    void SetUp() override {
        auto params = GetParam();
        dv = params.dv;
        shapeState = t * nv * dv * dk * sizeof(float);
        shapeQ = t * nk * dk * sizeof(float);
        shapeK = t * nk * dk * sizeof(float);
        shapeV = t * nv * dv * sizeof(float);
        shapeBeta = t * nv * sizeof(float);
        shapeGama = t * nv * sizeof(float);
        shapeActSeqLen = b * sizeof(int32_t);
        shapeSsmStaId = t * sizeof(int32_t);
        shapeNumAccTok = b * sizeof(int32_t);
        shapeAttnOut = t * nv * dv * sizeof(float);
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        stateGm = GmAllocWrapper<uint8_t>(shapeState);
        queryGm = GmAllocWrapper<uint8_t>(shapeQ);
        keyGm = GmAllocWrapper<uint8_t>(shapeK);
        valueGm = GmAllocWrapper<uint8_t>(shapeV);
        betaGm = GmAllocWrapper<uint8_t>(shapeBeta);
        gamaGm = GmAllocWrapper<uint8_t>(shapeGama);
        actSeqLenGM = GmAllocWrapper<uint8_t>(shapeActSeqLen);
        ssmStaIdGm = GmAllocWrapper<uint8_t>(shapeSsmStaId);
        numAccTokGm = GmAllocWrapper<uint8_t>(shapeNumAccTok);
        attnOutGM = GmAllocWrapper<uint8_t>(shapeAttnOut);
        workspace = GmAllocWrapper<uint8_t>(allWorkspaceSize);
        tiling = GmAllocWrapper<uint8_t>(tilingSize);

        InitInputData(stateGm, shapeState,
                      queryGm, shapeQ,
                      keyGm, shapeK,
                      valueGm, shapeV,
                      betaGm, shapeBeta,
                      gamaGm, shapeGama,
                      actSeqLenGM, b,
                      ssmStaIdGm, t,
                      numAccTokGm, b,
                      attnOutGM, shapeAttnOut);

        RecurrentGatedDeltaRuleTilingData* tilingData = reinterpret_cast<RecurrentGatedDeltaRuleTilingData*>(tiling);
        InitTilingData(tilingData, b, t, nk, nv, dk, dv,
                       params.hasGama,
                       params.hasGamaK,
                       params.hasAcceptedTokens,
                       params.vStep,
                       params.stateOutBufferNum,
                       params.attnOutBufferNum);
    }

    void TearDown() override {
        AscendC::GmFree(stateGm);
        AscendC::GmFree(queryGm);
        AscendC::GmFree(keyGm);
        AscendC::GmFree(valueGm);
        AscendC::GmFree(betaGm);
        AscendC::GmFree(gamaGm);
        AscendC::GmFree(actSeqLenGM);
        AscendC::GmFree(ssmStaIdGm);
        AscendC::GmFree(numAccTokGm);
        AscendC::GmFree(attnOutGM);
        AscendC::GmFree(workspace);
        AscendC::GmFree(tiling);
    }
};

INSTANTIATE_TEST_SUITE_P(
    FP32Tests,
    RecurrentGatedDeltaRuleFP32Test,
    testing::Values(
        RGDRTestParams{1, 0, 1, 32, 32, 1, 1},
        RGDRTestParams{0, 0, 1, 32, 32, 1, 1},
        RGDRTestParams{1, 0, 0, 32, 32, 1, 1},
        RGDRTestParams{1, 0, 1, 64, 32, 1, 1}
    )
);

TEST_P(RecurrentGatedDeltaRuleFP32Test, RunTest) {
    auto params = GetParam();
    std::cout << "fp32 test config: hasGama=" << params.hasGama
              << ", hasGamaK=" << params.hasGamaK
              << ", hasAcceptedTokens=" << params.hasAcceptedTokens << std::endl;

    uint32_t blockDim = 8;
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(recurrent_gated_delta_rule, blockDim,
                queryGm, keyGm, valueGm, betaGm, stateGm, actSeqLenGM, ssmStaIdGm,
                gamaGm, nullptr, numAccTokGm, attnOutGM, stateGm, workspace, tiling);
}
