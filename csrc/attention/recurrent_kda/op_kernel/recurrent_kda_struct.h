/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file recurrent_kda_struct.h
 * \brief Plain tiling struct shared by aclnn tiling and fast kernel launch.
 */

#ifndef RECURRENT_KDA_STRUCT_H
#define RECURRENT_KDA_STRUCT_H

#include <cstdint>

namespace RecurrentKda {

#pragma pack(push, 8)
struct alignas(8) RecurrentKdaTilingData {
    uint32_t vectorCoreNum;
    uint32_t ubCalSize;
    uint32_t ubRestBytes;
    uint32_t t;
    uint32_t seqLen;
    uint32_t nk;
    uint32_t dk;
    uint32_t nv;
    uint32_t dv;
    uint32_t sBlockNum;
    uint32_t ssmStateStride;
    uint32_t b;
    uint32_t vStep;
    uint32_t stateOutBufferNum;
    uint32_t attnOutBufferNum;
    float scale;
    float lowerBound;
    uint32_t layout;
    uint32_t hasSsmStateIndices;
    uint32_t hasALog;
    uint32_t hasDtBias;
    uint32_t hasAcceptedTokens;
    uint32_t useQkL2norm;
    uint32_t useGateInKernel;
    uint32_t useBetaSigmoid;
    uint32_t allowNegEigval;
    uint32_t safeGate;
    uint32_t stateVFirst;
};
#pragma pack(pop)

} // namespace RecurrentKda

#endif // RECURRENT_KDA_STRUCT_H
