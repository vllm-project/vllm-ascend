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
 * \file ffn_glu.h
 * \brief
 */

#ifndef ASCENDC_FFN_GLU_H
#define ASCENDC_FFN_GLU_H

#include "ffn_kernel.h"

namespace FFN {
using namespace matmul;

template <typename T>
class FFNGlu {
public:
    __aicore__ inline FFNGlu()
    {
    }
    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *weight1, __gm__ uint8_t *weight2,
                                __gm__ uint8_t *bias1, __gm__ uint8_t *bias2, __gm__ uint8_t *y,
                                __gm__ uint8_t *workSpace, const FFNTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // define matmul1
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using c1Type = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    Matmul<a1Type, b1Type, c1Type, bias1Type, CFG_MDL> mm1;
    // define matmul2
    using a2Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using c2Type = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    Matmul<a2Type, b2Type, c2Type, bias2Type, CFG_MDL> mm2;

protected:
    const FFNTilingData *__restrict tilingData;
    TPipe *pipe;
    // define the que
    TQue<QuePosition::VECIN, 1> vecInQueueLeft;
    TQue<QuePosition::VECIN, 1> vecInQueueRight;
    TQue<QuePosition::VECOUT, 1> vecOutQueue;

    GlobalTensor<T> xGm;
    GlobalTensor<T> weight1Gm;
    GlobalTensor<T> bias1Gm;
    GlobalTensor<T> weight2Gm;
    GlobalTensor<T> bias2Gm;
    GlobalTensor<T> yGm;

    // mm1 left and right result
    GlobalTensor<T> mm1ResLeft[2];  // 2: pingpong
    GlobalTensor<T> mm1ResRight[2]; // 2: pingpong

    GlobalTensor<T> mm2WorkspaceGm;

    GluActiveFuncPtr<T> activeFunc;

    // tiling data
    uint32_t m1;
    uint32_t k1;
    uint32_t n1;
    uint32_t k2;
    uint32_t n2;
    uint32_t coreNum;
    uint32_t gluActiveType;
    uint32_t baseM1;
    uint32_t baseN1;
    uint32_t dataTypeSize;

    bool hasBias1 = false;
    bool hasBias2 = false;
    uint32_t curBlockIdx;
    uint32_t singleM1;
    uint32_t singleM2;
    uint32_t singleM1Tail;
    uint32_t singleM2Tail;
    uint32_t singleN1;
    uint32_t singleN1Tail;
    uint32_t singleN2;
    uint32_t singleN2Tail;
    uint32_t m1Loops;
    uint32_t m2Loops;
    uint32_t n1Loops;
    uint32_t n2Loops;

    // mm1 geglu
    uint64_t xCoreOffset;
    uint64_t w1CoreOffset;
    uint64_t lastCoreMm1Offset;
    uint32_t curSingleM;
    uint32_t curSingleN;
    uint32_t mInnerLoops;
    uint32_t nInnerLoops;
    uint32_t aicMtail;
    uint32_t aicNtail;
    uint32_t singleTimeM;
    uint32_t singleTimeN;

    __aicore__ inline void initTilingData();
    __aicore__ inline void InitActivationFunction();
    __aicore__ inline void MM1GluSplit();
    __aicore__ inline void MM2Split();
    __aicore__ inline void CalcMM1GluParams();
    __aicore__ inline void MM1Compute(uint32_t curAicM, uint32_t curAicN, uint64_t w1Offset, uint32_t pingPongId);
    __aicore__ inline void GluSplit(uint32_t curAicM, uint32_t curAicN, uint64_t lastMm1Offset, uint32_t pingPongId);
    __aicore__ inline void GluCompute(uint32_t computeSize);
};
} // namespace FFN

#endif // ASCENDC_FFN_GLU_H
