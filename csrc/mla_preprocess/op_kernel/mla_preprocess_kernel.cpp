// Adapted from
//   https://gitee.com/ascend/ascend-transformer-boost
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//

#include "kernel_operator.h"
#include "../../kernels/types.h"

#include "mla_preprocess_mix_fp16.hpp"
#include "mla_preprocess_mix_bf16.hpp"
#include "mla_preprocess_mix_bf16_qdown.hpp"

#include "../op_host/tiling/mla_preprocess_tiling.h"

extern "C" __global__ __aicore__ void mla_preprocess(
    GM_ADDR hiddenState, GM_ADDR quantScale1, GM_ADDR quantOffset1, GM_ADDR wdqkv,
    GM_ADDR bias1, GM_ADDR gamma2, GM_ADDR beta2, GM_ADDR quantScale2, GM_ADDR quantOffset2, GM_ADDR gamma3,
    GM_ADDR sin1, GM_ADDR cos1, GM_ADDR sin2, GM_ADDR cos2, GM_ADDR keycache, GM_ADDR slotMapping, GM_ADDR wuq,
    GM_ADDR bias2, GM_ADDR wuk, GM_ADDR descale1, GM_ADDR descale2, GM_ADDR ctkvScale, GM_ADDR qnopeScale, GM_ADDR q,
    GM_ADDR keycacheOut, GM_ADDR q2, GM_ADDR keycacheOut2, GM_ADDR innerOut, GM_ADDR workspace, GM_ADDR tiling)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    PRELOAD(2);
#endif

    SetAtomicnone();
    SetMasknorm();
#ifdef __DAV_C220_CUBE__
    SetPadding<uint64_t>((uint64_t)0);
    SetNdpara(1, 0, 0);
#endif

    MlaTilingData mlaTilingData;
    __gm__ MlaTilingData *tilingData = reinterpret_cast<__gm__ MlaTilingData *>(tiling);

    mlaTilingData.tilingKey = tilingData->tilingKey;
    mlaTilingData.n = tilingData->n;

    mlaTilingData.mm1.numBatch = tilingData->mm1.numBatch;
    mlaTilingData.mm1.m = tilingData->mm1.m;
    mlaTilingData.mm1.k = tilingData->mm1.k;
    mlaTilingData.mm1.n = tilingData->mm1.n;
    mlaTilingData.mm1.m0 = tilingData->mm1.m0;
    mlaTilingData.mm1.k0 = tilingData->mm1.k0;
    mlaTilingData.mm1.n0 = tilingData->mm1.n0;
    mlaTilingData.mm1.mLoop = tilingData->mm1.mLoop;
    mlaTilingData.mm1.kLoop = tilingData->mm1.kLoop;
    mlaTilingData.mm1.nLoop = tilingData->mm1.nLoop;
    mlaTilingData.mm1.coreLoop = tilingData->mm1.coreLoop;
    mlaTilingData.mm1.swizzleCount = tilingData->mm1.swizzleCount;
    mlaTilingData.mm1.enShuffleK = tilingData->mm1.enShuffleK;
    mlaTilingData.mm1.blockDim = tilingData->mm1.blockDim;
    mlaTilingData.mm1.enLoadAllAmat = tilingData->mm1.enLoadAllAmat;
    mlaTilingData.mm1.b0matPingPongBufferLen = tilingData->mm1.b0matPingPongBufferLen;

    mlaTilingData.mm2.numBatch = tilingData->mm2.numBatch;
    mlaTilingData.mm2.m = tilingData->mm2.m;
    mlaTilingData.mm2.k = tilingData->mm2.k;
    mlaTilingData.mm2.n = tilingData->mm2.n;
    mlaTilingData.mm2.m0 = tilingData->mm2.m0;
    mlaTilingData.mm2.k0 = tilingData->mm2.k0;
    mlaTilingData.mm2.n0 = tilingData->mm2.n0;
    mlaTilingData.mm2.mLoop = tilingData->mm2.mLoop;
    mlaTilingData.mm2.kLoop = tilingData->mm2.kLoop;
    mlaTilingData.mm2.nLoop = tilingData->mm2.nLoop;
    mlaTilingData.mm2.coreLoop = tilingData->mm2.coreLoop;
    mlaTilingData.mm2.swizzleCount = tilingData->mm2.swizzleCount;
    mlaTilingData.mm2.enShuffleK = tilingData->mm2.enShuffleK;
    mlaTilingData.mm2.blockDim = tilingData->mm2.blockDim;
    mlaTilingData.mm2.enLoadAllAmat = tilingData->mm2.enLoadAllAmat;
    mlaTilingData.mm2.b0matPingPongBufferLen = tilingData->mm2.b0matPingPongBufferLen;

    mlaTilingData.mm3.numBatch = tilingData->mm3.numBatch;
    mlaTilingData.mm3.m = tilingData->mm3.m;
    mlaTilingData.mm3.k = tilingData->mm3.k;
    mlaTilingData.mm3.n = tilingData->mm3.n;
    mlaTilingData.mm3.m0 = tilingData->mm3.m0;
    mlaTilingData.mm3.k0 = tilingData->mm3.k0;
    mlaTilingData.mm3.n0 = tilingData->mm3.n0;
    mlaTilingData.mm3.mLoop = tilingData->mm3.mLoop;
    mlaTilingData.mm3.kLoop = tilingData->mm3.kLoop;
    mlaTilingData.mm3.nLoop = tilingData->mm3.nLoop;
    mlaTilingData.mm3.coreLoop = tilingData->mm3.coreLoop;
    mlaTilingData.mm3.swizzleCount = tilingData->mm3.swizzleCount;
    mlaTilingData.mm3.enShuffleK = tilingData->mm3.enShuffleK;
    mlaTilingData.mm3.blockDim = tilingData->mm3.blockDim;

    mlaTilingData.perTaskNum = tilingData->perTaskNum;
    mlaTilingData.resTaskNum = tilingData->resTaskNum;
    mlaTilingData.numCore = tilingData->numCore;

    mlaTilingData.rmsNumCore1 = tilingData->rmsNumCore1;
    mlaTilingData.rmsNumCol1 = tilingData->rmsNumCol1;
    mlaTilingData.rmsNumCore2 = tilingData->rmsNumCore2;
    mlaTilingData.rmsNumCol2 = tilingData->rmsNumCol2;

    mlaTilingData.hiddenSizeQ = tilingData->hiddenSizeQ;
    mlaTilingData.headNumQ = tilingData->headNumQ;
    mlaTilingData.headDim = tilingData->headDim;
    mlaTilingData.concatSize = tilingData->concatSize;
    mlaTilingData.rotaryCoeff = tilingData->rotaryCoeff;
    mlaTilingData.ntokens = tilingData->ntokens;
    mlaTilingData.realCore = tilingData->realCore;
    mlaTilingData.nlCoreRun = tilingData->nlCoreRun;
    mlaTilingData.lCoreRun = tilingData->lCoreRun;
    mlaTilingData.maxNPerLoopForUb = tilingData->maxNPerLoopForUb;
    mlaTilingData.preCoreLoopTime = tilingData->preCoreLoopTime;
    mlaTilingData.preCoreLoopNLast = tilingData->preCoreLoopNLast;
    mlaTilingData.lastCoreLoopTime = tilingData->lastCoreLoopTime;
    mlaTilingData.lastCoreLoopNLast = tilingData->lastCoreLoopNLast;

    mlaTilingData.esqFrontCore = tilingData->esqFrontCore;
    mlaTilingData.esqTailCore = tilingData->esqTailCore;
    mlaTilingData.esqFrontCoreBatch = tilingData->esqFrontCoreBatch;
    mlaTilingData.esqTailCoreBatch = tilingData->esqTailCoreBatch;
    mlaTilingData.esqHeadNum = tilingData->esqHeadNum;
    mlaTilingData.esqColNum = tilingData->esqColNum;
    mlaTilingData.esqUbHeadLoop = tilingData->esqUbHeadLoop;
    mlaTilingData.esqHeadPerLoop = tilingData->esqHeadPerLoop;
    mlaTilingData.esqHeadTail = tilingData->esqHeadTail;
    mlaTilingData.esqColLoop = tilingData->esqColLoop;
    mlaTilingData.esqColTail = tilingData->esqColTail;

    mlaTilingData.s1Offset = tilingData->s1Offset;
    mlaTilingData.s2Offset = tilingData->s2Offset;
    mlaTilingData.s3Offset = tilingData->s3Offset;
    mlaTilingData.s4Offset = tilingData->s4Offset;
    mlaTilingData.s5Offset = tilingData->s5Offset;

    GM_ADDR s1 = workspace + static_cast<uint64_t>(mlaTilingData.s1Offset);
    GM_ADDR s2 = workspace + static_cast<uint64_t>(mlaTilingData.s2Offset);
    GM_ADDR s3 = workspace + static_cast<uint64_t>(mlaTilingData.s3Offset);
    GM_ADDR s4 = workspace + static_cast<uint64_t>(mlaTilingData.s4Offset);
    GM_ADDR s5 = workspace + static_cast<uint64_t>(mlaTilingData.s5Offset);

    switch (mlaTilingData.tilingKey) {
        case KEY_FP16_CACHEMODE_0_QUANTMODE_0: {
            MLAPO_FP16::MLAOperation<CACHE_MODE_KVCACHE, DataFormat::NZ, DataFormat::NZ, DataFormat::ND> opFp16Cm0Qm0(
                mlaTilingData, tiling);
            opFp16Cm0Qm0.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                              quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                              bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                              s1, s2, s3);
            if ASCEND_IS_AIC {
                opFp16Cm0Qm0.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opFp16Cm0Qm0.ProcessVector();
            }
            break;
        }
        case KEY_FP16_CACHEMODE_1_QUANTMODE_0: {
            MLAPO_FP16::MLAOperation<CACHE_MODE_KROPE_CTKV, DataFormat::NZ, DataFormat::NZ, DataFormat::ND>
                opFp16Cm1Qm0(mlaTilingData, tiling);
            opFp16Cm1Qm0.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                              quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                              bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                              s1, s2, s3);
            if ASCEND_IS_AIC {
                opFp16Cm1Qm0.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opFp16Cm1Qm0.ProcessVector();
            }
            break;
        }
        case KEY_BF16_CACHEMODE_0_QUANTMODE_0: {
            MLAPO_BF16::MLAOperation<__bf16, 0, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                     QuantMode::PER_TENSOR_ASYMM_QUANT>
                opBf16Cm0Qm0(mlaTilingData, tiling);
            opBf16Cm0Qm0.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                              quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                              bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                              s1, s2, s3, s4, s5);
            if ASCEND_IS_AIC {
                opBf16Cm0Qm0.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opBf16Cm0Qm0.ProcessVector();
            }
            break;
        }
        case KEY_BF16_CACHEMODE_1_QUANTMODE_0: {
            MLAPO_BF16::MLAOperation<__bf16, 1, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                     QuantMode::PER_TENSOR_ASYMM_QUANT>
                opBf16Cm1Qm0(mlaTilingData, tiling);
            opBf16Cm1Qm0.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                              quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                              bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                              s1, s2, s3, s4, s5);
            if ASCEND_IS_AIC {
                opBf16Cm1Qm0.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opBf16Cm1Qm0.ProcessVector();
            }
            break;
        }
        case KEY_BF16_CACHEMODE_3_QUANTMODE_0: {
            MLAPO_BF16::MLAOperation<__bf16, 3, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                     QuantMode::PER_TENSOR_ASYMM_QUANT>
                opBf16Cm3Qm0(mlaTilingData, tiling);
            opBf16Cm3Qm0.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                              quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                              bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                              s1, s2, s3, s4, s5);
            if ASCEND_IS_AIC {
                opBf16Cm3Qm0.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opBf16Cm3Qm0.ProcessVector();
            }
            break;
        }
        case KEY_BF16_CACHEMODE_0_QUANTMODE_0_INNER: {
            MLAPO_BF16_INNER::MLAOperation<__bf16, 0, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                     QuantMode::PER_TENSOR_ASYMM_QUANT>
                opBf16Cm0Qm0Inner(mlaTilingData, tiling);
            opBf16Cm0Qm0Inner.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                              quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                              bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                              s1, s2, s3, s4, s5, innerOut);
            if ASCEND_IS_AIC {
                opBf16Cm0Qm0Inner.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opBf16Cm0Qm0Inner.ProcessVector();
            }
            break;
        }
        case KEY_BF16_CACHEMODE_1_QUANTMODE_0_INNER: {
            MLAPO_BF16_INNER::MLAOperation<__bf16, 1, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                     QuantMode::PER_TENSOR_ASYMM_QUANT>
                opBf16Cm1Qm0Inner(mlaTilingData, tiling);
            opBf16Cm1Qm0Inner.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                              quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                              bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                              s1, s2, s3, s4, s5, innerOut);
            if ASCEND_IS_AIC {
                opBf16Cm1Qm0Inner.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opBf16Cm1Qm0Inner.ProcessVector();
            }
            break;
        }
        case KEY_BF16_CACHEMODE_3_QUANTMODE_0_INNER: {
            MLAPO_BF16_INNER::MLAOperation<__bf16, 3, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                     QuantMode::PER_TENSOR_ASYMM_QUANT>
                opBf16Cm3Qm0Inner(mlaTilingData, tiling);
            opBf16Cm3Qm0Inner.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                              quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                              bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                              s1, s2, s3, s4, s5, innerOut);
            if ASCEND_IS_AIC {
                opBf16Cm3Qm0Inner.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opBf16Cm3Qm0Inner.ProcessVector();
            }
            break;
        }
        default: {
            break;
        }
    }
    return;
}

namespace vllm_ascend {

extern void mla_preprocess_impl(
    void* stream,
    void* hidden_state,
    void* quant_scale1,
    void* quant_offset1,
    void* wdqkv,
    void* bias1,
    void* gamma2,
    void* beta2,
    void* quant_scale2,
    void* quant_offset2,
    void* gamma3,
    void* sin1,
    void* cos1,
    void* sin2,
    void* cos2,
    void* keycache,
    void* slot_mapping,
    void* wuq,
    void* bias2,
    void* wuk,
    void* descale1,
    void* descale2,
    void* ctkv_scale,
    void* qnope_scale,
    void* q,
    void* keycache_out,
    void* q2,
    void* keycache_out2,
    void* inner_out,
    void* workspace,
    void* tiling,
    const uint32_t block_dim)
{
    mla_preprocess<<<block_dim, nullptr, stream>>>(
        hidden_state,
        quant_scale1,
        quant_offset1,
        wdqkv,
        bias1,
        gamma2,
        beta2,
        quant_scale2,
        quant_offset2,
        gamma3,
        sin1,
        cos1,
        sin2,
        cos2,
        keycache,
        slot_mapping,
        wuq,
        bias2,
        wuk,
        descale1,
        descale2,
        ctkv_scale,
        qnope_scale,
        q,
        keycache_out,
        q2,
        keycache_out2,
        inner_out,
        workspace,
        tiling);
}

}
