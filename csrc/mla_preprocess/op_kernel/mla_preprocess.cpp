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

#include "mla_preprocess_mix_fp16.hpp"
#include "mla_preprocess_mix_bf16.hpp"
#include "mla_preprocess_mix_bf16_qdown.hpp"
#include "mla_preprocess_mix_bf16_nq.hpp"

#include "mla_preprocess_tiling_data.h"

extern "C" __global__ __aicore__ void mla_preprocess(
    GM_ADDR hiddenState, GM_ADDR quantScale1, GM_ADDR quantOffset1, GM_ADDR wdqkv,
    GM_ADDR bias1, GM_ADDR gamma2, GM_ADDR beta2, GM_ADDR quantScale2, GM_ADDR quantOffset2, GM_ADDR gamma3,
    GM_ADDR sin1, GM_ADDR cos1, GM_ADDR sin2, GM_ADDR cos2, GM_ADDR keycache, GM_ADDR slotMapping, GM_ADDR wuq,
    GM_ADDR bias2, GM_ADDR wuk, GM_ADDR descale1, GM_ADDR descale2, GM_ADDR ctkvScale, GM_ADDR qnopeScale, GM_ADDR q,
    GM_ADDR keycacheOut, GM_ADDR q2, GM_ADDR keycacheOut2, GM_ADDR innerOut, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MlaTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    PRELOAD(2);
#endif

    SetAtomicnone();
    SetMasknorm();
#ifdef __DAV_C220_CUBE__
    SetPadding<uint64_t>((uint64_t)0);
    SetNdpara(1, 0, 0);
#endif

    GET_TILING_DATA_WITH_STRUCT(MlaTilingData, mlaTilingData, tiling);

    // ACLNN initializes the system-workspace state before entering the kernel.
    // Do not overwrite it with the workspace argument; this operator's custom
    // PpMatmul/FFTS path only uses the user portion.
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    GM_ADDR s1 = userWorkspace + static_cast<uint64_t>(mlaTilingData.s1Offset);
    GM_ADDR s2 = userWorkspace + static_cast<uint64_t>(mlaTilingData.s2Offset);
    GM_ADDR s3 = userWorkspace + static_cast<uint64_t>(mlaTilingData.s3Offset);
    GM_ADDR s4 = userWorkspace + static_cast<uint64_t>(mlaTilingData.s4Offset);
    GM_ADDR s5 = userWorkspace + static_cast<uint64_t>(mlaTilingData.s5Offset);

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
        case KEY_BF16_CACHEMODE_0_QUANTMODE_1: {
            MLAPO_BF16::MLAOperation<__bf16, 0, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                    QuantMode::PER_TOKEN_SYMM_QUANT>
                opBf16Cm0Qm1(mlaTilingData, tiling);
            opBf16Cm0Qm1.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                            quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                            bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                            s1, s2, s3, s4, s5);
            if ASCEND_IS_AIC {
                opBf16Cm0Qm1.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opBf16Cm0Qm1.ProcessVector();
            }
            break;
        }
        case KEY_BF16_CACHEMODE_1_QUANTMODE_1: {
            MLAPO_BF16::MLAOperation<__bf16, 1, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                    QuantMode::PER_TOKEN_SYMM_QUANT>
                opBf16Cm1Qm1(mlaTilingData, tiling);
            opBf16Cm1Qm1.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                            quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                            bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                            s1, s2, s3, s4, s5);
            if ASCEND_IS_AIC {
                opBf16Cm1Qm1.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opBf16Cm1Qm1.ProcessVector();
            }
            break;
        }
        case KEY_BF16_CACHEMODE_2_QUANTMODE_1: {
            MLAPO_BF16::MLAOperation<__bf16, 2, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                     QuantMode::PER_TOKEN_SYMM_QUANT>
                opBf16Cm2Qm1(mlaTilingData, tiling);
            opBf16Cm2Qm1.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                              quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                              bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                              s1, s2, s3, s4, s5);
            if ASCEND_IS_AIC {
                opBf16Cm2Qm1.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opBf16Cm2Qm1.ProcessVector();
            }
            break;
        }
        case KEY_BF16_CACHEMODE_3_QUANTMODE_1: {
            MLAPO_BF16::MLAOperation<__bf16, 3, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                     QuantMode::PER_TOKEN_SYMM_QUANT>
                opBf16Cm3Qm1(mlaTilingData, tiling);
            opBf16Cm3Qm1.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                              quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                              bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                              s1, s2, s3, s4, s5);
            if ASCEND_IS_AIC {
                opBf16Cm3Qm1.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opBf16Cm3Qm1.ProcessVector();
            }
            break;
        }
        case KEY_BF16_CACHEMODE_1_QUANTMODE_3: {
            MLAPO_BF16_NQ::MLAOperation<__bf16, 1, DataFormat::NZ, DataFormat::NZ, DataFormat::ND>
                opBf16Cm1Qm0(mlaTilingData, tiling);
            opBf16Cm1Qm0.Init(hiddenState, wdqkv, gamma2, beta2,
                            gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                            wuk, q, keycacheOut, q2, keycacheOut2,
                            s1, s2, s3);
            if ASCEND_IS_AIC {
                opBf16Cm1Qm0.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opBf16Cm1Qm0.ProcessVector();
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
        case KEY_BF16_CACHEMODE_0_QUANTMODE_1_INNER: {
            MLAPO_BF16_INNER::MLAOperation<__bf16, 0, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                     QuantMode::PER_TOKEN_SYMM_QUANT>
                opBf16Cm0Qm1Inner(mlaTilingData, tiling);
            opBf16Cm0Qm1Inner.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                              quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                              bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                              s1, s2, s3, s4, s5, innerOut);
            if ASCEND_IS_AIC {
                opBf16Cm0Qm1Inner.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opBf16Cm0Qm1Inner.ProcessVector();
            }
            break;
        }
        case KEY_BF16_CACHEMODE_1_QUANTMODE_1_INNER: {
            MLAPO_BF16_INNER::MLAOperation<__bf16, 1, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                     QuantMode::PER_TOKEN_SYMM_QUANT>
                opBf16Cm1Qm1Inner(mlaTilingData, tiling);
            opBf16Cm1Qm1Inner.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                              quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                              bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                              s1, s2, s3, s4, s5, innerOut);
            if ASCEND_IS_AIC {
                opBf16Cm1Qm1Inner.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opBf16Cm1Qm1Inner.ProcessVector();
            }
            break;
        }
        case KEY_BF16_CACHEMODE_2_QUANTMODE_1_INNER: {
            MLAPO_BF16_INNER::MLAOperation<__bf16, 2, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                     QuantMode::PER_TOKEN_SYMM_QUANT>
                opBf16Cm2Qm1Inner(mlaTilingData, tiling);
            opBf16Cm2Qm1Inner.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                              quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                              bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                              s1, s2, s3, s4, s5, innerOut);
            if ASCEND_IS_AIC {
                opBf16Cm2Qm1Inner.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opBf16Cm2Qm1Inner.ProcessVector();
            }
            break;
        }
        case KEY_BF16_CACHEMODE_3_QUANTMODE_1_INNER: {
            MLAPO_BF16_INNER::MLAOperation<__bf16, 3, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                     QuantMode::PER_TOKEN_SYMM_QUANT>
                opBf16Cm3Qm1Inner(mlaTilingData, tiling);
            opBf16Cm3Qm1Inner.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                              quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                              bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                              s1, s2, s3, s4, s5, innerOut);
            if ASCEND_IS_AIC {
                opBf16Cm3Qm1Inner.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opBf16Cm3Qm1Inner.ProcessVector();
            }
            break;
        }
        default: {
            break;
        }
    }
    return;
}
