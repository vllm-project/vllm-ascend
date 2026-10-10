/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "generic_block_sparse_attention_tilingkey.h"
#include "generic_block_sparse_attention_kernel_interface.cpp"

extern "C" __global__ __aicore__ void generic_block_sparse_attention(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sparseBlockIdx,
    __gm__ uint8_t *sparseBlockCount, __gm__ uint8_t *metaData, __gm__ uint8_t *attenMask,
    __gm__ uint8_t *qDequantScale, __gm__ uint8_t *kDequantScale, __gm__ uint8_t *vDequantScale,
    __gm__ uint8_t *pQuantScale, __gm__ uint8_t *cuSeqLengths, __gm__ uint8_t *cuSeqLengthsKv, __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *sequsedKv, __gm__ uint8_t *blockTable, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    if (TILING_KEY_VAR >= GBSA_BASE_TILING) {
        __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if (__CCE_AICORE__ == 220)
        TILING_KEY_IS(GBSA_FP16_TND_PA_BBND_TILING);
        TILING_KEY_IS(GBSA_BF16_TND_PA_BBND_TILING);
        TILING_KEY_IS(GBSA_FP16_TND_PA_BBND_HALFSM_TILING);
        TILING_KEY_IS(GBSA_FP16_TND_PA_BBND_TILING_LSE_OUT);
        TILING_KEY_IS(GBSA_BF16_TND_PA_BBND_TILING_LSE_OUT);
        TILING_KEY_IS(GBSA_FP16_TND_PA_BBND_HALFSM_TILING_LSE_OUT);

#if TILING_KEY_VAR == GBSA_FP16_TND_PA_BBND_TILING
        // softmaxPrecision=0: fp32 Softmax + Rescale
        GbsaInferIntfRegularArch22<half, float, float>(query, key, value, sparseBlockIdx, sparseBlockCount, metaData,
                                                       cuSeqLengths, cuSeqLengthsKv, sequsedQ, sequsedKv, blockTable,
                                                       attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_FP16_TND_PA_BBND_HALFSM_TILING
        // softmaxPrecision=1: half Softmax + fp32 Rescale (fp16 only)
        GbsaInferIntfRegularArch22<half, half, float>(query, key, value, sparseBlockIdx, sparseBlockCount, metaData,
                                                      cuSeqLengths, cuSeqLengthsKv, sequsedQ, sequsedKv, blockTable,
                                                      attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_BF16_TND_PA_BBND_TILING
        // softmaxPrecision=0 only; bf16+prec=1 rejected by host
        GbsaInferIntfRegularArch22<bfloat16_t, float, float>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_FP16_TND_PA_BBND_TILING_LSE_OUT
        GbsaInferIntfRegularArch22<half, float, float, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_FP16_TND_PA_BBND_HALFSM_TILING_LSE_OUT
        GbsaInferIntfRegularArch22<half, half, float, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_BF16_TND_PA_BBND_TILING_LSE_OUT
        GbsaInferIntfRegularArch22<bfloat16_t, float, float, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#endif
#elif (__CCE_AICORE__ == 310)
        TILING_KEY_IS(GBSA_FP16_TND_TND_FP32SM_TILING);
        TILING_KEY_IS(GBSA_FP16_TND_TND_FP16SM_TILING);
        TILING_KEY_IS(GBSA_BF16_TND_TND_FP32SM_TILING);
        TILING_KEY_IS(GBSA_BF16_TND_TND_BF16SM_TILING);
        TILING_KEY_IS(GBSA_FP16_TND_TND_FP32SM_TILING_LSE_OUT);
        TILING_KEY_IS(GBSA_FP16_TND_TND_FP16SM_TILING_LSE_OUT);
        TILING_KEY_IS(GBSA_BF16_TND_TND_FP32SM_TILING_LSE_OUT);
        TILING_KEY_IS(GBSA_BF16_TND_TND_BF16SM_TILING_LSE_OUT);
        TILING_KEY_IS(GBSA_FP8_TND_PA_BBND_FP16SM_TILING);
        TILING_KEY_IS(GBSA_FP8_TND_PA_BBND_BF16SM_TILING);
        TILING_KEY_IS(GBSA_FP16_TND_PA_BBND_FP32SM_TILING);
        TILING_KEY_IS(GBSA_BF16_TND_PA_BBND_FP32SM_TILING);
        TILING_KEY_IS(GBSA_FP16_TND_PA_BBND_FP16SM_TILING);
        TILING_KEY_IS(GBSA_BF16_TND_PA_BBND_BF16SM_TILING);
        TILING_KEY_IS(GBSA_FP16_TND_PA_BBND_FP32SM_TILING_LSE_OUT);
        TILING_KEY_IS(GBSA_FP16_TND_PA_BBND_FP16SM_TILING_LSE_OUT);
        TILING_KEY_IS(GBSA_BF16_TND_PA_BBND_FP32SM_TILING_LSE_OUT);
        TILING_KEY_IS(GBSA_BF16_TND_PA_BBND_BF16SM_TILING_LSE_OUT);

        TILING_KEY_IS(GBSA_FP16_TND_TND_FP32SM_TILING_RESIDUAL);
        TILING_KEY_IS(GBSA_FP16_TND_TND_FP16SM_TILING_RESIDUAL);
        TILING_KEY_IS(GBSA_BF16_TND_TND_FP32SM_TILING_RESIDUAL);
        TILING_KEY_IS(GBSA_BF16_TND_TND_BF16SM_TILING_RESIDUAL);
        TILING_KEY_IS(GBSA_FP16_TND_TND_FP32SM_TILING_LSE_OUT_RESIDUAL);
        TILING_KEY_IS(GBSA_FP16_TND_TND_FP16SM_TILING_LSE_OUT_RESIDUAL);
        TILING_KEY_IS(GBSA_BF16_TND_TND_FP32SM_TILING_LSE_OUT_RESIDUAL);
        TILING_KEY_IS(GBSA_BF16_TND_TND_BF16SM_TILING_LSE_OUT_RESIDUAL);
        TILING_KEY_IS(GBSA_FP16_TND_PA_BBND_FP32SM_TILING_RESIDUAL);
        TILING_KEY_IS(GBSA_BF16_TND_PA_BBND_FP32SM_TILING_RESIDUAL);
        TILING_KEY_IS(GBSA_FP16_TND_PA_BBND_FP16SM_TILING_RESIDUAL);
        TILING_KEY_IS(GBSA_BF16_TND_PA_BBND_BF16SM_TILING_RESIDUAL);
        TILING_KEY_IS(GBSA_FP16_TND_PA_BBND_FP32SM_TILING_LSE_OUT_RESIDUAL);
        TILING_KEY_IS(GBSA_FP16_TND_PA_BBND_FP16SM_TILING_LSE_OUT_RESIDUAL);
        TILING_KEY_IS(GBSA_BF16_TND_PA_BBND_FP32SM_TILING_LSE_OUT_RESIDUAL);
        TILING_KEY_IS(GBSA_BF16_TND_PA_BBND_BF16SM_TILING_LSE_OUT_RESIDUAL);

#if TILING_KEY_VAR == GBSA_FP16_TND_TND_FP32SM_TILING
        GbsaInferIntfRegular<half, float, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::TND, false>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_FP16_TND_TND_FP16SM_TILING
        GbsaInferIntfRegular<half, half, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::TND, false>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_BF16_TND_TND_FP32SM_TILING
        GbsaInferIntfRegular<bfloat16_t, float, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::TND, false>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_BF16_TND_TND_BF16SM_TILING
        GbsaInferIntfRegular<bfloat16_t, bfloat16_t, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::TND, false>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_FP16_TND_TND_FP32SM_TILING_LSE_OUT
        GbsaInferIntfRegular<half, float, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::TND, false, false,
                             NpuArch::Epilogue::LseMode::OUT_ONLY,
                             NpuArch::Epilogue::LseFormat::TN1>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_FP16_TND_TND_FP16SM_TILING_LSE_OUT
        GbsaInferIntfRegular<half, half, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::TND, false, false,
                             NpuArch::Epilogue::LseMode::OUT_ONLY,
                             NpuArch::Epilogue::LseFormat::TN1>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_BF16_TND_TND_FP32SM_TILING_LSE_OUT
        GbsaInferIntfRegular<bfloat16_t, float, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::TND, false, false,
                             NpuArch::Epilogue::LseMode::OUT_ONLY,
                             NpuArch::Epilogue::LseFormat::TN1>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_BF16_TND_TND_BF16SM_TILING_LSE_OUT
        GbsaInferIntfRegular<bfloat16_t, bfloat16_t, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::TND, false, false,
                             NpuArch::Epilogue::LseMode::OUT_ONLY,
                             NpuArch::Epilogue::LseFormat::TN1>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_FP8_TND_PA_BBND_FP16SM_TILING
        GbsaInferInterfaceFullQuant<fp8_e4m3fn_t, half, float, GbsaKernelArch35::Format::TND>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, qDequantScale, kDequantScale, vDequantScale, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_FP8_TND_PA_BBND_BF16SM_TILING
        GbsaInferInterfaceFullQuant<fp8_e4m3fn_t, bfloat16_t, float, GbsaKernelArch35::Format::TND>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, qDequantScale, kDequantScale, vDequantScale, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_FP16_TND_PA_BBND_FP32SM_TILING
        GbsaInferIntfRegular<half, float, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::BSND, true>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_BF16_TND_PA_BBND_FP32SM_TILING
        GbsaInferIntfRegular<bfloat16_t, float, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::BSND, true>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_FP16_TND_PA_BBND_FP16SM_TILING
        GbsaInferIntfRegular<half, half, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::BSND, true>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_BF16_TND_PA_BBND_BF16SM_TILING
        GbsaInferIntfRegular<bfloat16_t, bfloat16_t, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::BSND, true>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_FP16_TND_PA_BBND_FP32SM_TILING_LSE_OUT
        GbsaInferIntfRegular<half, float, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::BSND, true, false,
                             NpuArch::Epilogue::LseMode::OUT_ONLY,
                             NpuArch::Epilogue::LseFormat::TN1>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_FP16_TND_PA_BBND_FP16SM_TILING_LSE_OUT
        GbsaInferIntfRegular<half, half, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::BSND, true, false,
                             NpuArch::Epilogue::LseMode::OUT_ONLY,
                             NpuArch::Epilogue::LseFormat::TN1>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_BF16_TND_PA_BBND_FP32SM_TILING_LSE_OUT
        GbsaInferIntfRegular<bfloat16_t, float, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::BSND, true, false,
                             NpuArch::Epilogue::LseMode::OUT_ONLY,
                             NpuArch::Epilogue::LseFormat::TN1>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_BF16_TND_PA_BBND_BF16SM_TILING_LSE_OUT
        GbsaInferIntfRegular<bfloat16_t, bfloat16_t, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::BSND, true, false,
                             NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::LseFormat::TN1>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_FP16_TND_TND_FP32SM_TILING_RESIDUAL
        GbsaInferIntfRegular<half, float, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::TND, false, true>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_FP16_TND_TND_FP16SM_TILING_RESIDUAL
        GbsaInferIntfRegular<half, half, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::TND, false, true>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_BF16_TND_TND_FP32SM_TILING_RESIDUAL
        GbsaInferIntfRegular<bfloat16_t, float, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::TND, false, true>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_BF16_TND_TND_BF16SM_TILING_RESIDUAL
        GbsaInferIntfRegular<bfloat16_t, bfloat16_t, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::TND, false, true>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_FP16_TND_TND_FP32SM_TILING_LSE_OUT_RESIDUAL
        GbsaInferIntfRegular<half, float, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::TND, false, true,
                             NpuArch::Epilogue::LseMode::OUT_ONLY,
                             NpuArch::Epilogue::LseFormat::TN1>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_FP16_TND_TND_FP16SM_TILING_LSE_OUT_RESIDUAL
        GbsaInferIntfRegular<half, half, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::TND, false, true,
                             NpuArch::Epilogue::LseMode::OUT_ONLY,
                             NpuArch::Epilogue::LseFormat::TN1>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_BF16_TND_TND_FP32SM_TILING_LSE_OUT_RESIDUAL
        GbsaInferIntfRegular<bfloat16_t, float, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::TND, false, true,
                             NpuArch::Epilogue::LseMode::OUT_ONLY,
                             NpuArch::Epilogue::LseFormat::TN1>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_BF16_TND_TND_BF16SM_TILING_LSE_OUT_RESIDUAL
        GbsaInferIntfRegular<bfloat16_t, bfloat16_t, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::TND, false, true,
                             NpuArch::Epilogue::LseMode::OUT_ONLY,
                             NpuArch::Epilogue::LseFormat::TN1>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_FP16_TND_PA_BBND_FP32SM_TILING_RESIDUAL
        GbsaInferIntfRegular<half, float, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::BSND, true, true>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_BF16_TND_PA_BBND_FP32SM_TILING_RESIDUAL
        GbsaInferIntfRegular<bfloat16_t, float, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::BSND, true, true>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_FP16_TND_PA_BBND_FP16SM_TILING_RESIDUAL
        GbsaInferIntfRegular<half, half, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::BSND, true, true>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_BF16_TND_PA_BBND_BF16SM_TILING_RESIDUAL
        GbsaInferIntfRegular<bfloat16_t, bfloat16_t, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::BSND, true, true>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_FP16_TND_PA_BBND_FP32SM_TILING_LSE_OUT_RESIDUAL
        GbsaInferIntfRegular<half, float, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::BSND, true, true,
                             NpuArch::Epilogue::LseMode::OUT_ONLY,
                             NpuArch::Epilogue::LseFormat::TN1>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_FP16_TND_PA_BBND_FP16SM_TILING_LSE_OUT_RESIDUAL
        GbsaInferIntfRegular<half, half, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::BSND, true, true,
                             NpuArch::Epilogue::LseMode::OUT_ONLY,
                             NpuArch::Epilogue::LseFormat::TN1>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_BF16_TND_PA_BBND_FP32SM_TILING_LSE_OUT_RESIDUAL
        GbsaInferIntfRegular<bfloat16_t, float, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::BSND, true, true,
                             NpuArch::Epilogue::LseMode::OUT_ONLY,
                             NpuArch::Epilogue::LseFormat::TN1>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == GBSA_BF16_TND_PA_BBND_BF16SM_TILING_LSE_OUT_RESIDUAL
        GbsaInferIntfRegular<bfloat16_t, bfloat16_t, float, GbsaKernelArch35::Format::TND,
                             GbsaKernelArch35::Format::BSND, true, true,
                             NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::LseFormat::TN1>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, attentionOut, softmaxLse, user, tiling);
#endif
#endif
    }
}
