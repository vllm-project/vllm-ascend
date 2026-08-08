/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file batch_matmul_transpose.cpp
 * \brief
 */
#include "batch_matmul_transpose.h"
#include "batch_matmul_transpose_tiling_data.h"

extern "C" __global__ __aicore__ void batch_matmul_transpose(GM_ADDR gm_a, GM_ADDR gm_b, GM_ADDR gm_c,
                                                             GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    GET_TILING_DATA_WITH_STRUCT(BatchMatmulTransposeTilingData, tiling_data, tiling);

    PpMatmulEinSum<0, false, false, half, half, DataFormat::ND>
        einsum_0_n_fp16_nd;  // swizzleDir[0] transA[0] transB[0] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0]
                             // DataFormatB[0]
    PpMatmulEinSum<1, false, false, half, half, DataFormat::ND>
        einsum_1_n_fp16_nd;  // swizzleDir[1] transA[0] transB[0] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0]
                             // DataFormatB[0]
    PpMatmulEinSum<0, false, true, half, half, DataFormat::ND>
        einsum_0_t_fp16_nd;  // swizzleDir[0] transA[0] transB[1] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0]
                             // DataFormatB[0]
    PpMatmulEinSum<1, false, true, half, half, DataFormat::ND>
        einsum_1_t_fp16_nd;  // swizzleDir[1] transA[0] transB[1] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0]
                             // DataFormatB[0]
    PpMatmulEinSum<0, false, false, __bf16, __bf16, DataFormat::ND>
        einsum_0_n_bf16_nd;  // swizzleDir[0] transA[0] transB[0] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0]
                             // DataFormatB[0]
    PpMatmulEinSum<1, false, false, __bf16, __bf16, DataFormat::ND>
        einsum_1_n_bf16_nd;  // swizzleDir[1] transA[0] transB[0] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0]
                             // DataFormatB[0]
    PpMatmulEinSum<0, false, true, __bf16, __bf16, DataFormat::ND>
        einsum_0_t_bf16_nd;  // swizzleDir[0] transA[0] transB[1] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0]
                             // DataFormatB[0]
    PpMatmulEinSum<1, false, true, __bf16, __bf16, DataFormat::ND>
        einsum_1_t_bf16_nd;  // swizzleDir[1] transA[0] transB[1] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0]
                             // DataFormatB[0]

    PpMatmulEinSum<0, false, false, half, half, DataFormat::NZ>
        einsum_0_n_fp16_nz;  // swizzleDir[0] transA[0] transB[0] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0]
                             // DataFormatB[1]
    PpMatmulEinSum<1, false, false, half, half, DataFormat::NZ>
        einsum_1_n_fp16_nz;  // swizzleDir[1] transA[0] transB[0] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0]
                             // DataFormatB[1]
    PpMatmulEinSum<0, false, true, half, half, DataFormat::NZ>
        einsum_0_t_fp16_nz;  // swizzleDir[0] transA[0] transB[1] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0]
                             // DataFormatB[1]
    PpMatmulEinSum<1, false, true, half, half, DataFormat::NZ>
        einsum_1_t_fp16_nz;  // swizzleDir[1] transA[0] transB[1] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0]
                             // DataFormatB[1]
    PpMatmulEinSum<0, false, false, __bf16, __bf16, DataFormat::NZ>
        einsum_0_n_bf16_nz;  // swizzleDir[0] transA[0] transB[0] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0]
                             // DataFormatB[1]
    PpMatmulEinSum<1, false, false, __bf16, __bf16, DataFormat::NZ>
        einsum_1_n_bf16_nz;  // swizzleDir[1] transA[0] transB[0] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0]
                             // DataFormatB[1]
    PpMatmulEinSum<0, false, true, __bf16, __bf16, DataFormat::NZ>
        einsum_0_t_bf16_nz;  // swizzleDir[0] transA[0] transB[1] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0]
                             // DataFormatB[1]
    PpMatmulEinSum<1, false, true, __bf16, __bf16, DataFormat::NZ>
        einsum_1_t_bf16_nz;  // swizzleDir[1] transA[0] transB[1] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0]
                             // DataFormatB[1]

    SetPadding<uint64_t>((uint64_t)0);
    SetNdpara(1, 0, 0);
    SetAtomicnone();

    uint32_t masked_key = tiling_data.tilingKey >> 2;
    switch (masked_key) {
        case 0b00000100100100:
        case 0b01000100100100:
            einsum_0_n_fp16_nd.Init(gm_a, gm_b, gm_c, &tiling_data);
            einsum_0_n_fp16_nd.Process();
            break;
        case 0b00100100100100:
        case 0b01100100100100:
            einsum_0_t_fp16_nd.Init(gm_a, gm_b, gm_c, &tiling_data);
            einsum_0_t_fp16_nd.Process();
            break;
        case 0b10000100100100:
        case 0b11000100100100:
            einsum_1_n_fp16_nd.Init(gm_a, gm_b, gm_c, &tiling_data);
            einsum_1_n_fp16_nd.Process();
            break;
        case 0b10100100100100:
        case 0b11100100100100:
            einsum_1_t_fp16_nd.Init(gm_a, gm_b, gm_c, &tiling_data);
            einsum_1_t_fp16_nd.Process();
            break;
        case 0b00001001001000:
        case 0b01001001001000:
            einsum_0_n_bf16_nd.Init(gm_a, gm_b, gm_c, &tiling_data);
            einsum_0_n_bf16_nd.Process();
            break;
        case 0b00101001001000:
        case 0b01101001001000:
            einsum_0_t_bf16_nd.Init(gm_a, gm_b, gm_c, &tiling_data);
            einsum_0_t_bf16_nd.Process();
            break;
        case 0b10001001001000:
        case 0b11001001001000:
            einsum_1_n_bf16_nd.Init(gm_a, gm_b, gm_c, &tiling_data);
            einsum_1_n_bf16_nd.Process();
            break;
        case 0b10101001001000:
        case 0b11101001001000:
            einsum_1_t_bf16_nd.Init(gm_a, gm_b, gm_c, &tiling_data);
            einsum_1_t_bf16_nd.Process();
            break;

        case 0b00000100100101:
        case 0b01000100100101:
            einsum_0_n_fp16_nz.Init(gm_a, gm_b, gm_c, &tiling_data);
            einsum_0_n_fp16_nz.Process();
            break;
        case 0b00100100100101:
        case 0b01100100100101:
            einsum_0_t_fp16_nz.Init(gm_a, gm_b, gm_c, &tiling_data);
            einsum_0_t_fp16_nz.Process();
            break;
        case 0b10000100100101:
        case 0b11000100100101:
            einsum_1_n_fp16_nz.Init(gm_a, gm_b, gm_c, &tiling_data);
            einsum_1_n_fp16_nz.Process();
            break;
        case 0b10100100100101:
        case 0b11100100100101:
            einsum_1_t_fp16_nz.Init(gm_a, gm_b, gm_c, &tiling_data);
            einsum_1_t_fp16_nz.Process();
            break;
        case 0b00001001001001:
        case 0b01001001001001:
            einsum_0_n_bf16_nz.Init(gm_a, gm_b, gm_c, &tiling_data);
            einsum_0_n_bf16_nz.Process();
            break;
        case 0b00101001001001:
        case 0b01101001001001:
            einsum_0_t_bf16_nz.Init(gm_a, gm_b, gm_c, &tiling_data);
            einsum_0_t_bf16_nz.Process();
            break;
        case 0b10001001001001:
        case 0b11001001001001:
            einsum_1_n_bf16_nz.Init(gm_a, gm_b, gm_c, &tiling_data);
            einsum_1_n_bf16_nz.Process();
            break;
        case 0b10101001001001:
        case 0b11101001001001:
            einsum_1_t_bf16_nz.Init(gm_a, gm_b, gm_c, &tiling_data);
            einsum_1_t_bf16_nz.Process();
            break;
        default:
            break;
    }
}
