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
 * \file rotary_position_embedding.cpp
 * \brief
 */
#if defined(__DAV_C310__)
    #include "kernel_operator.h"
    #include "rotary_position_embedding_reg_bab.h"
    #include "rotary_position_embedding_reg_ab.h"
    #include "rotary_position_embedding_reg_aba_and_ba.h"
    #include "rotary_position_embedding_reg_a_and_b.h"
#else
    #include "kernel_operator.h"
    #include "kernel_tiling/kernel_tiling.h"
    #include "inplace_partial_rotary_mul.h"
    #include "rotate_interleaved_split_s.h"
    #include "rotate_interleaved_split_bs.h"
    #include "rotate_interleaved_split_bsn.h"
    #include "rotate_interleaved_split_s_pad.h"
    #include "rotate_interleaved_split_bs_pad.h"
    #include "rotate_interleaved_split_bsn_pad.h"
    using namespace AscendC;
    using namespace RotateInterleavedN;
#endif

#define TILING_KEY_ABA 20010
#define TILING_KEY_BA 20011
#define TILING_KEY_BAB 20020
#define TILING_KEY_AB 20030
#define TILING_KEY_A 20040
#define TILING_KEY_B 20041
#define TILING_KEY1 1
#define TILING_KEY2 2

using namespace AscendC;
using namespace InplacePartialRotaryMul;

extern "C" __global__ __aicore__ void inplace_partial_rotary_mul(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    #if defined(__DAV_C310__) 
        if (TILING_KEY_IS(TILING_KEY_ABA))
        {
            GET_TILING_DATA_WITH_STRUCT(RopeRegbaseTilingData, tiling_data_in, tiling);
            const RopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
            InplacePartialRotaryMul::RotaryPositionEmbeddingABAAndBA<DTYPE_X, false> op;
            op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
            op.Process();
        }
        else if (TILING_KEY_IS(TILING_KEY_BA))
        {
            GET_TILING_DATA_WITH_STRUCT(RopeRegbaseTilingData, tiling_data_in, tiling);
            const RopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
            InplacePartialRotaryMul::RotaryPositionEmbeddingABAAndBA<DTYPE_X, true> op;
            op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
            op.Process();
        }
        else if (TILING_KEY_IS(TILING_KEY_BAB))
        {
            GET_TILING_DATA_WITH_STRUCT(RopeRegbaseTilingData, tiling_data_in, tiling);
            const RopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
            InplacePartialRotaryMul::RotaryPositionEmbeddingBAB<DTYPE_X> op(&pipe, tilingData);
            op.Init(x, cos, sin, y);
            op.Process();
        }
        else if (TILING_KEY_IS(TILING_KEY_AB))
        {
            GET_TILING_DATA_WITH_STRUCT(RopeRegbaseTilingData, tiling_data_in, tiling);
            const RopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
            InplacePartialRotaryMul::RotaryPositionEmbeddingAB<DTYPE_X> op;
            op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
            op.Process();
        }
        else if (TILING_KEY_IS(TILING_KEY_A))
        {
            GET_TILING_DATA_WITH_STRUCT(RopeRegbaseTilingData, tiling_data_in, tiling);
            const RopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
            InplacePartialRotaryMul::RotaryPositionEmbeddingAAndB<DTYPE_X, false> op;
            op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
            op.Process();
        }
        else if (TILING_KEY_IS(TILING_KEY_B))
        {
            GET_TILING_DATA_WITH_STRUCT(RopeRegbaseTilingData, tiling_data_in, tiling);
            const RopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
            InplacePartialRotaryMul::RotaryPositionEmbeddingAAndB<DTYPE_X, true> op;
            op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
            op.Process();
        }
    #else
        GET_TILING_DATA_WITH_STRUCT(RopeRegbaseTilingData, tilingData, tiling);
        const RopeRegbaseTilingData* __restrict__ tilingData1 = &tilingData;
        if (TILING_KEY_IS(TILING_KEY1)) {
            InplacePartialRotaryMul::InplacePartialRotaryMulABA<DTYPE_X, true> op;
            op.Init(x, cos, sin, y, workspace, tilingData1, &pipe);
            op.Process();
            return;
        }
        if (TILING_KEY_IS(TILING_KEY2)) {
            InplacePartialRotaryMul::InplacePartialRotaryMulABA<DTYPE_X, false> op;
            op.Init(x, cos, sin, y, workspace, tilingData1, &pipe);
            op.Process();
            return;
        }
        // mode: rotate_interleaved
        if (TILING_KEY_IS(2000)) {
            InterleavedSplitS<half> interleavedSplitS;
            interleavedSplitS.Init(x, cos, sin, y, tilingData1, &pipe);
            interleavedSplitS.Process();
        } else if (TILING_KEY_IS(2010)) {
            InterleavedSplitS<bfloat16_t> interleavedSplitS;
            interleavedSplitS.Init(x, cos, sin, y, tilingData1, &pipe);
            interleavedSplitS.Process();
        } else if (TILING_KEY_IS(2020)) {
            InterleavedSplitS<float> interleavedSplitS;
            interleavedSplitS.Init(x, cos, sin, y, tilingData1, &pipe);
            interleavedSplitS.Process();
        } else if (TILING_KEY_IS(2100)) {
            InterleavedSplitBS<half> interleavedSplitBS;
            interleavedSplitBS.Init(x, cos, sin, y, tilingData1, &pipe);
            interleavedSplitBS.Process();
        } else if (TILING_KEY_IS(2110)) {
            InterleavedSplitBS<bfloat16_t> interleavedSplitBS;
            interleavedSplitBS.Init(x, cos, sin, y, tilingData1, &pipe);
            interleavedSplitBS.Process();
        } else if (TILING_KEY_IS(2120)) {
            InterleavedSplitBS<float> interleavedSplitBS;
            interleavedSplitBS.Init(x, cos, sin, y, tilingData1, &pipe);
            interleavedSplitBS.Process();
        } else if (TILING_KEY_IS(2200)) {
            InterleavedSplitBSN<half> interleavedSplitBSN;
            interleavedSplitBSN.Init(x, cos, sin, y, tilingData1, &pipe);
            interleavedSplitBSN.Process();
        } else if (TILING_KEY_IS(2210)) {
            InterleavedSplitBSN<bfloat16_t> interleavedSplitBSN;
            interleavedSplitBSN.Init(x, cos, sin, y, tilingData1, &pipe);
            interleavedSplitBSN.Process();
        } else if (TILING_KEY_IS(2220)) {
            InterleavedSplitBSN<float> interleavedSplitBSN;
            interleavedSplitBSN.Init(x, cos, sin, y, tilingData1, &pipe);
            interleavedSplitBSN.Process();
        } else if (TILING_KEY_IS(2001)) {
            InterleavedSplitSPad<half> interleavedSplitSPad;
            interleavedSplitSPad.Init(x, cos, sin, y, tilingData1, &pipe);
            interleavedSplitSPad.Process();
        } else if (TILING_KEY_IS(2011)) {
            InterleavedSplitSPad<bfloat16_t> interleavedSplitSPad;
            interleavedSplitSPad.Init(x, cos, sin, y, tilingData1, &pipe);
            interleavedSplitSPad.Process();
        } else if (TILING_KEY_IS(2021)) {
            InterleavedSplitSPad<float> interleavedSplitSPad;
            interleavedSplitSPad.Init(x, cos, sin, y, tilingData1, &pipe);
            interleavedSplitSPad.Process();
        } else if (TILING_KEY_IS(2101)) {
            InterleavedSplitBSPad<half> interleavedSplitBSPad;
            interleavedSplitBSPad.Init(x, cos, sin, y, tilingData1, &pipe);
            interleavedSplitBSPad.Process();
        } else if (TILING_KEY_IS(2111)) {
            InterleavedSplitBSPad<bfloat16_t> interleavedSplitBSPad;
            interleavedSplitBSPad.Init(x, cos, sin, y, tilingData1, &pipe);
            interleavedSplitBSPad.Process();
        } else if (TILING_KEY_IS(2121)) {
            InterleavedSplitBSPad<float> interleavedSplitBSPad;
            interleavedSplitBSPad.Init(x, cos, sin, y, tilingData1, &pipe);
            interleavedSplitBSPad.Process();
        } else if (TILING_KEY_IS(2201)) {
            InterleavedSplitBSNPad<half> interleavedSplitBSNPad;
            interleavedSplitBSNPad.Init(x, cos, sin, y, tilingData1, &pipe);
            interleavedSplitBSNPad.Process();
        } else if (TILING_KEY_IS(2211)) {
            InterleavedSplitBSNPad<bfloat16_t> interleavedSplitBSNPad;
            interleavedSplitBSNPad.Init(x, cos, sin, y, tilingData1, &pipe);
            interleavedSplitBSNPad.Process();
        } else if (TILING_KEY_IS(2221)) {
            InterleavedSplitBSNPad<float> interleavedSplitBSNPad;
            interleavedSplitBSNPad.Init(x, cos, sin, y, tilingData1, &pipe);
            interleavedSplitBSNPad.Process();
        }
    #endif
}

