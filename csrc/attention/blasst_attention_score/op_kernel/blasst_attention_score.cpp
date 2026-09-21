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
* \file blasst_attention_score.cpp
* \brief Kernel entry and tiling-key dispatch for VllmBlasstAttentionScore.
*/

#include "kernel_operator.h"
#include "adv_api/matmul/matmul_intf.h"
#include "blasst_attention_score_tiling_key.h"
#include "kernel/blasst_attention_score_kernel.h"

using namespace AscendC;
using namespace NpuArch;

namespace SplitFuse {
    template <
        typename InputDtypeQ = half,
        typename InputDtypeKv = half,
        typename IntermCalcPrec = float,
        bool PagedCacheFlag = false,
        FaiKernel::MaskType maskCategory = FaiKernel::MaskType::NO_MASK,
        FaiKernel::inputLayout inLayout = FaiKernel::inputLayout::TND,
        Epilogue::LseMode lseMode = Epilogue::LseMode::NONE,
        bool isFD = false>
    __global__ __aicore__ void FAInfer(
        GM_ADDR q,
        GM_ADDR k,
        GM_ADDR v,
        GM_ADDR mask,
        GM_ADDR blockTables,
        GM_ADDR o,
        GM_ADDR lse,
        GM_ADDR sparseStats,
        GM_ADDR actualQseqlen,
        GM_ADDR actualKvseqlen,
        GM_ADDR workspace,
        GM_ADDR tiling)
    {
        using ArchTag = Arch::AtlasA2;
        using ElementQ = InputDtypeQ;
        using LayoutQ = layout::RowMajor;
        using ElementK = InputDtypeKv;
        using LayoutK = layout::ColumnMajor;
        using ElementV = InputDtypeKv;
        using LayoutV = layout::RowMajor;
        // S (QK scores) handoff dtype: cube sub-core writes S to the GM workspace,
        // vector sub-core reads it back for softmax. fp16 halves the round-trip
        // bytes but forces a per-tile fp16->fp32 upcast pass in softmax, which
        // measured as the dominant occ0 prefill cost (26us of 256us wall: NOCAST
        // ablation 256.4->230.6 vs FIA 228.3) -- the vector passes + barriers
        // stall the cube/vec ring far beyond their busy time. fp32 S restores
        // FIX F322F32 + a direct MTE2 landing in the fp32 softmax buffer with no
        // upcast; the extra L2 bytes are absorbed (occ0 DDR traffic stays <1GB/s).
        using ElementS = float;
        using LayoutS = layout::RowMajor;
        using ElementP = InputDtypeQ;
        using LayoutP = layout::RowMajor;
        using ElementO = InputDtypeQ;
        using LayoutO = layout::RowMajor;
        using ElementLse = float;
        using LayoutLse = layout::RowMajor;
        using ElementMask = int8_t;
        using LayoutMask = layout::RowMajor;
        // OTmp (per-stack-tile PV accumulator) handoff dtype: cube FIX-writes
        // OTmp to GM and the vector epilogue reads it back every stack tile
        // (~128MB/call round-trip at fp32 for occ0 prefill shapes). fp16 halves
        // that traffic; FIX-pipe converts the fp32 accumulator natively
        // (F322F16, same path as the fp16 S workspace) and the epilogue upcasts
        // to fp32 in UB before the rescale math. ElementUpdate (the >64-row
        // row-loop round-trip) stays fp32: it carries the running O sum and is
        // read back through a plain MTE2 load, so it keeps the wider dtype.
        using ElementOTmp = half;
        using LayoutOTmp = layout::RowMajor;
        using ElementUpdate = IntermCalcPrec;
        using LayoutUpdate = layout::RowMajor;

        using L1TileShapeQK = GemmShape<Q_TILE_CEIL, 128, 128>;
        using L0TileShapeQK = GemmShape<128, 128, 128>;
        using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQK<PagedCacheFlag, false>;
        using QType = Gemm::GemmType<ElementQ, LayoutQ>;
        using KType = Gemm::GemmType<ElementK, LayoutK>;
        using SType = Gemm::GemmType<ElementS, LayoutS>;
        using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK,
                                                QType, KType, SType>;

        using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax<lseMode, static_cast<Epilogue::MaskMode>(maskCategory), IntermCalcPrec>;
        using PType = Gemm::GemmType<ElementP, LayoutP>;
        using maskType = Gemm::GemmType<ElementMask, LayoutMask>;
        using EpilogueOnlineSoftmax =
            Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, maskType>;

        using L1TileShapePV = GemmShape<128, 128, 256>;
        using L0TileShapePV = GemmShape<128, 128, 128>;
        using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPV<PagedCacheFlag, false>;
        using VType = Gemm::GemmType<ElementV, LayoutV>;
        using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
        using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV,
                                                PType, VType, OTmpType>;

        using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleO<lseMode, IntermCalcPrec>;
        using OType = Gemm::GemmType<ElementO, LayoutO>;
        using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
        using LseType = Gemm::GemmType<ElementLse, LayoutLse>;
        using EpilogueRescaleO =
            Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType, LseType>;

        using DispatchPolicyInitOutWhenZero = Epilogue::EpilogueAtlasA2InitOutWhenZero<lseMode>;
        using EpilogueInitOut =
            Epilogue::Block::BlockEpilogue<DispatchPolicyInitOutWhenZero, OType, LseType>;

        using FAInferKernelType = FAInferKernel<BlockMmadQK, BlockMmadPV,
                                                EpilogueOnlineSoftmax, EpilogueRescaleO, EpilogueInitOut,
                                                PagedCacheFlag, maskCategory, inLayout,
                                                Epilogue::Block::CombineScale<OType, LseType>, isFD>;

        // actualQseqlen / actualKvseqlen stay in the launch ABI (op inputs 5/6
        // remain declared OPTIONAL in the op def) but are absent in
        // host-list-only mode; seq lens come from fATilingData->actualQSeq /
        // actualKvSeq instead. Do not reintroduce reads of these pointers.
        (void)actualQseqlen;
        (void)actualKvseqlen;
        FAIKernelParams params{q, k, v, mask, blockTables, o, lse, sparseStats, workspace, tiling};
        FAInferKernelType flashAttnInfer;
        flashAttnInfer(params);
    }

}

#define DISPATCH_FA_INFER(KEY, DTYPE_Q, DTYPE_KV, MASK_ENUM) \
    if (TILING_KEY_VAR == KEY) { \
        SplitFuse::FAInfer<DTYPE_Q, DTYPE_KV, float, false, \
                           MASK_ENUM, FaiKernel::inputLayout::TND>( \
            query, key, value, attenMask, blocktable, attentionOut, softmaxLse, sparseStats, \
            actualSeqLengths, actualSeqLengthsKV, user, tiling); \
    }

#define DISPATCH_FA_INFER_LSE(KEY, DTYPE_Q, DTYPE_KV, MASK_ENUM) \
    if (TILING_KEY_VAR == KEY) { \
        SplitFuse::FAInfer<DTYPE_Q, DTYPE_KV, float, false, \
                           MASK_ENUM, FaiKernel::inputLayout::TND, \
                           NpuArch::Epilogue::LseMode::OUT_ONLY>( \
            query, key, value, attenMask, blocktable, attentionOut, softmaxLse, sparseStats, \
            actualSeqLengths, actualSeqLengthsKV, user, tiling); \
    }

#define DISPATCH_FA_INFER_PAGED_FD(KEY, DTYPE_Q, DTYPE_KV, MASK_ENUM) \
    if (TILING_KEY_VAR == KEY) { \
        SplitFuse::FAInfer<DTYPE_Q, DTYPE_KV, float, true, \
                           MASK_ENUM, FaiKernel::inputLayout::TND, \
                           NpuArch::Epilogue::LseMode::NONE, \
                           true>( \
            query, key, value, attenMask, blocktable, attentionOut, softmaxLse, sparseStats, \
            actualSeqLengths, actualSeqLengthsKV, user, tiling); \
    }

#define DISPATCH_FA_INFER_PAGED(KEY, DTYPE_Q, DTYPE_KV, MASK_ENUM) \
    if (TILING_KEY_VAR == KEY) { \
        SplitFuse::FAInfer<DTYPE_Q, DTYPE_KV, float, true, \
                           MASK_ENUM, FaiKernel::inputLayout::TND>( \
            query, key, value, attenMask, blocktable, attentionOut, softmaxLse, sparseStats, \
            actualSeqLengths, actualSeqLengthsKV, user, tiling); \
    }

#define DISPATCH_FA_INFER_PAGED_LSE(KEY, DTYPE_Q, DTYPE_KV, MASK_ENUM) \
    if (TILING_KEY_VAR == KEY) { \
        SplitFuse::FAInfer<DTYPE_Q, DTYPE_KV, float, true, \
                           MASK_ENUM, FaiKernel::inputLayout::TND, \
                           NpuArch::Epilogue::LseMode::OUT_ONLY>( \
            query, key, value, attenMask, blocktable, attentionOut, softmaxLse, sparseStats, \
            actualSeqLengths, actualSeqLengthsKV, user, tiling); \
    }

extern "C" __global__ __aicore__ void vllm_blasst_attention_score(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *pse_shift, __gm__ uint8_t *attenMask,
    __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKV,
    __gm__ uint8_t *blocktable, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *sparseStats,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    // Tiling-key discovery for the compile framework. These statements expand to
    // (g_tilingKey == (<key>)) in the preprocessed source so that the framework
    // registers every supported key and generates the FAInferTilingData struct.
    TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_FD_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_FD_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_FD_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_FD_TILING);

    // Dispatch using runtime-evaluated tiling key (the if-body is compiled per key above).
    DISPATCH_FA_INFER(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING,
                      half, half, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER_LSE(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING,
                          half, half, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING,
                      half, half, FaiKernel::MaskType::MASK_CAUSAL);
    DISPATCH_FA_INFER_LSE(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING,
                          half, half, FaiKernel::MaskType::MASK_CAUSAL);

    DISPATCH_FA_INFER(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING,
                      bfloat16_t, bfloat16_t, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER_LSE(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING,
                          bfloat16_t, bfloat16_t, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING,
                      bfloat16_t, bfloat16_t, FaiKernel::MaskType::MASK_CAUSAL);
    DISPATCH_FA_INFER_LSE(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING,
                          bfloat16_t, bfloat16_t, FaiKernel::MaskType::MASK_CAUSAL);

    DISPATCH_FA_INFER_PAGED_FD(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_FD_TILING,
                               half, half, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER_PAGED_FD(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_FD_TILING,
                               half, half, FaiKernel::MaskType::MASK_CAUSAL);
    DISPATCH_FA_INFER_PAGED_FD(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_FD_TILING,
                               bfloat16_t, bfloat16_t, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER_PAGED_FD(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_FD_TILING,
                               bfloat16_t, bfloat16_t, FaiKernel::MaskType::MASK_CAUSAL);

    DISPATCH_FA_INFER_PAGED(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING,
                            half, half, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER_PAGED_LSE(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING,
                                half, half, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER_PAGED(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING,
                            half, half, FaiKernel::MaskType::MASK_CAUSAL);
    DISPATCH_FA_INFER_PAGED_LSE(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING,
                                half, half, FaiKernel::MaskType::MASK_CAUSAL);
    DISPATCH_FA_INFER_PAGED(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING,
                            bfloat16_t, bfloat16_t, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER_PAGED_LSE(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING,
                                bfloat16_t, bfloat16_t, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER_PAGED(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING,
                            bfloat16_t, bfloat16_t, FaiKernel::MaskType::MASK_CAUSAL);
    DISPATCH_FA_INFER_PAGED_LSE(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING,
                                bfloat16_t, bfloat16_t, FaiKernel::MaskType::MASK_CAUSAL);
}
