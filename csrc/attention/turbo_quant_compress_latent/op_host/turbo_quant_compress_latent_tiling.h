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
 * \file turbo_quant_compress_latent_tiling.h
 * \brief
 */
#ifndef TURBO_QUANT_COMPRESS_LATENT_TILING_H
#define TURBO_QUANT_COMPRESS_LATENT_TILING_H

#include <cstdint>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "err/ops_err.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

// TurboQuant 4-bit codebook: one nibble per element, so exactly 16 centroids.
constexpr int64_t TQ_COMPRESS_N_CENT = 16;
// The fp16 L2 norm ("vecNorm") stored right behind the packed nibbles.
constexpr int64_t TQ_COMPRESS_SCALE_BYTES = 2;
// Slots are padded up to this many bytes so that consecutive slots stay 64B aligned.
constexpr int64_t TQ_COMPRESS_SLOT_ALIGN = 64;
// Only the MLA kv_lora_rank used in production is enabled for now. The whole implementation is
// written against headDim, so widening this list is a validation exercise, not a rewrite.
constexpr int64_t TQ_COMPRESS_SUPPORTED_HEAD_DIM = 512;

// The codebook scan is a chain of narrow vector instructions whose fixed issue cost dominates, so the
// kernel processes several tokens per instruction. Measured on Atlas A2: a 4x wider loop costs only
// 1.82x the time, i.e. 2.2x better efficiency; the gain flattens out past 8 and the buffers grow
// linearly, so the batch is capped here. Must stay <= TQ_COMPRESS_MAX_TOKENS_PER_BATCH in the kernel.
constexpr int64_t TQ_COMPRESS_MAX_TOKENS_PER_BATCH = 12;
// Each token's L2 norm is reduced into its own 64B-aligned slot so one V->S sync covers the whole batch.
constexpr int64_t TQ_COMPRESS_NORM_SLOT_FLOATS = 16;
constexpr int64_t TQ_COMPRESS_UB_RESERVE = 1024;

// slot layout: [0, headDim/2) packed nibbles | [headDim/2, headDim/2+2) vecNorm fp16 | pad
inline int64_t TqCompressSlotSize(int64_t headDim)
{
    int64_t used = headDim / 2 + TQ_COMPRESS_SCALE_BYTES;
    return (used + TQ_COMPRESS_SLOT_ALIGN - 1) / TQ_COMPRESS_SLOT_ALIGN * TQ_COMPRESS_SLOT_ALIGN;
}

inline int64_t TqCompressAlign64(int64_t value)
{
    return (value + TQ_COMPRESS_SLOT_ALIGN - 1) / TQ_COMPRESS_SLOT_ALIGN * TQ_COMPRESS_SLOT_ALIGN;
}

// UB held per token in a batch: inQ + u + nib + tmp + sel + one, plus the slot, the half staging
// buffer and the compare mask.
inline int64_t TqCompressBytesPerToken(int64_t headDim, int64_t slotSize)
{
    return TqCompressAlign64(headDim * static_cast<int64_t>(sizeof(float))) * 6 + TqCompressAlign64(slotSize) +
           TqCompressAlign64(headDim * 2) + TqCompressAlign64(headDim / 8);
}

// UB that does not scale with the batch: the ReduceSum work area, the codebook and the norm slots.
inline int64_t TqCompressFixedBytes(int64_t headDim)
{
    return TqCompressAlign64(headDim * static_cast<int64_t>(sizeof(float))) +
           TqCompressAlign64(TQ_COMPRESS_N_CENT * static_cast<int64_t>(sizeof(float))) +
           TqCompressAlign64(TQ_COMPRESS_MAX_TOKENS_PER_BATCH * TQ_COMPRESS_NORM_SLOT_FLOATS *
                             static_cast<int64_t>(sizeof(float)));
}

BEGIN_TILING_DATA_DEF(TurboQuantCompressLatentTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numTokens);
TILING_DATA_FIELD_DEF(uint32_t, tokensPerCore);
TILING_DATA_FIELD_DEF(uint32_t, headDim);
TILING_DATA_FIELD_DEF(uint32_t, slotSize);
TILING_DATA_FIELD_DEF(uint32_t, tokensPerBatch);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TurboQuantCompressLatent, TurboQuantCompressLatentTilingData)

struct TurboQuantCompressLatentCompileInfo {};

} // namespace optiling
#endif // TURBO_QUANT_COMPRESS_LATENT_TILING_H
