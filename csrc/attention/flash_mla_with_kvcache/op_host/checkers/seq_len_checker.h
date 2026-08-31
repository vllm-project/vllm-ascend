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
 * \file actual_seq_len_checker.h
 * \brief Checker for cache_seqlens (B), cu_seqlens_q (B+1) and seqused_q (B)
 *
 * seq-lens contract: the KV side is paged, so the flash_attn
 * KV-side cumulative/used seq-lens inputs do NOT exist; per-batch KV-cache
 * lengths are carried by cache_seqlens (INT32 [b]), which is REQUIRED (paged
 * KV). cu_seqlens_q is REQUIRED when layout_q is TND and must be empty
 * otherwise (BNSD/BSND query layouts carry no cumulative seqlens). seqused_q
 * remains optional [b].
 */

#ifndef FLASH_MLA_WITH_KVCACHE_ACTUAL_SEQ_LEN_CHECKER_H
#define FLASH_MLA_WITH_KVCACHE_ACTUAL_SEQ_LEN_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
namespace flash_mla_with_kvcache {

class ActualSeqLenChecker : public FlashMlaWithKvcacheBaseChecker {
public:
    ActualSeqLenChecker() = default;
    ~ActualSeqLenChecker() override = default;

    ge::graphStatus CheckSinglePara(const FlashMlaWithKvcacheTilingInfo &faInfo) override;
    ge::graphStatus CheckParaExistence(const FlashMlaWithKvcacheTilingInfo &faInfo) override;
    ge::graphStatus CheckFeature(const FlashMlaWithKvcacheTilingInfo &faInfo) override;
    ge::graphStatus CheckMultiPara(const FlashMlaWithKvcacheTilingInfo &faInfo) override;

private:
    ge::graphStatus CheckSingleParaSequsedQ(const FlashMlaWithKvcacheTilingInfo &faInfo);
    ge::graphStatus CheckSingleParaCuSeqlensQ(const FlashMlaWithKvcacheTilingInfo &faInfo);
    ge::graphStatus CheckSingleParaCacheSeqlens(const FlashMlaWithKvcacheTilingInfo &faInfo);
    ge::graphStatus CheckSingleParaMaxSeqlenQ(const FlashMlaWithKvcacheTilingInfo &faInfo);
    ge::graphStatus CheckSingleParaMaxSeqlenKv(const FlashMlaWithKvcacheTilingInfo &faInfo);
};

} // namespace flash_mla_with_kvcache
} // namespace optiling
#endif // FLASH_MLA_WITH_KVCACHE_ACTUAL_SEQ_LEN_CHECKER_H
