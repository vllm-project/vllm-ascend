/*
 * Copyright (c) 2026 TriangleMix contributors.
 * This program is licensed under CANN Open Software License Agreement
 * Version 2.0. See LICENSE in the repository root.
 *
 * Paged BSND address mapping extracted from the CANN 9.0.1
 * FusedInferAttentionScore QK/PV loaders:
 *
 *   fused_infer_attention_score/attn_infra/gemm/block/block_mmad_qk.hpp
 *   fused_infer_attention_score/attn_infra/gemm/block/block_mmad_pv.hpp
 *
 * Source revision: ops-transformer v9.0.1,
 * 8038339a99bae113a7ae07f4547306d6d15bbddf.
 *
 * This file establishes the exact logical-page -> physical-page address
 * contract used by the fused Cube/Vector path.
 */
#ifndef TRIANGLE_PAGED_ATTENTION_FIA_PAGED_KV_ADDRESS_H
#define TRIANGLE_PAGED_ATTENTION_FIA_PAGED_KV_ADDRESS_H

#include "kernel_operator.h"

namespace TrianglePaged {

struct PagedKvLocation {
    uint32_t logicalPage;
    uint32_t physicalPage;
    uint32_t offsetInPage;
    uint32_t elementOffset;
};

/*
 * Cache layout is BSND:
 *   [physical_page, page_size, kv_head, head_dim].
 *
 * Batch is fixed to one in the first production fast path, so blockTable is
 * the first and only row of [1, max_pages].
 */
__aicore__ inline PagedKvLocation ResolvePagedBsndLocation(
    AscendC::GlobalTensor<int32_t>& blockTable,
    uint32_t logicalToken,
    uint32_t kvHead,
    uint32_t pageSize,
    uint32_t kvHeads,
    uint32_t headDim)
{
    PagedKvLocation location{};
    location.logicalPage = logicalToken / pageSize;
    location.offsetInPage = logicalToken % pageSize;
    location.physicalPage =
        static_cast<uint32_t>(blockTable.GetValue(location.logicalPage));
    location.elementOffset =
        (((location.physicalPage * pageSize + location.offsetInPage) *
          kvHeads + kvHead) * headDim);
    return location;
}

}  // namespace TrianglePaged

#endif  // TRIANGLE_PAGED_ATTENTION_FIA_PAGED_KV_ADDRESS_H
