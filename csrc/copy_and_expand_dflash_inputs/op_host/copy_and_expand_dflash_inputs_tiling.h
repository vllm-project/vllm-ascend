#ifndef COPY_AND_EXPAND_DFLASH_INPUTS_TILING_H
#define COPY_AND_EXPAND_DFLASH_INPUTS_TILING_H

#include "register/tilingdata_base.h"
#include "error_log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(CopyAndExpandDflashInputsTilingData)
    // ---- 分核参数 ----
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);            // 实际使用的核数
    TILING_DATA_FIELD_DEF(uint32_t, numReqs);                // 总请求数
    TILING_DATA_FIELD_DEF(uint32_t, reqsPerCore);            // 每核基础请求数
    TILING_DATA_FIELD_DEF(uint32_t, remainderReqs);          // 余数（前 remainder 个核多处理 1 个请求）

    // ---- 算子属性 ----
    TILING_DATA_FIELD_DEF(int32_t, parallelDraftingTokenId); // 并行推测解码 mask token ID
    TILING_DATA_FIELD_DEF(uint32_t, numQueryPerReq);         // 每个请求的 query token 数 (1 + num_speculative_tokens)
    TILING_DATA_FIELD_DEF(uint32_t, numSpeculativeTokens);   // 投机 token 数
    TILING_DATA_FIELD_DEF(uint32_t, blockSize);              // KV cache block size
    TILING_DATA_FIELD_DEF(uint32_t, totalInputTokens);       // 输入 context token 总数
    TILING_DATA_FIELD_DEF(int32_t, hasNumRejected);          // 是否有 rejected tokens (0/1)
    TILING_DATA_FIELD_DEF(uint32_t, blockTableStride);       // block_table步长（每个请求的block数量）

    // ---- 输出尺寸 ----
    TILING_DATA_FIELD_DEF(uint32_t, totalQueryTokens);       // 输出 query token 总数
END_TILING_DATA_DEF;

struct CopyAndExpandDflashInputsCompileInfo {
    uint32_t totalCoreNum = 0;
};

REGISTER_TILING_DATA_CLASS(CopyAndExpandDflashInputs, CopyAndExpandDflashInputsTilingData)

}  // namespace optiling

#endif  // COPY_AND_EXPAND_DFLASH_INPUTS_TILING_H