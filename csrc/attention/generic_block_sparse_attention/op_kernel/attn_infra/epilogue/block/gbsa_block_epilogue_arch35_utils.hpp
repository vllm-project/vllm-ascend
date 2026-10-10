/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_EPILOGUE_ARCH35_UTILS_HPP
#define GBSA_BLOCK_EPILOGUE_ARCH35_UTILS_HPP

#include "../../../attn_infra/gbsa_base_defs.hpp"

namespace NpuArch::Epilogue::Block {

struct UBufTileHelper {
    // Statistics retain their maximum row capacity independently of S/P/LO tiling.
    static constexpr uint32_t STATS_ROW_NUM = 64U;
    uint32_t qBaseTilePerSubCore;
    uint32_t kvBaseTilePerSubCore;
    uint32_t embedPerSubCore;
    uint32_t loUbBufNum;
    uint32_t loUbRowNum;
    uint32_t kvOffsetUbBufNum = 1U;
    bool loGmTransit;
    uint32_t sStartOffset;
    uint32_t pStartOffset;
    uint32_t loStartOffset;
    uint32_t goStartOffset;
    uint32_t lmStartOffset;
    uint32_t gmStartOffset;
    uint32_t dmStartOffset;
    uint32_t llStartOffset;
    uint32_t glStartOffset;
    uint32_t lseStartOffset;
    uint32_t maskStartOffset;

    __aicore__ inline UBufTileHelper() {}

    __aicore__ inline UBufTileHelper(uint32_t qs, uint32_t kvs, uint32_t d, uint32_t s, uint32_t p, uint32_t lo,
                                     uint32_t go, uint32_t lm, uint32_t gm, uint32_t dm, uint32_t ll, uint32_t gl,
                                     uint32_t lse, uint32_t mask, uint32_t loBufNum = 2U)
        : qBaseTilePerSubCore(qs),
          kvBaseTilePerSubCore(kvs),
          embedPerSubCore(d),
          loUbBufNum(loBufNum),
          loUbRowNum(qs),
          loGmTransit(false),
          sStartOffset(s),
          pStartOffset(p),
          loStartOffset(lo),
          goStartOffset(go),
          lmStartOffset(lm),
          gmStartOffset(gm),
          dmStartOffset(dm),
          llStartOffset(ll),
          glStartOffset(gl),
          lseStartOffset(lse),
          maskStartOffset(mask)
    {}
};

} // namespace NpuArch::Epilogue::Block

#endif // GBSA_BLOCK_EPILOGUE_ARCH35_UTILS_HPP
