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
 * \file matmul_v3_l2_cache.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_L2_CACHE_H__
#define __OP_HOST_MATMUL_V3_L2_CACHE_H__

#include "matmul_v3_tiling.h"
#include "matmul_v3_common.h"

namespace optiling {
namespace mc2_matmul_v3 {
class Mc2L2Cache {
public:
    Mc2L2Cache(Mc2MatmulV3Args &args, Mc2MatmulV3TilingData &tilingData) : args_(args), tilingData_(tilingData)
    {
    }
    void SetL2CacheFlag(Mc2TilingEnable tilingEnable, uint64_t l2Size, uint32_t &l2CacheFlag);

private:
    void SetL2CacheFlag(bool aEnableL2Cache, bool bEnableL2Cache, bool cEnableL2Cache, bool biasEnableL2Cache,
                        uint32_t &l2CacheFlag);
    void SetL2CacheFlagMultiCoreSplitK(bool &aEnableL2Cache, bool &bEnableL2Cache) const;
    void SetL2CacheFlagSingleCoreSplitK(bool &aEnableL2Cache, bool &bEnableL2Cache) const;
    void SetL2CacheFlagBase(bool &aEnableL2Cache, bool &bEnableL2Cache) const;

private:
    Mc2MatmulV3Args &args_;
    Mc2MatmulV3TilingData &tilingData_;
};
} // namespace mc2_matmul_v3
} // namespace optiling
#endif // __OP_HOST_MATMUL_V3_L2_CACHE_H__
