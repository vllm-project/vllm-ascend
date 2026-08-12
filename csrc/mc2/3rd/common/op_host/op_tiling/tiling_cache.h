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
 * \file tiling_cache.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_HASH_TILING_CACHE_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_HASH_TILING_CACHE_H_

#include <map>
#include "lock.h"

namespace Ops {
namespace Transformer {
constexpr uint32_t kMaxTilingCacheEntryNum = 500;

template <typename HashInput, typename HashItem>
class TilingCache {
public:
    void Add(uint32_t key, const HashInput &hash_input, const HashItem &value)
    {
        rwlock_.wrlock();
        if (size_ >= kMaxTilingCacheEntryNum) {
            rwlock_.unlock();
            return;
        }

        if (map_.find(key) != map_.end()) {
            rwlock_.unlock();
            return;
        }

        map_.emplace(key, value);
        size_++;
        rwlock_.unlock();
        return;
    }

    void Replace(uint32_t key, const HashInput &hash_input, const HashItem &value)
    {
        rwlock_.wrlock();
        if (size_ >= kMaxTilingCacheEntryNum) {
            rwlock_.unlock();
            return;
        }

        if (map_.find(key) == map_.end()) {
            size_++;
        }
        map_.erase(key);
        map_.emplace(key, value);
        rwlock_.unlock();
        return;
    }

    bool Get(uint32_t key, const HashInput &hash_input, HashItem &value)
    {
        rwlock_.rdlock();
        auto iter = map_.find(key);
        if (iter == map_.end()) {
            rwlock_.unlock();
            return false;
        }
        if (!(hash_input == iter->second.input())) {
            rwlock_.unlock();
            OP_LOGD("CUBE", "inconsistent input data");
            return false;
        }

        value = iter->second;
        rwlock_.unlock();
        return true;
    }

private:
    std::map<uint32_t, HashItem> map_;
    uint32_t size_ = 0;
    Ops::Transformer::Optiling::RWLock rwlock_;
};
} // namespace Transformer
} // namespace Ops
#endif // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_HASH_TILING_CACHE_H_
