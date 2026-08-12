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
 * \file hash.cpp
 * \brief function of hash
 */
#include "hash.h"

namespace Ops {
namespace Transformer {
static constexpr uint32_t kRolScrambleLeft = 15;
static constexpr uint32_t kRolScrambleRight = 17;
static constexpr uint32_t kRolBodyLeft = 13;
static constexpr uint32_t kRolBodyRight = 19;
static constexpr uint32_t kChunkOffset = 5;
static constexpr uint32_t kRolTailLeft = 8;
static constexpr uint32_t kRolTailRight = 13;
static constexpr uint32_t kReadSize = 16;
static constexpr int32_t kMaxTilingCacheEntryNum = 500;

static inline uint32_t MurmurScramble(uint32_t key)
{
    key *= static_cast<uint32_t>(0xcc9e2d51);
    key = (key << kRolScrambleLeft) | (key >> kRolScrambleRight);
    key *= static_cast<uint32_t>(0x1b873593);
    return key;
}

uint32_t MurmurHash(const void *src, uint32_t len, uint32_t seed)
{
    const uint32_t *key = static_cast<const uint32_t *>(src);
    uint32_t hash_key = seed;
    uint32_t tmp_key;
    // Read in blocks of 4
    for (uint32_t i = len >> 2UL; i > 0UL; i--) {
        // Get a source of differing results across endiannesses.
        tmp_key = *key;
        key++;
        hash_key ^= MurmurScramble(tmp_key);
        hash_key = (hash_key << kRolBodyLeft) | (hash_key >> kRolBodyRight);
        hash_key = hash_key * kChunkOffset + 0xe6546b64;
    }
    // Process the rest
    const uint8_t *rest_key = static_cast<const uint8_t *>(src);
    tmp_key = 0U;
    for (uint32_t i = len & 3U; i != 0U; i--) {
        tmp_key <<= kRolTailLeft;
        tmp_key |= rest_key[i - 1U];
    }
    hash_key ^= MurmurScramble(tmp_key);
    // Finalize
    hash_key ^= len;
    hash_key ^= hash_key >> kReadSize;
    hash_key *= static_cast<uint32_t>(0x85ebca6b);
    hash_key ^= hash_key >> kRolTailRight;
    hash_key *= static_cast<uint32_t>(0xc2b2ae35);
    hash_key ^= hash_key >> kReadSize;
    return hash_key;
}
} // namespace Transformer
} // namespace Ops
