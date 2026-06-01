#ifndef HOT_TIER_H
#define HOT_TIER_H

#include "types.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// 开放寻址哈希表的辅助函数（被 local_hash.c 使用）

static inline uint32_t hot_tier_hash(uint64_t key, size_t size) {
    return (uint32_t)(key % size);
}

static inline uint32_t hot_tier_probe(uint32_t h, uint32_t i, size_t size) {
    return (h + i) % size;
}

#ifdef __cplusplus
}
#endif

#endif // HOT_TIER_H
