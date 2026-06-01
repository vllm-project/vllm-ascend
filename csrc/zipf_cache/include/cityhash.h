#ifndef CITYHASH_H
#define CITYHASH_H

#include <stdint.h>
#include <stddef.h>

// 简化版 CityHash64 实现 - 针对小数据优化
// 基于 Google CityHash 算法

#ifdef __cplusplus
extern "C" {
#endif

uint64_t cityhash64(const void* data, size_t len);
uint64_t cityhash64_with_seed(const void* data, size_t len, uint64_t seed);

// 针对 N-Gram 优化的哈希函数
uint64_t ngram_hash(const uint32_t* tokens, uint8_t length);

#ifdef __cplusplus
}
#endif

#endif // CITYHASH_H
