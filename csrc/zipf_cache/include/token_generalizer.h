#ifndef TOKEN_GENERALIZER_H
#define TOKEN_GENERALIZER_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Token Generalizer - 基于运行时频率的 token 泛化
 *
 * 核心思路：
 * - 高频 token（结构性 token：标点、介词、关键字等）保留原始 ID
 * - 低频 token（内容性 token：变量名、数字、字符串等）映射到通配符 ID
 * - 泛化后的 n-gram 可以匹配结构相似但内容不同的模式
 *
 * 实现：
 * - 使用哈希表记录每个 token 的出现频率
 * - 当频率 >= 阈值时，token 被视为"高频"，保留原始 ID
 * - 否则映射到 WILDCARD_TOKEN
 * - 阈值可以固定，也可以自适应（维护 top-K）
 */

#define WILDCARD_TOKEN 0xFFFFFFFE
#define GENERALIZER_EMPTY_KEY 0xFFFFFFFF

// 多桶泛化：低频 token 按共现分布分到不同的语义桶
#define NUM_WILDCARD_BUCKETS 16
#define WILDCARD_BASE 0xFFFF0000  // 桶 ID = WILDCARD_BASE + bucket (0..15)

// 频率表条目（含共现分布 hash）
typedef struct {
    uint32_t token;          // token ID
    uint32_t count;          // 出现次数
    uint32_t context_hash;   // 左邻居累积 hash（用于语义分桶）
} FreqEntry;

// Token 泛化器
typedef struct {
    FreqEntry* entries;       // 哈希表
    size_t capacity;          // 哈希表容量
    size_t count;             // 已使用的槽数

    uint32_t freq_threshold;  // 频率阈值：>= 此值的 token 保留原始 ID
    uint32_t total_updates;   // 总更新次数（用于自适应阈值）

    // 快速查找缓存：高频 token 的 bitmap（可选优化）
    // 对于 vocab_size <= 256K，用 bitmap 可以做到 O(1) 查找
    uint8_t* high_freq_bitmap; // bitmap[token / 8] & (1 << (token % 8))
    size_t bitmap_size;        // bitmap 字节数
    uint32_t bitmap_vocab_max; // bitmap 覆盖的最大 token ID
} TokenGeneralizer;

// 创建泛化器
// capacity: 频率表容量（建议 vocab_size * 2）
// freq_threshold: 初始频率阈值
// bitmap_vocab_max: bitmap 覆盖的最大 token ID（0 表示不使用 bitmap）
TokenGeneralizer* token_generalizer_create(size_t capacity,
                                           uint32_t freq_threshold,
                                           uint32_t bitmap_vocab_max);

// 释放泛化器
void token_generalizer_free(TokenGeneralizer* gen);

// 更新 token 频率（带左邻居信息，用于共现分布分桶）
// prev_token: 前一个 token（用于累积 context_hash），0xFFFFFFFF 表示无前驱
// 返回更新后的频率
uint32_t token_generalizer_update(TokenGeneralizer* gen, uint32_t token, uint32_t prev_token);

// 批量更新 token 频率（带左邻居信息）
void token_generalizer_update_batch(TokenGeneralizer* gen,
                                    const uint32_t* tokens,
                                    size_t count);

// 泛化单个 token
// 高频 token 返回原始 ID，低频 token 返回语义桶 ID
static inline uint32_t token_generalizer_map(const TokenGeneralizer* gen,
                                              uint32_t token) {
    if (!gen) return token;

    // 快速路径：bitmap 查找（只判断高频/低频）
    if (gen->high_freq_bitmap && token <= gen->bitmap_vocab_max) {
        size_t byte_idx = token >> 3;
        uint8_t bit_mask = 1U << (token & 7);
        if (gen->high_freq_bitmap[byte_idx] & bit_mask) {
            return token;  // 高频，保留
        }
        // 低频，需要查哈希表获取 context_hash 来分桶
        size_t cap = gen->capacity;
        uint32_t h = (token * 2654435761U) % cap;
        for (uint32_t i = 0; i < 16; i++) {
            size_t slot = (h + i) % cap;
            if (gen->entries[slot].token == GENERALIZER_EMPTY_KEY) {
                // 未见过，用 token ID 做 fallback 分桶
                return WILDCARD_BASE + (token % NUM_WILDCARD_BUCKETS);
            }
            if (gen->entries[slot].token == token) {
                uint32_t bucket = gen->entries[slot].context_hash % NUM_WILDCARD_BUCKETS;
                return WILDCARD_BASE + bucket;
            }
        }
        return WILDCARD_BASE + (token % NUM_WILDCARD_BUCKETS);
    }

    // 慢速路径：哈希表查找
    size_t cap = gen->capacity;
    uint32_t h = (token * 2654435761U) % cap;
    for (uint32_t i = 0; i < 16; i++) {
        size_t slot = (h + i) % cap;
        if (gen->entries[slot].token == GENERALIZER_EMPTY_KEY) {
            return WILDCARD_BASE + (token % NUM_WILDCARD_BUCKETS);
        }
        if (gen->entries[slot].token == token) {
            if (gen->entries[slot].count >= gen->freq_threshold) {
                return token;  // 高频，保留
            }
            uint32_t bucket = gen->entries[slot].context_hash % NUM_WILDCARD_BUCKETS;
            return WILDCARD_BASE + bucket;
        }
    }
    return WILDCARD_BASE + (token % NUM_WILDCARD_BUCKETS);
}

// 泛化 n-gram 并计算 hash
// 将 tokens 中的每个 token 泛化后计算 hash
// out_generalized: 可选输出，存放泛化后的 token 序列
// 返回泛化后的 n-gram hash
uint64_t token_generalizer_hash_ngram(const TokenGeneralizer* gen,
                                      const uint32_t* tokens,
                                      uint8_t length,
                                      uint32_t* out_generalized);

// 检查泛化后的 n-gram 是否与精确 n-gram 不同
// （如果所有 token 都是高频的，泛化后和原始相同，不需要额外存储）
static inline int token_generalizer_is_different(const TokenGeneralizer* gen,
                                                  const uint32_t* tokens,
                                                  uint8_t length) {
    if (!gen) return 0;
    for (uint8_t i = 0; i < length; i++) {
        if (token_generalizer_map(gen, tokens[i]) != tokens[i]) {
            return 1;  // 至少有一个 token 被泛化了
        }
    }
    return 0;  // 全部保留，泛化后相同
}

// 获取统计信息
typedef struct {
    size_t capacity;
    size_t count;           // 不同 token 数
    uint32_t freq_threshold;
    uint32_t total_updates;
    size_t high_freq_count; // 高频 token 数
    size_t num_wildcard_buckets; // 通配符桶数量
    size_t memory_bytes;
} GeneralizerStats;

GeneralizerStats token_generalizer_stats(const TokenGeneralizer* gen);

#ifdef __cplusplus
}
#endif

#endif // TOKEN_GENERALIZER_H
