#include "token_generalizer.h"
#include "cityhash.h"
#include <stdio.h>

// ============================================================
// 内部哈希函数
// ============================================================
static inline uint32_t freq_hash(uint32_t token, size_t cap) {
    return (token * 2654435761U) % cap;
}

static inline uint32_t freq_probe(uint32_t h, uint32_t i, size_t cap) {
    return (h + i) % cap;
}

// ============================================================
// 创建 / 释放
// ============================================================

TokenGeneralizer* token_generalizer_create(size_t capacity,
                                           uint32_t freq_threshold,
                                           uint32_t bitmap_vocab_max) {
    TokenGeneralizer* gen = calloc(1, sizeof(TokenGeneralizer));
    if (!gen) return NULL;

    // 确保容量合理
    if (capacity < 1024) capacity = 1024;

    gen->entries = malloc(capacity * sizeof(FreqEntry));
    if (!gen->entries) {
        free(gen);
        return NULL;
    }

    // 初始化所有槽为空
    for (size_t i = 0; i < capacity; i++) {
        gen->entries[i].token = GENERALIZER_EMPTY_KEY;
        gen->entries[i].count = 0;
        gen->entries[i].context_hash = 0;
    }

    gen->capacity = capacity;
    gen->count = 0;
    gen->freq_threshold = freq_threshold;
    gen->total_updates = 0;

    // 创建 bitmap（如果请求）
    if (bitmap_vocab_max > 0) {
        gen->bitmap_size = (bitmap_vocab_max + 8) / 8;
        gen->high_freq_bitmap = calloc(gen->bitmap_size, 1);
        gen->bitmap_vocab_max = bitmap_vocab_max;
        // bitmap 分配失败不是致命错误，退化到哈希表查找
    } else {
        gen->high_freq_bitmap = NULL;
        gen->bitmap_size = 0;
        gen->bitmap_vocab_max = 0;
    }

    return gen;
}

void token_generalizer_free(TokenGeneralizer* gen) {
    if (!gen) return;
    free(gen->entries);
    free(gen->high_freq_bitmap);
    free(gen);
}

// ============================================================
// 频率更新
// ============================================================

uint32_t token_generalizer_update(TokenGeneralizer* gen, uint32_t token, uint32_t prev_token) {
    if (!gen) return 0;

    size_t cap = gen->capacity;
    uint32_t h = freq_hash(token, cap);

    for (uint32_t i = 0; i < 16; i++) {
        size_t slot = freq_probe(h, i, cap);
        FreqEntry* e = &gen->entries[slot];

        if (e->token == GENERALIZER_EMPTY_KEY) {
            // 新 token，插入
            e->token = token;
            e->count = 1;
            // 初始化 context_hash（如果有前驱 token）
            if (prev_token != 0xFFFFFFFF) {
                e->context_hash = prev_token * 2654435761U;
            } else {
                e->context_hash = 0;
            }
            gen->count++;
            gen->total_updates++;
            return 1;
        }

        if (e->token == token) {
            // 已存在，增加计数
            e->count++;
            gen->total_updates++;

            // 累积左邻居信息到 context_hash
            if (prev_token != 0xFFFFFFFF) {
                e->context_hash = (e->context_hash * 31) + (prev_token * 2654435761U);
            }

            // 如果刚达到阈值，更新 bitmap
            if (e->count == gen->freq_threshold &&
                gen->high_freq_bitmap &&
                token <= gen->bitmap_vocab_max) {
                size_t byte_idx = token >> 3;
                uint8_t bit_mask = 1U << (token & 7);
                gen->high_freq_bitmap[byte_idx] |= bit_mask;
            }

            return e->count;
        }
    }

    // 探测失败（表太满），忽略
    gen->total_updates++;
    return 0;
}

void token_generalizer_update_batch(TokenGeneralizer* gen,
                                    const uint32_t* tokens,
                                    size_t count) {
    if (!gen || !tokens) return;
    for (size_t i = 0; i < count; i++) {
        uint32_t prev = (i > 0) ? tokens[i - 1] : 0xFFFFFFFF;
        token_generalizer_update(gen, tokens[i], prev);
    }
}

// ============================================================
// 泛化 n-gram hash
// ============================================================

uint64_t token_generalizer_hash_ngram(const TokenGeneralizer* gen,
                                      const uint32_t* tokens,
                                      uint8_t length,
                                      uint32_t* out_generalized) {
    if (!gen || !tokens || length == 0) return 0;

    uint32_t buf[8];  // MAX_NGRAM_ORDER = 8
    uint32_t* dest = out_generalized ? out_generalized : buf;

    for (uint8_t i = 0; i < length; i++) {
        dest[i] = token_generalizer_map(gen, tokens[i]);
    }

    return ngram_hash(dest, length);
}

// ============================================================
// 统计
// ============================================================

GeneralizerStats token_generalizer_stats(const TokenGeneralizer* gen) {
    GeneralizerStats stats = {0};
    if (!gen) return stats;

    stats.capacity = gen->capacity;
    stats.count = gen->count;
    stats.freq_threshold = gen->freq_threshold;
    stats.total_updates = gen->total_updates;

    // 计算高频 token 数
    for (size_t i = 0; i < gen->capacity; i++) {
        if (gen->entries[i].token != GENERALIZER_EMPTY_KEY &&
            gen->entries[i].count >= gen->freq_threshold) {
            stats.high_freq_count++;
        }
    }

    stats.memory_bytes = sizeof(TokenGeneralizer) +
                         gen->capacity * sizeof(FreqEntry) +
                         gen->bitmap_size;

    stats.num_wildcard_buckets = NUM_WILDCARD_BUCKETS;

    return stats;
}
