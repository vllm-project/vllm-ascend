#include "zipf_cache.h"
#include "cityhash.h"
#include "token_generalizer.h"
#include "async_worker.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================
// 1-gram 辅助函数（用于共享hash的初始化）
// ============================================================
static void init_unigram_entries(UnigramEntry* entries, size_t capacity) {
    for (size_t i = 0; i < capacity; i++) {
        entries[i].key_token = UNIGRAM_EMPTY_KEY;
        entries[i].tokens = NULL;
        entries[i].token_count = 0;
        entries[i].capacity = 0;
        entries[i].max_token = 0;
        entries[i].max_count = 0;
        entries[i].total_count = 0;
    }
}

static void free_unigram_entries(UnigramEntry* entries, size_t capacity) {
    if (!entries) return;
    for (size_t i = 0; i < capacity; i++) {
        if (entries[i].tokens) {
            free(entries[i].tokens);
        }
    }
    free(entries);
}

// 哈希函数
uint64_t zipf_hash(const uint32_t* tokens, uint8_t length) {
    return ngram_hash(tokens, length);
}

ZipfCache* zipf_cache_create(const ZipfConfig* config) {
    ZipfCache* cache = calloc(1, sizeof(ZipfCache));
    if (!cache) return NULL;

    cache->config = config ? *config : zipf_default_config();

    // 使用硬编码常量创建共享hash
    size_t shared_capacity = ZIPF_HARDCODED_SHARED_CAPACITY;
    cache->shared_hash = calloc(1, sizeof(LocalHash));
    if (!cache->shared_hash) {
        free(cache);
        return NULL;
    }

    // 分配n-gram存储
    cache->shared_hash->entries = calloc(shared_capacity, sizeof(NgramEntry));
    if (!cache->shared_hash->entries) {
        free(cache->shared_hash);
        free(cache);
        return NULL;
    }
    cache->shared_hash->capacity = shared_capacity;
    cache->shared_hash->count = 0;

    // 分配1-gram存储（硬编码启用）
    {
        size_t unigram_alloc = (size_t)(ZIPF_HARDCODED_UNIGRAM_CAPACITY / 0.7);
        cache->shared_hash->unigram_entries = calloc(unigram_alloc, sizeof(UnigramEntry));
        if (!cache->shared_hash->unigram_entries) {
            free(cache->shared_hash->entries);
            free(cache->shared_hash);
            free(cache);
            return NULL;
        }
        init_unigram_entries(cache->shared_hash->unigram_entries, unigram_alloc);
        cache->shared_hash->unigram_capacity = unigram_alloc;
        cache->shared_hash->unigram_threshold = ZIPF_HARDCODED_UNIGRAM_THRESHOLD;
    }
    cache->shared_hash->unigram_count = 0;
    cache->shared_hash->req_id = 0;
    cache->shared_hash->generation_count = 0;

    // 创建局部hash管理器
    size_t local_capacity = ZIPF_HARDCODED_LOCAL_CAPACITY;
    size_t gen_local_capacity = cache->config.enable_generalized
        ? local_capacity
        : 0;
    cache->local_manager = local_hash_manager_create(
        ZIPF_HARDCODED_MAX_LOCAL_HASHES,
        local_capacity,
        ZIPF_HARDCODED_UNIGRAM_CAPACITY,
        ZIPF_HARDCODED_UNIGRAM_THRESHOLD,
        gen_local_capacity
    );
    if (!cache->local_manager) {
        free_unigram_entries(cache->shared_hash->unigram_entries, cache->shared_hash->unigram_capacity);
        free(cache->shared_hash->entries);
        free(cache->shared_hash);
        free(cache);
        return NULL;
    }

    // 创建 Token 泛化器
    if (cache->config.enable_generalized) {
        cache->generalizer = token_generalizer_create(
            cache->config.gen_capacity,
            cache->config.gen_freq_threshold,
            cache->config.gen_bitmap_vocab_max
        );
        if (!cache->generalizer) {
            cache->config.enable_generalized = 0;
        }
    } else {
        cache->generalizer = NULL;
    }

    atomic_init(&cache->query_count, 0);
    atomic_init(&cache->shared_hit_count, 0);
    atomic_init(&cache->local_hit_count, 0);
    atomic_init(&cache->unigram_hit_count, 0);
    atomic_init(&cache->generalized_hit_count, 0);
    atomic_init(&cache->update_count, 0);

    // 创建异步工作线程
    cache->async_worker = async_worker_create(cache);
    // async_worker 创建失败不是致命错误，退化到同步模式

    return cache;
}

void zipf_cache_query_batch_with_req(ZipfCache* cache,
                                     const uint64_t* req_ids,
                                     const ZipfKey* keys,
                                     QueryResult* results,
                                     size_t batch_size) {
    if (!cache || batch_size == 0) return;

    uint64_t local_hits = 0;
    uint64_t shared_hits = 0;
    uint64_t unigram_hits = 0;
    uint64_t gen_hits = 0;

    for (size_t i = 0; i < batch_size; i++) {
        results[i].hit = 0;
        results[i].from_local = 0;

        LocalHash* local = local_hash_manager_get_or_create(cache->local_manager, req_ids[i]);
        uint8_t context_len = keys[i].length;

        for (uint8_t len = context_len; len >= 1 && !results[i].hit; len--) {
            const uint32_t* key_start = keys[i].tokens + (context_len - len);

            if (len == 1) {
                // 1-gram 使用概率判断
                uint32_t key_token = key_start[0];

                if (local && local_hash_query_unigram(local, key_token, &results[i].next_token)) {
                    results[i].hit = 1;
                    results[i].from_local = 4;
                    unigram_hits++;
                    break;
                }

                if (!cache->config.skip_shared && cache->shared_hash &&
                    local_hash_query_unigram(cache->shared_hash, key_token, &results[i].next_token)) {
                    results[i].hit = 1;
                    results[i].from_local = 3;
                    unigram_hits++;
                    break;
                }
            } else {
                uint64_t hash = zipf_hash(key_start, len);

                if (local && local_hash_query_with_level(local, hash, len, &results[i].next_token)) {
                    results[i].hit = 1;
                    results[i].from_local = 1;
                    local_hits++;
                    break;
                }

                if (cache->config.generalized_before_shared) {
                    // 泛化层优先于 shared
                    if (cache->config.enable_generalized && cache->generalizer &&
                        local && token_generalizer_is_different(cache->generalizer, key_start, len)) {
                        uint64_t gen_hash = token_generalizer_hash_ngram(
                            cache->generalizer, key_start, len, NULL);
                        if (local_hash_query_generalized_with_level(local, gen_hash, len,
                                &results[i].next_token)) {
                            results[i].hit = 1;
                            results[i].from_local = 5;
                            gen_hits++;
                            break;
                        }
                    }

                    if (!cache->config.skip_shared && cache->shared_hash &&
                        local_hash_query_with_level(cache->shared_hash, hash, len, &results[i].next_token)) {
                        results[i].hit = 1;
                        results[i].from_local = 2;
                        shared_hits++;
                        break;
                    }
                } else {
                    // shared 优先于泛化层
                    if (!cache->config.skip_shared && cache->shared_hash &&
                        local_hash_query_with_level(cache->shared_hash, hash, len, &results[i].next_token)) {
                        results[i].hit = 1;
                        results[i].from_local = 2;
                        shared_hits++;
                        break;
                    }

                    if (cache->config.enable_generalized && cache->generalizer &&
                        local && token_generalizer_is_different(cache->generalizer, key_start, len)) {
                        uint64_t gen_hash = token_generalizer_hash_ngram(
                            cache->generalizer, key_start, len, NULL);
                        if (local_hash_query_generalized_with_level(local, gen_hash, len,
                                &results[i].next_token)) {
                            results[i].hit = 1;
                            results[i].from_local = 5;
                            gen_hits++;
                            break;
                        }
                    }
                }
            }
        }
    }

    atomic_fetch_add_explicit(&cache->query_count, batch_size, memory_order_relaxed);
    atomic_fetch_add_explicit(&cache->local_hit_count, local_hits, memory_order_relaxed);
    atomic_fetch_add_explicit(&cache->shared_hit_count, shared_hits, memory_order_relaxed);
    atomic_fetch_add_explicit(&cache->unigram_hit_count, unigram_hits, memory_order_relaxed);
    atomic_fetch_add_explicit(&cache->generalized_hit_count, gen_hits, memory_order_relaxed);
}

void zipf_cache_update_with_req(ZipfCache* cache,
                                uint64_t req_id,
                                const uint32_t* context_tokens,
                                uint8_t context_len,
                                uint32_t next_token) {
    if (!cache || context_len == 0) return;

    LocalHash* local = local_hash_manager_get_or_create(cache->local_manager, req_id);

    if (context_len == 1) {
        // 1-gram 更新
        uint32_t key_token = context_tokens[0];
        if (local) {
            local_hash_update_unigram(local, key_token, next_token);
        }
        if (!cache->config.skip_shared && cache->shared_hash) {
            local_hash_update_unigram(cache->shared_hash, key_token, next_token);
        }
    } else {
        uint64_t hash = zipf_hash(context_tokens, context_len);
        if (local && local_hash_can_generate(local)) {
            local_hash_update(local, hash, next_token);
        }
        if (!cache->config.skip_shared && cache->shared_hash) {
            local_hash_update(cache->shared_hash, hash, next_token);
        }
    }

    atomic_fetch_add_explicit(&cache->update_count, 1, memory_order_relaxed);
}

// 多级 n-gram 更新（带历史窗口）
void zipf_cache_update_with_history(ZipfCache* cache,
                                    uint64_t req_id,
                                    const uint32_t* new_tokens,
                                    size_t new_token_count) {
    if (!cache || !new_tokens || new_token_count == 0) return;

    uint8_t min_win = cache->config.min_window;
    uint8_t max_win = cache->config.max_window;

    LocalHash* local = local_hash_manager_get_or_create(cache->local_manager, req_id);
    if (!local) return;

    uint8_t hist_len = local->history_len;
    uint32_t* history = local->history;

    int enable_gen = cache->config.enable_generalized && cache->generalizer;

    for (size_t i = 0; i < new_token_count; i++) {
        uint32_t next_token = new_tokens[i];

        if (enable_gen) {
            // 传入前一个 token 用于共现分布分桶
            uint32_t prev_token = (hist_len >= 1) ? history[hist_len - 1] : 0xFFFFFFFF;
            token_generalizer_update(cache->generalizer, next_token, prev_token);
        }

        // 更新 1-gram
        if (hist_len >= 1) {
            uint32_t key_token = history[hist_len - 1];
            local_hash_update_unigram(local, key_token, next_token);
            if (!cache->config.skip_shared && cache->shared_hash) {
                local_hash_update_unigram(cache->shared_hash, key_token, next_token);
            }
        }

        // 更新各级 n-gram
        for (uint8_t x = min_win; x <= max_win; x++) {
            if (hist_len >= x) {
                const uint32_t* key_start = history + (hist_len - x);
                uint64_t hash = zipf_hash(key_start, x);

                if (local_hash_can_generate(local)) {
                    local_hash_update(local, hash, next_token);
                }
                if (!cache->config.skip_shared && cache->shared_hash) {
                    local_hash_update(cache->shared_hash, hash, next_token);
                }
                if (enable_gen &&
                    token_generalizer_is_different(cache->generalizer, key_start, x)) {
                    uint64_t gen_hash = token_generalizer_hash_ngram(
                        cache->generalizer, key_start, x, NULL);
                    local_hash_update_generalized(local, gen_hash, next_token);
                }
            }
        }

        // 滑动历史窗口
        if (hist_len < max_win) {
            history[hist_len] = next_token;
            hist_len++;
        } else {
            for (uint8_t j = 0; j < max_win - 1; j++) {
                history[j] = history[j + 1];
            }
            history[max_win - 1] = next_token;
        }
    }

    local->history_len = hist_len;
    atomic_fetch_add_explicit(&cache->update_count, new_token_count, memory_order_relaxed);
}

void zipf_cache_set_history(ZipfCache* cache,
                            uint64_t req_id,
                            const uint32_t* tokens,
                            size_t token_count) {
    if (!cache || !tokens) return;
    LocalHash* local = local_hash_manager_get_or_create(cache->local_manager, req_id);
    if (!local) return;
    local_hash_set_history(local, tokens,
                           token_count > 255 ? 255 : (uint8_t)token_count,
                           cache->config.max_window);
}

void zipf_cache_warmup_from_prompt(ZipfCache* cache,
                                   uint64_t req_id,
                                   const uint32_t* prompt_tokens,
                                   size_t prompt_len) {
    if (!cache || !prompt_tokens || prompt_len == 0) return;
    uint8_t min_win = cache->config.min_window;
    uint8_t max_win = cache->config.max_window;
    LocalHash* local = local_hash_manager_get_or_create(cache->local_manager, req_id);
    if (!local) return;
    int enable_gen = cache->config.enable_generalized && cache->generalizer;
    for (size_t i = min_win; i < prompt_len; i++) {
        uint32_t next_token = prompt_tokens[i];
        if (enable_gen) {
            uint32_t prev_token = (i > 0) ? prompt_tokens[i - 1] : 0xFFFFFFFF;
            token_generalizer_update(cache->generalizer, next_token, prev_token);
        }
        if (i >= 1) {
            uint32_t key_token = prompt_tokens[i - 1];
            local_hash_update_unigram(local, key_token, next_token);
        }
        for (uint8_t x = min_win; x <= max_win; x++) {
            if (i >= x) {
                const uint32_t* key_start = prompt_tokens + (i - x);
                uint64_t hash = zipf_hash(key_start, x);
                local_hash_update(local, hash, next_token);
                if (enable_gen &&
                    token_generalizer_is_different(cache->generalizer, key_start, x)) {
                    uint64_t gen_hash = token_generalizer_hash_ngram(
                        cache->generalizer, key_start, x, NULL);
                    local_hash_update_generalized(local, gen_hash, next_token);
                }
            }
        }
    }
    local_hash_set_history(local, prompt_tokens,
                           prompt_len > 255 ? 255 : (uint8_t)prompt_len,
                           max_win);
}

void zipf_cache_reset_context(ZipfCache* cache, uint64_t req_id) {
    if (!cache) return;
    LocalHash* local = local_hash_manager_get_or_create(cache->local_manager, req_id);
    if (!local) return;
    local_hash_reset_history(local);
}

void zipf_cache_update_batch_with_req(ZipfCache* cache,
                                      const uint64_t* req_ids,
                                      const ZipfKey* contexts,
                                      const uint32_t* next_tokens,
                                      size_t batch_size) {
    if (!cache || !req_ids || !contexts || !next_tokens || batch_size == 0) return;
    for (size_t i = 0; i < batch_size; i++) {
        uint64_t hash = zipf_hash(contexts[i].tokens, contexts[i].length);
        LocalHash* local = local_hash_manager_get_or_create(cache->local_manager, req_ids[i]);
        if (local && local_hash_can_generate(local)) {
            local_hash_update(local, hash, next_tokens[i]);
        }
        if (cache->shared_hash && ZIPF_HARDCODED_SHARED_CAPACITY > 0) {
            local_hash_update(cache->shared_hash, hash, next_tokens[i]);
        }
    }
    atomic_fetch_add_explicit(&cache->update_count, batch_size, memory_order_relaxed);
}

void zipf_cache_update_with_history_batch(
    ZipfCache* cache,
    const uint64_t* req_ids,
    const uint32_t* flat_tokens,
    const size_t* offsets,
    const size_t* token_counts,
    size_t batch_size
) {
    if (!cache || !req_ids || !flat_tokens || !offsets || !token_counts
        || batch_size == 0) return;
    for (size_t i = 0; i < batch_size; i++) {
        if (token_counts[i] == 0) continue;
        zipf_cache_update_with_history(
            cache, req_ids[i],
            flat_tokens + offsets[i], token_counts[i]);
    }
}

ZipfCacheStats zipf_cache_stats(const ZipfCache* cache) {
    ZipfCacheStats stats = {0};
    if (!cache) return stats;

    stats.query_count = atomic_load(&cache->query_count);
    stats.shared_hit_count = atomic_load(&cache->shared_hit_count);
    stats.local_hit_count = atomic_load(&cache->local_hit_count);
    stats.unigram_hit_count = atomic_load(&cache->unigram_hit_count);
    stats.generalized_hit_count = atomic_load(&cache->generalized_hit_count);

    uint64_t total_hits = stats.shared_hit_count +
                          stats.local_hit_count + stats.unigram_hit_count +
                          stats.generalized_hit_count;
    stats.total_hit_rate = stats.query_count > 0 ?
        (double)total_hits / stats.query_count : 0.0;
    stats.shared_hit_rate = stats.query_count > 0 ?
        (double)stats.shared_hit_count / stats.query_count : 0.0;
    stats.local_hit_rate = stats.query_count > 0 ?
        (double)stats.local_hit_count / stats.query_count : 0.0;
    stats.unigram_hit_rate = stats.query_count > 0 ?
        (double)stats.unigram_hit_count / stats.query_count : 0.0;
    stats.generalized_hit_rate = stats.query_count > 0 ?
        (double)stats.generalized_hit_count / stats.query_count : 0.0;

    stats.update_count = atomic_load(&cache->update_count);
    stats.local_stats = local_hash_manager_stats(cache->local_manager);

    if (cache->shared_hash) {
        stats.shared_entries = cache->shared_hash->count;
        stats.shared_unigram = cache->shared_hash->unigram_count;
    }

    if (cache->generalizer) {
        stats.gen_stats = token_generalizer_stats(cache->generalizer);
    }

    return stats;
}

void zipf_cache_free(ZipfCache* cache) {
    if (!cache) return;
    // 先停止异步工作线程
    async_worker_free(cache->async_worker);
    if (cache->shared_hash) {
        free(cache->shared_hash->entries);
        free(cache->shared_hash->gen_entries);
        free_unigram_entries(cache->shared_hash->unigram_entries, cache->shared_hash->unigram_capacity);
        free(cache->shared_hash);
    }
    local_hash_manager_free(cache->local_manager);
    token_generalizer_free(cache->generalizer);
    free(cache);
}

// ============================================================
// 链式推测 API 实现（纯命中驱动，长度控制由 Python 层 adaptive-k 负责）
// ============================================================

static inline int query_single_level(
    const LocalHash* local,
    const LocalHash* shared,
    const ZipfCache* cache,
    const uint32_t* window,
    uint8_t window_len,
    uint8_t level,
    uint32_t* out_token
) {
    if (window_len < level) return 0;

    if (level == 1) {
        // 1-gram: local → shared
        uint32_t key_token = window[window_len - 1];
        if (local && local_hash_query_unigram(local, key_token, out_token)) return 1;
        if (shared && local_hash_query_unigram(shared, key_token, out_token)) return 1;
        return 0;
    }

    const uint32_t* key_start = window + (window_len - level);
    uint64_t hash = zipf_hash(key_start, level);

    // local 精确匹配（带级别阈值）
    if (local && local_hash_query_with_level(local, hash, level, out_token)) return 1;

    // local → 泛化 → shared 或 local → shared → 泛化，取决于配置
    int gen_before_shared = cache->config.generalized_before_shared;

    if (gen_before_shared) {
        // 泛化层优先于 shared
        if (cache->config.enable_generalized && cache->generalizer &&
            local && token_generalizer_is_different(cache->generalizer, key_start, level)) {
            uint64_t gen_hash = token_generalizer_hash_ngram(
                cache->generalizer, key_start, level, NULL);
            if (local_hash_query_generalized_with_level(local, gen_hash, level, out_token)) {
                atomic_fetch_add_explicit(
                    (_Atomic uint64_t*)&cache->generalized_hit_count, 1, memory_order_relaxed);
                return 1;
            }
        }
        if (shared && local_hash_query_with_level(shared, hash, level, out_token)) return 1;
    } else {
        // shared 优先于泛化层
        if (shared && local_hash_query_with_level(shared, hash, level, out_token)) return 1;
        if (cache->config.enable_generalized && cache->generalizer &&
            local && token_generalizer_is_different(cache->generalizer, key_start, level)) {
            uint64_t gen_hash = token_generalizer_hash_ngram(
                cache->generalizer, key_start, level, NULL);
            if (local_hash_query_generalized_with_level(local, gen_hash, level, out_token)) {
                atomic_fetch_add_explicit(
                    (_Atomic uint64_t*)&cache->generalized_hit_count, 1, memory_order_relaxed);
                return 1;
            }
        }
    }

    return 0;
}

int zipf_cache_speculate_chain_with_req(
    ZipfCache* cache,
    uint64_t req_id,
    const uint32_t* context,
    uint8_t context_len,
    uint32_t* out_tokens,
    int max_tokens
) {
    if (!cache || !context || !out_tokens || max_tokens <= 0 || context_len == 0) {
        return 0;
    }

    LocalHash* local = local_hash_manager_get_or_create(cache->local_manager, req_id);
    if (!local || !local_hash_can_generate(local)) {
        return 0;
    }

    LocalHash* shared = (!cache->config.skip_shared) ? cache->shared_hash : NULL;

    uint8_t min_win = cache->config.min_window;
    uint8_t max_win = cache->config.max_window;

    uint32_t window[MAX_NGRAM_ORDER];
    uint8_t window_len;

    if (context_len > max_win) {
        window_len = max_win;
        int start = context_len - max_win;
        for (int i = 0; i < max_win; i++) {
            window[i] = context[start + i];
        }
    } else {
        window_len = context_len;
        for (int i = 0; i < window_len; i++) {
            window[i] = context[i];
        }
    }

    int count = 0;
    while (count < max_tokens && local_hash_can_generate(local)) {
        uint32_t next_token = 0;
        int hit = 0;

        // 从最高级 n-gram 开始查询，逐级降级
        for (int x = max_win; x >= min_win; x--) {
            if (query_single_level(local, shared, cache,
                                   window, window_len, x, &next_token)) {
                hit = 1;
                break;
            }
        }

        // 1-gram fallback
        if (!hit && window_len >= 1) {
            if (query_single_level(local, shared, cache,
                                   window, window_len, 1, &next_token)) {
                hit = 1;
            }
        }

        if (!hit) break;

        out_tokens[count++] = next_token;

        // 滑动窗口
        if (window_len < max_win) {
            window[window_len++] = next_token;
        } else {
            for (int i = 0; i < max_win - 1; i++) {
                window[i] = window[i + 1];
            }
            window[max_win - 1] = next_token;
        }
    }

    atomic_fetch_add_explicit(&cache->query_count, 1, memory_order_relaxed);
    if (count > 0) {
        atomic_fetch_add_explicit(&cache->local_hit_count, 1, memory_order_relaxed);
    }
    return count;
}

void zipf_cache_speculate_chain_batch_with_req(
    ZipfCache* cache,
    const uint64_t* req_ids,
    const ZipfKey* contexts,
    uint32_t* out_tokens,
    int* out_counts,
    size_t batch_size,
    int max_tokens
) {
    if (!cache || !req_ids || !contexts || !out_tokens || !out_counts ||
        batch_size == 0 || max_tokens <= 0) return;
    for (size_t i = 0; i < batch_size; i++) {
        out_counts[i] = zipf_cache_speculate_chain_with_req(
            cache, req_ids[i],
            contexts[i].tokens, contexts[i].length,
            &out_tokens[i * max_tokens], max_tokens);
    }
}

// ============================================================
// Propose API 实现
// ============================================================

static inline int compute_adaptive_k(
    LocalHash* local,
    int num_sampled,
    const AdaptiveKConfig* ak
) {
    if (!local->is_initialized) {
        // 首次初始化
        local->ema_accepted = (float)ak->num_speculative_tokens;
        local->consecutive_fail = 0;
        local->consecutive_success = 0;
        local->last_adaptive_k = ak->num_speculative_tokens;
        local->is_initialized = 1;
    }

    // EMA 更新
    float ema = ak->ema_alpha * (float)num_sampled
              + (1.0f - ak->ema_alpha) * local->ema_accepted;
    local->ema_accepted = ema;

    // 接受率
    float acceptance_rate = (float)num_sampled / (float)(local->last_adaptive_k + 1);

    if (acceptance_rate < ak->fail_threshold) {
        local->consecutive_fail++;
        local->consecutive_success = 0;
    } else if (acceptance_rate >= ak->success_threshold) {
        local->consecutive_success++;
        local->consecutive_fail = 0;
    }

    int penalty = local->consecutive_fail;
    if (penalty > ak->max_penalty) penalty = ak->max_penalty;
    int bonus = local->consecutive_success;
    if (bonus > ak->max_bonus) bonus = ak->max_bonus;

    int adaptive_k = (int)(ema + 0.999999f) + 1 - penalty + bonus;
    if (adaptive_k < ak->min_k) adaptive_k = ak->min_k;
    if (adaptive_k > ak->max_k) adaptive_k = ak->max_k;

    local->last_adaptive_k = (int16_t)adaptive_k;
    return adaptive_k;
}

void zipf_cache_propose_batch(
    ZipfCache* cache,
    const ProposeRequest* requests,
    ProposeResult* results,
    size_t batch_size,
    const AdaptiveKConfig* ak_config
) {
    if (!cache || !requests || !results || batch_size == 0) return;

    uint8_t max_win = cache->config.max_window;

    // Step 0: Fence — 等待上一轮的异步 warmup/update 完成
    if (cache->async_worker) {
        async_worker_fence(cache->async_worker);
    }

    // Step 1: 对每个请求做 speculate（同步，快路径）
    //         同时收集需要 warmup/update 的任务
    for (size_t i = 0; i < batch_size; i++) {
        results[i].num_draft_tokens = 0;

        const ProposeRequest* req = &requests[i];
        if (!req->is_valid || req->num_sampled == 0) continue;
        if (req->num_tokens >= req->max_model_len) continue;

        LocalHash* local = local_hash_manager_get_or_create(
            cache->local_manager, req->req_hash);
        if (!local) continue;

        // 新请求：提交异步 warmup（不阻塞）
        if (!local->is_initialized) {
            if (req->num_prompt_tokens > 0 && req->token_ids && cache->async_worker) {
                async_worker_submit_warmup(
                    cache->async_worker, req->req_hash,
                    req->token_ids, (size_t)req->num_prompt_tokens);
            } else if (req->num_prompt_tokens > 0 && req->token_ids) {
                // 无异步工作线程，同步 warmup
                zipf_cache_warmup_from_prompt(
                    cache, req->req_hash,
                    req->token_ids, (size_t)req->num_prompt_tokens);
            }
        }

        // 提交异步 update（不阻塞）
        if (req->num_sampled > 0 && req->sampled_ids) {
            if (cache->async_worker) {
                async_worker_submit_update(
                    cache->async_worker, req->req_hash,
                    req->sampled_ids, (size_t)req->num_sampled);
            } else {
                zipf_cache_update_with_history(
                    cache, req->req_hash,
                    req->sampled_ids, (size_t)req->num_sampled);
            }
        }

        // Compute adaptive-k
        int adaptive_k = compute_adaptive_k(local, req->num_sampled, ak_config);

        // Prepare context for speculation
        int spec_start = req->num_tokens - max_win;
        if (spec_start < 0) spec_start = 0;
        int ctx_len = req->num_tokens - spec_start;
        if (ctx_len > max_win) ctx_len = max_win;
        if (ctx_len <= 0 || !req->token_ids) continue;

        const uint32_t* context = req->token_ids + spec_start;

        // Compute max draft length
        int max_draft = req->max_model_len - req->num_tokens - 1;
        if (adaptive_k < max_draft) max_draft = adaptive_k;
        if (max_draft <= 0) continue;

        // Speculate（同步，使用当前缓存状态）
        // 注意：新请求的 warmup 还在后台执行，所以第一次 speculate 可能 miss
        // 这是预期行为——延迟一步学习，换取不阻塞 propose
        int count = zipf_cache_speculate_chain_with_req(
            cache, req->req_hash,
            context, (uint8_t)ctx_len,
            results[i].draft_tokens, max_draft);

        results[i].num_draft_tokens = count;
    }
    // warmup/update 任务已提交到后台，会在 GPU forward 期间执行
    // 下一次 propose_batch 开头的 fence 会确保它们完成
}
