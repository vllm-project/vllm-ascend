#ifndef ZIPF_CACHE_H
#define ZIPF_CACHE_H

#include "types.h"
#include "local_hash.h"
#include "token_generalizer.h"
#include "async_worker.h"
#include <stdatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

// Zipf Cache - 完整的推测解码加速系统
// 优化版本：移除所有锁，适用于vLLM串行调用场景
// 1-gram 特殊处理已集成到 LocalHash 中

typedef struct {
    // 共享hash（所有req_id共享的动态存储，包含1-gram）
    LocalHash* shared_hash;

    // 局部hash管理器（每个req_id一个局部hash，包含1-gram）
    LocalHashManager* local_manager;

    // Token 泛化器（运行时频率统计 + 泛化映射）
    TokenGeneralizer* generalizer;

    // 统计
    _Atomic uint64_t query_count;
    _Atomic uint64_t shared_hit_count;
    _Atomic uint64_t local_hit_count;
    _Atomic uint64_t unigram_hit_count;
    _Atomic uint64_t generalized_hit_count;
    _Atomic uint64_t update_count;

    // 配置
    ZipfConfig config;

    // 异步工作线程（用于后台 warmup/update）
    AsyncWorker* async_worker;
} ZipfCache;

// 哈希函数声明
uint64_t zipf_hash(const uint32_t* tokens, uint8_t length);

// 创建缓存
ZipfCache* zipf_cache_create(const ZipfConfig* config);

// 推测查询 - 带req_id的版本（无锁优化版）
// 查询顺序：对于每个n-gram级别，依次查询 局部hash → 泛化层 → 共享hash
// 如果未命中，降级到更短的n-gram，直到1-gram
// 1-gram使用概率判断
static inline int zipf_cache_query_with_req(ZipfCache* cache,
                                            uint64_t req_id,
                                            const uint32_t* context_tokens,
                                            uint8_t context_len,
                                            uint32_t* out_next_token) {
    if (!cache || context_len == 0) return 0;

    atomic_fetch_add_explicit(&cache->query_count, 1, memory_order_relaxed);

    LocalHash* local = local_hash_manager_get_or_create(cache->local_manager, req_id);

    // 从最长的n-gram开始，逐级降级查询
    for (uint8_t len = context_len; len >= 1; len--) {
        const uint32_t* key_start = context_tokens + (context_len - len);

        if (len == 1) {
            // 1-gram 使用概率判断（硬编码启用 unigram）
            uint32_t key_token = key_start[0];

            // 查局部hash的1-gram
            if (local && local_hash_query_unigram(local, key_token, out_next_token)) {
                atomic_fetch_add_explicit(&cache->unigram_hit_count, 1, memory_order_relaxed);
                return 1;
            }

            // 查共享hash的1-gram
            if (!cache->config.skip_shared && cache->shared_hash &&
                local_hash_query_unigram(cache->shared_hash, key_token, out_next_token)) {
                atomic_fetch_add_explicit(&cache->unigram_hit_count, 1, memory_order_relaxed);
                return 1;
            }
        } else {
            // 2-gram及以上，使用 Space-Saving Top-2 查询
            uint64_t hash = zipf_hash(key_start, len);

            // 查局部hash（精确匹配，带级别阈值）
            if (local && local_hash_query_with_level(local, hash, len, out_next_token)) {
                atomic_fetch_add_explicit(&cache->local_hit_count, 1, memory_order_relaxed);
                return 1;
            }

            if (cache->config.generalized_before_shared) {
                // 泛化层优先于 shared
                if (cache->config.enable_generalized && cache->generalizer &&
                    local && token_generalizer_is_different(cache->generalizer, key_start, len)) {
                    uint64_t gen_hash = token_generalizer_hash_ngram(
                        cache->generalizer, key_start, len, NULL);
                    if (local_hash_query_generalized_with_level(local, gen_hash, len,
                            out_next_token)) {
                        atomic_fetch_add_explicit(&cache->generalized_hit_count, 1, memory_order_relaxed);
                        return 1;
                    }
                }

                if (!cache->config.skip_shared && cache->shared_hash &&
                    local_hash_query_with_level(cache->shared_hash, hash, len, out_next_token)) {
                    atomic_fetch_add_explicit(&cache->shared_hit_count, 1, memory_order_relaxed);
                    return 1;
                }
            } else {
                // shared 优先于泛化层
                if (!cache->config.skip_shared && cache->shared_hash &&
                    local_hash_query_with_level(cache->shared_hash, hash, len, out_next_token)) {
                    atomic_fetch_add_explicit(&cache->shared_hit_count, 1, memory_order_relaxed);
                    return 1;
                }

                if (cache->config.enable_generalized && cache->generalizer &&
                    local && token_generalizer_is_different(cache->generalizer, key_start, len)) {
                    uint64_t gen_hash = token_generalizer_hash_ngram(
                        cache->generalizer, key_start, len, NULL);
                    if (local_hash_query_generalized_with_level(local, gen_hash, len,
                            out_next_token)) {
                        atomic_fetch_add_explicit(&cache->generalized_hit_count, 1, memory_order_relaxed);
                        return 1;
                    }
                }
            }
        }
    }

    return 0;
}

// 兼容旧接口（不带req_id，无全局hash，直接返回0）
static inline int zipf_cache_query(ZipfCache* cache,
                                    const uint32_t* context_tokens,
                                    uint8_t context_len,
                                    uint32_t* out_next_token) {
    (void)context_tokens;
    (void)context_len;
    (void)out_next_token;

    atomic_fetch_add_explicit(&cache->query_count, 1, memory_order_relaxed);

    // 全局hash已移除，此接口不再有效
    return 0;
}

// 批量推测查询（带req_id）
void zipf_cache_query_batch_with_req(ZipfCache* cache,
                                     const uint64_t* req_ids,
                                     const ZipfKey* keys,
                                     QueryResult* results,
                                     size_t batch_size);

// 更新 - 带req_id，写入局部hash
void zipf_cache_update_with_req(ZipfCache* cache,
                                uint64_t req_id,
                                const uint32_t* context_tokens,
                                uint8_t context_len,
                                uint32_t next_token);

// 新版多级 n-gram 更新（带历史窗口）
// 对于每个新 token，同时更新 min_window 到 max_window 级别的 n-gram
// 自动维护历史窗口，支持连续调用
void zipf_cache_update_with_history(ZipfCache* cache,
                                    uint64_t req_id,
                                    const uint32_t* new_tokens,
                                    size_t new_token_count);

// 批量多级 n-gram 更新（带历史窗口）
// flat_tokens: 所有 request 的 token 拼接成的扁平数组
// offsets: 每个 request 在 flat_tokens 中的起始偏移
// token_counts: 每个 request 的 token 数量
void zipf_cache_update_with_history_batch(
    ZipfCache* cache,
    const uint64_t* req_ids,
    const uint32_t* flat_tokens,
    const size_t* offsets,
    const size_t* token_counts,
    size_t batch_size);

// 设置历史窗口（用于初始化上下文，如 prompt tokens）
void zipf_cache_set_history(ZipfCache* cache,
                            uint64_t req_id,
                            const uint32_t* tokens,
                            size_t token_count);

// Prompt-aware 预热：扫描整个 prompt，提取所有 n-gram 预填充到 local hash
// 同时设置历史窗口为 prompt 最后 max_window 个 token
void zipf_cache_warmup_from_prompt(ZipfCache* cache,
                                   uint64_t req_id,
                                   const uint32_t* prompt_tokens,
                                   size_t prompt_len);

// 重置历史窗口
void zipf_cache_reset_context(ZipfCache* cache, uint64_t req_id);

// 批量更新 - 带req_id
void zipf_cache_update_batch_with_req(ZipfCache* cache,
                                      const uint64_t* req_ids,
                                      const ZipfKey* contexts,
                                      const uint32_t* next_tokens,
                                      size_t batch_size);

// 获取统计信息
typedef struct {
    uint64_t query_count;
    uint64_t shared_hit_count;
    uint64_t local_hit_count;
    uint64_t unigram_hit_count;
    uint64_t generalized_hit_count;
    double shared_hit_rate;
    double local_hit_rate;
    double unigram_hit_rate;
    double generalized_hit_rate;
    double total_hit_rate;
    uint64_t update_count;
    LocalHashStats local_stats;
    size_t shared_entries;      // 共享hash中的n-gram条目数
    size_t shared_unigram;      // 共享hash中的1-gram条目数
    GeneralizerStats gen_stats; // 泛化器统计
} ZipfCacheStats;

ZipfCacheStats zipf_cache_stats(const ZipfCache* cache);

// 释放缓存
void zipf_cache_free(ZipfCache* cache);

// ============================================================
// 链式推测 API - 用于 vLLM 推测解码集成
// ============================================================

/**
 * 链式推测多个 token（带req_id版本）
 *
 * 从给定上下文开始，连续查询预测 token，直到遇到 miss 或达到 max_tokens。
 * 每次查询后，将预测的 token 加入上下文窗口（滑动窗口，最多 4 个）。
 *
 * @param cache         缓存实例
 * @param req_id        请求ID
 * @param context       初始上下文 token 数组
 * @param context_len   上下文长度（1-4，超过 4 会截取最后 4 个）
 * @param out_tokens    输出缓冲区，存放预测的 token
 * @param max_tokens    最大推测数量
 * @return              实际推测的 token 数量（0 表示第一次就 miss）
 */
int zipf_cache_speculate_chain_with_req(
    ZipfCache* cache,
    uint64_t req_id,
    const uint32_t* context,
    uint8_t context_len,
    uint32_t* out_tokens,
    int max_tokens
);

/**
 * 批量链式推测结果
 */
typedef struct {
    int count;              // 实际推测的 token 数量
} SpeculateChainResult;

/**
 * 批量链式推测（带req_id）
 */
void zipf_cache_speculate_chain_batch_with_req(
    ZipfCache* cache,
    const uint64_t* req_ids,
    const ZipfKey* contexts,
    uint32_t* out_tokens,
    int* out_counts,
    size_t batch_size,
    int max_tokens
);

// ============================================================
// Propose API - 完整的 propose 逻辑下沉到 C 层
// ============================================================

/**
 * Adaptive-K 配置
 */
typedef struct {
    float ema_alpha;             // EMA 平滑系数（默认 0.3）
    int num_speculative_tokens;  // 最大推测 token 数
    float fail_threshold;        // 接受率低于此值视为失败（默认 0.3）
    float success_threshold;     // 接受率高于此值视为成功（默认 0.5）
    int max_penalty;             // 最大惩罚值（默认 3）
    int max_bonus;               // 最大奖励值（默认 3）
    int min_k;                   // adaptive_k 下限（默认 1）
    int max_k;                   // adaptive_k 上限（默认 128）
} AdaptiveKConfig;

static inline AdaptiveKConfig adaptive_k_default_config(void) {
    return (AdaptiveKConfig){
        .ema_alpha = 0.3f,
        .num_speculative_tokens = 5,
        .fail_threshold = 0.3f,
        .success_threshold = 0.5f,
        .max_penalty = 3,
        .max_bonus = 3,
        .min_k = 1,
        .max_k = 128,
    };
}

/**
 * 单个请求的 propose 输入
 */
typedef struct {
    uint64_t req_hash;           // 请求 hash（用作 req_id）
    const uint32_t* token_ids;   // 该请求的完整 token 序列（prompt + generated）
    int num_tokens;              // 当前 token 总数（不含 spec tokens）
    int num_prompt_tokens;       // prompt token 数
    const uint32_t* sampled_ids; // 上一轮验证通过的 token（用于 update）
    int num_sampled;             // sampled_ids 长度
    int max_model_len;           // 模型最大长度
    uint8_t is_valid;            // 是否参与推测（0=跳过）
} ProposeRequest;

/**
 * 单个请求的 propose 输出
 */
typedef struct {
    uint32_t* draft_tokens;      // 输出的 draft token 缓冲区
    int num_draft_tokens;        // 实际输出的 draft token 数
} ProposeResult;

/**
 * 批量 propose - 完整的 propose 逻辑在 C 层执行
 *
 * 对每个有效请求：
 * 1. 新请求：warmup_from_prompt
 * 2. update_with_history（用 sampled_ids）
 * 3. 计算 adaptive-k
 * 4. speculate_chain
 * 5. 截断到 min(adaptive_k, max_model_len - num_tokens - 1)
 *
 * @param cache          缓存实例
 * @param requests       请求数组
 * @param results        结果数组（调用者预分配）
 * @param batch_size     批大小
 * @param ak_config      adaptive-k 配置
 */
void zipf_cache_propose_batch(
    ZipfCache* cache,
    const ProposeRequest* requests,
    ProposeResult* results,
    size_t batch_size,
    const AdaptiveKConfig* ak_config
);

#ifdef __cplusplus
}
#endif

#endif // ZIPF_CACHE_H
