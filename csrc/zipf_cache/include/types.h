#ifndef ZIPF_TYPES_H
#define ZIPF_TYPES_H

#include <stdint.h>
#include <stddef.h>

// ============================================================
// N-Gram 条目 - Space-Saving Top-2 设计（16 字节）
// ============================================================
// Misra-Gries 近似 Top-K：维护 2 个候选 token 及其计数
// 解决"最后写入覆盖"导致的翻转问题

typedef struct __attribute__((packed, aligned(16))) {
    uint32_t signature;      // 32 位哈希签名
    uint32_t token1;         // 最高频候选 token
    uint32_t token2;         // 次高频候选 token
    uint16_t count1;         // token1 计数
    uint16_t count2;         // token2 计数
} NgramEntry;  // 16 bytes

#define NGRAM_EMPTY_SIG 0

static inline int ngram_entry_is_empty(const NgramEntry* e) {
    return e->signature == NGRAM_EMPTY_SIG;
}

// N-gram 查询阈值：线性衰减公式
// 长 gram 更确定，阈值低；短 gram 不确定，阈值高
// threshold(level) = clamp(50 - 4 * level, 15, 50)
// level 2 → 42%, level 4 → 34%, level 8 → 18%
static inline uint16_t ngram_threshold_for_level(uint8_t level) {
    int t = 50 - 4 * (int)level;
    if (t < 15) t = 15;
    if (t > 50) t = 50;
    return (uint16_t)t;
}

// ============================================================
// 热数据层条目（保留兼容性）
// ============================================================
typedef struct __attribute__((packed, aligned(16))) {
    uint32_t signature;
    uint32_t token1;
    uint32_t token2;
    uint16_t count1;
    uint16_t count2;
} HotEntry;

// N-Gram 键 - 编译时上限
#define MAX_NGRAM_ORDER 8

// 默认窗口配置
#define DEFAULT_MIN_WINDOW 2
#define DEFAULT_MAX_WINDOW 4

// ============================================================
// 1-Gram 特殊存储结构
// ============================================================

typedef struct {
    uint32_t token;
    uint32_t count;
} UnigramTokenEntry;

typedef struct {
    uint32_t key_token;
    UnigramTokenEntry* tokens;
    uint16_t token_count;
    uint16_t capacity;
    uint32_t max_token;
    uint32_t max_count;
    uint32_t total_count;
} UnigramEntry;

#define UNIGRAM_EMPTY_KEY 0xFFFFFFFF

// 栈分配批量操作的最大大小
#define STACK_BATCH_SIZE 128

typedef struct {
    uint32_t tokens[MAX_NGRAM_ORDER];
    uint8_t length;
} ZipfKey;

// 查询结果
typedef struct {
    uint32_t next_token;
    uint8_t hit;
    uint8_t from_local;
} QueryResult;

// 局部hash表（每个req_id一个）
typedef struct {
    // n-gram (2-gram及以上) 存储
    NgramEntry* entries;
    size_t capacity;
    size_t count;

    // 1-gram 专用存储
    UnigramEntry* unigram_entries;
    size_t unigram_capacity;
    size_t unigram_count;
    float unigram_threshold;

    uint64_t req_id;
    uint64_t last_access;
    uint32_t generation_count;

    // 历史窗口
    uint32_t history[MAX_NGRAM_ORDER];
    uint8_t history_len;

    // 泛化层 n-gram 存储
    NgramEntry* gen_entries;
    size_t gen_capacity;
    size_t gen_count;

    // Adaptive-K 状态（per-request）
    float ema_accepted;          // EMA of accepted token count
    int16_t consecutive_fail;    // 连续失败次数
    int16_t consecutive_success; // 连续成功次数
    int16_t last_adaptive_k;    // 上一轮的 adaptive_k
    uint8_t is_initialized;      // 是否已初始化 adaptive-k 状态
} LocalHash;

// 硬编码常量
#define ZIPF_HARDCODED_SHARED_CAPACITY  16777216
#define ZIPF_HARDCODED_LOCAL_CAPACITY   262144
#define ZIPF_HARDCODED_MAX_LOCAL_HASHES 10000
#define ZIPF_HARDCODED_UNIGRAM_CAPACITY 131072
#define ZIPF_HARDCODED_UNIGRAM_THRESHOLD 0.50f

// 配置参数（仅保留可调参数）
typedef struct {
    uint8_t min_window;
    uint8_t max_window;
    uint8_t skip_shared;
    // 泛化层配置
    uint8_t enable_generalized;
    uint32_t gen_freq_threshold;
    size_t gen_capacity;
    uint32_t gen_bitmap_vocab_max;
    // 查询优先级：泛化层 vs 共享hash
    // 0 = shared 优先（local → shared → 泛化）
    // 1 = 泛化优先（local → 泛化 → shared）
    uint8_t generalized_before_shared;
} ZipfConfig;

static inline ZipfConfig zipf_default_config(void) {
    return (ZipfConfig){
        .min_window = DEFAULT_MIN_WINDOW,
        .max_window = DEFAULT_MAX_WINDOW,
        .skip_shared = 0,
        .enable_generalized = 1,
        .gen_freq_threshold = 8,
        .gen_capacity = 64 * 1024,
        .gen_bitmap_vocab_max = 256 * 1024,
        .generalized_before_shared = 1,
    };
}

#endif // ZIPF_TYPES_H
