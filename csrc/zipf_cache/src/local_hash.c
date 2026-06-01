#include "local_hash.h"
#include "hot_tier.h"
#include <stdlib.h>
#include <string.h>

// ============================================================
// NgramEntry 签名生成
// ============================================================
static inline uint32_t make_ngram_sig(uint64_t hash) {
    uint32_t sig = (uint32_t)(hash >> 32);
    if (sig == NGRAM_EMPTY_SIG) sig = 1;
    return sig;
}

// ============================================================
// 1-gram 哈希函数
// ============================================================
static inline uint32_t unigram_hash(uint32_t key, size_t size) {
    return (key * 2654435761U) % size;
}

static inline uint32_t unigram_probe(uint32_t h, uint32_t i, size_t size) {
    return (h + i) % size;
}

// ============================================================
// 1-gram 初始化辅助函数
// ============================================================
#define INITIAL_TOKEN_CAPACITY 4

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
        if (entries[i].tokens) free(entries[i].tokens);
    }
    free(entries);
}

static LocalHash* local_hash_create(uint64_t req_id, size_t capacity,
                                    size_t unigram_capacity, float unigram_threshold,
                                    size_t gen_capacity) {
    LocalHash* local = calloc(1, sizeof(LocalHash));
    if (!local) return NULL;

    local->entries = calloc(capacity, sizeof(NgramEntry));
    if (!local->entries) { free(local); return NULL; }

    if (unigram_capacity > 0) {
        size_t adjusted = (size_t)(unigram_capacity / 0.7);
        if (adjusted < unigram_capacity) adjusted = unigram_capacity;
        local->unigram_entries = calloc(adjusted, sizeof(UnigramEntry));
        if (!local->unigram_entries) { free(local->entries); free(local); return NULL; }
        init_unigram_entries(local->unigram_entries, adjusted);
        local->unigram_capacity = adjusted;
    }

    if (gen_capacity > 0) {
        local->gen_entries = calloc(gen_capacity, sizeof(NgramEntry));
        if (!local->gen_entries) {
            free_unigram_entries(local->unigram_entries, local->unigram_capacity);
            free(local->entries); free(local); return NULL;
        }
        local->gen_capacity = gen_capacity;
    }
    local->gen_count = 0;
    local->capacity = capacity;
    local->count = 0;
    local->unigram_count = 0;
    local->unigram_threshold = unigram_threshold;
    local->req_id = req_id;
    local->last_access = 0;
    local->generation_count = 0;
    local->history_len = 0;
    memset(local->history, 0, sizeof(local->history));
    // Adaptive-K 状态初始化
    local->ema_accepted = 0.0f;
    local->consecutive_fail = 0;
    local->consecutive_success = 0;
    local->last_adaptive_k = 0;
    local->is_initialized = 0;
    return local;
}

static void local_hash_free(LocalHash* local) {
    if (!local) return;
    free(local->entries);
    free(local->gen_entries);
    free_unigram_entries(local->unigram_entries, local->unigram_capacity);
    free(local);
}

LocalHashManager* local_hash_manager_create(size_t max_hashes,
                                            size_t local_capacity,
                                            size_t unigram_capacity,
                                            float unigram_threshold,
                                            size_t gen_capacity) {
    LocalHashManager* manager = calloc(1, sizeof(LocalHashManager));
    if (!manager) return NULL;
    manager->hashes = calloc(max_hashes, sizeof(LocalHash*));
    if (!manager->hashes) { free(manager); return NULL; }
    size_t index_cap = max_hashes * 2;
    if (index_cap < 16) index_cap = 16;
    manager->index = malloc(index_cap * sizeof(ReqIdIndexEntry));
    if (!manager->index) { free(manager->hashes); free(manager); return NULL; }
    for (size_t i = 0; i < index_cap; i++) manager->index[i].req_id = REQ_INDEX_EMPTY;
    manager->index_capacity = index_cap;
    manager->access_counter = 0;
    manager->capacity = max_hashes;
    manager->count = 0;
    manager->local_capacity = local_capacity;
    manager->unigram_capacity = unigram_capacity;
    manager->unigram_threshold = unigram_threshold;
    manager->gen_capacity = gen_capacity;
    return manager;
}

// 索引表内部哈希函数
static inline size_t req_index_hash(uint64_t req_id, size_t cap) {
    return (size_t)((req_id * 11400714819323198485ULL) >> 32) % cap;
}

static inline size_t req_index_find(const LocalHashManager* manager, uint64_t req_id) {
    size_t cap = manager->index_capacity;
    size_t h = req_index_hash(req_id, cap);
    for (size_t i = 0; i < cap; i++) {
        size_t slot = (h + i) % cap;
        if (manager->index[slot].req_id == REQ_INDEX_EMPTY) return (size_t)-1;
        if (manager->index[slot].req_id == req_id) return manager->index[slot].slot;
    }
    return (size_t)-1;
}

static inline void req_index_insert(LocalHashManager* manager, uint64_t req_id, size_t slot) {
    size_t cap = manager->index_capacity;
    size_t h = req_index_hash(req_id, cap);
    for (size_t i = 0; i < cap; i++) {
        size_t idx = (h + i) % cap;
        if (manager->index[idx].req_id == REQ_INDEX_EMPTY) {
            manager->index[idx].req_id = req_id;
            manager->index[idx].slot = slot;
            return;
        }
    }
}

static inline void req_index_remove(LocalHashManager* manager, uint64_t req_id) {
    size_t cap = manager->index_capacity;
    size_t h = req_index_hash(req_id, cap);
    for (size_t i = 0; i < cap; i++) {
        size_t idx = (h + i) % cap;
        if (manager->index[idx].req_id == REQ_INDEX_EMPTY) return;
        if (manager->index[idx].req_id == req_id) {
            manager->index[idx].req_id = REQ_INDEX_EMPTY;
            size_t next = (idx + 1) % cap;
            while (manager->index[next].req_id != REQ_INDEX_EMPTY) {
                size_t natural = req_index_hash(manager->index[next].req_id, cap);
                int needs_move;
                if (idx < next) needs_move = (natural <= idx || natural > next);
                else needs_move = (natural <= idx && natural > next);
                if (needs_move) {
                    manager->index[idx] = manager->index[next];
                    manager->index[next].req_id = REQ_INDEX_EMPTY;
                    idx = next;
                }
                next = (next + 1) % cap;
            }
            return;
        }
    }
}

static void req_index_rebuild(LocalHashManager* manager) {
    size_t cap = manager->index_capacity;
    for (size_t i = 0; i < cap; i++) manager->index[i].req_id = REQ_INDEX_EMPTY;
    for (size_t i = 0; i < manager->count; i++) {
        if (manager->hashes[i]) req_index_insert(manager, manager->hashes[i]->req_id, i);
    }
}

LocalHash* local_hash_manager_get_or_create(LocalHashManager* manager, uint64_t req_id) {
    if (!manager) return NULL;
    size_t found = req_index_find(manager, req_id);
    if (found != (size_t)-1) {
        manager->hashes[found]->last_access = ++manager->access_counter;
        return manager->hashes[found];
    }
    if (manager->count >= manager->capacity)
        local_hash_manager_cleanup(manager, manager->capacity * 3 / 4);
    LocalHash* local = local_hash_create(req_id, manager->local_capacity,
                                         manager->unigram_capacity,
                                         manager->unigram_threshold,
                                         manager->gen_capacity);
    if (!local) return NULL;
    local->last_access = ++manager->access_counter;
    size_t slot = manager->count;
    manager->hashes[slot] = local;
    manager->count++;
    req_index_insert(manager, req_id, slot);
    return local;
}

// Space-Saving Top-2 查询（带 gram 级别阈值）
int local_hash_query_with_level(const LocalHash* local, uint64_t ngram_hash, uint8_t level, uint32_t* out_next_token) {
    if (!local || !local->entries) return 0;
    uint32_t sig = make_ngram_sig(ngram_hash);
    uint32_t h = hot_tier_hash(ngram_hash, local->capacity);
    __builtin_prefetch(&local->entries[h], 0, 3);
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t slot = hot_tier_probe(h, i, local->capacity);
        const NgramEntry* entry = &local->entries[slot];
        if (ngram_entry_is_empty(entry)) return 0;
        if (entry->signature == sig) {
            if (entry->count1 == 0) return 0;
            uint16_t total = entry->count1 + entry->count2;
            uint16_t thresh = ngram_threshold_for_level(level);
            // 整数比较避免浮点：count1 * 100 >= total * thresh
            if ((uint32_t)entry->count1 * 100 >= (uint32_t)total * thresh) {
                *out_next_token = entry->token1;
                return 1;
            }
            return 0;
        }
    }
    return 0;
}

// 向后兼容：默认使用 level=0（使用默认阈值）
int local_hash_query(const LocalHash* local, uint64_t ngram_hash, uint32_t* out_next_token) {
    return local_hash_query_with_level(local, ngram_hash, 0, out_next_token);
}

int local_hash_query_unigram(const LocalHash* local, uint32_t key_token, uint32_t* out_next_token) {
    if (!local || !local->unigram_entries || local->unigram_capacity == 0) return 0;
    uint32_t h = unigram_hash(key_token, local->unigram_capacity);
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t slot = unigram_probe(h, i, local->unigram_capacity);
        const UnigramEntry* entry = &local->unigram_entries[slot];
        if (entry->key_token == UNIGRAM_EMPTY_KEY) return 0;
        if (entry->key_token == key_token) {
            if (entry->total_count == 0) return 0;
            float probability = (float)entry->max_count / (float)entry->total_count;
            if (probability > local->unigram_threshold) {
                *out_next_token = entry->max_token;
                return 1;
            }
            return 0;
        }
    }
    return 0;
}

// Space-Saving Misra-Gries 更新
int local_hash_update(LocalHash* local, uint64_t ngram_hash, uint32_t next_token) {
    if (!local || !local->entries) return -1;
    uint32_t sig = make_ngram_sig(ngram_hash);
    uint32_t h = hot_tier_hash(ngram_hash, local->capacity);
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t slot = hot_tier_probe(h, i, local->capacity);
        NgramEntry* entry = &local->entries[slot];
        if (ngram_entry_is_empty(entry)) {
            // 新条目
            entry->signature = sig;
            entry->token1 = next_token;
            entry->token2 = 0;
            entry->count1 = 1;
            entry->count2 = 0;
            local->count++;
            local->generation_count++;
            return 0;
        }
        if (entry->signature == sig) {
            // 已存在的条目，Space-Saving 更新
            if (next_token == entry->token1) {
                // 命中 token1
                if (entry->count1 < 65535) entry->count1++;
            } else if (next_token == entry->token2) {
                // 命中 token2
                if (entry->count2 < 65535) entry->count2++;
                // 如果 token2 超过 token1，交换
                if (entry->count2 > entry->count1) {
                    uint32_t tt = entry->token1; entry->token1 = entry->token2; entry->token2 = tt;
                    uint16_t tc = entry->count1; entry->count1 = entry->count2; entry->count2 = tc;
                }
            } else {
                // 第三个 token，Misra-Gries 衰减
                if (entry->count2 > 0) {
                    entry->count2--;
                    if (entry->count1 > 0) entry->count1--;
                } else {
                    // count2 已经是 0，替换 token2
                    entry->token2 = next_token;
                    entry->count2 = 1;
                }
            }
            return 0;
        }
    }
    return -1;
}

void local_hash_update_unigram(LocalHash* local, uint32_t key_token, uint32_t next_token) {
    if (!local || !local->unigram_entries || local->unigram_capacity == 0) return;
    uint32_t h = unigram_hash(key_token, local->unigram_capacity);
    for (uint32_t i = 0; i < local->unigram_capacity; i++) {
        uint32_t slot = unigram_probe(h, i, local->unigram_capacity);
        UnigramEntry* entry = &local->unigram_entries[slot];
        if (entry->key_token == UNIGRAM_EMPTY_KEY) {
            entry->key_token = key_token;
            entry->max_count = 0;
            entry->total_count = 0;
            local->unigram_count++;
        }
        if (entry->key_token == key_token) {
            for (uint16_t j = 0; j < entry->token_count; j++) {
                if (entry->tokens[j].token == next_token) {
                    entry->tokens[j].count++;
                    entry->total_count++;
                    if (entry->tokens[j].count > entry->max_count) {
                        entry->max_count = entry->tokens[j].count;
                        entry->max_token = next_token;
                    }
                    return;
                }
            }
            if (entry->token_count >= entry->capacity) {
                uint16_t new_cap = entry->capacity == 0 ? INITIAL_TOKEN_CAPACITY : entry->capacity * 2;
                UnigramTokenEntry* new_tokens = realloc(entry->tokens, new_cap * sizeof(UnigramTokenEntry));
                if (!new_tokens) return;
                entry->tokens = new_tokens;
                entry->capacity = new_cap;
            }
            entry->tokens[entry->token_count].token = next_token;
            entry->tokens[entry->token_count].count = 1;
            entry->token_count++;
            entry->total_count++;
            if (entry->max_count == 0) { entry->max_count = 1; entry->max_token = next_token; }
            return;
        }
    }
}

int local_hash_can_generate(const LocalHash* local) {
    if (!local) return 0;
    return local->generation_count < local->capacity;
}
void local_hash_reset_history(LocalHash* local) {
    if (!local) return;
    local->history_len = 0;
    memset(local->history, 0, sizeof(local->history));
}
void local_hash_set_history(LocalHash* local, const uint32_t* tokens, uint8_t len, uint8_t max_window) {
    if (!local || !tokens) return;
    uint8_t copy_len = len;
    const uint32_t* src = tokens;
    if (len > max_window) { copy_len = max_window; src = tokens + (len - max_window); }
    memcpy(local->history, src, copy_len * sizeof(uint32_t));
    local->history_len = copy_len;
}
const uint32_t* local_hash_get_history(const LocalHash* local, uint8_t* out_len) {
    if (!local) { if (out_len) *out_len = 0; return NULL; }
    if (out_len) *out_len = local->history_len;
    return local->history;
}

// 泛化层查询：Space-Saving Top-2 带级别阈值
int local_hash_query_generalized_with_level(const LocalHash* local, uint64_t gen_hash, uint8_t level, uint32_t* out_next_token) {
    if (!local || !local->gen_entries || local->gen_capacity == 0) return 0;
    uint32_t sig = make_ngram_sig(gen_hash);
    uint32_t h = hot_tier_hash(gen_hash, local->gen_capacity);
    __builtin_prefetch(&local->gen_entries[h], 0, 3);
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t slot = hot_tier_probe(h, i, local->gen_capacity);
        const NgramEntry* entry = &local->gen_entries[slot];
        if (ngram_entry_is_empty(entry)) return 0;
        if (entry->signature == sig) {
            if (entry->count1 == 0) return 0;
            uint16_t total = entry->count1 + entry->count2;
            uint16_t thresh = ngram_threshold_for_level(level);
            if ((uint32_t)entry->count1 * 100 >= (uint32_t)total * thresh) {
                *out_next_token = entry->token1;
                return 1;
            }
            return 0;
        }
    }
    return 0;
}

// 向后兼容
int local_hash_query_generalized(const LocalHash* local, uint64_t gen_hash, uint32_t* out_next_token) {
    return local_hash_query_generalized_with_level(local, gen_hash, 0, out_next_token);
}

// 泛化层更新：Space-Saving Misra-Gries
int local_hash_update_generalized(LocalHash* local, uint64_t gen_hash, uint32_t next_token) {
    if (!local || !local->gen_entries || local->gen_capacity == 0) return -1;
    uint32_t sig = make_ngram_sig(gen_hash);
    uint32_t h = hot_tier_hash(gen_hash, local->gen_capacity);
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t slot = hot_tier_probe(h, i, local->gen_capacity);
        NgramEntry* entry = &local->gen_entries[slot];
        if (ngram_entry_is_empty(entry)) {
            entry->signature = sig;
            entry->token1 = next_token;
            entry->token2 = 0;
            entry->count1 = 1;
            entry->count2 = 0;
            local->gen_count++;
            return 0;
        }
        if (entry->signature == sig) {
            if (next_token == entry->token1) {
                if (entry->count1 < 65535) entry->count1++;
            } else if (next_token == entry->token2) {
                if (entry->count2 < 65535) entry->count2++;
                if (entry->count2 > entry->count1) {
                    uint32_t tt = entry->token1; entry->token1 = entry->token2; entry->token2 = tt;
                    uint16_t tc = entry->count1; entry->count1 = entry->count2; entry->count2 = tc;
                }
            } else {
                if (entry->count2 > 0) {
                    entry->count2--;
                    if (entry->count1 > 0) entry->count1--;
                } else {
                    entry->token2 = next_token;
                    entry->count2 = 1;
                }
            }
            return 0;
        }
    }
    return -1;
}

static int compare_local_hash_last_access(const void* a, const void* b) {
    const LocalHash* access_a = *(const LocalHash* const*)a;
    const LocalHash* access_b = *(const LocalHash* const*)b;
    return (access_a->last_access < access_b->last_access) -
           (access_a->last_access > access_b->last_access);
}

void local_hash_manager_cleanup(LocalHashManager* manager, size_t keep_count) {
    if (!manager || manager->count <= keep_count) return;
    qsort(manager->hashes, manager->count, sizeof(LocalHash*), compare_local_hash_last_access);
    for (size_t i = keep_count; i < manager->count; i++) {
        local_hash_free(manager->hashes[i]);
        manager->hashes[i] = NULL;
    }
    manager->count = keep_count;
    req_index_rebuild(manager);
}

void local_hash_manager_free(LocalHashManager* manager) {
    if (!manager) return;
    for (size_t i = 0; i < manager->count; i++) local_hash_free(manager->hashes[i]);
    free(manager->hashes);
    free(manager->index);
    free(manager);
}

LocalHashStats local_hash_manager_stats(const LocalHashManager* manager) {
    LocalHashStats stats = {0};
    if (!manager) return stats;
    stats.total_hashes = manager->count;
    for (size_t i = 0; i < manager->count; i++) {
        if (manager->hashes[i]) {
            LocalHash* local = manager->hashes[i];
            stats.total_entries += local->count;
            stats.total_unigram += local->unigram_count;
            stats.total_gen_entries += local->gen_count;
            stats.memory_bytes += local->capacity * sizeof(NgramEntry);
            stats.memory_bytes += local->gen_capacity * sizeof(NgramEntry);
            stats.memory_bytes += local->unigram_capacity * sizeof(UnigramEntry);
            for (size_t j = 0; j < local->unigram_capacity; j++) {
                if (local->unigram_entries && local->unigram_entries[j].key_token != UNIGRAM_EMPTY_KEY)
                    stats.memory_bytes += local->unigram_entries[j].capacity * sizeof(UnigramTokenEntry);
            }
        }
    }
    stats.memory_bytes += sizeof(LocalHashManager) + manager->capacity * sizeof(LocalHash*);
    return stats;
}
