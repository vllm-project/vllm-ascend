#ifndef LOCAL_HASH_H
#define LOCAL_HASH_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

// req_id 索引表条目
typedef struct {
    uint64_t req_id;
    size_t   slot;
} ReqIdIndexEntry;

#define REQ_INDEX_EMPTY 0xFFFFFFFFFFFFFFFFULL

typedef struct {
    LocalHash** hashes;
    size_t capacity;
    size_t count;
    size_t local_capacity;
    size_t unigram_capacity;
    float unigram_threshold;
    size_t gen_capacity;
    // req_id 快速索引
    ReqIdIndexEntry* index;
    size_t index_capacity;
    uint64_t access_counter;
} LocalHashManager;

LocalHashManager* local_hash_manager_create(size_t max_hashes,
                                            size_t local_capacity,
                                            size_t unigram_capacity,
                                            float unigram_threshold,
                                            size_t gen_capacity);

LocalHash* local_hash_manager_get_or_create(LocalHashManager* manager,
                                            uint64_t req_id);

// 查询 n-gram（带 gram 级别，用于 Space-Saving 阈值判断）
int local_hash_query_with_level(const LocalHash* local,
                                uint64_t ngram_hash,
                                uint8_t level,
                                uint32_t* out_next_token);

// 查询 n-gram（默认级别，向后兼容）
int local_hash_query(const LocalHash* local,
                    uint64_t ngram_hash,
                    uint32_t* out_next_token);

int local_hash_query_unigram(const LocalHash* local,
                             uint32_t key_token,
                             uint32_t* out_next_token);

// 更新：Space-Saving Misra-Gries 策略
int local_hash_update(LocalHash* local,
                     uint64_t ngram_hash,
                     uint32_t next_token);

void local_hash_update_unigram(LocalHash* local,
                               uint32_t key_token,
                               uint32_t next_token);

int local_hash_can_generate(const LocalHash* local);
void local_hash_reset_history(LocalHash* local);
void local_hash_set_history(LocalHash* local, const uint32_t* tokens, uint8_t len, uint8_t max_window);
const uint32_t* local_hash_get_history(const LocalHash* local, uint8_t* out_len);

// 泛化层查询：带 gram 级别的 Space-Saving 阈值判断
int local_hash_query_generalized_with_level(const LocalHash* local,
                                            uint64_t gen_hash,
                                            uint8_t level,
                                            uint32_t* out_next_token);

// 泛化层查询：默认级别，向后兼容
int local_hash_query_generalized(const LocalHash* local,
                                 uint64_t gen_hash,
                                 uint32_t* out_next_token);

// 泛化层更新：Space-Saving Misra-Gries 策略
int local_hash_update_generalized(LocalHash* local,
                                  uint64_t gen_hash,
                                  uint32_t next_token);

void local_hash_manager_cleanup(LocalHashManager* manager, size_t keep_count);
void local_hash_manager_free(LocalHashManager* manager);

typedef struct {
    size_t total_hashes;
    size_t total_entries;
    size_t total_unigram;
    size_t total_gen_entries;
    size_t memory_bytes;
} LocalHashStats;

LocalHashStats local_hash_manager_stats(const LocalHashManager* manager);

#ifdef __cplusplus
}
#endif

#endif // LOCAL_HASH_H
