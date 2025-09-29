// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#include <stddef.h>
#include <stdint.h>

typedef enum
{
    KV_EVENT_BLOCK_STORED = 0,
    KV_EVENT_BLOCK_REMOVED = 1,
    KV_EVENT_ALL_BLOCKS_CLEARED = 2,
    KV_EVENT_UNKNOWN = -1
} KVEventType;

typedef struct
{
    double ts;
    void **events;
    size_t events_count;
} KVEventBatch;

typedef struct
{
    KVEventType type;
    union
    {
        struct
        {
            int64_t *block_hashes;
            size_t block_hashes_count;
            int64_t parent_block_hash;
            int64_t *token_ids;
            size_t token_ids_count;
            int32_t block_size;
            int64_t lora_id;
        } block_stored;

        struct
        {
            int64_t *block_hashes;
            size_t block_hashes_count;
        } block_removed;
    } data;
} KVCacheEvent;

struct omni_zmq_handler_t;

KVEventBatch *parse_kv_event_batch(const void *payload, size_t length);
void free_kv_event_batch(KVEventBatch *batch);
void print_kv_event_batch(const KVEventBatch *batch);