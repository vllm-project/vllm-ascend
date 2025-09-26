// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <ngx_http_upstream.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <ngx_atomic.h>
#include "../../../modules/ngx_http_prefill_module/jsmn.h"
#include "../ngx_http_upstream_least_total_load_module/ngx_http_upstream_least_total_load_module.h"

#define DEFAULT_BATCH_SIZE 16
#define MAX_PEERS          512
#define HASH_BLOCK_SIZE    640  // (128 * 5)
#define MAX_HASH_BLOCKS    1024
#define MAX_CACHE_SIZE     10000
#define REQUEST_DIFF_THRESHOLD     15

typedef struct ngx_http_node_t {
    uint64_t                key;
    ngx_uint_t              peer;
    struct ngx_http_node_t  *next_key;
    struct ngx_http_node_t  *next;
    struct ngx_http_node_t  *prev;
} ngx_http_node_t;

typedef struct {
    ngx_http_node_t         *head;
    ngx_shmtx_t             mutex;
} ngx_http_bucket_t;

typedef struct {
    ngx_http_node_t         *head;
    ngx_http_node_t         *tail;
    ngx_uint_t              size;
    ngx_shmtx_t             mutex;
} ngx_http_list_t;

typedef struct {
    ngx_uint_t              bucket_count;
    ngx_uint_t              peer_count;
    ngx_uint_t              max_size;
    ngx_http_bucket_t       *buckets;
    ngx_http_list_t         *lists;
    prefill_upstream_info_t *prefill_shm;
    ngx_shmtx_t             mutex;
} ngx_http_prefix_cache_affinity_data_t;

typedef struct {
    ngx_flag_t              enable;
    ngx_uint_t              batch_size;
    ngx_uint_t              cache_size;
    ngx_uint_t              block_size;
} ngx_http_prefix_cache_affinity_conf_t;

typedef struct {
    ngx_http_upstream_rr_peer_data_t *rrp;
    ngx_uint_t              chosen;
    ngx_uint_t              request_length;
    ngx_event_get_peer_pt   original_get;
    ngx_event_free_peer_pt  original_free;
    void                    *original_data;
} ngx_http_prefix_cache_affinity_peer_data_t;

typedef struct {
    uint64_t   blocks[MAX_HASH_BLOCKS];
    ngx_uint_t count;
} ngx_http_hash_chain_t;

static ngx_http_prefix_cache_affinity_data_t *global_cache = NULL;
static ngx_shm_zone_t *ngx_http_prefix_cache_affinity_shm_zone = NULL;
static ngx_uint_t ngx_http_prefix_cache_affinity_shm_size = 0;
static ngx_uint_t ngx_http_prefix_cache_affinity_dict_size = 0;
static ngx_uint_t ngx_http_prefix_cache_affinity_block_size = 0;
static ngx_uint_t ngx_http_least_total_load_batch_size = 0;
static const uint8_t siphash_key[16] = {0};

static ngx_int_t ngx_http_prefix_cache_affinity_get_peer(
    ngx_peer_connection_t *pc, void *data);
static void ngx_http_prefix_cache_affinity_free_peer(
    ngx_peer_connection_t *pc, void *data, ngx_uint_t state);
static ngx_int_t ngx_http_prefix_cache_affinity_postconfig(
    ngx_conf_t *cf);

static void 
*ngx_http_prefix_cache_affinity_create_srv_conf(ngx_conf_t *cf)
{
    ngx_http_prefix_cache_affinity_conf_t *conf =
        ngx_pcalloc(cf->pool, sizeof(*conf));
    if (conf == NULL) {
        return NULL;
    }
    conf->enable = NGX_CONF_UNSET;
    conf->batch_size = NGX_CONF_UNSET_UINT;
    conf->cache_size = NGX_CONF_UNSET_UINT;
    conf->block_size = NGX_CONF_UNSET_UINT;
    return conf;
}

static ngx_int_t ngx_http_prefix_cache_affinity_init_shm_zone(
    ngx_shm_zone_t *shm_zone, void *data)
{
    ngx_slab_pool_t                       *shpool;
    ngx_http_prefix_cache_affinity_data_t *cache;
    ngx_uint_t                            i, peer_count;
    ngx_http_upstream_srv_conf_t          *uscf;

    if (data) {
        shm_zone->data = data;
        global_cache = data;
        return NGX_OK;
    }

    uscf = shm_zone->tag;
    if (uscf && uscf->servers) {
        peer_count = uscf->servers->nelts;
    } else {
        peer_count = 0;
    }
    if (peer_count == 0) {
        return NGX_ERROR;
    }

    shpool = (ngx_slab_pool_t *)shm_zone->shm.addr;

    cache = ngx_slab_alloc(shpool, sizeof(ngx_http_prefix_cache_affinity_data_t));
    if (cache == NULL) {
        return NGX_ERROR;
    }

    cache->peer_count = peer_count;
    cache->max_size = ngx_http_prefix_cache_affinity_dict_size;
    cache->bucket_count = cache->peer_count * cache->max_size;

    ngx_http_bucket_t *buckets = ngx_slab_alloc(
        shpool, sizeof(ngx_http_bucket_t) * cache->bucket_count);
    if (buckets == NULL) {
        return NGX_ERROR;
    }

    size_t bucket_mutex_size = sizeof(ngx_shmtx_sh_t) * cache->bucket_count;
    ngx_shmtx_sh_t *bucket_mutexes = ngx_slab_alloc(shpool, bucket_mutex_size);
    if (bucket_mutexes == NULL) {
        return NGX_ERROR;
    }
    ngx_memzero(bucket_mutexes, bucket_mutex_size);

    for (i = 0; i < cache->bucket_count; i++) {
        ngx_shmtx_create(
            &buckets[i].mutex, &bucket_mutexes[i], (u_char *)"bucket_mutex");
        buckets[i].head = NULL;
    }

    ngx_http_list_t *lists = ngx_slab_alloc(
        shpool, sizeof(ngx_http_list_t) * cache->peer_count);
    if (lists == NULL) {
        return NGX_ERROR;
    }

    size_t list_mutex_size = sizeof(ngx_shmtx_sh_t) * cache->peer_count;
    ngx_shmtx_sh_t *list_mutexes = ngx_slab_alloc(shpool, list_mutex_size);
    if (list_mutexes == NULL) {
        return NGX_ERROR;
    }
    ngx_memzero(list_mutexes, list_mutex_size);

    for (i = 0; i < cache->peer_count; i++) {
        ngx_shmtx_create(
            &lists[i].mutex, &list_mutexes[i], (u_char *)"list_mutex");
            lists[i].head = lists[i].tail = NULL;
            lists[i].size = 0;
    }

    cache->buckets = buckets;
    cache->lists = lists;

    ngx_shmtx_sh_t *cache_mutex = ngx_slab_alloc(shpool, sizeof(ngx_shmtx_sh_t));
    ngx_shmtx_create(&cache->mutex, cache_mutex, (u_char *)"cache_mutex");

    size_t sz = sizeof(prefill_upstream_info_t) + 
                (peer_count - 1) * sizeof(ngx_http_least_total_load_shm_peer_t);
    cache->prefill_shm = ngx_slab_alloc(shpool, sz);
    if (cache->prefill_shm == NULL) {
        return NGX_ERROR;
    }
    cache->prefill_shm->peer_count = peer_count;
    for (i = 0; i < peer_count; i++) {
        cache->prefill_shm->peers[i].total_length_sum = 0;
        cache->prefill_shm->peers[i].total_request_sum = 0;
    }

    shm_zone->data = cache;
    global_cache = cache;

    return NGX_OK;
}

static ngx_command_t 
ngx_http_upstream_prefix_cache_affinity_commands[] = {
    {ngx_string("prefix_cache_affinity"),
     NGX_HTTP_UPS_CONF | NGX_CONF_FLAG,
     ngx_conf_set_flag_slot,
     NGX_HTTP_SRV_CONF_OFFSET,
     offsetof(ngx_http_prefix_cache_affinity_conf_t, enable),
     NULL},
    {ngx_string("least_total_load_batch_size"),
     NGX_HTTP_UPS_CONF | NGX_CONF_TAKE1,
     ngx_conf_set_num_slot,
     NGX_HTTP_SRV_CONF_OFFSET,
     offsetof(ngx_http_prefix_cache_affinity_conf_t, batch_size),
     NULL},
    {ngx_string("prefix_cache_affinity_dict_size"),
     NGX_HTTP_UPS_CONF | NGX_CONF_TAKE1,
     ngx_conf_set_num_slot,
     NGX_HTTP_SRV_CONF_OFFSET,
     offsetof(ngx_http_prefix_cache_affinity_conf_t, cache_size),
     NULL},
     {ngx_string("prefix_cache_affinity_block_size"),
        NGX_HTTP_UPS_CONF | NGX_CONF_TAKE1,
        ngx_conf_set_num_slot,
        NGX_HTTP_SRV_CONF_OFFSET,
        offsetof(ngx_http_prefix_cache_affinity_conf_t, block_size),
        NULL},
     ngx_null_command};

static ngx_http_module_t 
ngx_http_upstream_prefix_cache_affinity_module_ctx = {
    NULL,                                        /* preconfiguration */
    ngx_http_prefix_cache_affinity_postconfig,
    NULL,                                        /* create_main_conf */
    NULL,                                        /* init_main_conf */
    ngx_http_prefix_cache_affinity_create_srv_conf,
    NULL,                                        /* merge_srv_conf */
    NULL,                                        /* create_loc_conf */
    NULL};                                       /* merge_loc_conf */

ngx_module_t ngx_http_upstream_prefix_cache_affinity_module = {
    NGX_MODULE_V1,
    &ngx_http_upstream_prefix_cache_affinity_module_ctx,
    ngx_http_upstream_prefix_cache_affinity_commands,
    NGX_HTTP_MODULE,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NGX_MODULE_V1_PADDING};

static ngx_http_hash_chain_t 
compute_hash_chain(const uint8_t *data, size_t len, const uint8_t key[16]) 
{ 
    (void)key; /* unused */ 
    ngx_http_hash_chain_t hash_chain; 
    hash_chain.count = 0; 

    ngx_uint_t total_blocks = 
        (ngx_uint_t)(len / ngx_http_prefix_cache_affinity_block_size); 
    if (total_blocks == 0) { 
        return hash_chain; 
    } 
    if (total_blocks > MAX_HASH_BLOCKS) { 
        total_blocks = MAX_HASH_BLOCKS; 
    } 
    /* Compute per-block DJB2 hashes into fixed array */ 
    ngx_uint_t hash = 5381; 
    for (ngx_uint_t i = 0; i < total_blocks; i++) { 
        const uint8_t *block = 
            data + (size_t)i * ngx_http_prefix_cache_affinity_block_size; 
        for (size_t j = 0; j < ngx_http_prefix_cache_affinity_block_size; j++) {
            hash = ((hash << 5) + hash) + block[j]; /* hash * 33 + byte */ 
        } 
        hash_chain.blocks[i] = hash; 
    } 
    hash_chain.count = total_blocks; 
    return hash_chain; 
}

static ngx_int_t 
extract_content_field(u_char *json, size_t len, ngx_str_t *content)
{
    jsmn_parser parser;
    jsmntok_t tokens[128];
    int ret, i;

    jsmn_init(&parser);
    ret = jsmn_parse(
        &parser, 
        (const char *)json, 
        len, 
        tokens, 
        sizeof(tokens) / sizeof(tokens[0]));

    if (ret < 0 || ret < 1 || tokens[0].type != JSMN_OBJECT) {
        return NGX_ERROR;
    }

    for (i = 1; i < ret; i++) {
        if (tokens[i].type == JSMN_STRING &&
            (int)ngx_strlen("content") == tokens[i].end - tokens[i].start &&
            ngx_strncmp(
                json + tokens[i].start, "content", sizeof("content") - 1) == 0)
        {
            jsmntok_t *val = &tokens[i + 1];
            content->data = (u_char *)(json + val->start);
            content->len = val->end - val->start;
            return NGX_OK;
        }

        if (tokens[i].type == JSMN_STRING &&
            (int)ngx_strlen("prompt") == tokens[i].end - tokens[i].start &&
            ngx_strncmp(
                json + tokens[i].start, "prompt", sizeof("prompt") - 1) == 0) {
            jsmntok_t *val = &tokens[i + 1];
            content->data = (u_char *)(json + val->start);
            content->len = val->end - val->start;
            return NGX_OK;
        }
    }

    return NGX_DECLINED;
}

static ngx_http_hash_chain_t 
ngx_http_generate_request_hash_chain(ngx_http_request_t *r)
{
    ngx_http_hash_chain_t hash_chain;
    hash_chain.count = 0;

    if (r->request_body && r->request_body->bufs) {
        size_t        total = 0;
        for (ngx_chain_t *cl = r->request_body->bufs; cl; cl = cl->next) {
            total += cl->buf->last - cl->buf->pos;
        }

        u_char *raw = ngx_pnalloc(r->pool, total + 1);
        if (raw == NULL) {
            return hash_chain;
        }

        u_char *p = raw;
        for (ngx_chain_t *cl = r->request_body->bufs; cl; cl = cl->next) {
            p = ngx_copy(p, cl->buf->pos, cl->buf->last - cl->buf->pos);
        }
        raw[total] = '\0';

        ngx_str_t content = ngx_null_string;
        if (extract_content_field(raw, total, &content) == NGX_OK 
            && content.len > 0) {
            hash_chain = compute_hash_chain(
                content.data, 
                content.len, 
                siphash_key);
        } else {
            hash_chain = compute_hash_chain(
                r->uri.data, 
                r->uri.len, 
                siphash_key);
        }
    } else {
        hash_chain = compute_hash_chain(r->uri.data, r->uri.len, siphash_key);
    }

    return hash_chain;
}

static ngx_uint_t get(uint64_t key, ngx_uint_t peer_count, const double *scores)
{
    ngx_uint_t count = 0;
    ngx_uint_t peer = peer_count;
    double min_score;

    ngx_uint_t index = key % global_cache->bucket_count;
    ngx_http_bucket_t *bucket = &global_cache->buckets[index];

    ngx_shmtx_lock(&bucket->mutex);

    ngx_http_node_t *curr = bucket->head;
    while (curr) {
        if (curr->key == key && curr->peer < peer_count) {
            if (count == 0 || scores[curr->peer] < min_score) {
                count = 1;
                min_score = scores[curr->peer];
                peer = curr->peer;
            } else if (scores[curr->peer] == min_score) {
                count++;
                if (ngx_random() % count == 0)
                    peer = curr->peer;
            }
        }
        curr = curr->next_key;
    }
    ngx_shmtx_unlock(&bucket->mutex);
    return peer;
}

static ngx_int_t insert(uint64_t key, ngx_uint_t peer)
{
    ngx_slab_pool_t *shpool = 
        (ngx_slab_pool_t *)ngx_http_prefix_cache_affinity_shm_zone->shm.addr;

    ngx_uint_t index = key % global_cache->bucket_count;
    ngx_http_bucket_t *bucket = &global_cache->buckets[index];
    ngx_http_list_t *list = &global_cache->lists[peer];
    ngx_shmtx_lock(&bucket->mutex);

    ngx_http_node_t *curr = bucket->head;
    while (curr) {
        if (curr->key == key && curr->peer == peer) {
            ngx_shmtx_unlock(&bucket->mutex);
            ngx_shmtx_lock(&list->mutex);
            if (curr->next) {
                curr->next->prev = curr->prev;
                if (curr->prev) {
                    curr->prev->next = curr->next;
                } else {
                    list->head = curr->next;
                }
                curr->next = NULL;
                curr->prev = list->tail;
                list->tail->next = curr;
                list->tail = curr;
            }
            ngx_shmtx_unlock(&list->mutex);
            return NGX_OK;
        }
        curr = curr->next_key;
    }

    ngx_http_node_t *new_node = ngx_slab_alloc(shpool, sizeof(ngx_http_node_t));

    if (!new_node) {
        ngx_shmtx_unlock(&bucket->mutex);
        return NGX_ERROR;
    }

    new_node->key = key;
    new_node->peer = peer;
    new_node->next_key = bucket->head;
    bucket->head = new_node;
    ngx_shmtx_unlock(&bucket->mutex);

    ngx_shmtx_lock(&list->mutex);
    if (list->tail) {
        list->tail->next = new_node;
        new_node->prev = list->tail;
        list->tail = new_node;
        new_node->next = NULL;
    } else {
        list->head = list->tail = new_node;
        new_node->next = new_node->prev = NULL;
    }
    list->size++;

    if (list->size == global_cache->max_size + 1) {
        key = list->head->key;
        index = key % global_cache->bucket_count;

        curr = list->head;
        list->head = list->head->next;
        list->head->prev = NULL;
        curr->next = NULL;
        curr->prev = NULL;
        list->size--;

        bucket = &global_cache->buckets[index];
        ngx_shmtx_unlock(&list->mutex);
        ngx_shmtx_lock(&bucket->mutex);

        ngx_http_node_t *prev = NULL;
        curr = bucket->head;
        while (curr) {
            if (curr->key == key && curr->peer == peer) {
                if (prev) {
                    prev->next_key = curr->next_key;
                } else {
                    bucket->head = curr->next_key;
                } 
                curr->next_key = NULL;
                ngx_slab_free(shpool, curr);
                ngx_shmtx_unlock(&bucket->mutex);
                return NGX_OK;
            }
            prev = curr;
            curr = curr->next_key;
        }
        ngx_shmtx_unlock(&bucket->mutex);
        return NGX_ERROR;
    }

    ngx_shmtx_unlock(&list->mutex);
    return NGX_OK;
}

static ngx_uint_t 
prefix_cache_affinity_select(
    ngx_http_request_t *r,
    ngx_uint_t request_length,
    ngx_uint_t peer_count)
{
    const char                               *strategy;
    ngx_http_hash_chain_t                    hash_chain;
    ngx_uint_t                               chosen = 0;
    double *scores = (double*) malloc(peer_count * sizeof(double));
    ngx_uint_t                               min_len = (ngx_uint_t) - 1;  
    ngx_uint_t                               max_len = 0;
    ngx_uint_t                               max_hits = 0;

    for (ngx_uint_t i = 0; i < peer_count; ++i) {
        ngx_uint_t length_sum_peers = 
            global_cache->prefill_shm->peers[i].total_length_sum;
        ngx_uint_t request_sum_peers = 
            global_cache->prefill_shm->peers[i].total_request_sum;

        if (request_sum_peers < min_len) { 
            min_len = request_sum_peers; 
        }
        if (request_sum_peers > max_len) { 
            max_len = request_sum_peers; 
        }

        scores[i] = least_total_load_get_score(
            length_sum_peers, request_sum_peers,
            (double)ngx_http_least_total_load_batch_size);
    }

    hash_chain = ngx_http_generate_request_hash_chain(r);
    ngx_uint_t request_diff = max_len - min_len;
    if (request_diff > REQUEST_DIFF_THRESHOLD) {
        strategy = "least_total_load";
        least_total_load_select_solver(
            global_cache->prefill_shm, peer_count, request_length, &chosen);
    } else {
        strategy = "prefix_cache_affinity";
        chosen = peer_count;
        int L = 0, R = (int)hash_chain.count;
        while (R > L) {
            int mid = (L + R) / 2;
            ngx_uint_t peer = get(hash_chain.blocks[mid], peer_count, scores);
            if (peer == peer_count) {
                R = mid;
            } else {
                L = mid + 1;
                chosen = peer;
            }
        }
        max_hits = L;
        if (max_hits == 0) {
            strategy = "least_total_load";
            least_total_load_select_solver(
                global_cache->prefill_shm, peer_count, request_length, &chosen);
        }
    }

    for (ngx_int_t j = hash_chain.count - 1; j >= 0; j--) {
        insert(hash_chain.blocks[j], chosen);
    }

    ngx_shmtx_lock(&global_cache->mutex);
    ngx_atomic_fetch_add(
        &(global_cache->prefill_shm->peers[chosen].total_length_sum), 
        (ngx_atomic_int_t)request_length);
    ngx_atomic_fetch_add(
        &(global_cache->prefill_shm->peers[chosen].total_request_sum), 
        (ngx_atomic_int_t)1);
    ngx_shmtx_unlock(&global_cache->mutex);

    ngx_log_error(
        NGX_LOG_WARN, 
        r->connection->log, 
        0,
        "[prefix_cache_affinity][%s] chosen peer=%ui, "
        "hits=%ui/%ui, request_length=%ui",
        strategy,
        chosen,
        max_hits, 
        hash_chain.count, 
        request_length);
    free(scores);
    return chosen;
}

static ngx_int_t
ngx_http_prefix_cache_affinity_upstream_init(
    ngx_http_request_t *r,
    ngx_http_upstream_srv_conf_t *uscf)
{
    ngx_http_upstream_t                        *u;
    ngx_http_upstream_rr_peer_data_t           *rrp;
    ngx_http_prefix_cache_affinity_peer_data_t *pdata;
    ngx_uint_t                                 chosen = 0;
    ngx_uint_t                                 peer_count;

    u = r->upstream;
    if (ngx_http_upstream_init_round_robin_peer(r, uscf) != NGX_OK) {
        return NGX_ERROR;
    }
    rrp = u->peer.data;

    if (global_cache == NULL) {
        global_cache = ngx_http_prefix_cache_affinity_shm_zone->data;
    }

    peer_count = rrp->peers->number;
    peer_count = ngx_min(peer_count, global_cache->prefill_shm->peer_count);
    if (peer_count == 0) {
        return NGX_ERROR;
    }
    
    chosen = prefix_cache_affinity_select(
        r, (ngx_uint_t)r->request_length, peer_count);

    pdata = ngx_pcalloc(r->pool, sizeof(*pdata));
    if (pdata == NULL) {
        return NGX_ERROR;
    }
    pdata->original_get = u->peer.get;
    pdata->original_free = u->peer.free;
    pdata->original_data = u->peer.data;
    pdata->rrp = rrp;

    pdata->chosen = chosen;
    pdata->request_length = (ngx_uint_t)r->request_length;
    u->peer.data = pdata;
    u->peer.get = ngx_http_prefix_cache_affinity_get_peer;
    u->peer.free = ngx_http_prefix_cache_affinity_free_peer;

    return NGX_OK;
}

static ngx_int_t ngx_http_prefix_cache_affinity_get_peer(
    ngx_peer_connection_t *pc, void *data)
{
    ngx_http_prefix_cache_affinity_peer_data_t *pdata = data;
    ngx_http_upstream_rr_peer_data_t *rrp = pdata->rrp;
    ngx_http_upstream_rr_peers_t *peers = rrp->peers;
    ngx_uint_t idx = pdata->chosen;

    if (idx >= peers->number) {
        return ngx_http_upstream_get_round_robin_peer(pc, rrp);
    }
    if (peers->peer[idx].down) {
        return NGX_BUSY;
    }
    pc->sockaddr = peers->peer[idx].sockaddr;
    pc->socklen = peers->peer[idx].socklen;
    pc->name = &peers->peer[idx].name;
    rrp->current = &peers->peer[idx];
    return NGX_OK;
}

static void ngx_http_prefix_cache_affinity_free_peer(
    ngx_peer_connection_t *pc, void *data, ngx_uint_t state)
{
    ngx_http_prefix_cache_affinity_peer_data_t *pdata = data;

    if (global_cache != NULL && pdata->chosen < pdata->rrp->peers->number) {
        ngx_shmtx_lock(&global_cache->mutex);
        ngx_atomic_fetch_add(
            &global_cache->prefill_shm->peers[pdata->chosen].total_request_sum, 
            (ngx_atomic_int_t) - 1);
        ngx_atomic_fetch_add(
            &global_cache->prefill_shm->peers[pdata->chosen].total_length_sum, 
            (ngx_atomic_int_t) - (ngx_int_t)(pdata->request_length));
        ngx_shmtx_unlock(&global_cache->mutex);
        pdata->chosen = pdata->rrp->peers->number;
    }

    ngx_http_upstream_rr_peer_data_t *rrp = pdata->rrp;
    ngx_http_upstream_free_round_robin_peer(pc, rrp, state);
}

static ngx_int_t 
ngx_http_prefix_cache_affinity_postconfig(ngx_conf_t *cf)
{
    ngx_str_t prefix_cache_affinity_shm_name;
    ngx_http_upstream_main_conf_t         *upcf;
    ngx_http_upstream_srv_conf_t          **uscfp;
    ngx_http_prefix_cache_affinity_conf_t *conf;
    ngx_uint_t                            i;

    if (ngx_http_prefix_cache_affinity_shm_size == 0) {
        ngx_http_prefix_cache_affinity_shm_size = 2048 * ngx_pagesize;
    }

    prefix_cache_affinity_shm_name.len = 
        sizeof("upstream_prefill_prefix_cache_affinity") - 1;
    prefix_cache_affinity_shm_name.data = 
        (u_char *)"upstream_prefill_prefix_cache_affinity";

    upcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_upstream_module);
    if (upcf == NULL) {
        return NGX_OK;
    }

    uscfp = upcf->upstreams.elts;
    for (i = 0; i < upcf->upstreams.nelts; i++) {
        conf = ngx_http_conf_upstream_srv_conf(
            uscfp[i], ngx_http_upstream_prefix_cache_affinity_module);

        if (conf->enable != 1) {
            continue;
        }

        if (conf->batch_size == NGX_CONF_UNSET_UINT) {
            ngx_http_least_total_load_batch_size = DEFAULT_BATCH_SIZE;
        } else {
            ngx_http_least_total_load_batch_size = conf->batch_size;
        }

        if (conf->cache_size == NGX_CONF_UNSET_UINT) {
            ngx_http_prefix_cache_affinity_dict_size = MAX_CACHE_SIZE;
        } else {
            ngx_http_prefix_cache_affinity_dict_size = conf->cache_size;
        }

        if (conf->block_size == NGX_CONF_UNSET_UINT) {
            ngx_http_prefix_cache_affinity_block_size = HASH_BLOCK_SIZE;
        } else {
            ngx_http_prefix_cache_affinity_block_size = conf->block_size;
        }

        ngx_uint_t peer_count = 0;
        if (uscfp[i]->servers) {
            peer_count = uscfp[i]->servers->nelts;
        }

        ngx_http_prefix_cache_affinity_shm_zone = 
            ngx_shared_memory_add(
                cf,
                &prefix_cache_affinity_shm_name,
                ngx_http_prefix_cache_affinity_shm_size,
                &ngx_http_upstream_prefix_cache_affinity_module);

        if (ngx_http_prefix_cache_affinity_shm_zone == NULL) {
            return NGX_ERROR;
        }
        ngx_http_prefix_cache_affinity_shm_zone->init = 
            ngx_http_prefix_cache_affinity_init_shm_zone;
        ngx_http_prefix_cache_affinity_shm_zone->tag = uscfp[i];

        uscfp[i]->peer.init = ngx_http_prefix_cache_affinity_upstream_init;

        ngx_log_error(NGX_LOG_WARN, cf->log, 0,
                      "[prefix_cache_affinity] \"%V\" peers=%ui "
                      "least_total_load_batch_size=%ui "
                      "prefix_cache_affinity_dict_size=%ui "
                      "prefix_cache_affinity_block_size=%ui "
                      "prefix_cache_affinity_shm_size=%z",
                      &uscfp[i]->host,
                      peer_count,
                      ngx_http_least_total_load_batch_size,
                      ngx_http_prefix_cache_affinity_dict_size,
                      ngx_http_prefix_cache_affinity_block_size,
                      ngx_http_prefix_cache_affinity_shm_size);
    }

    return NGX_OK;
}