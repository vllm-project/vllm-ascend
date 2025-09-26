// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <ngx_http_upstream.h>
#include <ngx_atomic.h>
#include <stdlib.h>

typedef struct {
    ngx_flag_t enable;
} ngx_http_weighted_least_active_conf_t;

typedef struct {
    ngx_atomic_t active_requests;
    ngx_atomic_t total_decode_num;
} ngx_http_weighted_least_active_shm_peer_t;

typedef struct {
    ngx_http_upstream_rr_peer_data_t *rrp;
    ngx_uint_t chosen;
    int first_chunk;
    ngx_atomic_t decode_token_count;
} ngx_http_weighted_least_active_peer_data_t;

typedef struct {
    ngx_uint_t peer_count;
    ngx_http_weighted_least_active_shm_peer_t peers[1];
} ngx_http_weighted_least_active_shm_block_t;

static ngx_shm_zone_t *ngx_http_weighted_least_active_shm_zone = NULL;
static ngx_uint_t ngx_http_weighted_least_active_shm_size = 0;
static ngx_http_weighted_least_active_shm_block_t *wla_shm = NULL;
static ngx_http_output_body_filter_pt ngx_http_next_body_filter = NULL;

static void *ngx_http_weighted_least_active_create_srv_conf(ngx_conf_t *cf);
static char *ngx_http_weighted_least_active_merge_srv_conf(ngx_conf_t *cf, void *parent, void *child);

static ngx_int_t ngx_http_weighted_least_active_postconfig(ngx_conf_t *cf);
static ngx_int_t ngx_http_weighted_least_active_upstream_init(ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *uscf);
static ngx_int_t ngx_http_weighted_least_active_get_peer(ngx_peer_connection_t *pc, void *data);
static void ngx_http_weighted_least_active_free_peer(ngx_peer_connection_t *pc, void *data, ngx_uint_t state);
static ngx_int_t ngx_http_weighted_least_active_init_shm_zone(ngx_shm_zone_t *shm_zone, void *data);
static ngx_int_t ngx_http_weighted_least_active_body_filter(ngx_http_request_t *r, ngx_chain_t *in);

void ngx_http_weighted_least_active_add_decoded_tokens(ngx_http_request_t *r, ngx_uint_t num_tokens) {
    ngx_http_weighted_least_active_peer_data_t *pdata = r->upstream ? r->upstream->peer.data : NULL;
    ngx_slab_pool_t *shpool;
    if (wla_shm == NULL || pdata == NULL) {
        return;
    }
    shpool = (ngx_slab_pool_t *) ngx_http_weighted_least_active_shm_zone->shm.addr;
    ngx_shmtx_lock(&shpool->mutex);

    ngx_atomic_fetch_add(&wla_shm->peers[pdata->chosen].total_decode_num, (ngx_atomic_int_t)num_tokens);
    ngx_atomic_fetch_add(&pdata->decode_token_count, (ngx_atomic_int_t)num_tokens);

    ngx_log_error(NGX_LOG_WARN, r->connection->log, 0,
        "[WeightedLeastActive-Add] peer=%ui request=%p decode_token +%ui, peer_total_decode_token=%uA, request_total_decode_token=%uA",
        pdata->chosen,
        r,
        num_tokens,
        wla_shm->peers[pdata->chosen].total_decode_num,
        pdata->decode_token_count);

    ngx_shmtx_unlock(&shpool->mutex);
}

static char *ngx_http_weighted_least_active_set_flag(ngx_conf_t *cf, ngx_command_t *cmd, void *conf) {
    ngx_str_t *value = cf->args->elts;
    ngx_flag_t *fp = (ngx_flag_t *)((char *)conf + cmd->offset);
    *fp = ngx_atoi(value[1].data, value[1].len);
    return NGX_CONF_OK;
}

static ngx_command_t ngx_http_weighted_least_active_commands[] = {
    { ngx_string("weighted_least_active"),
      NGX_HTTP_UPS_CONF | NGX_CONF_FLAG,
      ngx_http_weighted_least_active_set_flag,
      NGX_HTTP_SRV_CONF_OFFSET,
      offsetof(ngx_http_weighted_least_active_conf_t, enable),
      NULL },

    { ngx_string("weighted_least_active_shm_size"),
      NGX_HTTP_MAIN_CONF | NGX_CONF_TAKE1,
      ngx_conf_set_size_slot,
      0,
      0,
      &ngx_http_weighted_least_active_shm_size },

    ngx_null_command
};

static ngx_http_module_t ngx_http_weighted_least_active_module_ctx = {
    NULL,
    ngx_http_weighted_least_active_postconfig,
    NULL, NULL,
    ngx_http_weighted_least_active_create_srv_conf,
    ngx_http_weighted_least_active_merge_srv_conf,
    NULL, NULL
};

ngx_module_t ngx_http_upstream_weighted_least_active_module = {
    NGX_MODULE_V1,
    &ngx_http_weighted_least_active_module_ctx,
    ngx_http_weighted_least_active_commands,
    NGX_HTTP_MODULE,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NGX_MODULE_V1_PADDING
};

static void *
ngx_http_weighted_least_active_create_srv_conf(ngx_conf_t *cf)
{
    ngx_http_weighted_least_active_conf_t *conf = ngx_pcalloc(cf->pool, sizeof(*conf));
    if (conf == NULL) {
        return NULL;
    }
    conf->enable = 0;
    return conf;
}

static char *
ngx_http_weighted_least_active_merge_srv_conf(ngx_conf_t *cf, void *parent, void *child)
{
    ngx_http_weighted_least_active_conf_t *prev = parent;
    ngx_http_weighted_least_active_conf_t *conf = child;
    ngx_conf_merge_value(conf->enable, prev->enable, 0);
    return NGX_CONF_OK;
}

static ngx_int_t
ngx_http_weighted_least_active_init_shm_zone(ngx_shm_zone_t *shm_zone, void *data)
{
    ngx_slab_pool_t *shpool;
    ngx_http_weighted_least_active_shm_block_t *shm_block;
    ngx_uint_t i; 
    ngx_uint_t n;

    if (data) {
        shm_zone->data = data;
        wla_shm = data;
        return NGX_OK;
    }

    shpool = (ngx_slab_pool_t *) shm_zone->shm.addr;

    n = 512;
    size_t sz = sizeof(ngx_http_weighted_least_active_shm_block_t) + (n - 1) * sizeof(ngx_http_weighted_least_active_shm_peer_t);
    shm_block = ngx_slab_alloc(shpool, sz);
    if (!shm_block) {
        return NGX_ERROR;
    }

    shm_block->peer_count = n;
    for (i = 0; i < n; i++) {
        shm_block->peers[i].active_requests = 0;
        shm_block->peers[i].total_decode_num = 0;
    }
    shm_zone->data = shm_block;
    wla_shm = shm_block;
    return NGX_OK;
}

static ngx_int_t
ngx_http_weighted_least_active_body_filter(ngx_http_request_t *r, ngx_chain_t *in)
{
    ngx_chain_t *cl;
    ngx_uint_t num_tokens = 0;
    ngx_uint_t prompt_tokens = 0;
    ngx_uint_t output_tokens = 0;
    ngx_http_weighted_least_active_peer_data_t *pdata;

    if (r->upstream == NULL || r->upstream->peer.data == NULL) {
        return ngx_http_next_body_filter(r, in);
    }
    pdata = r->upstream->peer.data;
    if (pdata == NULL) {
        return ngx_http_next_body_filter(r, in);
    }

    for (cl = in; cl; cl = cl->next) {
        if (cl->buf->last > cl->buf->pos) {
            u_char *p = cl->buf->pos;
            u_char *last = cl->buf->last;

            static char key_output[] = "\"output_num_token\":";
            static char key_prompt[] = "\"prompt_num_token\":";
            size_t keylen_output = sizeof(key_output) - 1;
            size_t keylen_prompt = sizeof(key_prompt) - 1;

            u_char *found = ngx_strnstr(p, key_output, last - p);
            if (found) {
                u_char *num_start = found + keylen_output;
                while (num_start < last && (*num_start == ' ' || *num_start == '\"')) {
                    num_start++;
                }
                ngx_uint_t val = 0;
                while (num_start < last && *num_start >= '0' && *num_start <= '9') {
                    val = val * 10 + (*num_start - '0');
                    num_start++;
                }
                output_tokens = val;
            }

            if (!pdata->first_chunk) {
                found = ngx_strnstr(p, key_prompt, last - p);
                if (found) {
                    u_char *num_start = found + keylen_prompt;
                    while (num_start < last && (*num_start == ' ' || *num_start == '\"')) {
                        num_start++;
                    }
                    ngx_uint_t val = 0;
                    while (num_start < last && *num_start >= '0' && *num_start <= '9') {
                        val = val * 10 + (*num_start - '0');
                        num_start++;
                    }
                    prompt_tokens = val;
                }
            }
        }
    }

    int is_first = (pdata->first_chunk == 0);

    if (is_first) {
        num_tokens = prompt_tokens + output_tokens;
        pdata->first_chunk = 1;
    } else {
        num_tokens = output_tokens;
    }

    if (num_tokens > 0) {
        ngx_log_error(NGX_LOG_WARN, r->connection->log, 0,
            "[WeightedLeastActive-Filter] chunk tokens: %ui (prompt=%ui, output=%ui, first=%d)",
            num_tokens, prompt_tokens, output_tokens, is_first);
        ngx_http_weighted_least_active_add_decoded_tokens(r, num_tokens);
    }

    return ngx_http_next_body_filter(r, in);
}

static ngx_int_t
ngx_http_weighted_least_active_postconfig(ngx_conf_t *cf)
{
    ngx_http_upstream_main_conf_t *upcf;
    ngx_http_upstream_srv_conf_t **uscfp;
    ngx_http_weighted_least_active_conf_t *conf;
    ngx_uint_t i;

    upcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_upstream_module);
    if (upcf == NULL) {
        return NGX_OK;
    }

    uscfp = upcf->upstreams.elts;

    for (i = 0; i < upcf->upstreams.nelts; i++) {
        conf = ngx_http_conf_upstream_srv_conf(uscfp[i],
                                               ngx_http_upstream_weighted_least_active_module);
        if (!conf->enable) {
            continue;
        }

        if (ngx_http_weighted_least_active_shm_zone == NULL) {
            ngx_http_next_body_filter = ngx_http_top_body_filter;
            ngx_http_top_body_filter = ngx_http_weighted_least_active_body_filter;

            if (ngx_http_weighted_least_active_shm_size == 0) {
                ngx_http_weighted_least_active_shm_size = 8 * ngx_pagesize;
            }

            ngx_str_t name = ngx_string("weighted_least_active");
            ngx_http_weighted_least_active_shm_zone = ngx_shared_memory_add(
                cf, &name, ngx_http_weighted_least_active_shm_size,
                &ngx_http_upstream_weighted_least_active_module);
            if (ngx_http_weighted_least_active_shm_zone == NULL) {
                return NGX_ERROR;
            }
            ngx_http_weighted_least_active_shm_zone->init = ngx_http_weighted_least_active_init_shm_zone;
        }

        uscfp[i]->peer.init = ngx_http_weighted_least_active_upstream_init;
        ngx_log_error(NGX_LOG_WARN, cf->log, 0,
                      "[WeightedLeastActive] enabled on upstream[%ui]", i);
    }

    return NGX_OK;
}

static ngx_int_t
ngx_http_weighted_least_active_upstream_init(ngx_http_request_t *r,
    ngx_http_upstream_srv_conf_t *uscf)
{
    ngx_http_upstream_t *u = r->upstream;
    ngx_http_upstream_rr_peer_data_t *rrp;
    ngx_http_weighted_least_active_peer_data_t *pdata;
    ngx_uint_t chosen = 0;
    ngx_uint_t i;
    ngx_uint_t n;
    ngx_slab_pool_t *shpool;

    if (ngx_http_upstream_init_round_robin_peer(r, uscf) != NGX_OK) {
        return NGX_ERROR;
    }
    rrp = u->peer.data;

    if (wla_shm == NULL) {
        wla_shm = ngx_http_weighted_least_active_shm_zone->data;
    }

    shpool = (ngx_slab_pool_t *) ngx_http_weighted_least_active_shm_zone->shm.addr;
    n = rrp->peers->number;
    if (n > wla_shm->peer_count) {
        n = wla_shm->peer_count;
    }

    ngx_shmtx_lock(&shpool->mutex);

    ngx_uint_t min_active = wla_shm->peers[0].active_requests;
    for (i = 1; i < n; i++) {
        if (wla_shm->peers[i].active_requests < min_active) {
            min_active = wla_shm->peers[i].active_requests;
        }
    }

    ngx_uint_t candidate[n];
    ngx_uint_t candidate_count = 0;
    for (i = 0; i < n; i++) {
        if (wla_shm->peers[i].active_requests == min_active) {
            candidate[candidate_count++] = i;
        }
    }

    ngx_uint_t min_decode = wla_shm->peers[candidate[0]].total_decode_num;
    chosen = candidate[rand() % candidate_count];
    for (i = 1; i < candidate_count; i++) {
        ngx_uint_t idx = candidate[i];
        if (wla_shm->peers[idx].total_decode_num < min_decode) {
            min_decode = wla_shm->peers[idx].total_decode_num;
            chosen = idx;
        }
    }

    ngx_atomic_fetch_add(&wla_shm->peers[chosen].active_requests, 1);

    ngx_shmtx_unlock(&shpool->mutex);

    pdata = ngx_pcalloc(r->pool, sizeof(*pdata));
    pdata->rrp = rrp;
    pdata->chosen = chosen;
    pdata->decode_token_count = 0;
    u->peer.data = pdata;
    u->peer.get = ngx_http_weighted_least_active_get_peer;
    u->peer.free = ngx_http_weighted_least_active_free_peer;

    ngx_log_error(NGX_LOG_WARN, r->connection->log, 0,
        "[WeightedLeastActive] assign request to peer #%ui", pdata->chosen);

    return NGX_OK;
}

static ngx_int_t
ngx_http_weighted_least_active_get_peer(ngx_peer_connection_t *pc, void *data)
{
    ngx_http_weighted_least_active_peer_data_t *pdata = data;
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

static void
ngx_http_weighted_least_active_free_peer(ngx_peer_connection_t *pc, void *data, ngx_uint_t state)
{
    ngx_http_weighted_least_active_peer_data_t *pdata = data;
    ngx_slab_pool_t *shpool;
    if (wla_shm == NULL) {
        return;
    }
    shpool = (ngx_slab_pool_t *) ngx_http_weighted_least_active_shm_zone->shm.addr;
    ngx_shmtx_lock(&shpool->mutex);
    ngx_atomic_fetch_add(&wla_shm->peers[pdata->chosen].active_requests, (ngx_atomic_int_t)-1);
    ngx_atomic_fetch_add(&wla_shm->peers[pdata->chosen].total_decode_num, (ngx_atomic_int_t)-(pdata->decode_token_count));
    ngx_shmtx_unlock(&shpool->mutex);

    ngx_http_upstream_rr_peer_data_t *rrp = pdata->rrp;
    ngx_http_upstream_free_round_robin_peer(pc, rrp, state);
}
