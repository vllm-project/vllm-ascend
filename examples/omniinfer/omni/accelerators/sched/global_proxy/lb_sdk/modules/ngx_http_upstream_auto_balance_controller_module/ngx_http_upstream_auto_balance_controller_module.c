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
    ngx_uint_t auto_balance_controller_batch_size;
} ngx_http_auto_balance_controller_conf_t;

typedef struct {
    ngx_atomic_t active_requests;
    ngx_atomic_t total_request_length;
} ngx_http_auto_balance_controller_shm_peer_t;

typedef struct {
    ngx_http_upstream_rr_peer_data_t *rrp;
    ngx_uint_t chosen;
    ngx_uint_t request_length;
} ngx_http_auto_balance_controller_peer_data_t;

typedef struct {
    ngx_uint_t peer_count;
    ngx_http_auto_balance_controller_shm_peer_t peers[1];
} ngx_http_auto_balance_controller_shm_block_t;

static ngx_shm_zone_t *ngx_http_auto_balance_controller_shm_zone = NULL;
static ngx_uint_t ngx_http_auto_balance_controller_shm_size = 0;
static ngx_uint_t ngx_http_auto_balance_controller_batch_size = 24;
static ngx_http_auto_balance_controller_shm_block_t
    *auto_balance_controller_shm = NULL;

static void *ngx_http_auto_balance_controller_create_srv_conf(ngx_conf_t *cf);

static ngx_int_t ngx_http_auto_balance_controller_postconfig(ngx_conf_t *cf);
static ngx_int_t ngx_http_auto_balance_controller_upstream_init(
    ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *uscf);
static ngx_int_t
ngx_http_auto_balance_controller_get_peer(ngx_peer_connection_t *pc,
                                          void *data);
static void
ngx_http_auto_balance_controller_free_peer(ngx_peer_connection_t *pc,
                                           void *data, ngx_uint_t state);
static ngx_int_t
ngx_http_auto_balance_controller_init_shm_zone(ngx_shm_zone_t *shm_zone,
                                               void *data);

static ngx_command_t ngx_http_auto_balance_controller_commands[] = {
    {ngx_string("auto_balance_controller"),
     NGX_HTTP_UPS_CONF | NGX_CONF_FLAG,
     ngx_conf_set_flag_slot, NGX_HTTP_SRV_CONF_OFFSET,
     offsetof(ngx_http_auto_balance_controller_conf_t, enable), 
     NULL},

    {ngx_string("auto_balance_controller_batch_size"),
     NGX_HTTP_UPS_CONF | NGX_CONF_TAKE1,
     ngx_conf_set_num_slot,
     NGX_HTTP_SRV_CONF_OFFSET,
     offsetof(ngx_http_auto_balance_controller_conf_t,
              auto_balance_controller_batch_size),
     NULL},
    ngx_null_command};

static ngx_http_module_t ngx_http_auto_balance_controller_module_ctx = {
    NULL,
    ngx_http_auto_balance_controller_postconfig,
    NULL,
    NULL,
    ngx_http_auto_balance_controller_create_srv_conf,
    NULL,
    NULL,
    NULL};

ngx_module_t ngx_http_upstream_auto_balance_controller_module = {
    NGX_MODULE_V1,
    &ngx_http_auto_balance_controller_module_ctx,
    ngx_http_auto_balance_controller_commands,
    NGX_HTTP_MODULE,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NGX_MODULE_V1_PADDING};

static void *ngx_http_auto_balance_controller_create_srv_conf(ngx_conf_t *cf)
{
    ngx_http_auto_balance_controller_conf_t *conf =
        ngx_pcalloc(cf->pool, sizeof(*conf));
    if (conf == NULL) {
        return NULL;
    }
    conf->enable = NGX_CONF_UNSET;
    conf->auto_balance_controller_batch_size = NGX_CONF_UNSET_UINT;
    return conf;
}

static ngx_int_t
ngx_http_auto_balance_controller_init_shm_zone(ngx_shm_zone_t *shm_zone,
                                               void *data)
{
    ngx_slab_pool_t *shpool;
    ngx_http_auto_balance_controller_shm_block_t *shm_block;
    ngx_uint_t i, n;

    if (data) {
        shm_zone->data = data;
        auto_balance_controller_shm = data;
        return NGX_OK;
    }

    shpool = (ngx_slab_pool_t *)shm_zone->shm.addr;

    n = 512;
    size_t sz = sizeof(ngx_http_auto_balance_controller_shm_block_t) +
                (n - 1) * sizeof(ngx_http_auto_balance_controller_shm_peer_t);
    shm_block = ngx_slab_alloc(shpool, sz);
    if (!shm_block) {
        return NGX_ERROR;
    }

    shm_block->peer_count = n;
    for (i = 0; i < n; i++) {
        shm_block->peers[i].active_requests = 0;
        shm_block->peers[i].total_request_length = 0;
    }
    shm_zone->data = shm_block;
    auto_balance_controller_shm = shm_block;
    return NGX_OK;
}

static ngx_int_t ngx_http_auto_balance_controller_postconfig(ngx_conf_t *cf)
{
    if (ngx_http_auto_balance_controller_batch_size == 0) {
        ngx_http_auto_balance_controller_batch_size = 24;
    }
    if (ngx_http_auto_balance_controller_shm_size == 0) {
        ngx_http_auto_balance_controller_shm_size = 8 * ngx_pagesize;
    }
    ngx_str_t *shm_name = ngx_palloc(cf->pool, sizeof(*shm_name));
    if (shm_name == NULL) {
        return NGX_ERROR;
    }
    shm_name->len = sizeof("auto_balance_controller") - 1;
    shm_name->data = (u_char *)"auto_balance_controller";
    ngx_http_auto_balance_controller_shm_zone = ngx_shared_memory_add(
        cf, shm_name, ngx_http_auto_balance_controller_shm_size,
        &ngx_http_upstream_auto_balance_controller_module);
    if (ngx_http_auto_balance_controller_shm_zone == NULL) {
        ngx_pfree(cf->pool, shm_name);
        return NGX_ERROR;
    }
    ngx_http_auto_balance_controller_shm_zone->init =
        ngx_http_auto_balance_controller_init_shm_zone;

    ngx_http_upstream_main_conf_t *upcf;
    ngx_http_upstream_srv_conf_t **uscfp;
    ngx_http_auto_balance_controller_conf_t *conf;
    ngx_uint_t i;

    upcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_upstream_module);
    if (upcf == NULL) {
        return NGX_OK;
    }
    uscfp = upcf->upstreams.elts;
    for (i = 0; i < upcf->upstreams.nelts; i++) {
        conf = ngx_http_conf_upstream_srv_conf(
            uscfp[i], ngx_http_upstream_auto_balance_controller_module);
        if (conf->enable == 1) {
            uscfp[i]->peer.init =
                ngx_http_auto_balance_controller_upstream_init;
            if (conf->auto_balance_controller_batch_size !=
                NGX_CONF_UNSET_UINT) {
                ngx_http_auto_balance_controller_batch_size =
                    conf->auto_balance_controller_batch_size;
            }
        }
    }
    return NGX_OK;
}

static ngx_int_t ngx_http_auto_balance_controller_upstream_init(
    ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *uscf)
{
    ngx_http_upstream_t *u = r->upstream;
    ngx_http_upstream_rr_peer_data_t *rrp;
    ngx_http_auto_balance_controller_peer_data_t *pdata;
    ngx_uint_t chosen = 0;
    ngx_uint_t i;
    ngx_uint_t n;

    if (ngx_http_upstream_init_round_robin_peer(r, uscf) != NGX_OK) {
        return NGX_ERROR;
    }
    rrp = u->peer.data;

    if (auto_balance_controller_shm == NULL) {
        auto_balance_controller_shm =
            ngx_http_auto_balance_controller_shm_zone->data;
    }

    n = rrp->peers->number;
    if (n > auto_balance_controller_shm->peer_count) {
        n = auto_balance_controller_shm->peer_count;
    }

    if (n == 0) {
        return NGX_ERROR;
    }

    ngx_uint_t active_requests[n];
    ngx_uint_t total_request_length[n];
    ngx_slab_pool_t *shpool;
    shpool =
        (ngx_slab_pool_t *)ngx_http_auto_balance_controller_shm_zone->shm.addr;
    ngx_shmtx_lock(&shpool->mutex);
    for (i = 0; i < n; i++) {
        active_requests[i] = ngx_atomic_fetch_add(
            &auto_balance_controller_shm->peers[i].active_requests, 0);
        total_request_length[i] = ngx_atomic_fetch_add(
            &auto_balance_controller_shm->peers[i].total_request_length, 0);
    }
    ngx_shmtx_unlock(&shpool->mutex);
    ngx_uint_t all_overload = 1;
    ngx_uint_t candidate[n];
    ngx_uint_t candidate_count = 0;
    for (i = 0; i < n; i++) {
        if (active_requests[i] < ngx_http_auto_balance_controller_batch_size) {
            all_overload = 0;
            break;
        }
    }
    if (all_overload) {
        ngx_uint_t min_active_request = 0;
        for (i = 0; i < n; i++) {
            if (candidate_count == 0 ||
                active_requests[i] < min_active_request) {
                candidate_count = 0;
                min_active_request = active_requests[i];
                candidate[candidate_count++] = i;
            } else if (active_requests[i] == min_active_request) {
                candidate[candidate_count++] = i;
            }
        }
    } else {
        ngx_uint_t min_total_request_length = 0;
        for (i = 0; i < n; i++) {
            if (active_requests[i] >=
                ngx_http_auto_balance_controller_batch_size) {
                continue;
            }
            if (candidate_count == 0 ||
                total_request_length[i] < min_total_request_length) {
                candidate_count = 0;
                min_total_request_length = total_request_length[i];
                candidate[candidate_count++] = i;
            } else if (total_request_length[i] == min_total_request_length) {
                candidate[candidate_count++] = i;
            }
        }
    }
    ngx_uint_t rand_idx = ngx_random() % candidate_count;
    chosen = candidate[rand_idx];

    ngx_atomic_fetch_add(
        &auto_balance_controller_shm->peers[chosen].active_requests, 1);
    ngx_atomic_fetch_add(
        &auto_balance_controller_shm->peers[chosen].total_request_length,
        (ngx_atomic_int_t)r->request_length);

    pdata = ngx_pcalloc(r->pool, sizeof(*pdata));
    if (pdata == NULL) {
        return NGX_ERROR;
    }
    pdata->rrp = rrp;
    pdata->chosen = chosen;
    pdata->request_length = (ngx_uint_t)r->request_length;
    u->peer.data = pdata;
    u->peer.get = ngx_http_auto_balance_controller_get_peer;
    u->peer.free = ngx_http_auto_balance_controller_free_peer;

    ngx_log_error(
        NGX_LOG_WARN, r->connection->log, 0,
        "[auto_balance_controller] assign request(len=%ui) to peer #%ui",
        pdata->request_length, pdata->chosen);

    return NGX_OK;
}

static ngx_int_t
ngx_http_auto_balance_controller_get_peer(ngx_peer_connection_t *pc, void *data)
{
    ngx_http_auto_balance_controller_peer_data_t *pdata = data;
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
ngx_http_auto_balance_controller_free_peer(ngx_peer_connection_t *pc,
                                           void *data, ngx_uint_t state)
{
    ngx_http_auto_balance_controller_peer_data_t *pdata = data;
    if (auto_balance_controller_shm == NULL) {
        return;
    }
    ngx_slab_pool_t *shpool;
    shpool =
        (ngx_slab_pool_t *)ngx_http_auto_balance_controller_shm_zone->shm.addr;
    ngx_shmtx_lock(&shpool->mutex);
    ngx_atomic_fetch_add(
        &auto_balance_controller_shm->peers[pdata->chosen].active_requests,
        (ngx_atomic_int_t)-1);
    ngx_atomic_fetch_add(
        &auto_balance_controller_shm->peers[pdata->chosen].total_request_length,
        (ngx_atomic_int_t) - (pdata->request_length));
    ngx_shmtx_unlock(&shpool->mutex);

    ngx_http_upstream_rr_peer_data_t *rrp = pdata->rrp;
    ngx_http_upstream_free_round_robin_peer(pc, rrp, state);
}