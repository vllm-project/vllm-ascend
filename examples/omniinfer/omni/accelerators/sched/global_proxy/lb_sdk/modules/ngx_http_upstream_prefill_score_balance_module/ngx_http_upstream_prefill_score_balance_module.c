// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <float.h>
#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <ngx_http_upstream.h>
#include <ngx_atomic.h>
#include <stdlib.h>

typedef struct {
    ngx_flag_t  enable;
} ngx_http_prefill_score_conf_t;

typedef struct {
    ngx_atomic_t active_requests;
    ngx_atomic_t total_time_cost;
} ngx_http_prefill_score_shm_peer_t;

typedef struct {
    ngx_uint_t peer_count;
    ngx_http_prefill_score_shm_peer_t peers[1];
} ngx_http_prefill_score_shm_block_t;

typedef struct {
    ngx_http_upstream_rr_peer_data_t *rrp;
    ngx_uint_t chosen;
    double my_time_cost;
} ngx_http_prefill_score_peer_data_t;

static ngx_shm_zone_t *ngx_http_prefill_score_shm_zone = NULL;
static ngx_uint_t ngx_http_prefill_score_shm_size = 0;
static ngx_http_prefill_score_shm_block_t *pfs_shm = NULL;

static void *ngx_http_prefill_score_create_srv_conf(ngx_conf_t *cf);
static char *ngx_http_prefill_score_merge_srv_conf(ngx_conf_t *cf, void *parent, void *child);

static ngx_int_t ngx_http_prefill_score_postconfig(ngx_conf_t *cf);
static ngx_int_t ngx_http_prefill_score_upstream_init(ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *uscf);
static ngx_int_t ngx_http_prefill_score_get_peer(ngx_peer_connection_t *pc, void *data);
static void ngx_http_prefill_score_free_peer(ngx_peer_connection_t *pc, void *data, ngx_uint_t state);
static ngx_int_t ngx_http_prefill_score_init_shm_zone(ngx_shm_zone_t *shm_zone, void *data);

static char *ngx_http_prefill_score_set_flag(ngx_conf_t *cf, ngx_command_t *cmd, void *conf) {
    ngx_str_t *value = cf->args->elts;
    ngx_flag_t *fp = (ngx_flag_t *)((char *)conf + cmd->offset);
    *fp = ngx_atoi(value[1].data, value[1].len);
    return NGX_CONF_OK;
}

static ngx_command_t ngx_http_upstream_prefill_score_commands[] = {
    { ngx_string("prefill_score_balance"),
      NGX_HTTP_UPS_CONF | NGX_CONF_FLAG,
      ngx_http_prefill_score_set_flag,
      NGX_HTTP_SRV_CONF_OFFSET,
      offsetof(ngx_http_prefill_score_conf_t, enable),
      NULL },

    { ngx_string("prefill_score_shm_size"),
      NGX_HTTP_MAIN_CONF | NGX_CONF_TAKE1,
      ngx_conf_set_size_slot,
      0,
      0,
      &ngx_http_prefill_score_shm_size },

    ngx_null_command
};

static ngx_http_module_t ngx_http_upstream_prefill_score_module_ctx = {
    NULL,
    ngx_http_prefill_score_postconfig,
    NULL, NULL,
    ngx_http_prefill_score_create_srv_conf,
    ngx_http_prefill_score_merge_srv_conf,
    NULL, NULL
};

ngx_module_t ngx_http_upstream_prefill_score_balance_module = {
    NGX_MODULE_V1,
    &ngx_http_upstream_prefill_score_module_ctx,
    ngx_http_upstream_prefill_score_commands,
    NGX_HTTP_MODULE,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NGX_MODULE_V1_PADDING
};

static void *
ngx_http_prefill_score_create_srv_conf(ngx_conf_t *cf)
{
    ngx_http_prefill_score_conf_t *conf = ngx_pcalloc(cf->pool, sizeof(*conf));
    if (conf == NULL) {
        return NULL;
    }
    conf->enable = 0;
    return conf;
}

static char* ngx_http_prefill_score_merge_srv_conf(ngx_conf_t* cf, void* parent, void* child) {
    ngx_http_prefill_score_conf_t* prev = parent;
    ngx_http_prefill_score_conf_t* conf = child;
    ngx_conf_merge_value(conf->enable, prev->enable, 0);
    return NGX_CONF_OK;
}

static ngx_int_t
ngx_http_prefill_score_init_shm_zone(ngx_shm_zone_t *shm_zone, void *data)
{
    ngx_slab_pool_t *shpool;
    ngx_http_prefill_score_shm_block_t *shm_block;
    ngx_uint_t i;
    ngx_uint_t n;

    if (data) {
        shm_zone->data = data;
        pfs_shm = data;
        return NGX_OK;
    }

    shpool = (ngx_slab_pool_t *)shm_zone->shm.addr;

    n = 512;
    size_t sz = sizeof(ngx_http_prefill_score_shm_block_t) + (n - 1) * sizeof(ngx_http_prefill_score_shm_peer_t);
    shm_block = ngx_slab_alloc(shpool, sz);
    if (!shm_block) {
        return NGX_ERROR;
    }

    shm_block->peer_count = n;
    for (i = 0; i < n; i++) {
        shm_block->peers[i].active_requests = 0;
        shm_block->peers[i].total_time_cost = 0;
    }
    shm_zone->data = shm_block;
    pfs_shm = shm_block;
    return NGX_OK;
}

static double
ngx_http_prefill_score_time_cost(ngx_uint_t request_length)
{
    double l = (double)request_length / 4.;
    return l * 0.0345 + 120.0745;
}

static ngx_int_t
ngx_http_prefill_score_postconfig(ngx_conf_t *cf)
{
    ngx_http_upstream_main_conf_t *upcf;
    ngx_http_upstream_srv_conf_t **uscfp;
    ngx_http_prefill_score_conf_t *conf;
    ngx_uint_t i;

    upcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_upstream_module);
    if (upcf == NULL) {
        return NGX_OK;
    }

    uscfp = upcf->upstreams.elts;

    for (i = 0; i < upcf->upstreams.nelts; i++) {
        conf = ngx_http_conf_upstream_srv_conf(uscfp[i],
                                               ngx_http_upstream_prefill_score_balance_module);
        if (!conf->enable) {
            continue;
        }

        if (ngx_http_prefill_score_shm_zone == NULL) {
            if (ngx_http_prefill_score_shm_size == 0) {
                ngx_http_prefill_score_shm_size = 8 * ngx_pagesize;
            }

            ngx_str_t name = ngx_string("prefill_score_balance");
            ngx_http_prefill_score_shm_zone = ngx_shared_memory_add(
                cf, &name, ngx_http_prefill_score_shm_size,
                &ngx_http_upstream_prefill_score_balance_module);
            if (ngx_http_prefill_score_shm_zone == NULL) {
                return NGX_ERROR;
            }
            ngx_http_prefill_score_shm_zone->init = ngx_http_prefill_score_init_shm_zone;
        }

        uscfp[i]->peer.init = ngx_http_prefill_score_upstream_init;
        ngx_log_error(NGX_LOG_WARN, cf->log, 0,
                      "[PrefillScoreBalance] enabled on upstream[%ui]", i);
    }

    return NGX_OK;
}

static ngx_int_t
ngx_http_prefill_score_upstream_init(ngx_http_request_t *r,
    ngx_http_upstream_srv_conf_t *uscf)
{
    ngx_http_upstream_t *u = r->upstream;
    ngx_http_upstream_rr_peer_data_t *rrp;
    ngx_http_prefill_score_peer_data_t *pdata;
    ngx_uint_t chosen = 0;
    ngx_uint_t i;
    ngx_uint_t n;
    double min_load;
    double my_time_cost;
    ngx_slab_pool_t *shpool;

    if (ngx_http_upstream_init_round_robin_peer(r, uscf) != NGX_OK) {
        return NGX_ERROR;
    }
    rrp = u->peer.data;

    if (pfs_shm == NULL) {
        pfs_shm = ngx_http_prefill_score_shm_zone->data;
    }

    shpool = (ngx_slab_pool_t *)ngx_http_prefill_score_shm_zone->shm.addr;
    n = rrp->peers->number;
    if (n > pfs_shm->peer_count) {
        n = pfs_shm->peer_count;
    }

    my_time_cost = ngx_http_prefill_score_time_cost((ngx_uint_t)r->request_length);

    ngx_shmtx_lock(&shpool->mutex);

    min_load = DBL_MAX;
    chosen = 0;

    for (i = 0; i < n; i++) {
        double peer_load = ((double)pfs_shm->peers[i].total_time_cost + my_time_cost) * ((double)pfs_shm->peers[i].active_requests + 1)
            - ((double)pfs_shm->peers[i].total_time_cost) * ((double)pfs_shm->peers[i].active_requests);

        ngx_log_error(NGX_LOG_WARN, r->connection->log, 0,
            "[PrefillScoreBalance] peer=%ui, active_requests=%uA, total_time_cost=%f, after_add=%f",
            i,
            pfs_shm->peers[i].active_requests,
            (double)pfs_shm->peers[i].total_time_cost,
            peer_load);

        if (peer_load < min_load) {
            min_load = peer_load;
            chosen = i;
        }
    }

    ngx_atomic_fetch_add(&pfs_shm->peers[chosen].active_requests, 1);
    ngx_atomic_fetch_add(&pfs_shm->peers[chosen].total_time_cost, (ngx_atomic_int_t)my_time_cost);

    ngx_log_error(NGX_LOG_WARN, r->connection->log, 0,
        "[PrefillScoreBalance] request assigned to peer=%ui, min_load=%f, request_time_cost=%.6f, request_length=%ui",
        chosen, min_load, my_time_cost, (ngx_uint_t)r->request_length);

    ngx_shmtx_unlock(&shpool->mutex);

    pdata = ngx_pcalloc(r->pool, sizeof(*pdata));
    pdata->rrp = rrp;
    pdata->chosen = chosen;
    pdata->my_time_cost = my_time_cost;
    u->peer.data = pdata;
    u->peer.get = ngx_http_prefill_score_get_peer;
    u->peer.free = ngx_http_prefill_score_free_peer;

    return NGX_OK;
}

static ngx_int_t
ngx_http_prefill_score_get_peer(ngx_peer_connection_t *pc, void *data)
{
    ngx_http_prefill_score_peer_data_t *pdata = data;
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
    pc->socklen  = peers->peer[idx].socklen;
    pc->name     = &peers->peer[idx].name;
    rrp->current = &peers->peer[idx];
    return NGX_OK;
}

static void
ngx_http_prefill_score_free_peer(ngx_peer_connection_t *pc, void *data, ngx_uint_t state)
{
    ngx_http_prefill_score_peer_data_t *pdata = data;
    ngx_slab_pool_t *shpool;
    if (pfs_shm == NULL)
        return;
    shpool = (ngx_slab_pool_t *)ngx_http_prefill_score_shm_zone->shm.addr;
    ngx_shmtx_lock(&shpool->mutex);
    ngx_atomic_fetch_add(&pfs_shm->peers[pdata->chosen].active_requests, (ngx_atomic_int_t)-1);
    ngx_atomic_fetch_add(&pfs_shm->peers[pdata->chosen].total_time_cost, (ngx_atomic_int_t)-(pdata->my_time_cost));
    ngx_shmtx_unlock(&shpool->mutex);

    ngx_http_upstream_rr_peer_data_t *rrp = pdata->rrp;
    ngx_http_upstream_free_round_robin_peer(pc, rrp, state);
}