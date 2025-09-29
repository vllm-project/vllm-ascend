// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.


#include <float.h>
#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <ngx_http_upstream.h>
#include <ngx_atomic.h>
#include <stdlib.h>
#include <math.h>


typedef struct {
    ngx_flag_t enable;
    ngx_uint_t batch_size;
} ngx_http_least_total_load_conf_t;

typedef struct {
    ngx_atomic_t total_length_sum;
    ngx_atomic_t total_request_sum;
} ngx_http_least_total_load_shm_peer_t;

typedef struct {
    ngx_uint_t peer_count;
    ngx_http_least_total_load_shm_peer_t peers[1];
} prefill_upstream_info_t;

typedef struct {
    ngx_http_upstream_rr_peer_data_t *rrp;
    ngx_uint_t chosen;
    ngx_uint_t request_length;
} ngx_http_least_total_load_peer_data_t;

#define DEFAULT_least_total_load_BATCH_SIZE 16
static ngx_shm_zone_t *ngx_http_least_total_load_shm_zone = NULL;
static prefill_upstream_info_t *prefill_shm = NULL;
static ngx_uint_t ngx_http_least_total_load_shm_size = 0;
static ngx_uint_t ngx_http_least_total_load_batch_size = 0;

static ngx_int_t ngx_http_least_total_load_get_peer(ngx_peer_connection_t *pc,
                                                    void *data);
static void ngx_http_least_total_load_free_peer(ngx_peer_connection_t *pc,
                                                void *data, ngx_uint_t state);

static void *ngx_http_least_total_load_create_srv_conf(ngx_conf_t *cf)
{
    ngx_http_least_total_load_conf_t *conf =
        ngx_pcalloc(cf->pool, sizeof(*conf));
    if (conf == NULL) {
        return NULL;
    }
    conf->enable = NGX_CONF_UNSET;
    conf->batch_size = NGX_CONF_UNSET_UINT;
    return conf;
}

static ngx_int_t
ngx_http_least_total_load_init_shm_zone(ngx_shm_zone_t *shm_zone, void *data)
{
    ngx_slab_pool_t *shpool;
    prefill_upstream_info_t *shm_block;
    ngx_uint_t i, n;

    if (data) {
        shm_zone->data = data;
        prefill_shm = data;
        return NGX_OK;
    }

    shpool = (ngx_slab_pool_t *)shm_zone->shm.addr;

    n = 512;
    size_t sz = sizeof(prefill_upstream_info_t) +
                (n - 1) * sizeof(ngx_http_least_total_load_shm_peer_t);
    shm_block = ngx_slab_alloc(shpool, sz);
    if (shm_block == NULL) {
        return NGX_ERROR;
    }
    shm_block->peer_count = n;
    for (i = 0; i < n; i++) {
        shm_block->peers[i].total_length_sum = 0;
        shm_block->peers[i].total_request_sum = 0;
    }
    shm_zone->data = shm_block;
    prefill_shm = shm_block;
    return NGX_OK;
}

static ngx_int_t ngx_http_least_total_load_postconfig(ngx_conf_t *cf);

static ngx_command_t ngx_http_upstream_least_total_load_commands[] = {
    {ngx_string("least_total_load"),
     NGX_HTTP_UPS_CONF | NGX_CONF_FLAG,
     ngx_conf_set_flag_slot,
     NGX_HTTP_SRV_CONF_OFFSET,
     offsetof(ngx_http_least_total_load_conf_t, enable),
     NULL},
    {ngx_string("least_total_load_batch_size"),
     NGX_HTTP_UPS_CONF | NGX_CONF_TAKE1,
     ngx_conf_set_num_slot,
     NGX_HTTP_SRV_CONF_OFFSET,
     offsetof(ngx_http_least_total_load_conf_t, batch_size),
     NULL},
     ngx_null_command};

static ngx_http_module_t ngx_http_upstream_least_total_load_module_ctx = {
    NULL,
    ngx_http_least_total_load_postconfig,
    NULL,
    NULL,
    ngx_http_least_total_load_create_srv_conf,
    NULL,
    NULL,
    NULL};

ngx_module_t ngx_http_upstream_least_total_load_module = {
    NGX_MODULE_V1,
    &ngx_http_upstream_least_total_load_module_ctx,
    ngx_http_upstream_least_total_load_commands,
    NGX_HTTP_MODULE,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NGX_MODULE_V1_PADDING};

static ngx_int_t
least_total_load_select_solver(prefill_upstream_info_t *prefill_shm,
                               ngx_uint_t worker_num, ngx_uint_t req_length,
                               ngx_uint_t *chosen)
{
    ngx_uint_t count = 0;
    double min_score = 0;
    const ngx_uint_t max_tie = worker_num;
    ngx_uint_t min_peers[max_tie];
    double least_total_load_batch_size =
        (double)ngx_http_least_total_load_batch_size;

    for (ngx_uint_t i = 0; i < worker_num; ++i) {
        ngx_uint_t length_sum_workers =
            ngx_atomic_fetch_add(&(prefill_shm->peers[i].total_length_sum), 0);
        ngx_uint_t request_sum_workers =
            ngx_atomic_fetch_add(&(prefill_shm->peers[i].total_request_sum), 0);
        double score =
            length_sum_workers *
            ((ceil(request_sum_workers / least_total_load_batch_size) + 1.0) /
             2.0);
        if (count == 0 || score < min_score) {
            min_score = score;
            count = 0;
            min_peers[count++] = i;
        } else if (score == min_score) {
            if (count < max_tie) {
                min_peers[count++] = i;
            }
        }
    }
    ngx_uint_t rand_idx = ngx_random() % count;
    *chosen = min_peers[rand_idx];
    return NGX_OK;
}

static ngx_int_t
ngx_http_least_total_load_upstream_init(ngx_http_request_t *r,
                                        ngx_http_upstream_srv_conf_t *uscf)
{
    ngx_http_upstream_t *u = r->upstream;
    ngx_http_upstream_rr_peer_data_t *rrp;
    ngx_http_least_total_load_peer_data_t *pdata;
    ngx_uint_t chosen = 0, n;
    ngx_slab_pool_t *shpool;

    if (ngx_http_upstream_init_round_robin_peer(r, uscf) != NGX_OK) {
        return NGX_ERROR;
    }
    rrp = u->peer.data;

    if (prefill_shm == NULL) {
        prefill_shm = ngx_http_least_total_load_shm_zone->data;
    }

    shpool = (ngx_slab_pool_t *)ngx_http_least_total_load_shm_zone->shm.addr;
    n = rrp->peers->number;
    if (n > prefill_shm->peer_count) {
        n = prefill_shm->peer_count;
    }

    if (n == 0) {
        return NGX_ERROR;
    }

    least_total_load_select_solver(prefill_shm, n,
                                   (ngx_uint_t)r->request_length, &chosen);

    ngx_shmtx_lock(&shpool->mutex);
    ngx_atomic_fetch_add(&(prefill_shm->peers[chosen].total_length_sum),
                         (ngx_atomic_int_t)r->request_length);
    ngx_atomic_fetch_add(&(prefill_shm->peers[chosen].total_request_sum),
                         (ngx_atomic_int_t)1);
    ngx_shmtx_unlock(&shpool->mutex);

    ngx_log_error(
        NGX_LOG_WARN, r->connection->log, 0,
        "[least_total_load] request assigned to peer=%ui, request_length=%ui",
        chosen, (ngx_uint_t)r->request_length);

    pdata = ngx_pcalloc(r->pool, sizeof(*pdata));
    if (pdata == NULL) {
        return NGX_ERROR;
    }
    pdata->rrp = rrp;
    pdata->chosen = chosen;
    pdata->request_length = (ngx_uint_t)r->request_length;
    u->peer.data = pdata;
    u->peer.get = ngx_http_least_total_load_get_peer;
    u->peer.free = ngx_http_least_total_load_free_peer;

    return NGX_OK;
}

static ngx_int_t ngx_http_least_total_load_get_peer(ngx_peer_connection_t *pc,
                                                    void *data)
{
    ngx_http_least_total_load_peer_data_t *pdata = data;
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

static void ngx_http_least_total_load_free_peer(ngx_peer_connection_t *pc,
                                                void *data, ngx_uint_t state)
{
    ngx_http_least_total_load_peer_data_t *pdata = data;

    if (prefill_shm != NULL && pdata->chosen < pdata->rrp->peers->number) {
        ngx_slab_pool_t *shpool;
        shpool =
            (ngx_slab_pool_t *)ngx_http_least_total_load_shm_zone->shm.addr;
        ngx_shmtx_lock(&shpool->mutex);
        ngx_atomic_fetch_add(
            &prefill_shm->peers[pdata->chosen].total_request_sum,
            (ngx_atomic_int_t)-1);
        ngx_atomic_fetch_add(
            &prefill_shm->peers[pdata->chosen].total_length_sum,
            (ngx_atomic_int_t) - (pdata->request_length));
        ngx_shmtx_unlock(&shpool->mutex);
        pdata->chosen = pdata->rrp->peers->number;
    }

    ngx_http_upstream_rr_peer_data_t *rrp = pdata->rrp;
    ngx_http_upstream_free_round_robin_peer(pc, rrp, state);
}

static ngx_int_t ngx_http_least_total_load_postconfig(ngx_conf_t *cf)
{
    if (ngx_http_least_total_load_shm_size == 0) {
        ngx_http_least_total_load_shm_size = 8 * ngx_pagesize;
    }
    ngx_str_t *shm_name = ngx_palloc(cf->pool, sizeof(*shm_name));
    if (shm_name == NULL) {
        return NGX_ERROR;
    }
    shm_name->len = sizeof("upstream_prefill_least_total_load") - 1;
    shm_name->data = (u_char *)"upstream_prefill_least_total_load";
    ngx_http_least_total_load_shm_zone =
        ngx_shared_memory_add(cf, shm_name, ngx_http_least_total_load_shm_size,
                              &ngx_http_upstream_least_total_load_module);
    if (ngx_http_least_total_load_shm_zone == NULL) {
        ngx_pfree(cf->pool, shm_name);
        return NGX_ERROR;
    }
    ngx_http_least_total_load_shm_zone->init =
        ngx_http_least_total_load_init_shm_zone;

    ngx_http_upstream_main_conf_t *upcf;
    ngx_http_upstream_srv_conf_t **uscfp;
    ngx_http_least_total_load_conf_t *conf;
    ngx_uint_t i;

    upcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_upstream_module);
    if (upcf == NULL) {
        return NGX_OK;
    }
    uscfp = upcf->upstreams.elts;
    for (i = 0; i < upcf->upstreams.nelts; i++) {
        conf = ngx_http_conf_upstream_srv_conf(
            uscfp[i], ngx_http_upstream_least_total_load_module);
        if (conf->enable == 1) {
            uscfp[i]->peer.init = ngx_http_least_total_load_upstream_init;
            if (conf->batch_size == NGX_CONF_UNSET_UINT) {
                ngx_http_least_total_load_batch_size =
                    DEFAULT_least_total_load_BATCH_SIZE;
            } else {
                ngx_http_least_total_load_batch_size = conf->batch_size;
            }
        }
    }
    ngx_log_error(NGX_LOG_WARN, cf->log, 0,
                  "[least_total_load] least_total_load_batch_size=%ui\n",
                  ngx_http_least_total_load_batch_size);
    return NGX_OK;
}