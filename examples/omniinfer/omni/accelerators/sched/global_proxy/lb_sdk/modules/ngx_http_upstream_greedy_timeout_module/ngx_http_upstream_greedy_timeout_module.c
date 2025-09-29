// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <ngx_http_upstream.h>
#include <math.h>

typedef struct {
    ngx_flag_t enable;
    ngx_uint_t warmup;
    double exp;
} ngx_http_gts_conf_t;

typedef struct {
    double available_time;
} ngx_http_gts_shm_processor_t;

typedef struct {
    ngx_uint_t warmup;
    double exp;
    ngx_uint_t p;
    ngx_http_gts_shm_processor_t procs[1];
} ngx_http_gts_shm_block_t;

typedef struct {
    ngx_http_upstream_rr_peer_data_t *rrp;
    ngx_uint_t chosen;
} ngx_http_gts_peer_data_t;

static ngx_shm_zone_t *ngx_http_gts_shm_zone = NULL;
static ngx_uint_t ngx_http_gts_shm_size = 0;
static ngx_http_gts_shm_block_t *gts_shm = NULL;

static void *ngx_http_gts_create_srv_conf(ngx_conf_t *cf);
static char *ngx_http_gts_merge_srv_conf(ngx_conf_t *cf, void *parent, void *child);

static char *ngx_http_gts_set_shm_size(ngx_conf_t *cf, ngx_command_t *cmd, void *conf);
static char *ngx_conf_set_double_slot(ngx_conf_t *cf, ngx_command_t *cmd, void *conf);

static ngx_int_t ngx_http_gts_postconfig(ngx_conf_t *cf);
static ngx_int_t ngx_http_gts_upstream_init(ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *uscf);
static ngx_int_t ngx_http_gts_get_peer(ngx_peer_connection_t *pc, void *data);
static void ngx_http_gts_free_peer(ngx_peer_connection_t *pc, void *data, ngx_uint_t state);
static ngx_int_t ngx_http_gts_init_shm_zone(ngx_shm_zone_t *shm_zone, void *data);

static char *ngx_http_gts_set_flag(ngx_conf_t *cf, ngx_command_t *cmd, void *conf) {
    ngx_str_t *value = cf->args->elts;
    ngx_flag_t *fp = (ngx_flag_t *)((char *)conf + cmd->offset);
    *fp = ngx_atoi(value[1].data, value[1].len);
    return NGX_CONF_OK;
}

static ngx_command_t ngx_http_gts_commands[] = {
    { ngx_string("greedy_timeout"),
      NGX_HTTP_UPS_CONF|NGX_CONF_FLAG,
      ngx_http_gts_set_flag,
      NGX_HTTP_SRV_CONF_OFFSET,
      offsetof(ngx_http_gts_conf_t, enable),
      NULL },

    { ngx_string("greedy_timeout_warmup"),
      NGX_HTTP_UPS_CONF|NGX_CONF_TAKE1,
      ngx_conf_set_num_slot,
      NGX_HTTP_SRV_CONF_OFFSET,
      offsetof(ngx_http_gts_conf_t, warmup),
      NULL },

    { ngx_string("greedy_timeout_exp"),
      NGX_HTTP_UPS_CONF|NGX_CONF_TAKE1,
      ngx_conf_set_double_slot,
      NGX_HTTP_SRV_CONF_OFFSET,
      offsetof(ngx_http_gts_conf_t, exp),
      NULL },

    { ngx_string("greedy_timeout_shm_size"),
      NGX_HTTP_MAIN_CONF|NGX_CONF_TAKE1,
      ngx_http_gts_set_shm_size,
      0,
      0,
      NULL },

    ngx_null_command
};

static ngx_http_module_t ngx_http_gts_module_ctx = {
    NULL,
    ngx_http_gts_postconfig,
    NULL, NULL,
    ngx_http_gts_create_srv_conf,
    ngx_http_gts_merge_srv_conf,
    NULL, NULL
};

ngx_module_t ngx_http_upstream_greedy_timeout_module = {
    NGX_MODULE_V1,
    &ngx_http_gts_module_ctx,
    ngx_http_gts_commands,
    NGX_HTTP_MODULE,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NGX_MODULE_V1_PADDING
};

static char *
ngx_conf_set_double_slot(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_str_t *value = cf->args->elts;
    double *dp = (double *)((char *)conf + cmd->offset);

    *dp = atof((const char *)value[1].data);

    return NGX_CONF_OK;
}

static char *
ngx_http_gts_set_shm_size(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_str_t *value = cf->args->elts;
    ssize_t size = ngx_parse_size(&value[1]);
    if (size == NGX_ERROR) {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0, "Invalid shared memory size `%V`", &value[1]);
        return NGX_CONF_ERROR;
    }
    size = ngx_align(size, ngx_pagesize);
    if ((size_t)size < 8 * ngx_pagesize) {
        ngx_conf_log_error(NGX_LOG_WARN, cf, 0, "greedy_timeout_shm_size must be at least %udKiB", (8 * ngx_pagesize) >> 10);
        size = 8 * ngx_pagesize;
    }
    if (ngx_http_gts_shm_size && ngx_http_gts_shm_size != (ngx_uint_t) size) {
        ngx_conf_log_error(NGX_LOG_WARN, cf, 0, "Cannot change memory area size without restart, ignoring change");
    } else {
        ngx_http_gts_shm_size = size;
    }
    ngx_conf_log_error(NGX_LOG_DEBUG, cf, 0, "Using %udKiB shared memory for greedy_timeout", size >> 10);
    return NGX_CONF_OK;
}

static void *
ngx_http_gts_create_srv_conf(ngx_conf_t *cf)
{
    ngx_http_gts_conf_t *conf = ngx_pcalloc(cf->pool, sizeof(*conf));
    if (conf == NULL) {
        return NULL;
    }
    conf->enable  = 0;
    conf->warmup  = NGX_CONF_UNSET_UINT;
    conf->exp     = NGX_CONF_UNSET;
    return conf;
}

static char *
ngx_http_gts_merge_srv_conf(ngx_conf_t *cf, void *parent, void *child)
{
    ngx_http_gts_conf_t *prev = parent;
    ngx_http_gts_conf_t *conf = child;
    ngx_conf_merge_value(conf->enable,  prev->enable,  0);
    ngx_conf_merge_uint_value(conf->warmup, prev->warmup, 5);
    ngx_conf_merge_value(conf->exp, prev->exp, 1.8);
    return NGX_CONF_OK;
}

static ngx_int_t
ngx_http_gts_init_shm_zone(ngx_shm_zone_t *shm_zone, void *data)
{
    ngx_slab_pool_t *shpool;
    ngx_http_gts_shm_block_t *shm_block;
    ngx_uint_t i;
    ngx_uint_t n;

    if (data) {
        shm_zone->data = data;
        gts_shm = data;
        return NGX_OK;
    }

    shpool = (ngx_slab_pool_t *)shm_zone->shm.addr;

    n = 512;
    size_t sz = sizeof(ngx_http_gts_shm_block_t) + (n - 1) * sizeof(ngx_http_gts_shm_processor_t);
    shm_block = ngx_slab_alloc(shpool, sz);
    if (!shm_block) return NGX_ERROR;

    shm_block->warmup = 5;
    shm_block->exp = 1.8;
    shm_block->p = n;
    for (i = 0; i < n; i++) {
        shm_block->procs[i].available_time = 0;
    }
    shm_zone->data = shm_block;
    gts_shm = shm_block;
    return NGX_OK;
}

static ngx_int_t
ngx_http_gts_postconfig(ngx_conf_t *cf)
{
    ngx_http_upstream_main_conf_t *upcf;
    ngx_http_upstream_srv_conf_t **uscfp;
    ngx_http_gts_conf_t *conf;
    ngx_uint_t i;

    upcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_upstream_module);
    if (upcf == NULL) {
        return NGX_OK;
    }

    uscfp = upcf->upstreams.elts;

    for (i = 0; i < upcf->upstreams.nelts; i++) {
        conf = ngx_http_conf_upstream_srv_conf(uscfp[i],
                                               ngx_http_upstream_greedy_timeout_module);
        if (!conf->enable) {
            continue;
        }

        if (ngx_http_gts_shm_zone == NULL) {
            if (ngx_http_gts_shm_size == 0) {
                ngx_http_gts_shm_size = 8 * ngx_pagesize;
            }

            ngx_str_t name = ngx_string("greedy_timeout");
            ngx_http_gts_shm_zone = ngx_shared_memory_add(
                cf, &name, ngx_http_gts_shm_size,
                &ngx_http_upstream_greedy_timeout_module);
            if (ngx_http_gts_shm_zone == NULL) {
                return NGX_ERROR;
            }
            ngx_http_gts_shm_zone->init = ngx_http_gts_init_shm_zone;
        }

        uscfp[i]->peer.init = ngx_http_gts_upstream_init;
        ngx_log_error(NGX_LOG_WARN, cf->log, 0,
                      "[GreedyTimeout] enabled on upstream[%ui]", i);
    }

    return NGX_OK;
}

static ngx_int_t
ngx_http_gts_upstream_init(ngx_http_request_t *r,
    ngx_http_upstream_srv_conf_t *uscf)
{
    ngx_http_upstream_t *u = r->upstream;
    ngx_http_upstream_rr_peer_data_t *rrp;
    ngx_http_gts_peer_data_t *gdata;
    ngx_uint_t chosen = 0;
    ngx_uint_t i;
    ngx_uint_t n;
    double z;
    double cost;
    double now;
    double earliest;
    ngx_slab_pool_t *shpool;

    if (ngx_http_upstream_init_round_robin_peer(r, uscf) != NGX_OK)
        return NGX_ERROR;
    rrp = u->peer.data;

    if (gts_shm == NULL)
        gts_shm = ngx_http_gts_shm_zone->data;

    shpool = (ngx_slab_pool_t *)ngx_http_gts_shm_zone->shm.addr;
    n = rrp->peers->number;
    if (n > gts_shm->p) n = gts_shm->p;

    ngx_shmtx_lock(&shpool->mutex);

    z = (double) r->request_length;
    cost = gts_shm->warmup + pow(z, gts_shm->exp);
    earliest = gts_shm->procs[0].available_time;
    for (i = 1; i < n; i++) {
        if (gts_shm->procs[i].available_time < earliest) {
            earliest = gts_shm->procs[i].available_time;
            chosen   = i;
        }
    }
    now = ngx_current_msec / 1000.0;
    if (earliest < now)
        earliest = now;
    gts_shm->procs[chosen].available_time = earliest + cost;

    ngx_shmtx_unlock(&shpool->mutex);

    gdata = ngx_pcalloc(r->pool, sizeof(*gdata));
    gdata->rrp    = rrp;
    gdata->chosen = chosen;
    u->peer.data  = gdata;
    u->peer.get   = ngx_http_gts_get_peer;
    u->peer.free  = ngx_http_gts_free_peer;

    struct sockaddr_in *sin = (struct sockaddr_in *)rrp->peers->peer[chosen].sockaddr;
    ngx_uint_t port = ntohs(sin->sin_port);

    ngx_log_error(NGX_LOG_WARN, r->connection->log, 0,
        "[GreedyTimeout] assign request to peer #%ui (port=%ui), request_length=%O",
        chosen, port, r->request_length);

    return NGX_OK;
}

static ngx_int_t
ngx_http_gts_get_peer(ngx_peer_connection_t *pc, void *data)
{
    ngx_http_gts_peer_data_t *gdata = data;
    ngx_http_upstream_rr_peer_data_t *rrp = gdata->rrp;
    ngx_http_upstream_rr_peers_t *peers = rrp->peers;
    ngx_uint_t idx = gdata->chosen;

    if (idx >= peers->number)
        return ngx_http_upstream_get_round_robin_peer(pc, rrp);

    if (peers->peer[idx].down)
        return NGX_BUSY;

    pc->sockaddr = peers->peer[idx].sockaddr;
    pc->socklen  = peers->peer[idx].socklen;
    pc->name     = &peers->peer[idx].name;
    rrp->current = &peers->peer[idx];
    ngx_log_debug(NGX_LOG_DEBUG_HTTP, pc->log, 0,
        "gts_get_peer: chosen=%ui addr=\"%V\"", idx, pc->name);
    return NGX_OK;
}

static void
ngx_http_gts_free_peer(ngx_peer_connection_t *pc, void *data, ngx_uint_t state)
{
    ngx_http_gts_peer_data_t *gdata = data;
    ngx_http_upstream_rr_peer_data_t *rrp = gdata->rrp;
    ngx_http_upstream_free_round_robin_peer(pc, rrp, state);
}