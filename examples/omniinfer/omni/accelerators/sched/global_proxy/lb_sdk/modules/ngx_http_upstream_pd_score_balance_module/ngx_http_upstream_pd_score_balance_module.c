// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <ngx_http_upstream.h>
#include <ngx_atomic.h>
#include <stdlib.h>
#include <float.h>


typedef enum {
    PD_MODE_NONE = 0,
    PD_MODE_PREFILL,
    PD_MODE_DECODE
} pd_score_mode_e;

typedef struct {
    size_t shm_size;
} ngx_http_pd_score_main_conf_t;

typedef struct {
    pd_score_mode_e mode;
    ngx_uint_t max_num_seqs;
} ngx_http_pd_score_srv_conf_t;

typedef struct {
    ngx_uint_t num_prefill_peers;
    ngx_uint_t num_decode_peers;
} ngx_http_pd_score_ctx_t;

typedef struct {
    ngx_atomic_t active_requests;
    ngx_atomic_t total_request_length;
} ngx_http_pd_score_shm_peer_P_t;

typedef struct {
    ngx_atomic_t active_requests;
    ngx_atomic_t total_decode_num;
    ngx_atomic_t total_request_length;
} ngx_http_pd_score_shm_peer_D_t;

typedef struct {
    ngx_queue_t queue;
    void *id_ptr;
    ngx_uint_t request_length;
    ngx_uint_t inque_time;
} ngx_http_pd_score_run_req_node_t;

typedef struct {
    ngx_uint_t total_active_request_count;
    ngx_queue_t running_requests_P;
    ngx_http_pd_score_shm_peer_P_t *peers_P;
    ngx_http_pd_score_shm_peer_D_t *peers_D;
    char data[0];
} ngx_http_pd_score_shm_block_t;

typedef struct {
    ngx_http_upstream_rr_peer_data_t *rrp;
    ngx_uint_t chosen;
    double my_time_cost;
    ngx_atomic_t decode_token_count;
    ngx_uint_t request_length;
    int first_chunk;
    ngx_uint_t last_total_tokens;
} ngx_http_pd_score_peer_data_t;

static ngx_shm_zone_t *ngx_http_pd_score_shm_zone = NULL;
static ngx_http_pd_score_shm_block_t *pd_shm = NULL;
static ngx_uint_t ngx_http_pd_score_shm_size = 0;
static ngx_uint_t ngx_http_pd_score_max_num_seqs_P = 0;
static ngx_uint_t ngx_http_pd_score_max_num_seqs_D = 0;
static ngx_uint_t max_predict_reqs = 4;
static ngx_uint_t LPT_max_min_thres = 1;
static ngx_http_output_body_filter_pt ngx_http_next_body_filter = NULL;

static void *ngx_http_pd_score_create_main_conf(ngx_conf_t *cf);
static void *ngx_http_pd_score_create_srv_conf(ngx_conf_t *cf);
static char *ngx_http_pd_score_set_mode(ngx_conf_t *cf, ngx_command_t *cmd, void *conf);
static ngx_int_t ngx_http_pd_score_init_shm_zone(ngx_shm_zone_t *shm_zone, void *data);
static ngx_int_t ngx_http_pd_score_postconfig(ngx_conf_t *cf);

static ngx_int_t ngx_http_pd_score_upstream_init(ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *uscf);
static ngx_int_t ngx_http_pd_score_get_peer_P(ngx_peer_connection_t *pc, void *data);
static ngx_int_t ngx_http_pd_score_get_peer_D(ngx_peer_connection_t *pc, void *data);
static void ngx_http_pd_score_free_peer_P(ngx_peer_connection_t *pc, void *data, ngx_uint_t state);
static void ngx_http_pd_score_free_peer_D(ngx_peer_connection_t *pc, void *data, ngx_uint_t state);
static ngx_int_t ngx_http_pd_score_prefill_strategy(ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *uscf);
static ngx_int_t ngx_http_pd_score_decode_strategy(ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *uscf);
static ngx_int_t ngx_http_pd_score_body_filter(ngx_http_request_t *r, ngx_chain_t *in);

void ngx_http_pd_score_add_decoded_tokens(ngx_http_request_t *r, ngx_uint_t num_tokens);

static ngx_command_t ngx_http_upstream_pd_score_commands[] = {
    { ngx_string("pd_score_balance_shm_size"),
      NGX_HTTP_MAIN_CONF | NGX_CONF_TAKE1, ngx_conf_set_size_slot,
      NGX_HTTP_MAIN_CONF_OFFSET, offsetof(ngx_http_pd_score_main_conf_t, shm_size),
      NULL },

    { ngx_string("pd_score_balance"),
      NGX_HTTP_UPS_CONF | NGX_CONF_TAKE1,
      ngx_http_pd_score_set_mode, NGX_HTTP_SRV_CONF_OFFSET,
      offsetof(ngx_http_pd_score_srv_conf_t, mode), NULL },

    { ngx_string("pd_score_balance_max_num_seqs"),
      NGX_HTTP_UPS_CONF | NGX_CONF_TAKE1,
      ngx_conf_set_num_slot, NGX_HTTP_SRV_CONF_OFFSET,
      offsetof(ngx_http_pd_score_srv_conf_t, max_num_seqs),
      NULL },

    ngx_null_command
};

static ngx_http_module_t ngx_http_upstream_pd_score_balance_module_ctx = {
    NULL,
    ngx_http_pd_score_postconfig,
    ngx_http_pd_score_create_main_conf,
    NULL,
    ngx_http_pd_score_create_srv_conf,
    NULL,
    NULL,
    NULL
};

ngx_module_t ngx_http_upstream_pd_score_balance_module = {
    NGX_MODULE_V1,
    &ngx_http_upstream_pd_score_balance_module_ctx,
    ngx_http_upstream_pd_score_commands,
    NGX_HTTP_MODULE,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NGX_MODULE_V1_PADDING
};

static void *ngx_http_pd_score_create_main_conf(ngx_conf_t *cf) {
    ngx_http_pd_score_main_conf_t *conf = ngx_pcalloc(cf->pool, sizeof(*conf));
    if (conf == NULL) {
        ngx_log_error(NGX_LOG_EMERG, cf->log, 0,
                      "Failed to allocate pd_score_balance main conf");
        return NULL;
    }

    conf->shm_size = NGX_CONF_UNSET_SIZE;
    ngx_log_debug0(NGX_LOG_DEBUG_HTTP, cf->log, 0,
                   "pd_score_balance main conf created");
    return conf;
}

static char *ngx_http_pd_score_set_mode(ngx_conf_t *cf, ngx_command_t *cmd,
                                        void *conf) {
    ngx_str_t *value = cf->args->elts;

    ngx_http_pd_score_srv_conf_t *c = conf;

    if (ngx_strcmp(value[1].data, "prefill") == 0) {
        c->mode = PD_MODE_PREFILL;
    } else if (ngx_strcmp(value[1].data, "decode") == 0) {
        c->mode = PD_MODE_DECODE;
    } else {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
                           "invalid pd_score_balance mode: %V", &value[1]);
        return NGX_CONF_ERROR;
    }
    return NGX_CONF_OK;
}

static void *ngx_http_pd_score_create_srv_conf(ngx_conf_t *cf) {
    ngx_http_pd_score_srv_conf_t *conf = ngx_pcalloc(cf->pool, sizeof(*conf));
    if (conf == NULL) {
        return NULL;
    }
    conf->mode = PD_MODE_NONE;
    conf->max_num_seqs = NGX_CONF_UNSET_UINT;
    return conf;
}

static ngx_int_t ngx_http_pd_score_init_shm_zone(ngx_shm_zone_t *shm_zone,
                                                 void *data) {
    ngx_slab_pool_t *shpool;
    ngx_http_pd_score_shm_block_t *shm_block;
    ngx_uint_t i, n;
    ngx_http_pd_score_ctx_t *ctx = shm_zone->data;
    shpool = (ngx_slab_pool_t *)shm_zone->shm.addr;

    size_t sz = sizeof(ngx_http_pd_score_shm_block_t)
                + sizeof(ngx_http_pd_score_shm_peer_P_t) * ctx->num_prefill_peers
                + sizeof(ngx_http_pd_score_shm_peer_D_t) * ctx->num_decode_peers;
    shm_block = ngx_slab_alloc(shpool, sz);

    if (!shm_block) {
        return NGX_ERROR;
    }
    ngx_queue_init(&shm_block->running_requests_P);
    shm_block->total_active_request_count = 0;
    shm_block->peers_P = (ngx_http_pd_score_shm_peer_P_t *)(shm_block->data);
    shm_block->peers_D = (ngx_http_pd_score_shm_peer_D_t *)(shm_block->peers_P + ctx->num_prefill_peers);
    
    for (i = 0; i < ctx->num_prefill_peers ; i++) {
        shm_block->peers_P[i].active_requests = 0;
        shm_block->peers_P[i].total_request_length = 0;
    }
    for (i = 0; i < ctx->num_decode_peers; i++) {
        shm_block->peers_D[i].active_requests = 0;
        shm_block->peers_D[i].total_decode_num = 0;
        shm_block->peers_D[i].total_request_length = 0;
    }
    pd_shm = shm_block;
    return NGX_OK;
}

static ngx_int_t ngx_http_pd_score_postconfig(ngx_conf_t *cf) {
    ngx_http_pd_score_main_conf_t *pmcf = ngx_http_conf_get_module_main_conf(
        cf, ngx_http_upstream_pd_score_balance_module);
    if (pmcf == NULL) {
        ngx_log_error(NGX_LOG_EMERG, cf->log, 0, "Failed to get main conf");
        return NGX_ERROR;
    }
    if (pmcf->shm_size == 0 || pmcf->shm_size == NGX_CONF_UNSET_SIZE) {
        pmcf->shm_size = 256 * ngx_pagesize;
    }
    ngx_http_pd_score_shm_size = pmcf->shm_size;
    ngx_log_error(NGX_LOG_WARN, cf->log, 0, "Set shm_size: %uz bytes", ngx_http_pd_score_shm_size);

    ngx_http_next_body_filter = ngx_http_top_body_filter;
    ngx_http_top_body_filter = ngx_http_pd_score_body_filter;

    ngx_str_t *shm_name = ngx_palloc(cf->pool, sizeof(*shm_name));
    shm_name->len = sizeof("pd_score_balance") - 1;
    shm_name->data = (u_char *)"pd_score_balance";
    ngx_http_pd_score_shm_zone =
        ngx_shared_memory_add(cf, shm_name, ngx_http_pd_score_shm_size,
                              &ngx_http_upstream_pd_score_balance_module);
    if (ngx_http_pd_score_shm_zone == NULL) {
        return NGX_ERROR;
    }
    ngx_http_upstream_main_conf_t *upcf;
    ngx_http_upstream_srv_conf_t **uscfp;
    ngx_http_pd_score_srv_conf_t *conf;
    ngx_http_pd_score_ctx_t *ctx;
    ctx = ngx_pcalloc(cf->pool, sizeof(ngx_http_pd_score_ctx_t));
    if (ctx == NULL) {
        return NGX_ERROR;
    }
    ngx_http_pd_score_shm_zone->init = ngx_http_pd_score_init_shm_zone;
    ngx_http_pd_score_shm_zone->data = ctx;
    ngx_uint_t i;
    upcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_upstream_module);
    if (upcf == NULL) {
        return NGX_OK;
    }
    uscfp = upcf->upstreams.elts;
    for (i = 0; i < upcf->upstreams.nelts; i++) {
        conf = ngx_http_conf_upstream_srv_conf(
            uscfp[i], ngx_http_upstream_pd_score_balance_module);
        if (conf->mode != PD_MODE_NONE) {
            uscfp[i]->peer.init = ngx_http_pd_score_upstream_init;
        } else {
            uscfp[i]->peer.init = ngx_http_upstream_init_round_robin_peer;
        }
        if (conf->mode == PD_MODE_PREFILL) {
            ctx->num_prefill_peers = uscfp[i]->servers->nelts;
            max_predict_reqs = 2 * uscfp[i]->servers->nelts;
            ngx_log_error(NGX_LOG_WARN, cf->log, 0,
                          "[PDScoreBalance] max request preallocated set to %ui", max_predict_reqs);
            ngx_http_pd_score_max_num_seqs_P = conf->max_num_seqs;
            if (ngx_http_pd_score_max_num_seqs_P == NGX_CONF_UNSET_UINT) {
                ngx_http_pd_score_max_num_seqs_P = 16;
            }
            ngx_log_error(NGX_LOG_WARN, cf->log, 0,
                          "[PDScoreBalance] upstream[%ui] mode: Prefill, max num seqs: %ui, num of peers: %ui", i, ngx_http_pd_score_max_num_seqs_P, ctx->num_prefill_peers);
        }
        if (conf->mode == PD_MODE_DECODE) {
            ctx->num_decode_peers = uscfp[i]->servers->nelts;
            ngx_http_pd_score_max_num_seqs_D = conf->max_num_seqs;
            if (ngx_http_pd_score_max_num_seqs_D == NGX_CONF_UNSET_UINT) {
                ngx_http_pd_score_max_num_seqs_D = 32;
            }
            ngx_log_error(NGX_LOG_WARN, cf->log, 0,
                          "[PDScoreBalance] upstream[%ui] mode: Decode, max num seqs: %ui, num of peers: %ui", i, ngx_http_pd_score_max_num_seqs_D, ctx->num_decode_peers);
        }
    }
    return NGX_OK;
}

void ngx_http_pd_score_add_decoded_tokens(ngx_http_request_t *r, ngx_uint_t num_tokens) {
    ngx_http_pd_score_peer_data_t *pdata = 
    r->upstream ? r->upstream->peer.data : NULL;
    ngx_slab_pool_t *shpool;
    if (pd_shm == NULL || pdata == NULL) {
        return;
    }
    shpool = (ngx_slab_pool_t *)ngx_http_pd_score_shm_zone->shm.addr;
    ngx_shmtx_lock(&shpool->mutex);

    ngx_atomic_fetch_add(&pd_shm->peers_D[pdata->chosen].total_decode_num,
                         (ngx_atomic_int_t)num_tokens);
    ngx_atomic_fetch_add(&pdata->decode_token_count,
                         (ngx_atomic_int_t)num_tokens);

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                  "[PDScore] peer=%ui request=%p decode_token +%ui, "
                  "peer_total_decode_token=%uA, request_total_decode_token=%uA",
                  pdata->chosen, r, num_tokens,
                  pd_shm->peers_D[pdata->chosen].total_decode_num,
                  pdata->decode_token_count);
    ngx_shmtx_unlock(&shpool->mutex);
}

static ngx_int_t ngx_http_pd_score_body_filter(ngx_http_request_t *r,
                                               ngx_chain_t *in) {
    ngx_http_pd_score_srv_conf_t *uscf;

    if (r->upstream == NULL) {
        return ngx_http_next_body_filter(r, in);
    }

    uscf = ngx_http_conf_upstream_srv_conf(
        r->upstream->upstream, ngx_http_upstream_pd_score_balance_module);
    if (uscf->mode == PD_MODE_PREFILL) {
        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                      "prefill mode, ignore filter");
        return ngx_http_next_body_filter(r, in);
    }
    ngx_chain_t *cl;
    ngx_uint_t total_tokens = 0;

    ngx_http_pd_score_peer_data_t *pdata;
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
            static char key_total[] = "\"total_tokens\":";
            size_t keylen_total = sizeof(key_total) - 1;
            u_char *found = ngx_strnstr(p, key_total, last - p);
            if (found) {
                u_char *num_start = found + keylen_total;
                while (num_start < last &&
                       (*num_start == ' ' || *num_start == '\"')) {
                    num_start++;
                }
                ngx_uint_t val = 0;
                while (num_start < last && *num_start >= '0' &&
                       *num_start <= '9') {
                    val = val * 10 + (*num_start - '0');
                    num_start++;
                }
                total_tokens = val;
            } else {
                if (pdata->first_chunk) {
                    total_tokens = pdata->decode_token_count + 1;
                } else {
                    total_tokens = pdata->last_total_tokens + 1;
                }
            }
        }
    }

    if (pdata->first_chunk) {
        pdata->first_chunk = 0;
        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, 
            "[PDScore-Filter] first chunk, peer #%ui, request_decode_token_count %ui, peer_total_decode_token %ui,",
            pdata->chosen, pdata->decode_token_count, pd_shm->peers_D[pdata->chosen].total_decode_num);
        ngx_atomic_fetch_add(&pd_shm->peers_D[pdata->chosen].total_decode_num,
            (ngx_atomic_int_t)-pdata->decode_token_count);
        pdata->decode_token_count = 0;
    }

    ngx_uint_t added_tokens = 0;

    if (total_tokens > pdata->last_total_tokens) {
        added_tokens = total_tokens - pdata->last_total_tokens;
    }

    pdata->last_total_tokens = total_tokens;

    if (added_tokens > 0) {
        ngx_http_pd_score_add_decoded_tokens(r, added_tokens);
    }
    return ngx_http_next_body_filter(r, in);
}

static int cmp_desc(const void *a, const void *b) {
    ngx_uint_t va = *(const ngx_uint_t *)a;
    ngx_uint_t vb = *(const ngx_uint_t *)b;
    return (vb > va) - (vb < va);
}

static ngx_int_t
ngx_http_pd_score_prefill_strategy(ngx_http_request_t *r,
                                   ngx_http_upstream_srv_conf_t *uscf) {
    ngx_http_upstream_t *u = r->upstream;
    ngx_http_upstream_rr_peer_data_t *rrp;
    ngx_http_pd_score_peer_data_t *pdata;
    ngx_uint_t chosen = 0, i, n;
    ngx_uint_t min_load, min_load_idx, min_req, min_req_idx;
    ngx_slab_pool_t *shpool;
    ngx_time_t *tp = ngx_timeofday();
    ngx_uint_t now = tp->sec * 1000 + tp->msec;
    if (ngx_http_upstream_init_round_robin_peer(r, uscf) != NGX_OK) {
        return NGX_ERROR;
    }
    rrp = u->peer.data;

    if (pd_shm == NULL) {
        pd_shm = ngx_http_pd_score_shm_zone->data;
    }
    shpool = (ngx_slab_pool_t *)ngx_http_pd_score_shm_zone->shm.addr;
    n = rrp->peers->number;

    ngx_shmtx_lock(&shpool->mutex);
    min_load = NGX_MAX_INT_T_VALUE;
    min_load_idx = NGX_CONF_UNSET_UINT;
    min_req = NGX_MAX_INT_T_VALUE;
    min_req_idx = NGX_CONF_UNSET_UINT;

    for (i = 0; i < n; i++) {
        if (pd_shm->peers_P[i].total_request_length < min_load
            && pd_shm->peers_P[i].active_requests < ngx_http_pd_score_max_num_seqs_P) {
            min_load = pd_shm->peers_P[i].total_request_length;
            min_load_idx = i;
        }
        if (pd_shm->peers_P[i].active_requests < min_req) {
            min_req = pd_shm->peers_P[i].active_requests;
            min_req_idx = i;
        }
    }
    if (min_load_idx != NGX_CONF_UNSET_UINT) {
        chosen = min_load_idx;
    } else {
        chosen = min_req_idx;
    }

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                  "[PDScore-PREFILL] request(len=%ui) assigned to peer #%ui",
                  (ngx_uint_t)r->request_length, chosen);

    pdata = ngx_pcalloc(r->pool, sizeof(*pdata));
    pdata->rrp = rrp;
    pdata->chosen = chosen;
    pdata->my_time_cost = 0;
    pdata->decode_token_count = (ngx_atomic_t)r->request_length / 4;
    pdata->first_chunk = 1;
    pdata->request_length = (ngx_uint_t)r->request_length;
    pdata->last_total_tokens = 0;
 
    ngx_http_pd_score_run_req_node_t *cur_req = ngx_slab_alloc_locked(shpool, sizeof(ngx_http_pd_score_run_req_node_t));
    if (cur_req == NULL) {
        ngx_shmtx_unlock(&shpool->mutex);
        return NGX_ERROR;
    }

    cur_req->id_ptr = pdata;
    cur_req->inque_time = now;
    cur_req->request_length = (ngx_uint_t)r->request_length;

    pd_shm->total_active_request_count++;
    // find suitable position to insert, keep the queue sorted by inque_time ascendingly
    ngx_queue_t *q = ngx_queue_last(&pd_shm->running_requests_P);
    while (q != ngx_queue_sentinel(&pd_shm->running_requests_P) && ((ngx_http_pd_score_run_req_node_t *)q)->inque_time > now) {
        q = ngx_queue_prev(q);
    }
    ngx_queue_insert_after(q, &cur_req->queue);

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                  "[PDScore-PREFILL] request(len=%ui) enqueued, total_active_request_count=%ui",
                  (ngx_uint_t)r->request_length, pd_shm->total_active_request_count);
    
    ngx_shmtx_unlock(&shpool->mutex);

    u->peer.data = pdata;
    u->peer.get = ngx_http_pd_score_get_peer_P;
    u->peer.free = ngx_http_pd_score_free_peer_P;
    return NGX_OK;
}

static ngx_int_t
ngx_http_pd_score_decode_strategy(ngx_http_request_t *r,
                                  ngx_http_upstream_srv_conf_t *uscf) {
    ngx_http_upstream_t *u = r->upstream;
    ngx_http_upstream_rr_peer_data_t *rrp;
    ngx_http_pd_score_peer_data_t *pdata;
    ngx_uint_t chosen = 0, i, n;
    ngx_uint_t min_req, min_req_idx, max_req, max_req_idx, peer_req;
    ngx_uint_t min_load, min_load_idx;
    ngx_slab_pool_t *shpool;

    if (ngx_http_upstream_init_round_robin_peer(r, uscf) != NGX_OK) {
        return NGX_ERROR;
    }
    rrp = u->peer.data;

    if (pd_shm == NULL) {
        pd_shm = ngx_http_pd_score_shm_zone->data;
    }

    shpool = (ngx_slab_pool_t *)ngx_http_pd_score_shm_zone->shm.addr;
    n = rrp->peers->number;

    ngx_shmtx_lock(&shpool->mutex);

    min_req = NGX_MAX_INT_T_VALUE;
    min_req_idx = NGX_CONF_UNSET_UINT;
    max_req = 0;
    max_req_idx = NGX_CONF_UNSET_UINT;

    for (i = 0; i < n; i++) {
        peer_req = pd_shm->peers_D[i].active_requests;
        if (peer_req < min_req) {
            min_req = peer_req;
            min_req_idx = i;
        }
        if (peer_req > max_req) {
            max_req = peer_req;
            max_req_idx = i;
        }
    }

    // apply LPT if max_req - min_req <= threshold and there is at least one peer with fewer than max_num_seqs_D active requests
    if (min_req < ngx_http_pd_score_max_num_seqs_D &&  max_req - min_req <= LPT_max_min_thres) {
        ngx_uint_t filtered_count = pd_shm->total_active_request_count < max_predict_reqs ?
                                    pd_shm->total_active_request_count : max_predict_reqs;

        ngx_uint_t *filtered_req_lengths = ngx_pcalloc(r->pool, sizeof(ngx_uint_t) * (filtered_count + 1));
        if (filtered_req_lengths == NULL) {
            ngx_shmtx_unlock(&shpool->mutex);
            return NGX_ERROR;
        }

        ngx_uint_t j = 0;
        ngx_queue_t *q = ngx_queue_head(&pd_shm->running_requests_P);
        for (j = 0; j < filtered_count; j++) {
            filtered_req_lengths[j] = ((ngx_http_pd_score_run_req_node_t *)q)->request_length;
            q = ngx_queue_next(q);
        }
        filtered_req_lengths[filtered_count] = (ngx_uint_t)r->request_length;
        qsort(filtered_req_lengths, filtered_count + 1, sizeof(ngx_uint_t), cmp_desc);
        for (j = 0; j < filtered_count + 1; j++) {
            ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                        "sorted filtered_req_lengths[%ui]: %ui", j,
                        filtered_req_lengths[j]);
        }

        ngx_uint_t *peer_loads = ngx_pcalloc(r->pool, sizeof(ngx_uint_t) * n);
        ngx_uint_t *peer_reqs = ngx_pcalloc(r->pool, sizeof(ngx_uint_t) * n);
        if (peer_loads == NULL || peer_reqs == NULL) {
            ngx_shmtx_unlock(&shpool->mutex);
            return NGX_ERROR;
        }

        for (i = 0; i < n; i++) {
            peer_loads[i] = pd_shm->peers_D[i].total_decode_num * 4;
            peer_reqs[i] = pd_shm->peers_D[i].active_requests;
        }

        chosen = 0;

        for (i = 0; i < filtered_count + 1; i++) {
            min_load = NGX_MAX_INT_T_VALUE;
            min_load_idx = NGX_CONF_UNSET_UINT;
            min_req = NGX_MAX_INT_T_VALUE;
            min_req_idx = NGX_CONF_UNSET_UINT;
            for (j = 0; j < n; j++) {
                if (peer_loads[j] < min_load && pd_shm->peers_D[j].active_requests < ngx_http_pd_score_max_num_seqs_D) {
                    min_load = peer_loads[j];
                    min_load_idx = j;
                }
                if (peer_reqs[j] < min_req) {
                    min_req = peer_reqs[j];
                    min_req_idx = j;
                }
            }
            if (min_load_idx != NGX_CONF_UNSET_UINT) {
                chosen = min_load_idx;
            } else {
                chosen = min_req_idx;
            }

            peer_loads[chosen] += filtered_req_lengths[i];
            peer_reqs[chosen] += 1;
            ngx_log_error(
                NGX_LOG_INFO, r->connection->log, 0,
                "[PDScore-DECODE-LPT] simulate assigning reqs round %ui: req_len=%ui to peer #%ui",
                i, filtered_req_lengths[i], chosen);
            if (filtered_req_lengths[i] == (ngx_uint_t)r->request_length) {
                break;
            }
        }
    } else {
        chosen = min_req_idx;
    }
    
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                  "[PDScore-DECODE-LPT] request(len=%ui) assigned to peer #%ui",
                  (ngx_uint_t)r->request_length, chosen);

    ngx_shmtx_unlock(&shpool->mutex);

    pdata = ngx_pcalloc(r->pool, sizeof(*pdata));
    pdata->rrp = rrp;

    pdata->chosen = chosen;

    pdata->my_time_cost = 0;
    pdata->decode_token_count = (ngx_uint_t)r->request_length / 4;
    pdata->first_chunk = 1;
    pdata->request_length = (ngx_uint_t)r->request_length;
    pdata->last_total_tokens = 0;
    u->peer.data = pdata;
    u->peer.get = ngx_http_pd_score_get_peer_D;
    u->peer.free = ngx_http_pd_score_free_peer_D;
    return NGX_OK;
}

static ngx_int_t ngx_http_pd_score_get_peer_P(ngx_peer_connection_t *pc,
                                            void *data) {
    ngx_http_pd_score_peer_data_t *pdata = data;
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

    ngx_http_pd_score_shm_peer_P_t *peer_P = &pd_shm->peers_P[idx];
    ngx_atomic_fetch_add(&peer_P->active_requests, 1);
    ngx_atomic_fetch_add(&peer_P->total_request_length,
                         (ngx_atomic_int_t)pdata->request_length);
    return NGX_OK;
}

static ngx_int_t ngx_http_pd_score_get_peer_D(ngx_peer_connection_t *pc,
                                            void *data) {
    ngx_http_pd_score_peer_data_t *pdata = data;
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

    ngx_http_pd_score_shm_peer_D_t *peer_D = &pd_shm->peers_D[idx];
    ngx_atomic_fetch_add(&peer_D->active_requests, 1);
    ngx_atomic_fetch_add(&peer_D->total_decode_num, (ngx_atomic_int_t)pdata->decode_token_count);

    return NGX_OK;
}

static ngx_int_t ngx_http_pd_score_upstream_init(ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *uscf) {
    ngx_http_pd_score_srv_conf_t *conf = ngx_http_conf_upstream_srv_conf(
        uscf, ngx_http_upstream_pd_score_balance_module);
    switch (conf->mode) {
    case PD_MODE_PREFILL:
        return ngx_http_pd_score_prefill_strategy(r, uscf);
    case PD_MODE_DECODE:
        return ngx_http_pd_score_decode_strategy(r, uscf);
    default:
        return NGX_ERROR;
    }
}

static void ngx_http_pd_score_free_peer_P(ngx_peer_connection_t *pc, void *data,
                                          ngx_uint_t state) {
    ngx_log_error(NGX_LOG_INFO, pc->log, 0, "Freeing peer P req.%p", pc->data);
    ngx_http_pd_score_peer_data_t *pdata = data;
    ngx_slab_pool_t *shpool;
    if (pd_shm == NULL) {
        return;
    }
    shpool = (ngx_slab_pool_t *)ngx_http_pd_score_shm_zone->shm.addr;
    ngx_shmtx_lock(&shpool->mutex);
    ngx_http_pd_score_shm_peer_P_t *peer_P = &pd_shm->peers_P[pdata->chosen];
    ngx_log_error(NGX_LOG_INFO, pc->log, 0, "total active request count: %ui",
                  pd_shm->total_active_request_count);

    // find and remove from running_requests_P queue
    ngx_queue_t *q;
    ngx_http_pd_score_run_req_node_t *cur_req;
    for (q = ngx_queue_head(&pd_shm->running_requests_P);
         q != ngx_queue_sentinel(&pd_shm->running_requests_P);
         q = ngx_queue_next(q)) {
        cur_req = (ngx_http_pd_score_run_req_node_t *)q;
        if (cur_req->id_ptr == (void *)pc->data) {
            ngx_log_error(NGX_LOG_INFO, pc->log, 0,
                          "found and removing req.%p from running_requests_P queue",
                          pc->data);
            ngx_queue_remove(q);
            ngx_slab_free_locked(shpool, cur_req);
            pd_shm->total_active_request_count--;
            break;
        }
    }
    ngx_atomic_fetch_add(&peer_P->active_requests, (ngx_atomic_int_t)-1);
    ngx_atomic_fetch_add(&peer_P->total_request_length,
                         (ngx_atomic_int_t) - (pdata->request_length));

    ngx_shmtx_unlock(&shpool->mutex);

    ngx_http_upstream_rr_peer_data_t *rrp = pdata->rrp;
    ngx_http_upstream_free_round_robin_peer(pc, rrp, state);
}

static void ngx_http_pd_score_free_peer_D(ngx_peer_connection_t *pc, void *data,
                                          ngx_uint_t state) {
    ngx_log_error(NGX_LOG_INFO, pc->log, 0, "Freeing peer D");
    ngx_http_pd_score_peer_data_t *pdata = data;
    ngx_slab_pool_t *shpool;
    if (pd_shm == NULL) {
        return;
    }
    shpool = (ngx_slab_pool_t *)ngx_http_pd_score_shm_zone->shm.addr;
    ngx_shmtx_lock(&shpool->mutex);

    ngx_http_pd_score_shm_peer_D_t *peer_D = &pd_shm->peers_D[pdata->chosen];

    ngx_atomic_fetch_add(&peer_D->active_requests, (ngx_atomic_int_t)-1);
    ngx_atomic_fetch_add(&peer_D->total_decode_num,
                         (ngx_atomic_int_t) - (pdata->decode_token_count));

    ngx_shmtx_unlock(&shpool->mutex);

    ngx_http_upstream_rr_peer_data_t *rrp = pdata->rrp;
    ngx_http_upstream_free_round_robin_peer(pc, rrp, state);
}