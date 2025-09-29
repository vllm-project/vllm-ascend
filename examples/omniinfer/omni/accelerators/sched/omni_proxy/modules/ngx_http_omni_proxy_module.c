// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <omni_proxy.h>
#include <omni_pd_body_rewrite.h>
#include <omni_scheduler.h>
#include <omni_utils.h>
#include <omni_tokenizer.h>
#include <stdbool.h>
#include <omni_apc.h>
#include <omni_metrics.h>

ngx_module_t ngx_http_omni_proxy_module;
#define PREFILL_ENDPOINTS "prefill_endpoints"
#define DECODE_ENDPOINTS "decode_endpoints"

#define TIMER_INTERVAL 1

static const char *PREFILL_URI = "/prefill_sub";
static const size_t PREFILL_URI_LEN = sizeof("/prefill_sub") - 1;

static omni_global_state_t *g_state;          // In share memory
static omni_worker_local_state_t local_state; // In local process memory space

omni_global_state_t *omni_get_global_state()
{
    return g_state;
}

omni_worker_local_state_t *omni_get_local_state()
{
    return &local_state;
}

static omni_req_t *omni_req_init(ngx_http_request_t *r)
{
    ngx_shmtx_lock(&g_state->shmtx);
    omni_req_t *req = omni_allocate_request(&g_state->request_pool, r);
    ngx_shmtx_unlock(&g_state->shmtx);

    omni_req_context_t *ctx = ngx_pcalloc(r->pool, sizeof(omni_req_context_t));
    ctx->req = req;
    ngx_http_set_ctx(r, ctx, ngx_http_omni_proxy_module);

    omni_req_enter_phase(req, 0);
    omni_add_req_to_group(req->slot_index, &local_state.groups[0]);

    ngx_shmtx_lock(&g_state->shmtx);
    omni_add_req_to_group(req->slot_index, &g_state->groups[0]);
    ngx_shmtx_unlock(&g_state->shmtx);

    req->worker_pid = ngx_pid;

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "Allocate Req:%d",
                  req->slot_index);

    return req;
}

static void omni_req_free(omni_req_t *req)
{
    ngx_http_request_t *r = omni_get_http_request(req);
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "Free Req:%d, at:%x",
                  req->slot_index, req->phase_state);
    omni_free_request(&g_state->request_pool, req);
}

void omni_phase_transition_all(omni_req_t *req, omni_proxy_request_phase_t from, omni_proxy_request_phase_t to)
{
    omni_local_phase_change_to(req, from, to);

    ngx_shmtx_lock(&g_state->shmtx);
    omni_global_phase_change_to(req, from, to);
    omni_req_leave_phase(req, from);
    omni_req_enter_phase(req, to);
    ngx_shmtx_unlock(&g_state->shmtx);
}

static inline omni_req_context_t *omni_get_req_ctx(ngx_http_request_t *r)
{
    return ngx_http_get_module_ctx(r, ngx_http_omni_proxy_module);
}


static inline omni_req_t *omni_get_req(ngx_http_request_t *r)
{
    return omni_get_req_ctx(r)->req;
}

static void omni_proxy_post_tokenized(omni_req_t *req)
{
    /* 1) snapshot pointers under lock */
    ngx_shmtx_lock(&g_state->shmtx);
    /* snapshot radix_tree pointers */
    uint16_t num_prefill = g_state->num_prefill_endpoints;
    omni_radix_tree_t *local_trees[MAX_PREFILL_UPSTREAMS];
    for (uint16_t i = 0; i < num_prefill; ++i)
    {
        local_trees[i] = g_state->prefill_states[i].radix_tree;
    }
    ngx_shmtx_unlock(&g_state->shmtx);

    /* 2) Compute matches without holding g_state lock */
    uint32_t local_match_depths[MAX_PREFILL_UPSTREAMS];
    ngx_uint_t computed_max = 0;

    for (uint16_t i = 0; i < num_prefill; ++i)
    {
        omni_radix_tree_t *tree = local_trees[i];
        ngx_uint_t match_depth = 0;
        if (tree != NULL)
        {
            match_depth = omni_radix_tree_match_optimistic(tree,
                                                            (uint64_t *)req->tokenizer_req.block_hashes,
                                                            req->tokenizer_req.block_hashes_len);
        }
        local_match_depths[i] = (uint32_t)match_depth;
        if ((ngx_uint_t)match_depth > computed_max)
            computed_max = match_depth;
    }
    

    /* 3) write results to shared request */
    for (uint16_t i = 0; i < num_prefill; ++i)
    {
        req->match_depths[i] = local_match_depths[i];
    }
    req->max_match_depth = (uint32_t)computed_max;

    /* Finally perform phase transition and timestamp updates */
    omni_phase_transition_all(req, 0, PHASE_PREFILL_WAITING_SCHEDULE);
    req->metrics.time_enter_wait_prefill = ngx_current_msec;

    if (g_state->pd_policy == PD_PARALLEL)
    {
        omni_add_req_to_group(req->slot_index, &local_state.groups[PHASE_DECODE_WAITING_SCHEDULE]);
        req->metrics.time_enter_wait_decode = ngx_current_msec;
    }
}

static void omni_proxy_req_body_handler(ngx_http_request_t *r)
{
    omni_req_t *req = omni_get_req(r);
    omni_req_context_t *ctx = omni_get_req_ctx(r);
    req->metrics.time_contents_received = ngx_current_msec;
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[Prefill-%d]: request body received", req->slot_index);

    omni_proxy_save_origin_body(r, ctx);

    req->metrics.prompt_num_tokens = r->request_length / 8; // will be updated after tokenization

    if (!g_state->has_tokenizer)
    {
        omni_proxy_post_tokenized(req);
        return;
    }

    req->tokenizer_req.input_data = ctx->origin_body_data;
    req->tokenizer_req.input_len = ctx->origin_body_data_size;

    // Chat template expansion buffer
    req->tokenizer_req.prompt_buf_size = req->tokenizer_req.input_len + 512;
    req->tokenizer_req.prompt = ngx_palloc(r->pool, req->tokenizer_req.prompt_buf_size);

    req->tokenizer_req.input_ids_buf_size = req->tokenizer_req.prompt_buf_size;
    req->tokenizer_req.input_ids = ngx_palloc(r->pool, sizeof(int64_t) * req->tokenizer_req.input_ids_buf_size);

    req->tokenizer_req.block_hashes_buf_size = req->tokenizer_req.input_ids_buf_size / local_state.loc_conf->kv_block_size;
    req->tokenizer_req.block_hashes = ngx_palloc(r->pool, sizeof(int64_t) * req->tokenizer_req.block_hashes_buf_size);
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                  "buf sizes: prompt %zu, input_ids %zu, block_hashes %zu, %p, %p, %p\n",
                  req->tokenizer_req.prompt_buf_size,
                  req->tokenizer_req.input_ids_buf_size,
                  req->tokenizer_req.block_hashes_buf_size,
                  req->tokenizer_req.prompt,
                  req->tokenizer_req.input_ids,
                  req->tokenizer_req.block_hashes);
    omni_tokenizer_worker_submit(&local_state.tokenize_worker, req->slot_index);
}
static inline void omni_proxy_cleanup_req(omni_req_t *req)
{
    omni_proxy_request_phase_t phases[PHASE_MAX];
    size_t count = 0;
    omni_req_get_phases(req, phases, &count);

    for (size_t i = 0; i < count; ++i)
    {
        omni_proxy_request_phase_t phase = phases[i];
        ngx_shmtx_lock(&g_state->shmtx);

        if (phase == PHASE_PREFILLING)
        {
            omni_upstream_prefill_t *ps =
                &g_state->prefill_states[req->prefill_upstream_endpoint_idx];

            ps->num_running--;

            if (ps->num_tokens >= req->metrics.prompt_num_tokens)
            {
                ps->num_tokens -= req->metrics.prompt_num_tokens;
            }
            else
            {
                ps->num_tokens = 0;
            }
            ngx_log_error(NGX_LOG_INFO, omni_get_http_request(req)->connection->log, 0,
                          "[Prefill Release Stats] req %d prompt_tokens=%ui, decoded_tokens=%ui; "
                          "prefill idx=%d num_running=%ui, num_tokens(after)=%ui",
                          req->slot_index,
                          req->metrics.prompt_num_tokens,
                          req->metrics.decoded_tokens,
                          req->prefill_upstream_endpoint_idx,
                          ps->num_running,
                          ps->num_tokens);
        }

        if (phase == PHASE_DECODING)
        {
            omni_upstream_decode_t *ds =
                &g_state->decode_states[req->decode_upstream_endpoint_idx];

            ds->num_running--;

            if (ds->num_tokens >= req->metrics.prompt_num_tokens)
            {
                ds->num_tokens -= req->metrics.prompt_num_tokens + req->metrics.decoded_tokens;
            }
            else
            {
                ds->num_tokens = 0;
            }
            ngx_log_error(NGX_LOG_INFO, omni_get_http_request(req)->connection->log, 0,
                          "[Decode Release Stats] req %d prompt_tokens=%ui, decoded_tokens=%ui; "
                          "decode idx=%d num_running=%ui, num_tokens(after)=%ui",
                          req->slot_index,
                          req->metrics.prompt_num_tokens,
                          req->metrics.decoded_tokens,
                          req->decode_upstream_endpoint_idx,
                          ds->num_running,
                          ds->num_tokens);
        }
        omni_remove_req_from_group_by_req_index(req->slot_index, &g_state->groups[phase]);

        ngx_shmtx_unlock(&g_state->shmtx);

        omni_remove_req_from_group_by_req_index(req->slot_index, &local_state.groups[phase]);
    }
}

static void omni_proxy_main_req_cleanup(void *data)
{
    omni_req_t *req = data;
    omni_proxy_cleanup_req(req);

    ngx_log_error(NGX_LOG_INFO, omni_get_http_request(req)->connection->log, 0,
                  "[Decode-%d]: Done from %d.",
                  req->slot_index, req->decode_upstream_endpoint_idx);

    omni_req_free(req);
}

static ngx_int_t omni_proxy_handler(ngx_http_request_t *r)
{
    if (r->parent != NULL)
    {
        return NGX_DECLINED;
    }

    omni_req_t *req = omni_req_init(r);
    if (req == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "omni_proxy: allocate omni req failed.");
        ngx_http_finalize_request(r, NGX_ERROR);
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    ngx_http_omni_loc_conf_t *olcf;

    olcf = ngx_http_get_module_loc_conf(r, ngx_http_omni_proxy_module);
    if (olcf == NULL || olcf->upstream_name.len == 0)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "omni_proxy: no upstream_name in loc conf");
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    g_state->pd_policy = olcf->pd_policy;
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                  "[Prefill-%d]: enter main request handler, policy:%d",
                  req->slot_index, g_state->pd_policy);

    req->metrics.time_received = ngx_current_msec;

    ngx_http_cleanup_t *cleanup = ngx_http_cleanup_add(r, 0);
    cleanup->handler = omni_proxy_main_req_cleanup;
    cleanup->data = req;

    ngx_int_t rc = ngx_http_read_client_request_body(r, omni_proxy_req_body_handler);
    if (rc == NGX_AGAIN)
    {
        return NGX_DONE;
    }
    else if (rc >= NGX_HTTP_SPECIAL_RESPONSE)
    {
        return rc;
    }

    return NGX_DONE;
}

static ngx_int_t ngx_http_prefill_post_subrequest(ngx_http_request_t *subr, void *data, ngx_int_t rc)
{
    omni_req_t *req = data;
    ngx_chain_t *cl;
    size_t total = 0;
    u_char *p;
    ngx_http_request_t *r = omni_get_http_request(req);
    omni_req_context_t *ctx = omni_get_req_ctx(subr);
    req->metrics.time_prefilled = ngx_current_msec;

    ngx_connection_t *c = r->connection;

    ctx = ngx_http_get_module_ctx(r, ngx_http_omni_proxy_module);

    if (c->read->timer_set)
    {
        ngx_del_timer(c->read);
    }
    if (c->write->timer_set)
    {
        ngx_del_timer(c->write);
    }

    r->read_event_handler = ngx_http_request_empty_handler;
    r->write_event_handler = ngx_http_request_empty_handler;

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                  "[Prefill-%d] Done from %d", req->slot_index, req->prefill_upstream_endpoint_idx);

    if (rc != NGX_OK)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "[Prefill-%d] subrequest failed with code %i", req->slot_index, rc);
        ngx_http_finalize_request(r->main, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return rc;
    }

    if (subr->headers_out.status != NGX_HTTP_OK)
    {
        r->headers_out.status = subr->headers_out.status;
        ngx_http_finalize_request(r, subr->headers_out.status);
        return NGX_OK;
    }

    for (cl = subr->out; cl; cl = cl->next)
    {
        total += ngx_buf_size(cl->buf);
    }

    ctx->prefill_response_body = ngx_palloc(r->main->pool, total + 1);
    if (ctx->prefill_response_body == NULL)
    {
        ngx_http_finalize_request(r, NGX_ERROR);
        ngx_log_error(
            NGX_LOG_ERR, r->connection->log, 0,
            "done prefill subrequest, malloc prefill response body buffer failed");
        return rc;
    }

    p = ctx->prefill_response_body;
    for (cl = subr->out; cl; cl = cl->next)
    {
        size_t buf_size = ngx_buf_size(cl->buf);
        if (buf_size > 0)
        {
            p = ngx_cpymem(p, cl->buf->pos, buf_size);
        }
    }
    *p = '\0';
    ctx->prefill_response_body_size = total;

    omni_upstream_prefill_t *us = &g_state->prefill_states[req->prefill_upstream_endpoint_idx];
    us->num_running--;
    us->num_tokens -= req->metrics.prompt_num_tokens;

    omni_batch_metrics_t *current_batch = &us->his.his[us->his.head];
    ngx_msec_t delta = (current_batch->last_response_receive_time > 0) ? 
                     (ngx_current_msec - current_batch->last_response_receive_time) : 
                     (21); // If first，force delta > 20 to get a new batch

    // Need a smarter value from statistics or work out by the number of tokens scheduled
    if (delta > 20)
    {
        // 1. **calculate last batch**
        if (current_batch->num_requests > 0) 
        { // not a empty batch
            current_batch->time_taken += current_batch->last_response_receive_time - current_batch->first_response_receive_time;
             ngx_log_error(NGX_LOG_INFO, omni_get_http_request(req)->connection->log, 0,
                          "[Prefill-Batch-End] Batch at head %ui finalized. Duration: %ui ms, Tokens: %ui",
                          us->his.head, current_batch->time_taken, current_batch->num_tokens);
        }

        // 2. **start a new batch**
        if (us->his.count < NUM_PREFILL_BATCH_METRICS_HIS) { 
            us->his.count++;
        }
        us->his.head = (us->his.head + 1) % NUM_PREFILL_BATCH_METRICS_HIS;

        omni_batch_metrics_t *new_batch = &us->his.his[us->his.head];
        ngx_memzero(new_batch, sizeof(omni_batch_metrics_t));

        new_batch->first_response_receive_time = ngx_current_msec;
        new_batch->last_response_receive_time = ngx_current_msec;
        new_batch->num_requests = 1;
        new_batch->num_tokens = req->metrics.prompt_num_tokens;
        new_batch->time_taken = ngx_current_msec - req->metrics.time_to_prefill;
    }
    else
    {
        // **add to current batch**
        current_batch->last_response_receive_time = ngx_current_msec;
        current_batch->num_requests++;
        current_batch->num_tokens += req->metrics.prompt_num_tokens;
        // average_delta 
        current_batch->average_delta = current_batch->average_delta * (current_batch->num_requests - 1) +
                                       delta / (current_batch->num_requests);
    }

    // check policy
    if (g_state->pd_policy == PD_SEQUENTIAL)
    {
        omni_phase_transition_all(req, PHASE_PREFILLING, PHASE_DECODE_WAITING_SCHEDULE);
        req->metrics.time_enter_wait_decode = ngx_current_msec;
    }

    return NGX_DONE;
}

static ngx_int_t ngx_http_prefill_wakeup(omni_req_t *req)
{
    ngx_http_request_t *r = omni_get_http_request(req);

    size_t prelen = PREFILL_URI_LEN + r->uri.len;
    if (r->args.len)
    {
        prelen += 1 + r->args.len;
    }
    u_char *prefill_uri = ngx_pnalloc(r->pool, prelen + 1);
    if (prefill_uri == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "omni_proxy: allocate prefill uri failed.");
        ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    u_char *p = prefill_uri;
    p = ngx_cpymem(p, PREFILL_URI, PREFILL_URI_LEN);
    p = ngx_cpymem(p, r->uri.data, r->uri.len);
    if (r->args.len)
    {
        *p++ = '?';
        p = ngx_cpymem(p, r->args.data, r->args.len);
    }
    *p = '\0';

    ngx_str_t sub_uri;
    sub_uri.len = ngx_strlen(prefill_uri);
    sub_uri.data = prefill_uri;

    ngx_http_post_subrequest_t *psr = ngx_pcalloc(r->pool, sizeof(ngx_http_post_subrequest_t));
    psr->handler = ngx_http_prefill_post_subrequest;
    psr->data = req;

    ngx_str_t args = ngx_null_string;
    ngx_http_request_t *sr;

    ngx_int_t rc = ngx_http_subrequest(r, &sub_uri, &args, &sr, psr,
                                       NGX_HTTP_SUBREQUEST_IN_MEMORY);
    if (rc != NGX_OK)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "omni_proxy: create subrequest failed.");
        ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return rc;
    }
    sr->method = r->method;
    sr->method_name = r->method_name;

    omni_req_context_t *ctx = omni_get_req_ctx(r);
    ngx_http_set_ctx(sr, ctx, ngx_http_omni_proxy_module);

    omni_proxy_prepare_prefill_subrequest(r, sr, ctx);

    omni_phase_transition_all(req, PHASE_PREFILL_SCHEDULED, PHASE_PREFILLING);

    req->metrics.time_to_prefill = ngx_current_msec;

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[Prefill-%d] Submit to:%d",
                  req->slot_index, req->prefill_upstream_endpoint_idx);

    // ngx_http_core_run_phases(sr);
    ngx_http_run_posted_requests(r->connection);

    return NGX_OK;
}

static ngx_int_t omni_proxy_get_peer(ngx_peer_connection_t *pc,
                                     void *data)
{
    ngx_http_upstream_rr_peer_data_t *rrp = data;
    ngx_http_request_t *r = (ngx_http_request_t *)rrp->data;
    omni_req_t *req = omni_get_req(r);
    ngx_http_upstream_rr_peers_t *peers = rrp->peers;

    assert(omni_req_is_in_phase(req, PHASE_PREFILLING));

    ngx_uint_t idx = req->prefill_upstream_endpoint_idx;
    // TODO: Need to consider the liveness of the upstream in case it crashed or lag
    assert(idx < rrp->peers->number);

    pc->sockaddr = peers->peer[idx].sockaddr;
    pc->socklen = peers->peer[idx].socklen;
    pc->name = &peers->peer[idx].name;
    rrp->current = &peers->peer[idx];

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                  "[Prefill-%d] Upstream set: %d, running:%d",
                  req->slot_index, idx, g_state->prefill_states[idx].num_running);

    return NGX_OK;
}

static ngx_int_t omni_proxy_upstream_init(ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *uscf)
{

    omni_req_t *req = omni_get_req(r);
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[-%d] init upstream at phase: %x",
                  req->slot_index, req->phase_state);

    ngx_http_upstream_t *u = r->upstream;
    u->conf->send_lowat = 0;
    ngx_http_upstream_rr_peer_data_t *rrp;
    if (ngx_http_upstream_init_round_robin_peer(r, uscf) != NGX_OK)
    {
        return NGX_ERROR;
    }
    rrp = u->peer.data;

    u->peer.get = omni_proxy_get_peer;
    u->peer.data = rrp;
    rrp->data = (uintptr_t)r;

    return NGX_OK;
}

static void omni_proxy_update_decode_stats(ngx_http_request_t *r, ngx_buf_t *buf, ssize_t bytes)
{
    omni_req_t *req = omni_get_req(r);

    // Update request level statistics

    if (req->metrics.time_last_reponse)
    {
        req->metrics.decoded_tokens++;
        ngx_uint_t decode_idx = req->decode_upstream_endpoint_idx;
        ngx_atomic_fetch_add(&g_state->decode_states[decode_idx].num_tokens, 1);

        if (req->metrics.decoded_tokens != 0)
        {
            req->metrics.tpot =
                ((req->metrics.tpot * req->metrics.decoded_tokens - 1) +
                 ngx_current_msec - req->metrics.time_last_reponse) /
                req->metrics.decoded_tokens;
        }
        ngx_log_error(NGX_LOG_DEBUG, r->connection->log, 0,
                      "[Decode Update Stats]  req %d prompt_tokens= %ui decoded_tokens= %ui; decode upstream %d num_tokens=%ui",
                      req->slot_index,
                      req->metrics.prompt_num_tokens,
                      req->metrics.decoded_tokens,
                      decode_idx,
                      (unsigned long long)g_state->decode_states[decode_idx].num_tokens);
    }
    else
    {
        req->metrics.ttft = ngx_current_msec - req->metrics.time_received;
        req->metrics.time_first_token = req->metrics.tpot = ngx_current_msec - req->metrics.time_to_decode;
        req->metrics.decoded_tokens++;

        ngx_uint_t decode_idx = req->decode_upstream_endpoint_idx;
        ngx_atomic_fetch_add(&g_state->decode_states[decode_idx].num_tokens, 1);

        ngx_log_error(NGX_LOG_DEBUG, r->connection->log, 0,
                      "[Decode Update Stats]  req %d prompt_tokens= %ui decoded_tokens= %ui; decode upstream %d num_tokens=%ui",
                      req->slot_index,
                      req->metrics.prompt_num_tokens,
                      req->metrics.decoded_tokens,
                      decode_idx,
                      (unsigned long long)g_state->decode_states[decode_idx].num_tokens);
    }

    req->metrics.time_last_reponse = ngx_current_msec;

    omni_upstream_decode_t *us = &g_state->decode_states[req->decode_upstream_endpoint_idx];
    // Update batch level statistics
    omni_batch_metrics_t *batch = &us->his.his[us->his.head];
    ngx_msec_t delta = ngx_current_msec - batch->last_response_receive_time;

    // Need a smarter value from statistics or work out by the number of tokens scheduled
    if (delta > 10)
    {
        // An new batch comes back
        if (us->his.count < NUM_PREFILL_BATCH_METRICS_HIS - 1)
        {
            us->his.count++;
        }

        us->his.head++;
        if (us->his.head == NUM_PREFILL_BATCH_METRICS_HIS)
        {
            us->his.head = 0;
        }

        batch = &us->his.his[us->his.head];
        ngx_memzero(batch, sizeof(omni_batch_metrics_t));

        batch->first_response_receive_time = ngx_current_msec;
        batch->last_response_receive_time = ngx_current_msec;
        batch->num_requests = 1;
        batch->num_tokens = req->metrics.prompt_num_tokens + req->metrics.decoded_tokens;
    }
    else
    {
        batch->average_delta = batch->average_delta * (batch->num_requests - 1) +
                               delta / (batch->num_requests);
        batch->last_response_receive_time = ngx_current_msec;
        batch->num_requests++;
        batch->num_tokens += req->metrics.prompt_num_tokens + req->metrics.decoded_tokens;
    }
}

static ngx_int_t ngx_http_omni_create_request(ngx_http_request_t *r)
{
    ngx_chain_t *cl;
    size_t body_len = 0;
    ngx_str_t host;
    omni_req_context_t *ctx = ngx_http_get_module_ctx(r, ngx_http_omni_proxy_module);

    omni_proxy_prepare_decode_request_body(r, ctx);

    if (r->request_body == NULL || r->request_body->bufs == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "ngx_http_omni_create_request: request body is NULL");
        return NGX_ERROR;
    }

    for (cl = r->request_body->bufs; cl; cl = cl->next)
    {
        body_len += ngx_buf_size(cl->buf);
    }

    if (r->headers_in.host && r->headers_in.host->value.len)
    {
        host = r->headers_in.host->value;
    }
    else
    {
        host = r->headers_in.server;
    }

    ngx_buf_t *hdr = ngx_create_temp_buf(r->pool, 256 + r->uri.len + host.len);
    if (hdr == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "ngx_http_omni_create_request: failed to create temp buffer for request header");
        return NGX_ERROR;
    }

    hdr->last = ngx_snprintf(hdr->last, hdr->end - hdr->last,
                             "POST %V HTTP/1.1\r\n", &r->uri);

    ngx_list_part_t *part = &r->headers_in.headers.part;
    ngx_table_elt_t *h = part->elts;
    ngx_uint_t i;

    for (;;)
    {
        for (i = 0; i < part->nelts; i++)
        {
            if (h[i].key.len == sizeof("Content-Length") - 1 &&
                ngx_strncasecmp(h[i].key.data,
                                (u_char *)"Content-Length",
                                h[i].key.len) == 0)
            {
                continue;
            }

            hdr->last = ngx_snprintf(hdr->last, hdr->end - hdr->last,
                                     "%V: %V\r\n", &h[i].key, &h[i].value);
        }

        if (part->next == NULL)
        {
            break;
        }
        part = part->next;
        h = part->elts;
    }

    hdr->last = ngx_snprintf(hdr->last, hdr->end - hdr->last,
                             "Content-Length: %O\r\n\r\n", (off_t)body_len);

    cl = ngx_alloc_chain_link(r->pool);
    if (cl == NULL)
    {
        return NGX_ERROR;
    }

    cl->buf = hdr;
    cl->next = r->request_body->bufs;
    r->upstream->request_bufs = cl;

    return NGX_OK;
}

static ngx_int_t ngx_http_omni_reinit_request(ngx_http_request_t *r)
{
    ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                  "ngx_http_omni_reinit_request(ngx_http_request_t *r): [Line %d]", __LINE__);
    return NGX_OK;
}

static ngx_int_t ngx_http_omni_process_header(ngx_http_request_t *r)
{
    ngx_int_t rc;
    ngx_http_upstream_t *u;
    ngx_table_elt_t *h;

    u = r->upstream;

    for (;;)
    {
        rc = ngx_http_parse_header_line(r, &u->buffer, 1);

        if (rc == NGX_OK)
        {
            h = ngx_list_push(&r->headers_out.headers);
            if (h == NULL)
            {
                ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                              "omni_proxy: push header failed.");
                return NGX_HTTP_INTERNAL_SERVER_ERROR;
            }

            h->key.len = r->header_name_end - r->header_name_start;
            h->key.data = ngx_pnalloc(r->pool, h->key.len);
            if (h->key.data == NULL)
            {
                return NGX_ERROR;
            }
            ngx_memcpy(h->key.data, r->header_name_start, h->key.len);

            h->value.len = r->header_end - r->header_start;
            h->value.data = ngx_pnalloc(r->pool, h->value.len);
            if (h->value.data == NULL)
            {
                return NGX_ERROR;
            }
            ngx_memcpy(h->value.data, r->header_start, h->value.len);

            h->lowcase_key = ngx_pnalloc(r->pool, h->key.len);
            if (h->lowcase_key == NULL)
            {
                return NGX_ERROR;
            }
            ngx_strlow(h->lowcase_key, h->key.data, h->key.len);

            h->hash = ngx_hash_key_lc(h->lowcase_key, h->key.len);

            continue;
        }

        if (rc == NGX_HTTP_PARSE_HEADER_DONE)
        {
            // Mark as processed to avoid forward to filter below
            // u->buffer.last = u->buffer.pos;

            ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                          "omni_process_header: finished parsing header");

            return NGX_OK;
        }

        if (rc == NGX_AGAIN)
        {
            return NGX_AGAIN;
        }

        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "omni_process_header: invalid header from upstream");
        return NGX_HTTP_UPSTREAM_INVALID_HEADER;
    }
}

static ngx_int_t ngx_http_omni_process_status_line(ngx_http_request_t *r)
{
    ngx_int_t rc;
    ngx_http_upstream_t *u = r->upstream;
    ngx_http_status_t st;
    ngx_memzero(&st, sizeof(st));

    rc = ngx_http_parse_status_line(r, &u->buffer, &st);

    if (rc == NGX_AGAIN)
    {
        return NGX_AGAIN;
    }

    if (rc == NGX_ERROR)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "omni_proxy: invalid status line from upstream");
        return NGX_HTTP_UPSTREAM_INVALID_HEADER;
    }

    u->headers_in.status_n = st.code;

    u->headers_in.status_line.len = st.end - st.start;
    u->headers_in.status_line.data = ngx_pnalloc(r->pool, u->headers_in.status_line.len);
    if (u->headers_in.status_line.data == NULL)
    {
        return NGX_ERROR;
    }
    ngx_memcpy(u->headers_in.status_line.data, st.start, u->headers_in.status_line.len);

    return ngx_http_omni_process_header(r);
}

static ngx_int_t ngx_http_omni_input_filter_init(void *data)
{
    return NGX_OK;
}

static ngx_int_t ngx_http_omni_input_filter(void *data, ssize_t bytes)
{
    ngx_http_request_t *r = data;
    ngx_http_upstream_t *u = r->upstream;

    ngx_log_error(NGX_LOG_DEBUG, r->connection->log, 0,
                  "ngx_http_omni_input_filter: bytes %lu", bytes);

    u->buffer.last = u->buffer.pos + bytes;
    if (u->buffer.pos != NULL && u->buffer.pos < u->buffer.last)
    {
        size_t len = u->buffer.last - u->buffer.pos;
        ngx_buf_t *b = ngx_create_temp_buf(r->pool, len);
        if (b == NULL)
        {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                          "ngx_http_omni_input_filter, create buf failed");
            return NGX_ERROR;
        }
        ngx_memcpy(b->pos, u->buffer.pos, len);
        b->last = b->pos + len;
        b->memory = 1;
        b->flush = 1;

        ngx_chain_t *cl = ngx_alloc_chain_link(r->pool);
        if (cl == NULL)
        {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                          "ngx_http_omni_input_filter, allocate chain failed");
            return NGX_ERROR;
        }
        cl->buf = b;
        cl->next = NULL;

        ngx_int_t rc = ngx_http_output_filter(r, cl);
        if (rc == NGX_ERROR)
        {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                          "ngx_http_omni_input_filter, putput failed");
            return NGX_ERROR;
        }
    }

    omni_proxy_update_decode_stats(r, &u->buffer, bytes);

    return NGX_OK;
}

static void ngx_http_omni_finalize_request(ngx_http_request_t *r, ngx_int_t rc)
{
}

static ngx_int_t ngx_http_omni_start_decode_upstream(ngx_http_request_t *r)
{
    omni_req_t *req = omni_get_req(r);
    ngx_http_omni_loc_conf_t *olcf;

    if (ngx_http_upstream_create(r) != NGX_OK)
    {
        return NGX_ERROR;
    }
    olcf = ngx_http_get_module_loc_conf(r, ngx_http_omni_proxy_module);
    if (olcf->upstream == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "omni_proxy: no upstream in loc conf");
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    ngx_http_upstream_t *u = r->upstream;

    u->conf = olcf->upstream->srv_conf[ngx_http_upstream_module.ctx_index];
    u->conf->buffer_size = 8192;
    u->conf->send_lowat = 0;

    u->create_request = ngx_http_omni_create_request;
    u->reinit_request = ngx_http_omni_reinit_request;
    u->process_header = ngx_http_omni_process_status_line;
    u->input_filter_init = ngx_http_omni_input_filter_init;
    u->input_filter = ngx_http_omni_input_filter;
    u->input_filter_ctx = r;
    u->finalize_request = ngx_http_omni_finalize_request;

    u->output.tag = (ngx_buf_tag_t)&ngx_http_omni_proxy_module;

    if (req->decode_upstream_endpoint_idx > g_state->num_decode_endpoints - 1)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "omni_proxy: decode upstream endpoint index %d out of bounds for configured size %d",
                      req->decode_upstream_endpoint_idx, g_state->num_decode_endpoints);
        return NGX_ERROR;
    }
    omni_upstream_address_t *addr = &g_state->decode_states[req->decode_upstream_endpoint_idx].address;

    u->resolved = ngx_pcalloc(r->pool, sizeof(ngx_http_upstream_resolved_t));
    if (u->resolved == NULL)
    {
        return NGX_ERROR;
    }

    struct sockaddr *sa = ngx_pcalloc(r->pool, addr->socklen);
    if (sa == NULL)
    {
        return NGX_ERROR;
    }
    ngx_memcpy(sa, &addr->sockaddr, addr->socklen);

    u->resolved->sockaddr = sa;
    u->resolved->port = addr->port;
    u->resolved->socklen = addr->socklen;
    u->resolved->naddrs = 1;

    u->resolved->host.len = addr->text_len;
    u->resolved->host.data = addr->text;

    ngx_http_upstream_init(r);

    return NGX_OK;
}

static ngx_int_t ngx_http_decode_wakeup(omni_req_t *req)
{
    ngx_http_request_t *r = omni_get_http_request(req);
    ngx_int_t rc = ngx_http_omni_start_decode_upstream(r);

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[Decode-%d]: wakeup", req->slot_index);

    omni_phase_transition_all(req, PHASE_DECODE_SCHEDULED, PHASE_DECODING);
    req->metrics.time_to_decode = ngx_current_msec;

    if (rc == NGX_OK)
    {
        /* ngx_http_upstream_init will drive request; return NGX_OK from post_subrequest */
        return NGX_OK;
    }

    ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                  "omni_proxy: start decode upstream failed.");
    ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
    return NGX_OK;
}

typedef ngx_int_t (*omni_run_handle_t)(omni_req_t *);

static void omni_proxy_run_group(int phase_from, omni_run_handle_t handle)
{
    omni_req_group_t *group = &local_state.groups[phase_from];

    for (uint32_t i = 0; i < group->watermark; ++i)
    {
        omni_req_info_t *info = &group->requests[i];
        if (info->in_use)
        {
            omni_req_t *req = omni_info_to_req(info);
            ngx_http_request_t *r = omni_get_http_request(req);
            ngx_int_t rc = handle(req);
            if (!(rc == NGX_OK || rc == NGX_DONE))
            {
                ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[Wakeup-%d] Failed: %d",
                              req->slot_index, rc);
            }
        }
    }
}

void print_summary()
{
    if (ngx_current_msec - g_state->last_summary < 10000)
    {
        return;
    }
    g_state->last_summary = ngx_current_msec;

    ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                  "Active requests: %d, prefill waiting: %d, prefill running: %d "
                  "decode waiting: %d, decode running: %d, pid: %d\n",
                  g_state->request_pool.num_requests,
                  g_state->groups[PHASE_PREFILL_WAITING_SCHEDULE].num_requests,
                  g_state->groups[PHASE_PREFILLING].num_requests,
                  g_state->groups[PHASE_DECODE_WAITING_SCHEDULE].num_requests,
                  g_state->groups[PHASE_DECODING].num_requests,
                  ngx_pid);
}
static void omni_update_local_waiting(omni_worker_local_state_t *local_state,
                                      omni_req_group_t *group,
                                      omni_proxy_request_phase_t to)
{
    for (uint32_t i = 0; i < group->watermark; ++i)
    {
        omni_req_info_t *info = &group->requests[i];
        omni_req_t *req = omni_info_to_req(info);
        if (req->in_use && omni_req_is_in_phase(req, to))
        {
            omni_remove_from_group_by_req_info(info, group);
            omni_add_req_to_group(req->slot_index,
                                  &local_state->groups[to]);
        }
    }

    omni_sort_compact_group(group);
}

// Global scheduler has changed the req->phase to PHASE_PREFILL_SCHEDULED, here to update local state
// to make sure local state is consistent with global state
static inline void omni_update_local_prefill_waiting(omni_worker_local_state_t *local_state)
{
    omni_update_local_waiting(local_state, &local_state->groups[PHASE_PREFILL_WAITING_SCHEDULE],
                              PHASE_PREFILL_SCHEDULED);
}

static inline void omni_update_local_decode_waiting(omni_worker_local_state_t *local_state)
{
    omni_update_local_waiting(local_state, &local_state->groups[PHASE_DECODE_WAITING_SCHEDULE],
                              PHASE_DECODE_SCHEDULED);
}

static void omni_proxy_schedule(omni_global_state_t *gs)
{
    if (omni_is_master_worker(gs))
    {
        ngx_shmtx_lock(&gs->shmtx);
        omni_proxy_schedule_prefill(gs);
        omni_proxy_schedule_decode(gs);
        ngx_shmtx_unlock(&gs->shmtx);
    }
}

static void omni_proxy_timer_handler(ngx_event_t *ev)
{
    omni_proxy_schedule(g_state);

    // Global state has moved on, local state needs to be updated
    omni_update_local_prefill_waiting(&local_state);
    omni_update_local_decode_waiting(&local_state);

    omni_proxy_run_group(PHASE_PREFILL_SCHEDULED, ngx_http_prefill_wakeup);
    omni_proxy_run_group(PHASE_DECODE_SCHEDULED, ngx_http_decode_wakeup);

    ngx_add_timer(&local_state.omni_proxy_timer_event, TIMER_INTERVAL);

    // print_summary();
}

static void omni_proxy_init_req_groups(omni_req_group_t groups[])
{
    for (int i = 0; i < PHASE_MAX; i++)
    {
        groups[i].phase = i;
    }
}

static ngx_int_t omni_proxy_global_state_init(ngx_shm_zone_t *zone, void *data)
{
    ngx_uint_t i;

    if (data)
    {
        zone->data = data;
        return NGX_OK;
    }

    if (zone->shm.addr == NULL)
    {
        return NGX_ERROR;
    }

    g_state = (omni_global_state_t *)zone->shm.addr;
    ngx_memzero(g_state, GLOBAL_STATE_SIZE);
    g_state->magic = 47;

    omni_proxy_init_req_groups(g_state->groups);

    ngx_shmtx_create(&g_state->shmtx, &g_state->lock,
                     (u_char *)"omni_proxy_lock");

    printf("shared memory initialed: %p\n", zone->shm.addr);

    return NGX_OK;
}

ngx_int_t omni_proxy_init_global_state(ngx_conf_t *cf)
{
    ngx_str_t name = ngx_string("omni_proxy_state");
    printf("Init global state with size:%ldK\n", GLOBAL_STATE_SIZE / 1024);

    ngx_shm_zone_t *zone = ngx_shared_memory_add(
        cf,
        &name,
        GLOBAL_STATE_SIZE,
        &ngx_http_omni_proxy_module);

    if (zone == NULL)
    {
        return NGX_ERROR;
    }

    zone->shm.log->data = cf->cycle;

    zone->init = omni_proxy_global_state_init;

    return NGX_OK;
}

static void *ngx_http_omni_create_loc_conf(ngx_conf_t *cf)
{
    ngx_http_omni_loc_conf_t *conf;

    conf = ngx_pcalloc(cf->pool, sizeof(ngx_http_omni_loc_conf_t));
    if (conf == NULL)
    {
        return NULL;
    }

    conf->upstream_name.data = NULL;
    conf->upstream_name.len = 0;

    conf->pd_policy = NGX_CONF_UNSET;
    conf->model_path.data = NULL;
    conf->model_path.len = 0;
    conf->kv_block_size = 128;

    conf->vllm_kv_port_offset = NGX_CONF_UNSET;

    return conf;
}

static char *ngx_http_omni_merge_loc_conf(ngx_conf_t *cf, void *parent, void *child)
{
    ngx_http_omni_loc_conf_t *prev = parent;
    ngx_http_omni_loc_conf_t *conf = child;

    ngx_conf_merge_str_value(conf->upstream_name, prev->upstream_name, "");

    if (conf->pd_policy == NGX_CONF_UNSET)
    {
        conf->pd_policy = (prev->pd_policy != NGX_CONF_UNSET) ? prev->pd_policy : NGX_CONF_UNSET;
    }

    ngx_conf_merge_str_value(conf->model_path, prev->model_path, "");

    if (conf->vllm_kv_port_offset == NGX_CONF_UNSET)
    {
        conf->vllm_kv_port_offset = (prev->vllm_kv_port_offset != NGX_CONF_UNSET) ? prev->vllm_kv_port_offset : NGX_CONF_UNSET;
    }

    if (conf->metrics_enabled == NGX_CONF_UNSET)
    {
        conf->metrics_enabled = (prev->metrics_enabled != NGX_CONF_UNSET) ? prev->metrics_enabled : NGX_CONF_UNSET;
    }

    if (conf->model_path.len != 0 || conf->vllm_kv_port_offset != NGX_CONF_UNSET || conf->pd_policy != NGX_CONF_UNSET)
    {
        local_state.loc_conf = conf;
    }

    return NGX_CONF_OK;
}

static ngx_int_t ngx_http_omni_init_upstreams()
{
    if (g_state->upstream_initialized)
    {
        return NGX_OK;
    }

    g_state->upstream_initialized = 1;

    ngx_uint_t i;
    ngx_http_upstream_main_conf_t *umcf = ngx_http_cycle_get_module_main_conf(ngx_cycle, ngx_http_upstream_module);

    if (umcf == NULL)
    {
        return NGX_OK;
    }

    ngx_array_t *upstreams = &umcf->upstreams;
    ngx_uint_t nupstreams = upstreams->nelts;
    ngx_http_upstream_srv_conf_t **uscfp = upstreams->elts;

    for (i = 0; i < nupstreams; i++)
    {
        ngx_http_upstream_srv_conf_t *uscf = uscfp[i];
        uscfp[i]->peer.init = omni_proxy_upstream_init;
        bool is_prefill = false;
        if (ngx_strncmp(uscf->host.data, PREFILL_ENDPOINTS, sizeof(PREFILL_ENDPOINTS) - 1) == 0)
        {
            g_state->num_prefill_endpoints = uscf->servers->nelts;
            is_prefill = true;
        }
        else if (ngx_strncmp(uscf->host.data, DECODE_ENDPOINTS, sizeof(DECODE_ENDPOINTS) - 1) == 0)
        {
            g_state->num_decode_endpoints = uscf->servers->nelts;
        }

        ngx_http_upstream_rr_peers_t *peers = uscf->peer.data;

        ngx_uint_t j;
        ngx_http_upstream_rr_peer_t *peer = peers->peer;
        for (j = 0; j < peers->number; j++)
        {
            omni_upstream_address_t *address = NULL;
            if (is_prefill)
            {
                g_state->prefill_states[j].index = j;
                address = &g_state->prefill_states[j].address;
            }
            else
            {
                g_state->decode_states[j].index = j;
                address = &g_state->decode_states[j].address;
            }
            address->socklen = peer[j].socklen;

            ngx_memcpy(&address->sockaddr, peer[j].sockaddr, peer[j].socklen);

            address->text_len = peer[j].name.len;
            ngx_memcpy(address->text, peer[j].name.data, peer[j].name.len);
            address->text[address->text_len] = '\0';
            struct sockaddr_in *ipv4 = (struct sockaddr_in *)&address->sockaddr;
            inet_ntop(AF_INET, &(ipv4->sin_addr), address->ip, UPSTREAM_IP_MAX);
            address->port = ntohs(ipv4->sin_port);
        }
    }

    return NGX_OK;
}

static ngx_int_t omni_proxy_apc_shm_initialized(ngx_shm_zone_t *shm_zone, void *data)
{
    if (local_state.num_prefill_endpoints > g_state->num_prefill_endpoints)
    {
        g_state->prefill_states[g_state->num_prefill_endpoints].radix_pool = (ngx_slab_pool_t *)shm_zone->shm.addr;
        g_state->num_prefill_endpoints++;
    }
    else if (local_state.num_decode_endpoints > g_state->num_decode_endpoints)
    {
        g_state->decode_states[g_state->num_decode_endpoints].radix_pool = (ngx_slab_pool_t *)shm_zone->shm.addr;
        g_state->num_decode_endpoints++;
    }
    else
    {
        return NGX_ERROR;
    }

    return NGX_OK;
}

static ngx_int_t omni_proxy_init_apc_shm(ngx_conf_t *cf)
{
    ngx_http_upstream_main_conf_t *umcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_upstream_module);
    ngx_http_upstream_srv_conf_t **uscfp;
    ngx_uint_t i, j;

    if (umcf == NULL)
    {
        return NGX_ERROR;
    }

    uscfp = umcf->upstreams.elts;

    for (i = 0; i < umcf->upstreams.nelts; i++)
    {
        ngx_http_upstream_srv_conf_t *uscf = uscfp[i];
        if (uscf == NULL)
        {
            continue;
        }

        if (ngx_strncmp(uscf->host.data, PREFILL_ENDPOINTS, sizeof(PREFILL_ENDPOINTS) - 1) == 0)
        {
            local_state.num_prefill_endpoints = uscf->servers->nelts;
        }
        else if (ngx_strncmp(uscf->host.data, DECODE_ENDPOINTS, sizeof(DECODE_ENDPOINTS) - 1) == 0)
        {
            local_state.num_decode_endpoints = uscf->servers->nelts;
        }

        int count = 0;
        if (uscf->servers)
        {
            ngx_http_upstream_server_t *servers = uscf->servers->elts;

            for (j = 0; j < uscf->servers->nelts; j++, count++)
            {
                ngx_http_upstream_server_t *server = &servers[j];
                ngx_shm_zone_t *shm_zone;
                ngx_str_t shm_name;

                u_char *buffer = ngx_pcalloc(cf->pool, 64);
                u_char *end = ngx_snprintf(buffer, sizeof(buffer), "omni_proxy_%d",
                                           count);
                *end = 0;

                shm_name.data = buffer;
                shm_name.len = end - buffer;

                shm_zone = ngx_shared_memory_add(cf, &server->name, 20 * 1024 * 1024,
                                                 &ngx_http_omni_proxy_module);
                if (shm_zone == NULL)
                {
                    ngx_conf_log_error(NGX_LOG_ERR, cf, 0,
                                       "Failed to create shared memory for server %V",
                                       &servers[j].addrs->name);
                    continue;
                }

                shm_zone->init = omni_proxy_apc_shm_initialized;

                ngx_conf_log_error(NGX_LOG_ERR, cf, 0,
                                   "    Created 20MB shared memory zone: %V",
                                   &shm_name);
            }
        }
    }
}

static ngx_int_t ngx_http_omni_proxy_metrics_handler(ngx_http_request_t *r)
{
    ngx_http_omni_loc_conf_t *plcf;
    ngx_int_t rc;
    ngx_buf_t *b;
    ngx_chain_t out;

    plcf = ngx_http_get_module_loc_conf(r, ngx_http_omni_proxy_module);

    if (!plcf || !plcf->metrics_enabled)
    {
        return NGX_DECLINED;
    }

    ngx_str_t metrics = omni_metrics_export(g_state);

    r->headers_out.status = NGX_HTTP_OK;
    r->headers_out.content_length_n = metrics.len;

    r->headers_out.content_type.len = sizeof("text/plain") - 1;
    r->headers_out.content_type.data = (u_char *)"text/plain";
    r->headers_out.content_type_len = sizeof("text/plain") - 1;

    rc = ngx_http_send_header(r);
    if (rc == NGX_ERROR || rc > NGX_OK)
    {
        return rc;
    }

    b = ngx_calloc_buf(r->pool);
    if (b == NULL)
    {
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    b->pos = metrics.data;
    b->last = metrics.data + metrics.len;
    b->memory = 1;
    b->last_buf = 1;
    b->last_in_chain = 1;

    out.buf = b;
    out.next = NULL;

    return ngx_http_output_filter(r, &out);
}

static ngx_int_t omni_proxy_post_config(ngx_conf_t *cf)
{
    if (omni_proxy_init_global_state(cf) != NGX_OK)
    {
        return NGX_ERROR;
    }

    omni_proxy_init_apc_shm(cf);

    omni_proxy_init_req_groups(local_state.groups);

    ngx_http_core_main_conf_t *cmcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_core_module);

    ngx_http_handler_pt *h = ngx_array_push(&cmcf->phases[NGX_HTTP_CONTENT_PHASE].handlers);
    if (h == NULL)
    {
        return NGX_ERROR;
    }
    *h = ngx_http_omni_proxy_metrics_handler;

    return NGX_OK;
}

static void ngx_omni_tokenizer_pipe_handler(ngx_event_t *ev)
{
    ngx_omni_tokenize_worker_t *worker = ev->data;
    uint32_t slot_id;
    ssize_t nread;

    while ((nread = read(worker->resp_pipe[OMNI_PIPE_READ], &slot_id, sizeof(slot_id))) > 0)
    {
        ngx_log_error(NGX_LOG_DEBUG, ev->log, 0, "Tokenize done for%u\n", slot_id);
        omni_req_t *req = omni_id_to_req(slot_id);
        if (req != NULL)
        {
            ngx_log_error(NGX_LOG_DEBUG, ev->log, 0,
                          "Tokenizer completed for slot: %ui, prompt: %s",
                          slot_id, req->tokenizer_req.prompt);

            req->metrics.time_tokenized = ngx_current_msec;
            printf("Tokenize %ld, time taken:%lu\n", req->tokenizer_req.input_len, ngx_current_msec - req->metrics.time_contents_received);
            print_tokenize_result(&req->tokenizer_req);

            omni_proxy_post_tokenized(req);
        }
        else
        {
            ngx_log_error(NGX_LOG_DEBUG, ev->log, 0,
                          "Receive illegal slot id from tokenizer thread: %u",
                          slot_id);
        }
    }

    if (nread == -1 && errno != EAGAIN && errno != EWOULDBLOCK)
    {
        ngx_log_error(NGX_LOG_ERR, ev->log, ngx_errno,
                      "Error reading from response pipe");
        exit(-1);
    }
}

static ngx_int_t omni_proxy_init_tokenizer_worker(ngx_cycle_t *cycle)
{
    ngx_connection_t *c;

    if (local_state.loc_conf->model_path.len == 0)
    {
        return NGX_OK;
    }

    local_state.tokenize_worker.model_path = local_state.loc_conf->model_path;
    local_state.tokenize_worker.kv_block_size = local_state.loc_conf->kv_block_size;

    if (omni_tokenizer_worker_init(cycle, &local_state.tokenize_worker) != NGX_OK)
    {
        ngx_log_error(NGX_LOG_EMERG, cycle->log, 0,
                      "Failed to initialize tokenizer worker");
        return NGX_ERROR;
    }

    c = ngx_get_connection(local_state.tokenize_worker.resp_pipe[OMNI_PIPE_READ], cycle->log);
    if (c == NULL)
    {
        omni_tokenizer_worker_exit(&local_state.tokenize_worker);
        ngx_log_error(NGX_LOG_EMERG, cycle->log, 0,
                      "Failed to get connection for response pipe");
        return NGX_ERROR;
    }

    if (ngx_nonblocking(local_state.tokenize_worker.resp_pipe[OMNI_PIPE_READ]) == -1)
    {
        ngx_free_connection(c);
        omni_tokenizer_worker_exit(&local_state.tokenize_worker);
        ngx_log_error(NGX_LOG_EMERG, cycle->log, ngx_errno,
                      "Failed to set response pipe non-blocking");
        return NGX_ERROR;
    }

    c->read->handler = ngx_omni_tokenizer_pipe_handler;
    c->read->data = &local_state.tokenize_worker;
    c->read->log = cycle->log;

    if (ngx_add_conn(c) != NGX_OK)
    {
        ngx_free_connection(c);
        omni_tokenizer_worker_exit(&local_state.tokenize_worker);
        ngx_log_error(NGX_LOG_EMERG, cycle->log, 0,
                      "Failed to add response pipe connection to event loop");
        return NGX_ERROR;
    }

    local_state.tokenize_worker.resp_connection = c;

    ngx_log_error(NGX_LOG_INFO, cycle->log, 0,
                  "Omni proxy process initialized with tokenizer worker, model: %V",
                  &local_state.loc_conf->model_path);

    g_state->has_tokenizer = true;
    return NGX_OK;
}

static void omni_proxy_kv_event_handler(struct omni_zmq_handler_s *handler,
                      const char *topic,
                      const void *message,
                      size_t length)
{
    KVEventBatch *batch = parse_kv_event_batch(message, length);
    if (!batch)
    {
        ngx_log_error(NGX_LOG_INFO, handler->log, 0, "Failed to parse KV event batch");
        return;
    }

    ngx_log_error(NGX_LOG_INFO, handler->log, 0, "[%ld - %ld] Received KV event batch with %ld events",
           handler->index, ngx_worker, batch->events_count);

    omni_upstream_prefill_t *prefill = &g_state->prefill_states[handler->index];
    // assert(!prefill->radix_tree==null)
    if (!prefill->radix_tree)
    {
        ngx_log_error(NGX_LOG_EMERG, handler->log, 0,
                      "omni_proxy_kv_event_handler: Radix tree for prefill %d is not initialized; shared state corrupted. Aborting.",
                      handler->index);

        free_kv_event_batch(batch);

        /* Best-effort unlock if global mutex held elsewhere; unlocking here reduces chance of leaving a mutex locked.
           Note: if g_state itself is corrupted this may be a no-op or unsafe, but we attempt to be tidy. */
        ngx_shmtx_unlock(&g_state->shmtx);

        /* Ensure failure is loud in both debug and release builds */
        assert(prefill->radix_tree != NULL && "omni_proxy_kv_event_handler: radix_tree for prefill not initialized");
        abort();
        return;
    }

    for (size_t i = 0; i < batch->events_count; i++)
    {
        KVCacheEvent *event = (KVCacheEvent *)batch->events[i];
        if (!event)
            continue;

        switch (event->type)
        {
        case KV_EVENT_BLOCK_STORED:
        {
            int64_t *block_hashes = event->data.block_stored.block_hashes;
            size_t block_hashes_count = event->data.block_stored.block_hashes_count;
            int64_t parent_block_hash = event->data.block_stored.parent_block_hash;
            if (block_hashes_count > 0 && block_hashes != NULL)
            {
                omni_radix_tree_add_chain(prefill->radix_tree,
                                          (uint64_t *)block_hashes,
                                          (ngx_uint_t)block_hashes_count);
            }
            break;
        }
        case KV_EVENT_BLOCK_REMOVED:
        {
            int64_t *hashes = event->data.block_removed.block_hashes;
            size_t count = event->data.block_removed.block_hashes_count;
            if (hashes && count > 0)
            {
                for (size_t k = 0; k < count; ++k)
                {
                    uint64_t h = (uint64_t)hashes[k];
                    if (omni_radix_tree_remove(prefill->radix_tree, h) == NGX_OK)
                    {
                        ngx_log_error(NGX_LOG_INFO, handler->log, 0,
                                      "Removed block hash %" PRIu64 " from radix tree (prefill %d)",
                                      h, handler->index);
                    }
                    else
                    {
                        ngx_log_error(NGX_LOG_DEBUG, handler->log, 0,
                                      "Block hash %" PRIu64 " not found in radix tree (prefill %d)",
                                      h, handler->index);
                    }
                }
            }
            break;
        }
        case KV_EVENT_ALL_BLOCKS_CLEARED:
        {
            // Re-initialize the radix tree for this upstream
            omni_radix_tree_destroy(prefill->radix_tree);
            prefill->radix_tree = omni_radix_tree_init(prefill->radix_pool);
            break;
        }
        default:
            break;
        }
    }

    free_kv_event_batch(batch);
}

static ngx_int_t omni_proxy_init_kv_listener(ngx_cycle_t *cycle)
{
    if (local_state.loc_conf->vllm_kv_port_offset == NGX_CONF_UNSET)
    {
        return NGX_OK;
    }

    ngx_core_conf_t *ccf = (ngx_core_conf_t *)ngx_get_conf(ngx_cycle->conf_ctx, ngx_core_module);
    for (int i = 0; i < g_state->num_prefill_endpoints; i++)
    {
        omni_upstream_prefill_t *prefill = &g_state->prefill_states[i];
        prefill->kv_handler.index = i;

        if (i % ccf->worker_processes == ngx_worker)
        {
            prefill->radix_tree = omni_radix_tree_init(prefill->radix_pool);
            if (prefill->radix_tree == NULL)
            {
                ngx_log_error(NGX_LOG_EMERG, cycle->log, 0,
                              "Failed to create radix hash_tree");
                return NGX_ERROR;
            }

            printf("Start radix tree unit test...\n");
            omni_radix_tree_test(prefill->radix_tree);
            printf("Radix tree unit test done.\n");

            u_char *buf = ngx_palloc(cycle->pool, 64);
            u_char *last = ngx_snprintf((u_char *)buf, 64, "%s:%d",
                                        prefill->address.ip,
                                        prefill->address.port + local_state.loc_conf->vllm_kv_port_offset);

            printf("address: %s\n", buf);
            ngx_str_t addr = {last - buf, buf};

            ngx_str_t topic = ngx_string("");
            omni_zmq_handler_init(cycle, &prefill->kv_handler, addr, topic, omni_proxy_kv_event_handler);
        }
    }
}

static ngx_int_t omni_proxy_init_process(ngx_cycle_t *cycle)
{

    local_state.pid = ngx_pid;
    local_state.worker = ngx_worker;

    local_state.omni_proxy_timer_event.handler = omni_proxy_timer_handler;
    local_state.omni_proxy_timer_event.log = cycle->log;
    local_state.omni_proxy_timer_event.data = NULL;

    if (ngx_http_omni_init_upstreams(cycle) != NGX_OK)
    {
        ngx_log_error(NGX_LOG_EMERG, cycle->log, 0,
                      "Failed to update global upstream states");
        return NGX_ERROR;
    }

    if (omni_proxy_init_tokenizer_worker(cycle) != NGX_OK)
    {
        ngx_log_error(NGX_LOG_EMERG, cycle->log, 0,
                      "Failed to initialize tokenizer worker");
        return NGX_ERROR;
    }

    omni_register_worker(g_state, &g_state->shmtx);

    printf("Init timer, pid: %u, worker: %lu, g_state:%p\n", ngx_pid, ngx_worker, g_state);

    ngx_add_timer(&local_state.omni_proxy_timer_event, TIMER_INTERVAL);

    omni_proxy_init_kv_listener(cycle);

    return NGX_OK;
}

static void omni_proxy_exit_process(ngx_cycle_t *cycle)
{
    if (local_state.omni_proxy_timer_event.timer_set)
    {
        ngx_del_timer(&local_state.omni_proxy_timer_event);
    }

    if (local_state.tokenize_worker.resp_connection != NULL)
    {
        ngx_del_conn(local_state.tokenize_worker.resp_connection, 0);
        ngx_free_connection(local_state.tokenize_worker.resp_connection);
    }

    if (local_state.loc_conf->model_path.len == 0)
    {
        omni_tokenizer_worker_exit(&local_state.tokenize_worker);
    }

    ngx_log_error(NGX_LOG_INFO, cycle->log, 0,
                  "Omni proxy process exited, tokenizer worker cleaned up");
}

static char *omni_proxy_init_conf(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_http_core_loc_conf_t *clcf = ngx_http_conf_get_module_loc_conf(cf, ngx_http_core_module);
    clcf->handler = omni_proxy_handler;

    ngx_memzero(&local_state, sizeof(omni_worker_local_state_t));

    ngx_http_omni_loc_conf_t *olcf = conf;
    ngx_str_t *value = cf->args->elts;

    if (cf->args->nelts != 2)
    {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0, "invalid number of arguments in \"%V\"", &cmd->name);
        return NGX_CONF_ERROR;
    }

    olcf->upstream_name = value[1];

    ngx_url_t u;
    ngx_memzero(&u, sizeof(ngx_url_t));
    u.url = value[1];
    u.no_resolve = 1;

    olcf->upstream = ngx_http_upstream_add(cf, &u, 0);
    if (olcf->upstream == NULL)
    {
        return NGX_CONF_ERROR;
    }

    return NGX_CONF_OK;
}

static char *ngx_http_omni_proxy_pd_policy(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_http_omni_loc_conf_t *olcf = conf;
    ngx_str_t *value = cf->args->elts;

    if (cf->args->nelts != 2)
    {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0, "invalid number of arguments in \"%V\"", &cmd->name);
        return NGX_CONF_ERROR;
    }

    if (ngx_strcmp(value[1].data, "sequential") == 0)
    {
        olcf->pd_policy = 0;
    }
    else if (ngx_strcmp(value[1].data, "parallel") == 0)
    {
        olcf->pd_policy = 1;
    }
    else
    {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0, "invalid policy \"%V\"", &value[1]);
        return NGX_CONF_ERROR;
    }

    return NGX_CONF_OK;
}

static char *ngx_http_omni_proxy_model_path(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_http_omni_loc_conf_t *olcf = conf;
    ngx_str_t *value = cf->args->elts;

    if (cf->args->nelts != 2)
    {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0, "invalid number of arguments in \"%V\"", &cmd->name);
        return NGX_CONF_ERROR;
    }

    olcf->model_path = value[1];
    return NGX_CONF_OK;
}

static char *omni_proxy_metrics(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_http_omni_loc_conf_t *plcf = conf;

    plcf->metrics_enabled = 1;
    return NGX_CONF_OK;
}

static ngx_command_t omni_proxy_commands[] = {

    {ngx_string("omni_proxy"),
     NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
     omni_proxy_init_conf,
     NGX_HTTP_LOC_CONF_OFFSET,
     0,
     NULL},

    {ngx_string("omni_proxy_metrics"),
     NGX_HTTP_LOC_CONF | NGX_CONF_FLAG, // Now takes an argument
     omni_proxy_metrics,
     NGX_HTTP_LOC_CONF_OFFSET,
     offsetof(ngx_http_omni_loc_conf_t, metrics_enabled),
     NULL},

    {ngx_string("omni_proxy_pd_policy"),
     NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
     ngx_http_omni_proxy_pd_policy,
     NGX_HTTP_LOC_CONF_OFFSET,
     0,
     NULL},

    {ngx_string("omni_proxy_model_path"),
     NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
     ngx_http_omni_proxy_model_path,
     NGX_HTTP_LOC_CONF_OFFSET,
     0,
     NULL},

    {ngx_string("omni_proxy_vllm_kv_port_offset"),
     NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
     ngx_conf_set_num_slot,
     NGX_HTTP_LOC_CONF_OFFSET,
     offsetof(ngx_http_omni_loc_conf_t, vllm_kv_port_offset),
     NULL},

    ngx_null_command};

static ngx_http_module_t omni_proxy_module_ctx = {
    NULL,                   // preconfiguration
    omni_proxy_post_config, // postconfiguration
    NULL,                   /* create main configuration */
    NULL,                   /* init main configuration */

    NULL, /* create server configuration */
    NULL, /* merge server configuration */

    ngx_http_omni_create_loc_conf, /* create location configuration */
    ngx_http_omni_merge_loc_conf   /* merge location configuration */
};

ngx_module_t ngx_http_omni_proxy_module = {
    NGX_MODULE_V1,
    &omni_proxy_module_ctx,  /* module context */
    omni_proxy_commands,     /* module directives */
    NGX_HTTP_MODULE,         /* module type */
    NULL,                    /* init master */
    NULL,                    /* init module */
    omni_proxy_init_process, /* init process */
    NULL,                    /* init thread */
    NULL,                    /* exit thread */
    omni_proxy_exit_process, /* exit process */
    NULL,                    /* exit master */
    NGX_MODULE_V1_PADDING,
};
