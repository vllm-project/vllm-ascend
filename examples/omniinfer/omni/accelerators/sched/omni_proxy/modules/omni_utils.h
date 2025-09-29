
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#include <omni_shared_state.h>
#include <omni_proxy.h>

#define BIT(n) (1U << (n))

static inline void omni_req_enter_phase(omni_req_t *req, omni_proxy_request_phase_t phase)
{
    req->phase_state |= BIT(phase);
}

static inline void omni_req_leave_phase(omni_req_t *req, omni_proxy_request_phase_t phase)
{
    req->phase_state &= ~BIT(phase);
}

static inline int omni_req_is_in_phase(omni_req_t *req, omni_proxy_request_phase_t phase)
{
    return (req->phase_state & BIT(phase)) != 0;
}

static inline void omni_req_get_phases(omni_req_t *req, omni_proxy_request_phase_t *phases, size_t *count)
{
    *count = 0;
    for (omni_proxy_request_phase_t i = 0; i < PHASE_MAX; ++i)
    {
        if (omni_req_is_in_phase(req, i))
        {
            phases[*count] = i;
            (*count)++;
        }
    }
}

omni_req_t *omni_allocate_request(omni_request_pool_t *pool, void *data);

/* Free a previously‐allocated request slot */
void omni_free_request(omni_request_pool_t *pool, omni_req_t *req);

static inline omni_req_info_t *omni_add_req_to_group(uint32_t req_index, omni_req_group_t *group)
{
    assert(group->watermark < MAX_REQUEST_SLOTS);
    uint32_t idx = group->watermark++;
    group->requests[idx].in_use = 1;
    group->requests[idx].slot_index = req_index;
    group->requests[idx].weight = 0.0; // initial weight; user may adjust later
    group->num_requests++;

    return &group->requests[idx];
}

static inline void omni_remove_from_group_by_req_info(omni_req_info_t *req_info, omni_req_group_t *group)
{
    assert(req_info != NULL);
    req_info->in_use = 0;
    // keep the weight for debug
    // req_info->weight = -1;
    assert(group->num_requests > 0);
    group->num_requests--;
}

static int cmp_req_info_desc(const void *pa, const void *pb)
{
    const omni_req_info_t *a = pa, *b = pb;
    if (a->in_use != b->in_use)
        return (int)b->in_use - (int)a->in_use; // in_use=1 precedes in_use=0
    if (a->weight < b->weight)
        return 1;
    if (a->weight > b->weight)
        return -1;
    return 0;
}

static inline void omni_sort_compact_group(omni_req_group_t *group)
{
    if (group->watermark > 1 && group->num_requests)
    {
        qsort(group->requests,
              group->watermark,
              sizeof(omni_req_info_t),
              cmp_req_info_desc);
    }
    // now the first num_requests entries are in_use==1;
    // drop the rest
    group->watermark = group->num_requests;
}

static inline void omni_remove_req_from_group_by_req_index(uint32_t req_index, omni_req_group_t *group)
{
    for (uint32_t i = 0; i < group->watermark; ++i)
    {
        if (group->requests[i].in_use &&
            group->requests[i].slot_index == req_index)
        {
            omni_remove_from_group_by_req_info(&group->requests[i], group);
            // TODO: performance optimization required
            omni_sort_compact_group(group);
            return;
        }
    }
    // not found ⇒ user error
    assert(!"omni_remove_req_from_group: slot_index not in group");
}

static inline void omni_phase_change_to(
    omni_req_t *req, omni_req_group_t groups[],
    omni_proxy_request_phase_t from,
    omni_proxy_request_phase_t to)
{
    omni_remove_req_from_group_by_req_index(req->slot_index, &groups[from]);
    omni_add_req_to_group(req->slot_index, &groups[to]);
}

static inline void omni_register_worker(omni_global_state_t *gs, ngx_shmtx_t *g_shmtx)
{
    // need to protect by g_shmtx
    ngx_shmtx_lock(g_shmtx);

    for (size_t i = 0; i < MAX_WORKERS; ++i)
    {
        if (gs->workers[i] == 0)
        {
            gs->workers[i] = ngx_pid;
            gs->num_workers++;
            ngx_shmtx_unlock(g_shmtx);
            return;
        }
    }
    ngx_shmtx_unlock(g_shmtx);

    assert(!"No space left for new worker");
}

// The first work is the master and will do the global schduling
static inline int omni_is_master_worker(omni_global_state_t *gs)
{
    return gs->workers[0] == ngx_pid;
}

omni_global_state_t *omni_get_global_state();
omni_worker_local_state_t *omni_get_local_state();

static inline omni_req_t *omni_id_to_req(uint32_t slot)
{
    return &omni_get_global_state()->request_pool.slots[slot];
}

static inline omni_req_t *omni_info_to_req(omni_req_info_t *info)
{
    return omni_id_to_req(info->slot_index);
}

static inline ngx_http_request_t *omni_get_http_request(omni_req_t *req)
{
    ngx_http_request_t *r = req->backend;
    if (r == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, ngx_cycle->log, 0, "Request backend is NULL for req %d", req->slot_index);
        return NULL;
    }
    return r;
}

static inline void omni_global_phase_change_to(omni_req_t *req, omni_proxy_request_phase_t from, omni_proxy_request_phase_t to)
{
    // printf("[Phase-%d]: Global from: %d To: %d.\n", req->slot_index, from, to);
    omni_phase_change_to(req, omni_get_global_state()->groups, from, to);
}

static inline void omni_local_phase_change_to(omni_req_t *req, omni_proxy_request_phase_t from, omni_proxy_request_phase_t to)
{
    // printf("[Phase-%d]: Local from: %d To: %d.\n", req->slot_index, from, to);
    omni_phase_change_to(req, omni_get_local_state()->groups, from, to);
}
