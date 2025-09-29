// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <omni_proxy.h>
#include <omni_scheduler.h>
#include <omni_utils.h>

static void update_prefill_weights(omni_req_group_t *group)
{
    uint32_t max_prompt_tokens = 0;
    ngx_msec_t max_wait_time = 0;
    for (uint32_t i = 0; i < group->watermark; i++)
    {
        omni_req_info_t *info = &group->requests[i];
        if (!info->in_use)
        {
            continue;
        }
        omni_req_t *req = omni_info_to_req(info);

        if (max_prompt_tokens < req->metrics.prompt_num_tokens)
        {
            max_prompt_tokens = req->metrics.prompt_num_tokens;
        }

        ngx_msec_t waited = ngx_current_msec - req->metrics.time_received;

        if (max_wait_time < waited)
        {
            max_wait_time = waited;
        }
    }

    if (max_wait_time < 50)
    {
        max_wait_time = 50;
    }

    for (uint32_t i = 0; i < group->watermark; i++)
    {
        omni_req_info_t *info = &group->requests[i];
        if (!info->in_use)
        {
            continue;
        }
        omni_req_t *req = omni_info_to_req(info);
        ngx_msec_t waited = ngx_current_msec - req->metrics.time_received;

        double token_weight = (double)(max_prompt_tokens - req->metrics.prompt_num_tokens) / max_prompt_tokens;
        double time_weight = (double)waited / max_wait_time;

        info->weight = token_weight * 0.8 + time_weight * 0.2;
    }

    omni_sort_compact_group(group);

    for (uint32_t idx = 0; idx < group->num_requests; idx++)
    {
        omni_req_info_t *info = &group->requests[idx];
        if (!info->in_use)
        {
            continue;
        }
        omni_req_t *req = omni_info_to_req(info);
        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                      "[Prefill-Sort] Order %ui: slot=%ui tokens=%ui weight=%.2f",
                      idx,
                      info->slot_index,
                      req->metrics.prompt_num_tokens,
                      info->weight);
    }
}

static void update_decode_weights(omni_req_group_t *group)
{
    uint32_t max_total_tokens = 1;

    for (uint32_t i = 0; i < group->watermark; i++) {
        omni_req_info_t *info = &group->requests[i];
        if (!info->in_use) continue;
        omni_req_t *req = omni_info_to_req(info);
        if ((req->metrics.prompt_num_tokens + req->metrics.max_tokens) > max_total_tokens) {
            max_total_tokens = req->metrics.prompt_num_tokens + req->metrics.max_tokens;
        }
    }

    for (uint32_t i = 0; i < group->watermark; i++) {
        omni_req_info_t *info = &group->requests[i];
        if (!info->in_use) continue;
        omni_req_t *req = omni_info_to_req(info);
        info->weight = ((double)req->metrics.prompt_num_tokens + (double)req->metrics.max_tokens)/ max_total_tokens;
    }

    omni_sort_compact_group(group);

    for (uint32_t idx = 0; idx < group->num_requests; idx++) {
        omni_req_info_t *info = &group->requests[idx];
        if (!info->in_use) {
            continue;
        }
        omni_req_t *req = omni_info_to_req(info);
        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                      "[Decode-Sort] Order %ui: slot=%ui total_tokens=%ui prompt_num_tokens=%ui max_tokens=%ui weight=%.2f",
                      idx,
                      info->slot_index,
                      req->metrics.prompt_num_tokens + req->metrics.max_tokens,
                      req->metrics.prompt_num_tokens,
                      req->metrics.max_tokens,
                      info->weight);
    }
}

void omni_proxy_schedule_prefill(omni_global_state_t *gs)
{
    omni_req_group_t *group = &gs->groups[PHASE_PREFILL_WAITING_SCHEDULE];

    // TODO: Check should schedule or wait based on upstream expected come back time

    update_prefill_weights(group);

    for (uint32_t i = 0; i < group->num_requests; i++)
    {
        omni_req_info_t *info = &group->requests[i];
        omni_req_t *req = omni_info_to_req(info);

        assert(omni_req_is_in_phase(req, PHASE_PREFILL_WAITING_SCHEDULE));

        // uint32_t least_load = UINT32_MAX;
        uint32_t selected = UINT32_MAX;
        uint32_t best_match = 0;
        uint32_t best_load_tokens = UINT32_MAX;
        uint32_t best_running = UINT32_MAX;
        uint32_t best_idx = UINT32_MAX;

        for (uint32_t j = 0; j < gs->num_prefill_endpoints; j++)
        {
            uint32_t m = 0;

            m = req->match_depths[j];

            uint32_t load_tokens = gs->prefill_states[j].num_tokens;
            uint32_t running = gs->prefill_states[j].num_running;
            if (load_tokens > 30000 || running > 32)//overload
                continue;
            if (m > best_match ||
                (m == best_match && load_tokens < best_load_tokens) ||
                (m == best_match && load_tokens == best_load_tokens && running < best_running))
            {
                best_match = m;
                best_load_tokens = load_tokens;
                best_running = running;
                best_idx = j;
            }
        }

        if (best_match > 0 && best_idx != UINT32_MAX)
        {
            selected = best_idx;
        }
        else
        {
            uint32_t least_load = UINT32_MAX;
            for (uint32_t m = gs->last_selected_prefill;
                 m < gs->num_prefill_endpoints + gs->last_selected_prefill;
                 m++)
            {
                uint32_t j = m % gs->num_prefill_endpoints;
                if (gs->prefill_states[j].num_tokens < least_load)
                {
                    least_load = gs->prefill_states[j].num_tokens;
                    selected = j;
                    if (least_load == 0)
                    {
                        break;
                    }
                }
            }
        }

        req->prefill_upstream_endpoint_idx = selected;
        gs->last_selected_prefill = selected + 1;
        gs->prefill_states[selected].num_running++;
        gs->prefill_states[selected].num_tokens += req->metrics.prompt_num_tokens;

        omni_global_phase_change_to(req, PHASE_PREFILL_WAITING_SCHEDULE, PHASE_PREFILL_SCHEDULED);
        omni_req_leave_phase(req, PHASE_PREFILL_WAITING_SCHEDULE);
        omni_req_enter_phase(req, PHASE_PREFILL_SCHEDULED);

        // If policy is parallel, we can change to DECODE_SCHEDULED directly
        if (gs->pd_policy == PD_PARALLEL)
        {
            req->decode_upstream_endpoint_idx = 0;
            gs->decode_states[selected].num_running++;

            omni_add_req_to_group(req->slot_index, &gs->groups[PHASE_DECODE_SCHEDULED]);
            omni_req_enter_phase(req, PHASE_DECODE_SCHEDULED);
        }

        req->metrics.time_prefill_scheduled = ngx_current_msec;

        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0, "[Prefill-%d] Schedule to: %d",
                      req->slot_index, req->prefill_upstream_endpoint_idx);
    }

    // TODO: estimated expected next schedule time
}

void omni_proxy_schedule_decode(omni_global_state_t *gs)
{
    omni_req_group_t *group = &gs->groups[PHASE_DECODE_WAITING_SCHEDULE];
    // TODO: Check should schedule or wait based on upstream expected come back time
    // TODO: Here we can do some estimation of pull kv finish time to make sure pull kv
    // workloads are balanced

    update_decode_weights(group);

    for (size_t i = 0; i < group->watermark; i++)
    {
        omni_req_info_t *info = &group->requests[i];
        if (!info->in_use)
        {
            continue;
        }
        omni_req_t *req = omni_info_to_req(info);
        assert(omni_req_is_in_phase(req, PHASE_DECODE_WAITING_SCHEDULE));

        uint32_t least_load = UINT32_MAX;
        uint32_t selected = UINT32_MAX;
        for (int m = gs->last_selected_decode; m < gs->num_decode_endpoints + gs->last_selected_decode; m++)
        {
            int j = m % gs->num_decode_endpoints;
            if (gs->decode_states[j].num_tokens < least_load)
            {
                least_load = gs->decode_states[j].num_tokens;
                selected = j;
                if (least_load == 0)
                {
                    break;
                }
            }
        }

        req->decode_upstream_endpoint_idx = selected;
        gs->last_selected_decode = selected + 1;
        gs->decode_states[selected].num_running++;
        gs->decode_states[selected].num_tokens += req->metrics.prompt_num_tokens;

        omni_global_phase_change_to(req, PHASE_DECODE_WAITING_SCHEDULE, PHASE_DECODE_SCHEDULED);
        omni_req_leave_phase(req, PHASE_DECODE_WAITING_SCHEDULE);
        omni_req_enter_phase(req, PHASE_DECODE_SCHEDULED);

        req->metrics.time_decode_scheduled = ngx_current_msec;

        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                      "[Decode-%d] Schedule to: %d (load=%ui)",
                      req->slot_index,
                      req->decode_upstream_endpoint_idx,
                      gs->decode_states[selected].num_tokens);
    }
}