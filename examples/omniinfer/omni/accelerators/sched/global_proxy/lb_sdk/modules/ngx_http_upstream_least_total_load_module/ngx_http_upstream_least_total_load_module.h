// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#ifndef NGX_HTTP_UPSTREAM_LEAST_TOTAL_LOAD_MODULE_H
#define NGX_HTTP_UPSTREAM_LEAST_TOTAL_LOAD_MODULE_H
#include <ngx_atomic.h>

typedef struct {
    ngx_atomic_t total_length_sum;
    ngx_atomic_t total_request_sum;
} ngx_http_least_total_load_shm_peer_t;

typedef struct {
    ngx_uint_t peer_count;
    ngx_http_least_total_load_shm_peer_t peers[1];
} prefill_upstream_info_t;

double least_total_load_get_score(
    ngx_uint_t length_sum_workers, ngx_uint_t request_sum_workers,
    double least_total_load_batch_size);

ngx_int_t least_total_load_select_solver(
    prefill_upstream_info_t *prefill_shm, ngx_uint_t worker_num,
    ngx_uint_t req_length, ngx_uint_t *chosen);

#endif // NGX_HTTP_UPSTREAM_LEAST_TOTAL_LOAD_MODULE_H