// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#include <omni_proxy.h>

ngx_int_t omni_proxy_save_origin_body(
    ngx_http_request_t *r,
    omni_req_context_t *ctx);

ngx_int_t omni_proxy_prepare_prefill_subrequest(
    ngx_http_request_t *r,
    ngx_http_request_t *sr,
    omni_req_context_t *ctx);

void omni_proxy_prepare_decode_request_body(
    ngx_http_request_t *r,
    omni_req_context_t *ctx);