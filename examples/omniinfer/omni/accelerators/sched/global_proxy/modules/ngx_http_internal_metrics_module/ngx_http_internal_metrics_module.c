// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>

#include "../ngx_http_metrics_common.h"

#define NGX_HTTP_INTERNAL_METRICS_URI "/internal/metrics"

static const char *ngx_http_internal_metrics_format =
    "# HELP vllm:requests_success_total Total number of successful global proxy requests\n"
    "# TYPE vllm:requests_success_total counter\n"
    "vllm:requests_success_total %uA\n"
    "# HELP vllm:requests_failure_total Total number of failed global proxy requests\n"
    "# TYPE vllm:requests_failure_total counter\n"
    "vllm:requests_failure_total %uA\n"
    "# HELP vllm:time_to_first_token_seconds Time to first token histogram\n"
    "# TYPE vllm:time_to_first_token_seconds histogram\n"
    "%s"
    "vllm:time_to_first_token_seconds_sum{engine=\"0\",model_name=\"%s\"} %.3f\n"
    "vllm:time_to_first_token_seconds_count{engine=\"0\",model_name=\"%s\"} %uA\n"
    "# HELP vllm:time_per_output_token_seconds Time per output token histogram\n"
    "# TYPE vllm:time_per_output_token_seconds histogram\n"
    "%s"
    "vllm:time_per_output_token_seconds_sum{engine=\"0\",model_name=\"%s\"} %.3f\n"
    "vllm:time_per_output_token_seconds_count{engine=\"0\",model_name=\"%s\"} %uA\n"
    "# HELP vllm:e2e_request_latency_seconds End-to-end request latency histogram\n"
    "# TYPE vllm:e2e_request_latency_seconds histogram\n"
    "%s"
    "vllm:e2e_request_latency_seconds_sum{engine=\"0\",model_name=\"%s\"} %.3f\n"
    "vllm:e2e_request_latency_seconds_count{engine=\"0\",model_name=\"%s\"} %uA\n";

typedef struct {
    ngx_slab_pool_t *shpool;
    ngx_http_metrics_data_t *metrics;
} ngx_http_internal_metrics_loc_conf_t;

static ngx_int_t ngx_http_internal_metrics_handler(ngx_http_request_t *r);
static void *ngx_http_internal_metrics_create_loc_conf(ngx_conf_t *cf);
static char *ngx_http_internal_metrics_merge_loc_conf(ngx_conf_t *cf, void *parent, void *child);
static ngx_int_t ngx_http_internal_metrics_init(ngx_conf_t *cf);
static ngx_str_t ngx_http_internal_metrics_generate_histogram_buckets(ngx_http_request_t *r,
                                                                    const char *metric_name,
                                                                    ngx_atomic_t *buckets,
                                                                    ngx_uint_t buckets_count,
                                                                    const ngx_uint_t *bucket_definitions,
                                                                    const char *log_histogram_name,
                                                                    const char *model_name);

static ngx_command_t ngx_http_internal_metrics_commands[] = {
    ngx_null_command
};

static ngx_http_module_t ngx_http_internal_metrics_module_ctx = {
    NULL,                                        /* preconfiguration */
    ngx_http_internal_metrics_init,              /* postconfiguration */

    NULL,                                        /* create main configuration */
    NULL,                                        /* init main configuration */

    NULL,                                        /* create server configuration */
    NULL,                                        /* merge server configuration */

    ngx_http_internal_metrics_create_loc_conf,   /* create location configuration */
    ngx_http_internal_metrics_merge_loc_conf     /* merge location configuration */
};

ngx_module_t ngx_http_internal_metrics_module = {
    NGX_MODULE_V1,
    &ngx_http_internal_metrics_module_ctx,       /* module context */
    ngx_http_internal_metrics_commands,          /* module directives */
    NGX_HTTP_MODULE,                             /* module type */
    NULL,                                        /* init master */
    NULL,                                        /* init module */
    NULL,                                        /* init process */
    NULL,                                        /* init thread */
    NULL,                                        /* exit thread */
    NULL,                                        /* exit process */
    NULL,                                        /* exit master */
    NGX_MODULE_V1_PADDING
};

static ngx_int_t ngx_http_internal_metrics_handler(ngx_http_request_t *r)
{
    ngx_buf_t    *b;
    ngx_chain_t   out;
    ngx_str_t     metrics_response;
    ngx_str_t     time_to_first_token_buckets_str, time_per_output_token_buckets_str, e2e_request_latency_buckets_str;
    ngx_shm_zone_t *shm_zone;
    ngx_http_metrics_data_t *metrics = NULL;
    ngx_atomic_t success_count = 0, failure_count = 0;
    ngx_atomic_t time_to_first_token_sum = 0, time_to_first_token_count = 0;
    ngx_atomic_t time_per_output_token_sum = 0, time_per_output_token_count = 0; 
    ngx_atomic_t e2e_request_latency_sum = 0, e2e_request_latency_count = 0;
    ngx_uint_t i;

    ngx_log_debug2(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "internal metrics: checking URI '%V' (len=%uz)", &r->uri, r->uri.len);

    // Only handle requests to /internal/metrics
    if (r->uri.len != sizeof(NGX_HTTP_INTERNAL_METRICS_URI) - 1 ||
        ngx_strncmp(r->uri.data, NGX_HTTP_INTERNAL_METRICS_URI, sizeof(NGX_HTTP_INTERNAL_METRICS_URI) - 1) != 0) {
        ngx_log_debug1(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                       "internal metrics: declining URI '%V'", &r->uri);
        return NGX_DECLINED;
    }

    if (!(r->method & (NGX_HTTP_GET|NGX_HTTP_HEAD))) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "internal metrics: method not allowed for URI '%V', method: %ui", &r->uri, r->method);
        return NGX_HTTP_NOT_ALLOWED;
    }

    ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, "internal metrics handler");

    // Find the shared memory zone for global proxy metrics
    ngx_list_part_t *part = &ngx_cycle->shared_memory.part;
    shm_zone = part->elts;

    ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "internal metrics: searching for shared memory zone");

    for (i = 0; /* void */; i++) {
        if (i >= part->nelts) {
            if (part->next == NULL) {
                break;
            }
            part = part->next;
            shm_zone = part->elts;
            i = 0;
        }

        ngx_log_debug2(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                       "internal metrics: checking shm zone '%V' (len=%uz)",
                       &shm_zone[i].shm.name, shm_zone[i].shm.name.len);

        if (shm_zone[i].shm.name.len == sizeof(NGX_HTTP_METRICS_SHM_NAME) - 1 &&
            ngx_strncmp(shm_zone[i].shm.name.data, NGX_HTTP_METRICS_SHM_NAME,
                       sizeof(NGX_HTTP_METRICS_SHM_NAME) - 1) == 0) {

            ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                           "internal metrics: found matching shared memory zone");

            // Found the global proxy metrics zone, get metrics from shared memory
            if (shm_zone[i].data) {
                ngx_slab_pool_t *shpool = (ngx_slab_pool_t *) shm_zone[i].shm.addr;
                if (shpool && shpool->data) {
                    metrics = (ngx_http_metrics_data_t *) shpool->data;
                    success_count = metrics->success_count;
                    failure_count = metrics->failure_count;
                    time_to_first_token_sum = metrics->ttft_sum;
                    time_to_first_token_count = metrics->ttft_count;
                    time_per_output_token_sum = metrics->tpot_sum;
                    time_per_output_token_count = metrics->tpot_count;
                    e2e_request_latency_sum = metrics->e2e_latency_sum;
                    e2e_request_latency_count = metrics->e2e_latency_count;

                    ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                                   "internal metrics: successfully loaded metrics from shared memory");
                } else {
                    ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                                   "internal metrics: shared memory pool or data is null");
                }
            } else {
                ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                               "internal metrics: shared memory zone data is null");
            }
            break;
        }
    }

    if (metrics == NULL) {
        ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                       "internal metrics: no metrics data found, using default values");
    }

    // Get model name for labels, default to "unknown" if not available
    const char *model_name = "unknown";
    if (metrics && metrics->model_name_len > 0) {
        model_name = (const char *)metrics->model_name;
    }

    // Generate histogram bucket strings
    time_to_first_token_buckets_str = ngx_http_internal_metrics_generate_histogram_buckets(r, 
                                                           "vllm:time_to_first_token_seconds",
                                                           metrics ? metrics->ttft_buckets : NULL,
                                                           NGX_HTTP_TTFT_BUCKETS_COUNT,
                                                           ngx_http_metrics_ttft_buckets,
                                                           "TTFT histogram",
                                                           model_name);
    if (time_to_first_token_buckets_str.data == NULL) {
        time_to_first_token_buckets_str.data = (u_char *)"";
        time_to_first_token_buckets_str.len = 0;
    }

    time_per_output_token_buckets_str = ngx_http_internal_metrics_generate_histogram_buckets(r, 
                                                           "vllm:time_per_output_token_seconds",
                                                           metrics ? metrics->tpot_buckets : NULL,
                                                           NGX_HTTP_TPOT_BUCKETS_COUNT,
                                                           ngx_http_metrics_tpot_buckets,
                                                           "TPOT histogram",
                                                           model_name);
    if (time_per_output_token_buckets_str.data == NULL) {
        time_per_output_token_buckets_str.data = (u_char *)"";
        time_per_output_token_buckets_str.len = 0;
    }

    e2e_request_latency_buckets_str = ngx_http_internal_metrics_generate_histogram_buckets(r, 
                                                           "vllm:e2e_request_latency_seconds",
                                                           metrics ? metrics->e2e_latency_buckets : NULL,
                                                           NGX_HTTP_E2E_LATENCY_BUCKETS_COUNT,
                                                           ngx_http_metrics_e2e_latency_buckets,
                                                           "E2E Latency histogram",
                                                           model_name);
    if (e2e_request_latency_buckets_str.data == NULL) {
        e2e_request_latency_buckets_str.data = (u_char *)"";
        e2e_request_latency_buckets_str.len = 0;
    }

    // Generate Prometheus-style metrics response
    ngx_log_debug6(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "internal metrics: format args - success:%uA, failure:%uA, "
                   "time_to_first_token_sum:%uA, time_to_first_token_count:%uA, "
                   "time_per_output_token_sum:%uA, time_per_output_token_count:%uA",
                   success_count, failure_count, time_to_first_token_sum, 
                   time_to_first_token_count, time_per_output_token_sum, time_per_output_token_count);

    ngx_log_debug2(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "internal metrics: histogram strings - time_to_first_token_len:%uz, tpot_len:%uz",
                   time_to_first_token_buckets_str.len, time_per_output_token_buckets_str.len);

    // Calculate response length manually for better reliability
    size_t static_content_len = ngx_strlen(ngx_http_internal_metrics_format) - 8; // subtract format specifiers
    size_t numbers_len = 6 * 20; // 6 numbers, max 20 chars each (very conservative)
    size_t histograms_len = time_to_first_token_buckets_str.len + time_per_output_token_buckets_str.len + 
                           e2e_request_latency_buckets_str.len;

    metrics_response.len = static_content_len + numbers_len + histograms_len + 100; // extra buffer for safety

    ngx_log_debug4(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "internal metrics: length calculation - static:%uz, numbers:%uz, histograms:%uz, total:%uz",
                   static_content_len, numbers_len, histograms_len, metrics_response.len);

    if (metrics_response.len == 0) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "internal metrics: generated response length is zero");
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    metrics_response.data = ngx_pnalloc(r->pool, metrics_response.len);
    if (metrics_response.data == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "internal metrics: failed to allocate %uz bytes for response data", metrics_response.len);
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    // Generate the actual response content
    u_char *end = ngx_snprintf(metrics_response.data, metrics_response.len,
        ngx_http_internal_metrics_format,
        success_count, failure_count,
        time_to_first_token_buckets_str.data,
        model_name, time_to_first_token_sum/1000.0, model_name, time_to_first_token_count,
        time_per_output_token_buckets_str.data,
        model_name, time_per_output_token_sum/1000.0, model_name, time_per_output_token_count,
        e2e_request_latency_buckets_str.data,
        model_name, e2e_request_latency_sum/1000.0, model_name, e2e_request_latency_count);

    // Update the actual length used
    metrics_response.len = end - metrics_response.data;

    ngx_log_debug1(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "internal metrics: actual response length: %uz", metrics_response.len);

    r->headers_out.status = NGX_HTTP_OK;
    r->headers_out.content_length_n = metrics_response.len;

    // Set content type to text/plain for Prometheus compatibility
    ngx_str_set(&r->headers_out.content_type, "text/plain; charset=utf-8");
    r->headers_out.content_type_lowcase = NULL;

    if (r->method == NGX_HTTP_HEAD) {
        return ngx_http_send_header(r);
    }

    b = ngx_pcalloc(r->pool, sizeof(ngx_buf_t));
    if (b == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "internal metrics: failed to allocate buffer");
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    b->pos = metrics_response.data;
    b->last = metrics_response.data + metrics_response.len;
    b->memory = 1;
    b->last_buf = 1;

    out.buf = b;
    out.next = NULL;

    ngx_int_t rc = ngx_http_send_header(r);
    if (rc == NGX_ERROR || rc > NGX_OK || r->header_only) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "internal metrics: failed to send header, rc: %i", rc);
        return rc;
    }

    return ngx_http_output_filter(r, &out);
}

static void *ngx_http_internal_metrics_create_loc_conf(ngx_conf_t *cf)
{
    ngx_http_internal_metrics_loc_conf_t *conf;

    conf = ngx_pcalloc(cf->pool, sizeof(ngx_http_internal_metrics_loc_conf_t));
    if (conf == NULL) {
        return NULL;
    }

    conf->shpool = NULL;
    conf->metrics = NULL;

    return conf;
}

static char *ngx_http_internal_metrics_merge_loc_conf(ngx_conf_t *cf, void *parent, void *child)
{
    ngx_http_internal_metrics_loc_conf_t *prev = parent;
    ngx_http_internal_metrics_loc_conf_t *conf = child;

    if (conf->shpool == NULL) {
        conf->shpool = prev->shpool;
    }

    if (conf->metrics == NULL) {
        conf->metrics = prev->metrics;
    }

    return NGX_CONF_OK;
}

static ngx_int_t ngx_http_internal_metrics_init(ngx_conf_t *cf)
{
    ngx_http_handler_pt        *h;
    ngx_http_core_main_conf_t  *cmcf;

    cmcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_core_module);

    h = ngx_array_push(&cmcf->phases[NGX_HTTP_CONTENT_PHASE].handlers);
    if (h == NULL) {
        return NGX_ERROR;
    }

    *h = ngx_http_internal_metrics_handler;

    return NGX_OK;
}

static ngx_str_t ngx_http_internal_metrics_generate_histogram_buckets(ngx_http_request_t *r,
                                                                    const char *metric_name,
                                                                    ngx_atomic_t *buckets,
                                                                    ngx_uint_t buckets_count,
                                                                    const ngx_uint_t *bucket_definitions,
                                                                    const char *log_histogram_name,
                                                                    const char *model_name)
{
    ngx_str_t result;
    u_char *p;
    size_t len = 0;
    ngx_uint_t i;
    ngx_atomic_t bucket_value;
    size_t model_name_len = ngx_strlen(model_name);

    // Calculate required length for buckets manually
    size_t metric_name_len = ngx_strlen(metric_name);
    for (i = 0; i < buckets_count; i++) {
        if (i < buckets_count - 1) {
            // Format: metric_name_bucket{engine="0",model_name="model_name",le="N"} value\n
            len += metric_name_len + 35 + model_name_len + 40 + 10; // bucket format + buffer
        } else {
            // Format: metric_name_bucket{engine="0",model_name="model_name",le="+Inf"} value\n
            len += metric_name_len + 40 + model_name_len + 40 + 10; // +Inf bucket format + buffer
        }
    }

    ngx_log_debug3(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "internal metrics: %s calculated length: %uz for %ui buckets", 
                   log_histogram_name, len, buckets_count);

    result.data = ngx_pnalloc(r->pool, len);
    if (result.data == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "internal metrics: failed to allocate %uz bytes for %s buckets", len, log_histogram_name);
        result.len = 0;
        return result;
    }

    p = result.data;
    for (i = 0; i < buckets_count; i++) {
        bucket_value = buckets ? buckets[i] : 0;
        if (i < buckets_count - 1) {
            // Format bucket threshold, using integer format when possible
            double threshold = bucket_definitions[i] / 1000.0;
            p = ngx_snprintf(p, len - (p - result.data),
                            "%s_bucket{engine=\"0\",model_name=\"%s\",le=\"%.3f\"} %uA\n",
                            metric_name, model_name, threshold, bucket_value);

        } else {
            p = ngx_snprintf(p, len - (p - result.data),
                           "%s_bucket{engine=\"0\",model_name=\"%s\",le=\"+Inf\"} %uA\n",
                           metric_name, model_name, bucket_value);
        }
    }

    // Update with actual length used
    result.len = p - result.data;

    ngx_log_debug3(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "internal metrics: %s actual length: %uz (allocated: %uz)", log_histogram_name, result.len, len);

    return result;
}
