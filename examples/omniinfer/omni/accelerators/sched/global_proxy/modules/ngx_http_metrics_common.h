// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#ifndef _NGX_HTTP_METRICS_COMMON_H_INCLUDED_
#define _NGX_HTTP_METRICS_COMMON_H_INCLUDED_

#include <ngx_config.h>
#include <ngx_core.h>

// TTFT histogram bucket count
typedef enum {
    NGX_HTTP_TTFT_BUCKET_1MS = 0,
    NGX_HTTP_TTFT_BUCKET_5MS,
    NGX_HTTP_TTFT_BUCKET_10MS,
    NGX_HTTP_TTFT_BUCKET_20MS,
    NGX_HTTP_TTFT_BUCKET_40MS,
    NGX_HTTP_TTFT_BUCKET_60MS,
    NGX_HTTP_TTFT_BUCKET_80MS,
    NGX_HTTP_TTFT_BUCKET_100MS,
    NGX_HTTP_TTFT_BUCKET_250MS,
    NGX_HTTP_TTFT_BUCKET_500MS,
    NGX_HTTP_TTFT_BUCKET_750MS,
    NGX_HTTP_TTFT_BUCKET_1000MS,
    NGX_HTTP_TTFT_BUCKET_2500MS,
    NGX_HTTP_TTFT_BUCKET_5000MS,
    NGX_HTTP_TTFT_BUCKET_7500MS,
    NGX_HTTP_TTFT_BUCKET_10000MS,
    NGX_HTTP_TTFT_BUCKET_20000MS,
    NGX_HTTP_TTFT_BUCKET_40000MS,
    NGX_HTTP_TTFT_BUCKET_80000MS,
    NGX_HTTP_TTFT_BUCKET_160000MS,
    NGX_HTTP_TTFT_BUCKET_640000MS,
    NGX_HTTP_TTFT_BUCKET_2560000MS,
    NGX_HTTP_TTFT_BUCKET_INF,
    NGX_HTTP_TTFT_BUCKETS_COUNT
} ngx_http_ttft_bucket_e;

// TPOT histogram bucket count
typedef enum {
    NGX_HTTP_TPOT_BUCKET_10MS = 0,
    NGX_HTTP_TPOT_BUCKET_25MS,
    NGX_HTTP_TPOT_BUCKET_50MS,
    NGX_HTTP_TPOT_BUCKET_75MS,
    NGX_HTTP_TPOT_BUCKET_100MS,
    NGX_HTTP_TPOT_BUCKET_150MS,
    NGX_HTTP_TPOT_BUCKET_200MS,
    NGX_HTTP_TPOT_BUCKET_300MS,
    NGX_HTTP_TPOT_BUCKET_400MS,
    NGX_HTTP_TPOT_BUCKET_500MS,
    NGX_HTTP_TPOT_BUCKET_750MS,
    NGX_HTTP_TPOT_BUCKET_1000MS,
    NGX_HTTP_TPOT_BUCKET_2500MS,
    NGX_HTTP_TPOT_BUCKET_5000MS,
    NGX_HTTP_TPOT_BUCKET_7500MS,
    NGX_HTTP_TPOT_BUCKET_10000MS,
    NGX_HTTP_TPOT_BUCKET_20000MS,
    NGX_HTTP_TPOT_BUCKET_40000MS,
    NGX_HTTP_TPOT_BUCKET_80000MS,
    NGX_HTTP_TPOT_BUCKET_INF,
    NGX_HTTP_TPOT_BUCKETS_COUNT
} ngx_http_tpot_bucket_e;

// E2E Latency histogram bucket count
typedef enum {
    NGX_HTTP_E2E_LATENCY_BUCKET_300MS = 0,
    NGX_HTTP_E2E_LATENCY_BUCKET_500MS,
    NGX_HTTP_E2E_LATENCY_BUCKET_800MS,
    NGX_HTTP_E2E_LATENCY_BUCKET_1000MS,
    NGX_HTTP_E2E_LATENCY_BUCKET_1500MS,
    NGX_HTTP_E2E_LATENCY_BUCKET_2000MS,
    NGX_HTTP_E2E_LATENCY_BUCKET_2500MS,
    NGX_HTTP_E2E_LATENCY_BUCKET_5000MS,
    NGX_HTTP_E2E_LATENCY_BUCKET_10000MS,
    NGX_HTTP_E2E_LATENCY_BUCKET_15000MS,
    NGX_HTTP_E2E_LATENCY_BUCKET_20000MS,
    NGX_HTTP_E2E_LATENCY_BUCKET_30000MS,
    NGX_HTTP_E2E_LATENCY_BUCKET_40000MS,
    NGX_HTTP_E2E_LATENCY_BUCKET_50000MS,
    NGX_HTTP_E2E_LATENCY_BUCKET_60000MS,
    NGX_HTTP_E2E_LATENCY_BUCKET_120000MS,
    NGX_HTTP_E2E_LATENCY_BUCKET_240000MS,
    NGX_HTTP_E2E_LATENCY_BUCKET_480000MS,
    NGX_HTTP_E2E_LATENCY_BUCKET_960000MS,
    NGX_HTTP_E2E_LATENCY_BUCKET_1920000MS,
    NGX_HTTP_E2E_LATENCY_BUCKET_7680000MS,
    NGX_HTTP_E2E_LATENCY_BUCKET_INF,
    NGX_HTTP_E2E_LATENCY_BUCKETS_COUNT
} ngx_http_e2e_latency_bucket_e;

typedef struct {
    ngx_atomic_t success_count;
    ngx_atomic_t failure_count;
    
    // TTFT histogram buckets
    ngx_atomic_t ttft_buckets[NGX_HTTP_TTFT_BUCKETS_COUNT];
    ngx_atomic_t ttft_sum;      // Sum of all TTFT values (in milliseconds)
    ngx_atomic_t ttft_count;    // Total count of TTFT measurements
    
    // TPOT histogram buckets
    ngx_atomic_t tpot_buckets[NGX_HTTP_TPOT_BUCKETS_COUNT];
    ngx_atomic_t tpot_sum;      // Sum of all TPOT values (in milliseconds)
    ngx_atomic_t tpot_count;    // Total count of TPOT measurements

    // E2E Latency histogram buckets
    ngx_atomic_t e2e_latency_buckets[NGX_HTTP_E2E_LATENCY_BUCKETS_COUNT];
    ngx_atomic_t e2e_latency_sum;      // Sum of all E2E Latency values (in milliseconds)
    ngx_atomic_t e2e_latency_count;    // Total count of E2E Latency measurements
    
    // Model name for histogram labels
    u_char model_name[2048];     // Store model name for histogram labels
    ngx_uint_t model_name_len;  // Length of model name
} ngx_http_metrics_data_t;

#define NGX_HTTP_METRICS_SHM_NAME "gp_metrics_shm_zone"

// TTFT histogram bucket thresholds in milliseconds
static const ngx_uint_t ngx_http_metrics_ttft_buckets[NGX_HTTP_TTFT_BUCKETS_COUNT] = {
    1, 5, 10, 20, 40, 60, 80, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 
    20000, 40000, 80000, 160000, 640000, 2560000
};

// TPOT histogram bucket thresholds in milliseconds
static const ngx_uint_t ngx_http_metrics_tpot_buckets[NGX_HTTP_TPOT_BUCKETS_COUNT] = {
    10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 2500, 5000, 7500, 10000, 
    20000, 40000, 80000
};

// E2E Latency histogram bucket thresholds in milliseconds
static const ngx_uint_t ngx_http_metrics_e2e_latency_buckets[NGX_HTTP_E2E_LATENCY_BUCKETS_COUNT] = {
    300, 500, 800, 1000, 1500, 2000, 2500, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 
    60000, 120000, 240000, 480000, 960000, 1920000, 7680000
};

// Helper function to record TTFT histogram value (will be exported in seconds)
static ngx_inline void
ngx_http_metrics_record_ttft_histogram(ngx_atomic_t *buckets, ngx_atomic_t *sum, 
                                      ngx_atomic_t *count, ngx_uint_t value_ms)
{
    ngx_uint_t i;
    
    // Record in appropriate bucket
    for (i = 0; i < NGX_HTTP_TTFT_BUCKETS_COUNT - 1; i++) {
        if (value_ms <= ngx_http_metrics_ttft_buckets[i]) {
            ngx_atomic_fetch_add(&buckets[i], 1);
            break;
        }
    }
    
    // If not in any bucket, it goes to the +Inf bucket (last one)
    if (i == NGX_HTTP_TTFT_BUCKETS_COUNT - 1) {
        ngx_atomic_fetch_add(&buckets[i], 1);
    }
    
    // Update sum (keep in milliseconds) and count
    ngx_atomic_fetch_add(sum, value_ms);
    ngx_atomic_fetch_add(count, 1);
}

// Helper function to record TPOT histogram value (already in milliseconds)
static ngx_inline void
ngx_http_metrics_record_tpot_histogram(ngx_atomic_t *buckets, ngx_atomic_t *sum, 
                                      ngx_atomic_t *count, ngx_uint_t value_ms)
{
    ngx_uint_t i;
    
    // Record in appropriate bucket
    for (i = 0; i < NGX_HTTP_TPOT_BUCKETS_COUNT - 1; i++) {
        if (value_ms <= ngx_http_metrics_tpot_buckets[i]) {
            ngx_atomic_fetch_add(&buckets[i], 1);
            break;
        }
    }
    
    // If not in any bucket, it goes to the +Inf bucket (last one)
    if (i == NGX_HTTP_TPOT_BUCKETS_COUNT - 1) {
        ngx_atomic_fetch_add(&buckets[i], 1);
    }
    
    // Update sum (keep in milliseconds) and count
    ngx_atomic_fetch_add(sum, value_ms);
    ngx_atomic_fetch_add(count, 1);
}

// Helper function to record E2E Latency histogram value (already in milliseconds)
static ngx_inline void
ngx_http_metrics_record_e2e_latency_histogram(ngx_atomic_t *buckets, ngx_atomic_t *sum, 
                                             ngx_atomic_t *count, ngx_uint_t value_ms)
{
    ngx_uint_t i;
    
    // Record in appropriate bucket
    for (i = 0; i < NGX_HTTP_E2E_LATENCY_BUCKETS_COUNT - 1; i++) {
        if (value_ms <= ngx_http_metrics_e2e_latency_buckets[i]) {
            ngx_atomic_fetch_add(&buckets[i], 1);
            break;
        }
    }
    
    // If not in any bucket, it goes to the +Inf bucket (last one)
    if (i == NGX_HTTP_E2E_LATENCY_BUCKETS_COUNT - 1) {
        ngx_atomic_fetch_add(&buckets[i], 1);
    }
    
    // Update sum (keep in milliseconds) and count
    ngx_atomic_fetch_add(sum, value_ms);
    ngx_atomic_fetch_add(count, 1);
}

// Helper function to safely update model name in shared memory
static ngx_inline void
ngx_http_metrics_update_model_name(ngx_http_metrics_data_t *metrics, const char *model_name, ngx_uint_t len)
{
    if (metrics && model_name && len > 0 && len < sizeof(metrics->model_name) - 1) {
        // Only update if model name is not already set or is different
        if (metrics->model_name_len == 0 || 
            ngx_strncmp(metrics->model_name, model_name, len) != 0) {
            ngx_memcpy(metrics->model_name, model_name, len);
            metrics->model_name[len] = '\0';
            metrics->model_name_len = len;
            
            // Basic sanitization: replace problematic characters with underscores
            for (ngx_uint_t i = 0; i < len; i++) {
                if (metrics->model_name[i] == '"' || metrics->model_name[i] == '\\' || 
                    metrics->model_name[i] == '\n' || metrics->model_name[i] == '\r') {
                    metrics->model_name[i] = '_';
                }
            }
        }
    }
}

#endif /* _NGX_HTTP_METRICS_COMMON_H_INCLUDED_ */
