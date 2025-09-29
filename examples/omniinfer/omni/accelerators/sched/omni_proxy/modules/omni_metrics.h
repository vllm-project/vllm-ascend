// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#include <ngx_core.h>
#include "omni_proxy.h"

// Metric types
typedef enum
{
    OMNI_METRIC_GAUGE,
    OMNI_METRIC_COUNTER,
    OMNI_METRIC_HISTOGRAM
} omni_metric_type_t;

// Value types
typedef enum
{
    OMNI_VALUE_INT32,
    OMNI_VALUE_INT64,
    OMNI_VALUE_DOUBLE
} omni_value_type_t;

#define OMNI_METRICS_MAX_LABEL_LEN 128
#define OMNI_METRICS_MAX_LABELS 5

// Metric descriptor
typedef struct omni_metric_desc_s
{
    const char *name;                                                      // Metric name
    const char *help;                                                      // Help text
    omni_metric_type_t type;                                               // Metric type
    char label_names[OMNI_METRICS_MAX_LABELS][OMNI_METRICS_MAX_LABEL_LEN]; // Array of label names
    size_t label_count;                                                    // Number of labels

    union
    {
        int32_t *int_values;   // Pointer to int values in global_state
        int64_t *int64_values;
        double *double_values; // Pointer to double values
    } value;

    omni_value_type_t value_type; // Value type (int or double)
    int32_t value_count;          // Number of values (for arrays/histograms)

} omni_metric_desc_t;

// Get the metrics registry (singleton pattern)
const omni_metric_desc_t *omni_metrics_get_registry(omni_global_state_t *global_state, size_t *count);

// Export metrics in Prometheus format
ngx_str_t omni_metrics_export(omni_global_state_t *global_state);
