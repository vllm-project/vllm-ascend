// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <ngx_core.h>
#include <omni_metrics.h>

// Label definitions
static const char *LABEL_ENDPOINT[] = {"endpoint"};
static const char *LABEL_PHASE[] = {"phase"};

// Singleton metrics registry
static omni_metric_desc_t *metrics_registry = NULL;
static size_t metrics_count = 0;

#define UPSTREAM_METRIC(endpoint, name, help, value)                                                          \
    metrics_registry[index] = (omni_metric_desc_t){                                                           \
        name,                                                                                                 \
        help,                                                                                                 \
        OMNI_METRIC_GAUGE,                                                                                    \
        {{0}},                                                                                                \
        2,                                                                                                    \
        {.int_values = &endpoint->value},                                                                     \
        OMNI_VALUE_INT32,                                                                                     \
        1};                                                                                                   \
    snprintf(&metrics_registry[index].label_names[0][0],                                                      \
             OMNI_METRICS_MAX_LABEL_LEN, "endpoint=\"%s:%d\"", endpoint->address.ip, endpoint->address.port); \
    snprintf(&metrics_registry[index].label_names[1][0],                                                      \
             OMNI_METRICS_MAX_LABEL_LEN, "role=\"prefill\"");                                                 \
    index++;

#define PHASE_METRIC(phase, name, help, value)                    \
    metrics_registry[index] = (omni_metric_desc_t){               \
        name,                                                     \
        help,                                                     \
        OMNI_METRIC_GAUGE,                                        \
        {{0}},                                                    \
        1,                                                        \
        {.int_values = &global_state->groups[phase].value},       \
        OMNI_VALUE_INT32,                                         \
        1};                                                       \
    snprintf(&metrics_registry[index].label_names[0][0],          \
             OMNI_METRICS_MAX_LABEL_LEN, "phase=\"%s\"", #phase); \
    index++;

#define UPSTREAM_METRIC_MSEC(endpoint, name, help, value)                                                     \
    metrics_registry[index] = (omni_metric_desc_t){                                                           \
        name,                                                                                                 \
        help,                                                                                                 \
        OMNI_METRIC_GAUGE,                                                                                    \
        {{0}},                                                                                                \
        2,                                                                                                    \
        {.int64_values = (int64_t *)&endpoint->value},                                                        \
        OMNI_VALUE_INT64,                                                                                     \
        1};                                                                                                   \
    snprintf(&metrics_registry[index].label_names[0][0],                                                      \
             OMNI_METRICS_MAX_LABEL_LEN, "endpoint=\"%s:%d\"", endpoint->address.ip, endpoint->address.port); \
    snprintf(&metrics_registry[index].label_names[1][0],                                                      \
             OMNI_METRICS_MAX_LABEL_LEN, "role=\"prefill\"");                                                 \
    index++;

// Get or create the metrics registry
const omni_metric_desc_t *omni_metrics_get_registry(omni_global_state_t *global_state, size_t *count)
{
    if (metrics_registry != NULL)
    {
        if (count)
            *count = metrics_count;
        return metrics_registry;
    }

    if (!global_state)
    {
        return NULL;
    }

    // Calculate number of metrics needed
    size_t num_prefill = global_state->num_prefill_endpoints;
    size_t num_decode = global_state->num_decode_endpoints;

    // 9 metrics per endpoint type + 20 others, update the last value when add more
    metrics_count = (num_prefill * 4) + (num_decode * 5) + 20;

    // Allocate registry
    metrics_registry = ngx_alloc(metrics_count * sizeof(omni_metric_desc_t), ngx_cycle->log);
    if (!metrics_registry)
    {
        return NULL;
    }

    size_t index = 0;

    // Setup prefill metrics for each endpoint
    for (int i = 0; i < num_prefill; i++)
    {
        omni_upstream_prefill_t *prefill = &global_state->prefill_states[i];

        // Prefill running requests
        UPSTREAM_METRIC(
            prefill,
            "omni_prefill_running_requests",
            "Number of running requests on prefill upstream",
            num_running);

        // Prefill number of tokens
        UPSTREAM_METRIC(
            prefill,
            "omni_prefill_num_tokens",
            "Number of tokens on prefill upstream",
            num_tokens);

        // Prefill last scheduled time
        UPSTREAM_METRIC_MSEC(
            prefill,
            "omni_prefill_last_scheduled_time",
            "Last scheduled time on prefill upstream",
            last_scheduled_time);

        // Prefill expected next time
        UPSTREAM_METRIC_MSEC(
            prefill,
            "omni_prefill_expected_next_time",
            "Expected next schedule time on prefill upstream",
            expected_next_schedule_time);
    }

    // Setup decode metrics for each endpoint
    for (int i = 0; i < num_decode; i++)
    {
        omni_upstream_decode_t *decode = &global_state->decode_states[i];

        UPSTREAM_METRIC(
            decode,
            "omni_decode_running_requests",
            "Number of running requests on decode upstream",
            num_running);

        UPSTREAM_METRIC(
            decode,
            "omni_decode_num_tokens",
            "Number of tokens on decode upstream",
            num_tokens);

        UPSTREAM_METRIC(
            decode,
            "omni_decode_generated_tokens_total",
            "Total tokens generated by decode upstream",
            generated_tokens);

        UPSTREAM_METRIC_MSEC(
            decode,
            "omni_decode_last_scheduled_time",
            "Last scheduled time on decode upstream",
            last_scheduled_time);

        UPSTREAM_METRIC_MSEC(
            decode,
            "omni_decode_expected_next_time",
            "Expected next schedule time on decode upstream",
            expected_next_schedule_time);
    }

    PHASE_METRIC(PHASE_TOKENIZING,
                 "omni_phase_waiting",
                 "Num of requests waiting at given phase", num_requests);
    PHASE_METRIC(PHASE_APC_MATCHING,
                 "omni_phase_waiting",
                 "Num of requests waiting at given phase", num_requests);
    PHASE_METRIC(PHASE_PREFILL_WAITING_SCHEDULE,
                 "omni_phase_waiting",
                 "Num of requests waiting at given phase", num_requests);
    PHASE_METRIC(PHASE_PREFILL_SCHEDULED,
                 "omni_phase_waiting",
                 "Num of requests waiting at given phase", num_requests);
    PHASE_METRIC(PHASE_PREFILLING,
                 "omni_phase_waiting",
                 "Num of requests waiting at given phase", num_requests);
    PHASE_METRIC(PHASE_DECODE_WAITING_SCHEDULE,
                 "omni_phase_waiting",
                 "Num of requests waiting at given phase", num_requests);
    PHASE_METRIC(PHASE_DECODE_SCHEDULED,
                 "omni_phase_waiting", "Num of requests waiting at given phase", num_requests);
    PHASE_METRIC(PHASE_DECODING,
                 "omni_phase_waiting",
                 "Num of requests waiting at given phase", num_requests);

    if (count)
    {
        metrics_count = index;
        *count = metrics_count;
    }

    return metrics_registry;
}

// Format a single metric value
static u_char *format_metric_value(u_char *p, u_char *end,
                                   const omni_metric_desc_t *desc)
{
    double value = 0.0;

    // Get the value based on type
    if (desc->value_type == OMNI_VALUE_INT32)
    {
        value = (double)(*desc->value.int_values);
    }
    else
    {
        value = *desc->value.double_values;
    }

    // Output metric name
    p = ngx_snprintf(p, end - p, "%s", desc->name);

    // Output labels if present
    if (desc->label_count > 0)
    {
        p = ngx_snprintf(p, end - p, "{");
        for (size_t i = 0; i < desc->label_count; i++)
        {
            p = ngx_snprintf(p, end - p, "%s", &desc->label_names[i]);
            if (i < desc->label_count - 1)
            {
                p = ngx_snprintf(p, end - p, ",");
            }
        }
        p = ngx_snprintf(p, end - p, "}");
    }

    // Output value
    p = ngx_snprintf(p, end - p, " %f\n", value);

    return p;
}

// Export all metrics in Prometheus format
ngx_str_t omni_metrics_export(omni_global_state_t *global_state)
{
    static u_char buffer[65536];
    u_char *p = buffer;
    u_char *end = buffer + sizeof(buffer);

    if (!global_state)
    {
        p = ngx_snprintf(p, end - p, "# ERROR: Global state not available\n");
        goto done;
    }

    // Get or create metrics registry
    size_t count = 0;
    const omni_metric_desc_t *registry = omni_metrics_get_registry(global_state, &count);
    if (!registry)
    {
        p = ngx_snprintf(p, end - p, "# ERROR: Failed to get metrics registry\n");
        goto done;
    }

    // Track which metrics we've already output HELP/TYPE for
    int help_output[256] = {0}; // Simple deduplication array

    // Export all metrics
    for (size_t i = 0; i < count; i++)
    {
        const omni_metric_desc_t *desc = &registry[i];

        // Output HELP and TYPE only once per metric name
        if (!help_output[i])
        {
            p = ngx_snprintf(p, end - p, "# HELP %s %s\n", desc->name, desc->help);

            const char *type_str = "";
            switch (desc->type)
            {
            case OMNI_METRIC_GAUGE:
                type_str = "gauge";
                break;
            case OMNI_METRIC_COUNTER:
                type_str = "counter";
                break;
            case OMNI_METRIC_HISTOGRAM:
                type_str = "histogram";
                break;
            }
            p = ngx_snprintf(p, end - p, "# TYPE %s %s\n", desc->name, type_str);

            help_output[i] = 1;

            // Mark other instances of same metric name as already documented
            for (size_t j = i + 1; j < count; j++)
            {
                if (ngx_strcmp(registry[j].name, desc->name) == 0)
                {
                    help_output[j] = 1;
                }
            }
        }

        // Format the metric value
        p = format_metric_value(p, end, desc);
    }

done:
    {
        ngx_str_t result;
        result.data = buffer;
        result.len = p - buffer;

        return result;
    }
}
