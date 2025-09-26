// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <time.h>

typedef enum {
    NGX_HTTP_PREFILL_REFACTOR_BACKEND_VLLM = 0,
    NGX_HTTP_PREFILL_REFACTOR_BACKEND_SGLANG = 1
} ngx_http_prefill_refactor_backend_e;

typedef enum {
    NGX_HTTP_PREFILL_REFACTOR_DIST_P_BEFORE_D = 0,
    NGX_HTTP_PREFILL_REFACTOR_DIST_P_AFTER_D = 1,
    NGX_HTTP_PREFILL_REFACTOR_DIST_PD_TOGETHER = 2
} ngx_http_prefill_refactor_distribution_e;

typedef struct {
    ngx_str_t server;
    ngx_str_t port;
} ngx_http_prefill_refactor_server_info_t;

typedef struct {
    ngx_str_t prefill_location;
    ngx_str_t prefill_servers_list;
    ngx_str_t decode_servers_list;
    ngx_str_t bootstrap_ports_list;
    ngx_http_prefill_refactor_backend_e pd_backend;
    ngx_http_prefill_refactor_distribution_e pd_distribution;
    ngx_array_t *prefill_servers_info;   /* array of ngx_http_prefill_refactor_server_info_t */
    ngx_array_t *decode_servers;         /* array of ngx_str_t */
} ngx_http_prefill_refactor_loc_conf_t;

typedef struct {
    ngx_str_t selected_prefill_server;
    ngx_str_t selected_decode_server;
    ngx_str_t selected_bootstrap_port;
    ngx_str_t bootstrap_host;
    ngx_str_t bootstrap_room;
    ngx_uint_t done;
    ngx_uint_t status;
    u_char *origin_body_data;
    ngx_uint_t origin_body_data_size;
    u_char *prefill_response_body;
    ngx_uint_t prefill_response_body_size;
} ngx_http_prefill_refactor_ctx_t;

// Function declarations
static ngx_int_t ngx_http_prefill_refactor_handler(ngx_http_request_t *r);
static void *ngx_http_prefill_refactor_create_loc_conf(ngx_conf_t *cf);
static char *ngx_http_prefill_refactor_merge_loc_conf(ngx_conf_t *cf, void *parent, void *child);
static ngx_int_t ngx_http_prefill_refactor_init(ngx_conf_t *cf);
static char *ngx_http_prefill_refactor_set_directive(ngx_conf_t *cf, ngx_command_t *cmd, void *conf);
static char *ngx_http_prefill_refactor_set_backend(ngx_conf_t *cf, ngx_command_t *cmd, void *conf);
static char *ngx_http_prefill_refactor_set_distribution(ngx_conf_t *cf, ngx_command_t *cmd, void *conf);
static ngx_int_t ngx_http_prefill_refactor_prefill_server_variable(ngx_http_request_t *r, 
                                                                   ngx_http_variable_value_t *v, 
                                                                   uintptr_t data);
static ngx_int_t ngx_http_prefill_refactor_decode_server_variable(ngx_http_request_t *r, 
                                                                  ngx_http_variable_value_t *v, 
                                                                  uintptr_t data);
static ngx_int_t ngx_http_prefill_refactor_add_variables(ngx_conf_t *cf);
static ngx_int_t ngx_http_prefill_refactor_parse_servers(ngx_conf_t *cf, ngx_str_t *server_list, ngx_array_t **servers);

static ngx_command_t ngx_http_prefill_refactor_commands[] = {
    {ngx_string("prefill_servers_list"),
        NGX_HTTP_MAIN_CONF | NGX_HTTP_SRV_CONF | NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
        ngx_conf_set_str_slot,
        NGX_HTTP_LOC_CONF_OFFSET,
        offsetof(ngx_http_prefill_refactor_loc_conf_t, prefill_servers_list),
        NULL},

    {ngx_string("decode_servers_list"),
        NGX_HTTP_MAIN_CONF | NGX_HTTP_SRV_CONF | NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
        ngx_conf_set_str_slot,
        NGX_HTTP_LOC_CONF_OFFSET,
        offsetof(ngx_http_prefill_refactor_loc_conf_t, decode_servers_list),
        NULL},

    {ngx_string("bootstrap_ports_list"),
        NGX_HTTP_MAIN_CONF | NGX_HTTP_SRV_CONF | NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
        ngx_conf_set_str_slot,
        NGX_HTTP_LOC_CONF_OFFSET,
        offsetof(ngx_http_prefill_refactor_loc_conf_t, bootstrap_ports_list),
        NULL},

    {ngx_string("prefill_refactor"),
        NGX_HTTP_MAIN_CONF | NGX_HTTP_SRV_CONF | NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
        ngx_http_prefill_refactor_set_directive,
        NGX_HTTP_LOC_CONF_OFFSET,
        0,
        NULL},

    {ngx_string("pd_backend"),
        NGX_HTTP_MAIN_CONF | NGX_HTTP_SRV_CONF | NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
        ngx_http_prefill_refactor_set_backend,
        NGX_HTTP_LOC_CONF_OFFSET,
        0,
        NULL},

    {ngx_string("pd_distribution"),
        NGX_HTTP_MAIN_CONF | NGX_HTTP_SRV_CONF | NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
        ngx_http_prefill_refactor_set_distribution,
        NGX_HTTP_LOC_CONF_OFFSET,
        0,
        NULL},

    ngx_null_command
};

static ngx_http_module_t ngx_http_prefill_refactor_module_ctx = {
    ngx_http_prefill_refactor_add_variables,      /* preconfiguration */
    ngx_http_prefill_refactor_init,               /* postconfiguration */

    NULL,                               /* create main configuration */
    NULL,                               /* init main configuration */

    NULL,                               /* create server configuration */
    NULL,                               /* merge server configuration */

    ngx_http_prefill_refactor_create_loc_conf,    /* create location configuration */
    ngx_http_prefill_refactor_merge_loc_conf      /* merge location configuration */
};

ngx_module_t ngx_http_prefill_refactor_module = {
    NGX_MODULE_V1,
    &ngx_http_prefill_refactor_module_ctx,        /* module context */
    ngx_http_prefill_refactor_commands,           /* module directives */
    NGX_HTTP_MODULE,                    /* module type */
    NULL,                               /* init master */
    NULL,                               /* init module */
    NULL,                               /* init process */
    NULL,                               /* init thread */
    NULL,                               /* exit thread */
    NULL,                               /* exit process */
    NULL,                               /* exit master */
    NGX_MODULE_V1_PADDING
};

static ngx_http_variable_t ngx_http_prefill_refactor_variables[] = {
    {ngx_string("prefill_server"), NULL, ngx_http_prefill_refactor_prefill_server_variable, 
     0, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("decode_server"), NULL, ngx_http_prefill_refactor_decode_server_variable, 
     0, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_null_string, NULL, NULL, 0, 0, 0}
};

// Utility function to select a random server from a pre-parsed array
static ngx_str_t ngx_http_prefill_refactor_select_server_from_array(ngx_array_t *servers) {
    ngx_str_t result = ngx_null_string;
    ngx_str_t *server_list;
    ngx_uint_t selected;
    
    if (servers == NULL || servers->nelts == 0) {
        return result;
    }
    
    server_list = (ngx_str_t *)servers->elts;
    selected = rand() % servers->nelts;
    
    return server_list[selected];
}

// Generate random 63-bit integer as string
static ngx_str_t ngx_http_prefill_refactor_generate_bootstrap_room(ngx_pool_t *pool) {
    ngx_str_t result;
    uint64_t room_id;
    
    // Generate random 63-bit number (ensuring MSB is 0)
    room_id = ((uint64_t)rand() << 32) | (uint64_t)rand();
    room_id &= 0x7FFFFFFFFFFFFFFFULL; // Clear MSB to ensure it's 63 bits
    
    result.data = ngx_palloc(pool, 21); // Enough for 64-bit decimal
    if (result.data) {
        result.len = ngx_sprintf(result.data, "%uL", room_id) - result.data;
    } else {
        result.len = 0;
    }
    
    return result;
}

// Extract host from server address (host:port -> host)
static ngx_str_t ngx_http_prefill_refactor_extract_host(ngx_pool_t *pool, ngx_str_t server) {
    ngx_str_t result = ngx_null_string;
    u_char *colon;
    
    if (server.len == 0) {
        return result;
    }
    
    colon = ngx_strlchr(server.data, server.data + server.len, ':');
    if (colon) {
        result.len = colon - server.data;
    } else {
        result.len = server.len;
    }
    
    result.data = ngx_palloc(pool, result.len + 1);
    if (result.data) {
        ngx_memcpy(result.data, server.data, result.len);
        result.data[result.len] = '\0';
    }
    
    return result;
}

// Load balance and select servers/ports
static ngx_int_t ngx_http_prefill_refactor_load_balance(ngx_http_request_t *r, 
                                                        ngx_http_prefill_refactor_ctx_t *ctx, 
                                                        ngx_http_prefill_refactor_loc_conf_t *conf) {
    // Select prefill server and bootstrap port together
    if (conf->prefill_servers_info == NULL || conf->prefill_servers_info->nelts == 0) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, 
                      "prefill_refactor: no prefill servers configured");
        return NGX_ERROR;
    }
    
    ngx_http_prefill_refactor_server_info_t *server_info_list = 
        (ngx_http_prefill_refactor_server_info_t *)conf->prefill_servers_info->elts;
    ngx_uint_t selected = rand() % conf->prefill_servers_info->nelts;
    ngx_http_prefill_refactor_server_info_t *selected_info = &server_info_list[selected];
    
    ctx->selected_prefill_server = selected_info->server;
    ctx->selected_bootstrap_port = selected_info->port;
    
    if (ctx->selected_prefill_server.len == 0) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill_refactor: failed to select prefill server");
        return NGX_ERROR;
    }
    
    // Select decode server
    ctx->selected_decode_server = ngx_http_prefill_refactor_select_server_from_array(conf->decode_servers);
    if (ctx->selected_decode_server.len == 0) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill_refactor: failed to select decode server");
        return NGX_ERROR;
    }
    
    // Extract bootstrap host from selected prefill server
    ctx->bootstrap_host = ngx_http_prefill_refactor_extract_host(r->pool, ctx->selected_prefill_server);
    
    // Generate bootstrap room
    ctx->bootstrap_room = ngx_http_prefill_refactor_generate_bootstrap_room(r->pool);
    
    ngx_log_debug5(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
        "prefill_refactor: selected prefill_server=%V, decode_server=%V, bootstrap_port=%V, "
        "bootstrap_host=%V, bootstrap_room=%V",
        &ctx->selected_prefill_server, &ctx->selected_decode_server, &ctx->selected_bootstrap_port,
        &ctx->bootstrap_host, &ctx->bootstrap_room);
    
    return NGX_OK;
}

// Parse comma-separated server list into array
static ngx_int_t ngx_http_prefill_refactor_parse_servers(ngx_conf_t *cf, 
                                                         ngx_str_t *server_list, 
                                                         ngx_array_t **servers) {
    ngx_array_t *array;
    ngx_str_t *server;
    u_char *start, *p, *end;
    size_t len;
    
    if (server_list->len == 0) {
        *servers = NULL;
        return NGX_OK;
    }
    
    // Create array with initial capacity of 8 servers
    array = ngx_array_create(cf->pool, 8, sizeof(ngx_str_t));
    if (array == NULL) {
        return NGX_ERROR;
    }
    
    start = server_list->data;
    end = server_list->data + server_list->len;
    
    for (p = start; p <= end; p++) {
        if (*p == ',' || p == end) {
            // Skip empty entries
            if (p == start) {
                start = p + 1;
                continue;
            }
            
            // Calculate length and trim whitespace
            len = p - start;
            while (len > 0 && (start[0] == ' ' || start[0] == '\t')) {
                start++;
                len--;
            }
            while (len > 0 && (start[len-1] == ' ' || start[len-1] == '\t')) {
                len--;
            }
            
            if (len > 0) {
                server = ngx_array_push(array);
                if (server == NULL) {
                    return NGX_ERROR;
                }
                
                server->len = len;
                server->data = ngx_palloc(cf->pool, len + 1);
                if (server->data == NULL) {
                    return NGX_ERROR;
                }
                
                ngx_memcpy(server->data, start, len);
                server->data[len] = '\0';
            }
            
            start = p + 1;
        }
    }
    
    *servers = array;
    return NGX_OK;
}

// Build prefill server info array combining servers and bootstrap ports
static ngx_int_t ngx_http_prefill_refactor_build_server_info(ngx_conf_t *cf, 
                                                             ngx_array_t *servers, 
                                                             ngx_array_t *bootstrap_ports, 
                                                             ngx_array_t **server_info) {
    ngx_array_t *info_array;
    ngx_http_prefill_refactor_server_info_t *info;
    ngx_str_t *server_list, *port_list;
    ngx_uint_t i;
    
    if (servers == NULL || servers->nelts == 0) {
        *server_info = NULL;
        return NGX_OK;
    }
    
    // Create server info array
    info_array = ngx_array_create(cf->pool, servers->nelts, 
                                   sizeof(ngx_http_prefill_refactor_server_info_t));
    if (info_array == NULL) {
        return NGX_ERROR;
    }
    
    server_list = (ngx_str_t *)servers->elts;
    port_list = (bootstrap_ports != NULL) ? (ngx_str_t *)bootstrap_ports->elts : NULL;
    
    for (i = 0; i < servers->nelts; i++) {
        info = ngx_array_push(info_array);
        if (info == NULL) {
            return NGX_ERROR;
        }
        
        // Set server
        info->server = server_list[i];
        
        // Set bootstrap port based on pattern
        if (bootstrap_ports == NULL || bootstrap_ports->nelts == 0) {
            // Pattern 1: Empty - no bootstrap port
            info->port.len = 0;
            info->port.data = NULL;
        } else if (bootstrap_ports->nelts == 1) {
            // Pattern 2: Single port for all servers
            info->port = port_list[0];
        } else {
            // Pattern 3: Port list matching servers
            info->port = port_list[i];
        }
    }
    
    *server_info = info_array;
    return NGX_OK;
}

// Modify request body to add bootstrap parameters
static ngx_int_t ngx_http_prefill_refactor_modify_request_body(ngx_http_request_t *r, 
                                                               ngx_http_prefill_refactor_ctx_t *ctx) {
    ngx_chain_t *cl;
    size_t len = 0;
    u_char *body_data = NULL;
    u_char *p;
    ngx_buf_t *b;
    ngx_chain_t *new_chain;
    size_t new_len;
    u_char *new_body;
    u_char *insertion_point;
    
    if (r->request_body == NULL || r->request_body->bufs == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill_refactor: request body is empty");
        return NGX_ERROR;
    }

    // Calculate total body size
    for (cl = r->request_body->bufs; cl != NULL; cl = cl->next) {
        len += ngx_buf_size(cl->buf);
    }

    if (len == 0) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill_refactor: request body length is zero");
        return NGX_ERROR;
    }

    // Allocate memory for the original body
    body_data = ngx_palloc(r->pool, len + 1);
    if (body_data == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill_refactor: failed to allocate memory for body");
        return NGX_ERROR;
    }

    ctx->origin_body_data = body_data;
    ctx->origin_body_data_size = len;

    // Copy body data
    p = body_data;
    for (cl = r->request_body->bufs; cl != NULL; cl = cl->next) {
        size_t buf_size = ngx_buf_size(cl->buf);
        if (buf_size > 0) {
            p = ngx_cpymem(p, cl->buf->pos, buf_size);
        }
    }
    *p = '\0';

    // Find insertion point (before the closing brace)
    insertion_point = body_data + len - 1;
    while (insertion_point > body_data && *insertion_point != '}') {
        insertion_point--;
    }
    
    if (*insertion_point != '}') {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill_refactor: invalid JSON format");
        return NGX_ERROR;
    }

    // Calculate new body size (original + bootstrap fields)
    new_len = len + 256; // Extra space for bootstrap fields
    new_body = ngx_palloc(r->pool, new_len);
    if (new_body == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill_refactor: failed to allocate memory for new body");
        return NGX_ERROR;
    }

    // Copy original body up to insertion point
    size_t prefix_len = insertion_point - body_data;
    ngx_memcpy(new_body, body_data, prefix_len);
    p = new_body + prefix_len;

    // Add bootstrap fields
    if (prefix_len > 1 && *(p-1) != '{') {
        *p++ = ',';
    }
    
    p += ngx_sprintf(p, "\"bootstrap_host\":\"%V\"", &ctx->bootstrap_host) - p;
    
    if (ctx->selected_bootstrap_port.len > 0) {
        p += ngx_sprintf(p, ",\"bootstrap_port\":\"%V\"", &ctx->selected_bootstrap_port) - p;
    } else {
        p += ngx_sprintf(p, ",\"bootstrap_port\":null") - p;
    }
    
    p += ngx_sprintf(p, ",\"bootstrap_room\":\"%V\"", &ctx->bootstrap_room) - p;
    
    // Add closing brace and any remaining content
    ngx_memcpy(p, insertion_point, len - prefix_len);
    p += len - prefix_len;
    
    new_len = p - new_body;

    // Create new buffer
    b = ngx_pcalloc(r->pool, sizeof(ngx_buf_t));
    if (b == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill_refactor: failed to allocate buffer");
        return NGX_ERROR;
    }
    
    b->pos = new_body;
    b->last = new_body + new_len;
    b->memory = 1;
    b->last_buf = 1;
    b->last_in_chain = 1;

    // Create new chain
    new_chain = ngx_alloc_chain_link(r->pool);
    if (new_chain == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill_refactor: failed to allocate chain link");
        return NGX_ERROR;
    }
    new_chain->buf = b;
    new_chain->next = NULL;

    // Update request body
    r->request_body->bufs = new_chain;
    r->request_body->buf = b;
    r->headers_in.content_length_n = new_len;
    
    if (r->headers_in.content_length) {
        r->headers_in.content_length->value.len =
            ngx_sprintf(r->headers_in.content_length->value.data, "%uz", new_len) -
            r->headers_in.content_length->value.data;
    }

    ngx_log_debug2(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
        "prefill_refactor: modified request body length=%uz, body=%s", new_len, new_body);

    return NGX_OK;
}

// Subrequest completion handler
static ngx_int_t ngx_http_prefill_refactor_subrequest_done(ngx_http_request_t *r, void *data, ngx_int_t rc) {
    ngx_http_prefill_refactor_ctx_t *ctx = (ngx_http_prefill_refactor_ctx_t *)data;
    ngx_http_prefill_refactor_loc_conf_t *conf;
    ngx_chain_t *cl;
    size_t total = 0;
    u_char *p;

    ctx->done = 1;
    ctx->status = r->headers_out.status;

    if (rc != NGX_OK) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill_refactor: subrequest failed with code %i", rc);
        ctx->status = NGX_HTTP_INTERNAL_SERVER_ERROR;
        ngx_http_finalize_request(r->main, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return rc;
    }

    conf = ngx_http_get_module_loc_conf(r->main, ngx_http_prefill_refactor_module);
    if (conf == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, 
                      "prefill_refactor: failed to get module location configuration in subrequest handler");
        ngx_http_finalize_request(r->main, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return rc;
    }

    // Calculate total response size
    for (cl = r->out; cl; cl = cl->next) {
        total += ngx_buf_size(cl->buf);
    }

    if (total > 0) {
        // Allocate memory for response body
        ctx->prefill_response_body = ngx_palloc(r->main->pool, total + 1);
        if (ctx->prefill_response_body == NULL) {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, 
                          "prefill_refactor: failed to allocate response body buffer");
            ngx_http_finalize_request(r->main, NGX_ERROR);
            return rc;
        }

        // Copy response data
        p = ctx->prefill_response_body;
        for (cl = r->out; cl; cl = cl->next) {
            size_t buf_size = ngx_buf_size(cl->buf);
            if (buf_size > 0) {
                p = ngx_cpymem(p, cl->buf->pos, buf_size);
            }
        }
        *p = '\0';
        ctx->prefill_response_body_size = total;
    }

    ngx_log_debug2(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
        "prefill_refactor: subrequest completed, status=%ui, response_size=%uz", ctx->status, total);

    if (conf->pd_distribution == NGX_HTTP_PREFILL_REFACTOR_DIST_P_BEFORE_D) {
        if (ctx->status >= NGX_HTTP_OK && ctx->status < NGX_HTTP_SPECIAL_RESPONSE) {
            ngx_http_core_run_phases(r->main);  // status < 300: resume processing the main request
        } else {
            // subrequest status >= 300: set client response header from subrequest
            r->main->headers_out.status = ctx->status;
            r->main->headers_out.content_length_n = ctx->prefill_response_body_size;
            r->main->headers_out.content_type.data = r->headers_out.content_type.data;
            r->main->headers_out.content_type.len = r->headers_out.content_type.len;

            ngx_http_send_header(r->main);
        }
    }

    return rc;
}

// Request handler after body is read
static void ngx_http_prefill_refactor_request_handler(ngx_http_request_t *r) {
    ngx_http_prefill_refactor_ctx_t *ctx;
    ngx_http_prefill_refactor_loc_conf_t *conf;
    ngx_http_post_subrequest_t *ps;
    ngx_str_t uri;
    ngx_http_request_t *sr;
    ngx_int_t flags;

    ctx = ngx_http_get_module_ctx(r, ngx_http_prefill_refactor_module);
    if (ctx == NULL) {
        ngx_http_finalize_request(r, NGX_ERROR);
        return;
    }

    conf = ngx_http_get_module_loc_conf(r, ngx_http_prefill_refactor_module);
    if (conf == NULL || conf->prefill_location.len == 0) {
        ngx_http_finalize_request(r, NGX_ERROR);
        return;
    }

    // Perform load balancing
    if (ngx_http_prefill_refactor_load_balance(r, ctx, conf) != NGX_OK) {
        ngx_http_finalize_request(r, NGX_ERROR);
        return;
    }

    // Modify request body to include bootstrap parameters
    if (ngx_http_prefill_refactor_modify_request_body(r, ctx) != NGX_OK) {
        ngx_http_finalize_request(r, NGX_ERROR);
        return;
    }

    // Set up subrequest
    ps = ngx_palloc(r->pool, sizeof(ngx_http_post_subrequest_t));
    if (ps == NULL) {
        ngx_http_finalize_request(r, NGX_ERROR);
        return;
    }

    ps->handler = ngx_http_prefill_refactor_subrequest_done;
    ps->data = ctx;

    // Create subrequest URI
    uri.len = conf->prefill_location.len + r->uri.len;
    uri.data = ngx_palloc(r->pool, uri.len);
    if (uri.data == NULL) {
        ngx_http_finalize_request(r, NGX_ERROR);
        return;
    }
    
    ngx_memcpy(uri.data, conf->prefill_location.data, conf->prefill_location.len);
    ngx_memcpy(uri.data + conf->prefill_location.len, r->uri.data, r->uri.len);

    flags = NGX_HTTP_SUBREQUEST_IN_MEMORY;
    if (conf->pd_distribution == NGX_HTTP_PREFILL_REFACTOR_DIST_P_BEFORE_D) {
        // If distribution is P_BEFORE_D, we need to wait for the subrequest to complete
        flags |= NGX_HTTP_SUBREQUEST_WAITED;
    }

    if (ngx_http_subrequest(r, &uri, &r->args, &sr, ps, flags) != NGX_OK) {
        ngx_http_finalize_request(r, NGX_ERROR);
        return;
    }

    // Copy request properties to subrequest
    sr->method = r->method;
    sr->method_name = r->method_name;
    sr->headers_in.content_length_n = r->headers_in.content_length_n;
    sr->headers_in.content_type = r->headers_in.content_type;
    sr->request_body = r->request_body;

    if (conf->pd_distribution == NGX_HTTP_PREFILL_REFACTOR_DIST_PD_TOGETHER) {
        ngx_http_core_run_phases(r);
    }

    ngx_log_debug1(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, "prefill_refactor: subrequest created for URI: %V", &uri);
}

// Main handler
static ngx_int_t ngx_http_prefill_refactor_handler(ngx_http_request_t *r) {
    ngx_http_prefill_refactor_loc_conf_t *conf;
    ngx_http_prefill_refactor_ctx_t *ctx;

    ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, "prefill_refactor: handler called");

    conf = ngx_http_get_module_loc_conf(r, ngx_http_prefill_refactor_module);
    if (conf == NULL || conf->prefill_location.len == 0) {
        return NGX_DECLINED;
    }

    ctx = ngx_http_get_module_ctx(r, ngx_http_prefill_refactor_module);
    if (ctx != NULL) {
        return NGX_OK;
    }

    // Create context
    ctx = ngx_pcalloc(r->pool, sizeof(ngx_http_prefill_refactor_ctx_t));
    if (ctx == NULL) {
        return NGX_ERROR;
    }

    ngx_http_set_ctx(r, ctx, ngx_http_prefill_refactor_module);

    // Read client request body
    ngx_int_t rc = ngx_http_read_client_request_body(r, ngx_http_prefill_refactor_request_handler);
    if (rc >= NGX_HTTP_SPECIAL_RESPONSE) {
        return rc;
    }

    return NGX_AGAIN;
}

// Variable getters
static ngx_int_t ngx_http_prefill_refactor_prefill_server_variable(ngx_http_request_t *r, 
                                                                   ngx_http_variable_value_t *v, 
                                                                   uintptr_t data) {
    ngx_http_prefill_refactor_ctx_t *ctx;

    ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, 
                   "prefill_refactor: accessing prefill_server variable");

    ctx = ngx_http_get_module_ctx(r, ngx_http_prefill_refactor_module);
    if (ctx == NULL && r != r->main) {
        ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, 
                       "prefill_refactor: trying to get ctx from r->main");
        ctx = ngx_http_get_module_ctx(r->main, ngx_http_prefill_refactor_module);
    }
    
    if (ctx == NULL || ctx->selected_prefill_server.len == 0) {
        ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, 
                       "prefill_refactor: prefill_server variable not found - no context or empty server");
        v->not_found = 1;
        return NGX_OK;
    }

    v->len = ctx->selected_prefill_server.len;
    v->data = ctx->selected_prefill_server.data;
    v->valid = 1;
    v->no_cacheable = 1;
    v->not_found = 0;

    ngx_log_debug1(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, 
                    "prefill_refactor: prefill_server variable value: %V", &ctx->selected_prefill_server);

    return NGX_OK;
}

static ngx_int_t ngx_http_prefill_refactor_decode_server_variable(ngx_http_request_t *r, 
                                                                  ngx_http_variable_value_t *v, 
                                                                  uintptr_t data) {
    ngx_http_prefill_refactor_ctx_t *ctx;

    ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, 
                   "prefill_refactor: accessing decode_server variable");

    ctx = ngx_http_get_module_ctx(r, ngx_http_prefill_refactor_module);
    if (ctx == NULL && r != r->main) {
        ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, 
                       "prefill_refactor: trying to get ctx from r->main");
        ctx = ngx_http_get_module_ctx(r->main, ngx_http_prefill_refactor_module);
    }
    
    if (ctx == NULL || ctx->selected_decode_server.len == 0) {
        ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, 
                       "prefill_refactor: decode_server variable not found - no context or empty server");
        v->not_found = 1;
        return NGX_OK;
    }

    v->len = ctx->selected_decode_server.len;
    v->data = ctx->selected_decode_server.data;
    v->valid = 1;
    v->no_cacheable = 1;
    v->not_found = 0;

    ngx_log_debug1(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, 
                    "prefill_refactor: decode_server variable value: %V", &ctx->selected_decode_server);

    return NGX_OK;
}

// Configuration functions
static void *ngx_http_prefill_refactor_create_loc_conf(ngx_conf_t *cf) {
    ngx_http_prefill_refactor_loc_conf_t *conf;

    conf = ngx_pcalloc(cf->pool, sizeof(ngx_http_prefill_refactor_loc_conf_t));
    if (conf == NULL) {
        return NULL;
    }

    // Initialize arrays as NULL (will be created during merge if needed)
    conf->prefill_servers_info = NULL;
    conf->decode_servers = NULL;
    
    // Initialize enum values to invalid/unset state
    conf->pd_backend = NGX_CONF_UNSET_UINT;
    conf->pd_distribution = NGX_CONF_UNSET_UINT;

    return conf;
}

static char *ngx_http_prefill_refactor_merge_loc_conf(ngx_conf_t *cf, void *parent, void *child) {
    ngx_http_prefill_refactor_loc_conf_t *prev = parent;
    ngx_http_prefill_refactor_loc_conf_t *conf = child;
    ngx_array_t *prefill_servers = NULL;
    ngx_array_t *bootstrap_ports = NULL;

    ngx_conf_merge_str_value(conf->prefill_location, prev->prefill_location, "");
    ngx_conf_merge_str_value(conf->prefill_servers_list, prev->prefill_servers_list, "");
    ngx_conf_merge_str_value(conf->decode_servers_list, prev->decode_servers_list, "");
    ngx_conf_merge_str_value(conf->bootstrap_ports_list, prev->bootstrap_ports_list, "");
    ngx_conf_merge_uint_value(conf->pd_backend, prev->pd_backend, 
                              NGX_HTTP_PREFILL_REFACTOR_BACKEND_VLLM);
    ngx_conf_merge_uint_value(conf->pd_distribution, prev->pd_distribution, 
                              NGX_HTTP_PREFILL_REFACTOR_DIST_P_BEFORE_D);

    // Parse prefill servers list
    if (conf->prefill_servers_list.len > 0 && conf->prefill_servers_info == NULL) {
        if (ngx_http_prefill_refactor_parse_servers(cf, &conf->prefill_servers_list, 
                                                     &prefill_servers) != NGX_OK) {
            return NGX_CONF_ERROR;
        }
    }

    // Parse decode servers list
    if (conf->decode_servers_list.len > 0 && conf->decode_servers == NULL) {
        if (ngx_http_prefill_refactor_parse_servers(cf, &conf->decode_servers_list, 
                                                     &conf->decode_servers) != NGX_OK) {
            return NGX_CONF_ERROR;
        }
    }

    // Parse bootstrap ports list and validate
    if (conf->bootstrap_ports_list.len > 0) {
        if (ngx_http_prefill_refactor_parse_servers(cf, &conf->bootstrap_ports_list, 
                                                     &bootstrap_ports) != NGX_OK) {
            return NGX_CONF_ERROR;
        }
        
        // Validate bootstrap ports pattern
        if (bootstrap_ports != NULL && bootstrap_ports->nelts > 0) {
            ngx_uint_t bootstrap_count = bootstrap_ports->nelts;
            ngx_uint_t prefill_count = (prefill_servers != NULL) ? prefill_servers->nelts : 0;
            
            // Check if bootstrap ports is a single port or matches prefill server count
            if (bootstrap_count != 1 && bootstrap_count != prefill_count) {
                ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
                    "bootstrap_ports_list must be: empty, single port, or match "
                    "prefill_servers_list count (%ui ports for %ui servers)",
                    bootstrap_count, prefill_count);
                return NGX_CONF_ERROR;
            }
            
            // Validate each port is a valid integer
            ngx_str_t *ports = (ngx_str_t *)bootstrap_ports->elts;
            for (ngx_uint_t i = 0; i < bootstrap_count; i++) {
                ngx_int_t port = ngx_atoi(ports[i].data, ports[i].len);
                if (port == NGX_ERROR || port < 1 || port > 65535) {
                    ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
                        "invalid port number \"%V\" in bootstrap_ports_list, must be 1-65535", &ports[i]);
                    return NGX_CONF_ERROR;
                }
            }
        }
    }

    // Build combined server info array
    if (prefill_servers != NULL && conf->prefill_servers_info == NULL) {
        if (ngx_http_prefill_refactor_build_server_info(cf, prefill_servers, bootstrap_ports, 
                                                        &conf->prefill_servers_info) != NGX_OK) {
            return NGX_CONF_ERROR;
        }
    }

    return NGX_CONF_OK;
}

static char *ngx_http_prefill_refactor_set_directive(ngx_conf_t *cf, ngx_command_t *cmd, void *conf) {
    ngx_http_prefill_refactor_loc_conf_t *slcf = conf;
    ngx_str_t *value;

    value = cf->args->elts;
    slcf->prefill_location = value[1];

    return NGX_CONF_OK;
}

static char *ngx_http_prefill_refactor_set_backend(ngx_conf_t *cf, ngx_command_t *cmd, void *conf) {
    ngx_http_prefill_refactor_loc_conf_t *slcf = conf;
    ngx_str_t *value;

    value = cf->args->elts;
    
    if (ngx_strcmp(value[1].data, "vllm") == 0) {
        slcf->pd_backend = NGX_HTTP_PREFILL_REFACTOR_BACKEND_VLLM;
    } else if (ngx_strcmp(value[1].data, "sglang") == 0) {
        slcf->pd_backend = NGX_HTTP_PREFILL_REFACTOR_BACKEND_SGLANG;
    } else {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
            "invalid pd_backend value \"%V\", must be \"vllm\" or \"sglang\"", &value[1]);
        return NGX_CONF_ERROR;
    }

    return NGX_CONF_OK;
}

static char *ngx_http_prefill_refactor_set_distribution(ngx_conf_t *cf, ngx_command_t *cmd, void *conf) {
    ngx_http_prefill_refactor_loc_conf_t *slcf = conf;
    ngx_str_t *value;

    value = cf->args->elts;
    
    if (ngx_strcmp(value[1].data, "p_before_d") == 0) {
        slcf->pd_distribution = NGX_HTTP_PREFILL_REFACTOR_DIST_P_BEFORE_D;
    } else if (ngx_strcmp(value[1].data, "p_after_d") == 0) {
        slcf->pd_distribution = NGX_HTTP_PREFILL_REFACTOR_DIST_P_AFTER_D;
    } else if (ngx_strcmp(value[1].data, "pd_together") == 0) {
        slcf->pd_distribution = NGX_HTTP_PREFILL_REFACTOR_DIST_PD_TOGETHER;
    } else {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
            "invalid pd_distribution value \"%V\", must be \"p_before_d\", "
            "\"p_after_d\", or \"pd_together\"", &value[1]);
        return NGX_CONF_ERROR;
    }

    return NGX_CONF_OK;
}

static ngx_int_t ngx_http_prefill_refactor_add_variables(ngx_conf_t *cf) {
    ngx_http_variable_t *var, *v;

    for (v = ngx_http_prefill_refactor_variables; v->name.len; v++) {
        var = ngx_http_add_variable(cf, &v->name, v->flags);
        if (var == NULL) {
            return NGX_ERROR;
        }

        var->get_handler = v->get_handler;
        var->data = v->data;
    }

    return NGX_OK;
}

static ngx_int_t ngx_http_prefill_refactor_init(ngx_conf_t *cf) {
    ngx_http_handler_pt *h;
    ngx_http_core_main_conf_t *cmcf;

    cmcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_core_module);

    h = ngx_array_push(&cmcf->phases[NGX_HTTP_ACCESS_PHASE].handlers);
    if (h == NULL) {
        return NGX_ERROR;
    }

    *h = ngx_http_prefill_refactor_handler;

    // Initialize random seed for load balancing
    srand((unsigned int)time(NULL));

    return NGX_OK;
}
