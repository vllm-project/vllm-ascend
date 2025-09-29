// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <pthread.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/epoll.h>
#include "omni_tokenizer.h"

#define OMNI_PIPE_READ 0
#define OMNI_PIPE_WRITE 1
#define OMNI_SLOT_EXIT UINT32_MAX
#define OMNI_MAX_BATCH 256

typedef struct
{
    pthread_t thread_id;
    int cmd_pipe[2];  // 命令管道: nginx worker → tokenizer worker
    int resp_pipe[2]; // 响应管道: tokenizer worker → nginx worker
    int epoll_fd;
    ngx_int_t active;
    ngx_str_t model_path;
    ngx_int_t kv_block_size;
    ngx_log_t *log;
    ngx_connection_t *resp_connection; // 响应管道的连接
} ngx_omni_tokenize_worker_t;

ngx_int_t omni_tokenizer_worker_init(ngx_cycle_t *cycle, ngx_omni_tokenize_worker_t *worker);
void omni_tokenizer_worker_exit(ngx_omni_tokenize_worker_t *worker);
ngx_int_t omni_tokenizer_worker_submit(ngx_omni_tokenize_worker_t *worker, uint32_t slot_id);