// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "omni_tokenizer_worker.h"
#include "omni_utils.h"

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>

static void *omni_tokenizer_worker_func(void *data)
{
    ngx_omni_tokenize_worker_t *worker = (ngx_omni_tokenize_worker_t *)data;
    struct epoll_event events[1];
    uint32_t slot_ids[OMNI_MAX_BATCH];
    omni_tokenizer_request *batch_requests[OMNI_MAX_BATCH];
    ngx_uint_t batch_count = 0;

    if (omni_tokenizer_init() != 0)
    {
        ngx_log_error(NGX_LOG_ERR, worker->log, 0, "Failed to initialize tokenizer in worker thread");
        return NULL;
    }

    char model_path_str[256];
    ngx_snprintf((u_char *)model_path_str, sizeof(model_path_str), "%V", &worker->model_path);

    if (omni_init_tokenizer(model_path_str) != 0)
    {
        ngx_log_error(NGX_LOG_ERR, worker->log, 0, "Failed to init tokenizer with model: %V", &worker->model_path);
        omni_tokenizer_cleanup();
        return NULL;
    }

    struct epoll_event ev;
    ev.events = EPOLLIN | EPOLLET;
    ev.data.fd = worker->cmd_pipe[OMNI_PIPE_READ];
    if (epoll_ctl(worker->epoll_fd, EPOLL_CTL_ADD, worker->cmd_pipe[OMNI_PIPE_READ], &ev) == -1)
    {
        ngx_log_error(NGX_LOG_ERR, worker->log, ngx_errno, "epoll_ctl add cmd pipe failed");
        omni_tokenizer_cleanup();
        return NULL;
    }

    while (worker->active)
    {
        int nfds = epoll_wait(worker->epoll_fd, events, 1, -1);
        if (nfds == -1)
        {
            if (errno == EINTR)
            {
                continue;
            }
            ngx_log_error(NGX_LOG_ERR, worker->log, ngx_errno, "epoll_wait failed");
            break;
        }

        ssize_t nread;
        batch_count = 0;

        while ((nread = read(worker->cmd_pipe[OMNI_PIPE_READ],
                             slot_ids,
                             sizeof(uint32_t) * OMNI_MAX_BATCH)) > 0)
        {
            batch_count = nread / sizeof(uint32_t);
            for (uint32_t i = 0; i < batch_count; i++)
            {
                uint32_t slot_id = slot_ids[i];
                printf("Slot id:%u\n", slot_id);
                if (slot_id == OMNI_SLOT_EXIT)
                {
                    worker->active = 0;
                    break;
                }

                omni_req_t *req = omni_id_to_req(slot_id);
                if (req != NULL)
                {
                    batch_requests[i] = &req->tokenizer_req;
                }
                else
                {
                    ngx_log_error(NGX_LOG_ERR, worker->log, ngx_errno,
                                  "Receive invalid slot_id: %u", slot_id);
                    exit(-1);
                }
            }

            printf("Tokenize batch %ld\n", batch_count);

            if (omni_batch_chat_encode(batch_requests, batch_count) == 0)
            {
                printf("Write response back:%ld\n", nread);
            }
            else
            {
                for (int i = 0; i < batch_count; i++)
                {
                    batch_requests[i]->failed = true;
                }
                printf("Tokenize failed.\n");
            }
            write(worker->resp_pipe[OMNI_PIPE_WRITE], slot_ids, nread);
        }

        if (nread == -1 && errno != EAGAIN && errno != EWOULDBLOCK)
        {
            ngx_log_error(NGX_LOG_ERR, worker->log, ngx_errno,
                          "read from cmd pipe failed");
            exit(-1);
        }
    }

    omni_tokenizer_cleanup();
    return NULL;
}

ngx_int_t omni_tokenizer_worker_init(ngx_cycle_t *cycle, ngx_omni_tokenize_worker_t *worker)
{
    if (setenv("LANG", "en_US.UTF-8", 1) != 0)
    {
        perror("Failed to set LANG environment variable");
        return NGX_ERROR;
    }

    setenv("LC_ALL", "en_US.UTF-8", 1);
    setenv("LC_CTYPE", "en_US.UTF-8", 1);

    if (pipe(worker->cmd_pipe) == -1)
    {
        ngx_log_error(NGX_LOG_ERR, cycle->log, ngx_errno, "Failed to create command pipe");
        return NGX_ERROR;
    }

    if (pipe(worker->resp_pipe) == -1)
    {
        ngx_log_error(NGX_LOG_ERR, cycle->log, ngx_errno, "Failed to create response pipe");
        close(worker->cmd_pipe[OMNI_PIPE_READ]);
        close(worker->cmd_pipe[OMNI_PIPE_WRITE]);
        return NGX_ERROR;
    }

    worker->epoll_fd = epoll_create1(0);
    if (worker->epoll_fd == -1)
    {
        ngx_log_error(NGX_LOG_ERR, cycle->log, ngx_errno, "Failed to create epoll");
        close(worker->cmd_pipe[OMNI_PIPE_READ]);
        close(worker->cmd_pipe[OMNI_PIPE_WRITE]);
        close(worker->resp_pipe[OMNI_PIPE_READ]);
        close(worker->resp_pipe[OMNI_PIPE_WRITE]);
        return NGX_ERROR;
    }

    int flags;
    int fds[] = {
        worker->cmd_pipe[OMNI_PIPE_READ], worker->cmd_pipe[OMNI_PIPE_WRITE],
        worker->resp_pipe[OMNI_PIPE_READ], worker->resp_pipe[OMNI_PIPE_WRITE]};

    for (int i = 0; i < 4; i++)
    {
        flags = fcntl(fds[i], F_GETFL, 0);
        if (flags == -1 || fcntl(fds[i], F_SETFL, flags | O_NONBLOCK) == -1)
        {
            ngx_log_error(NGX_LOG_ERR, cycle->log, ngx_errno, "Failed to set pipe non-blocking");
            close(worker->epoll_fd);
            close(worker->cmd_pipe[OMNI_PIPE_READ]);
            close(worker->cmd_pipe[OMNI_PIPE_WRITE]);
            close(worker->resp_pipe[OMNI_PIPE_READ]);
            close(worker->resp_pipe[OMNI_PIPE_WRITE]);
            return NGX_ERROR;
        }
    }

    worker->log = cycle->log;
    worker->active = 1;

    int err = pthread_create(&worker->thread_id, NULL, omni_tokenizer_worker_func, worker);
    if (err != 0)
    {
        ngx_log_error(NGX_LOG_ERR, cycle->log, 0, "Failed to create worker thread: %s", strerror(err));
        close(worker->epoll_fd);
        close(worker->cmd_pipe[OMNI_PIPE_READ]);
        close(worker->cmd_pipe[OMNI_PIPE_WRITE]);
        close(worker->resp_pipe[OMNI_PIPE_READ]);
        close(worker->resp_pipe[OMNI_PIPE_WRITE]);
        return NGX_ERROR;
    }

    return NGX_OK;
}

void omni_tokenizer_worker_exit(ngx_omni_tokenize_worker_t *worker)
{
    if (!worker->active)
    {
        return;
    }

    worker->active = 0;
    uint32_t exit_cmd = OMNI_SLOT_EXIT;

    ssize_t nwritten = write(worker->cmd_pipe[OMNI_PIPE_WRITE], &exit_cmd, sizeof(exit_cmd));
    if (nwritten != sizeof(exit_cmd))
    {
        ngx_log_error(NGX_LOG_WARN, worker->log, 0, "Failed to send exit command to worker");
    }

    pthread_join(worker->thread_id, NULL);

    close(worker->epoll_fd);
    close(worker->cmd_pipe[OMNI_PIPE_READ]);
    close(worker->cmd_pipe[OMNI_PIPE_WRITE]);
    close(worker->resp_pipe[OMNI_PIPE_READ]);
    close(worker->resp_pipe[OMNI_PIPE_WRITE]);

    ngx_log_error(NGX_LOG_INFO, worker->log, 0, "Tokenizer worker exited");
}

ngx_int_t omni_tokenizer_worker_submit(ngx_omni_tokenize_worker_t *worker, uint32_t slot_id)
{
    ssize_t nwritten = write(worker->cmd_pipe[OMNI_PIPE_WRITE], &slot_id, sizeof(slot_id));
    if (nwritten != sizeof(slot_id))
    {
        if (errno == EAGAIN || errno == EWOULDBLOCK)
        {
            ngx_log_error(NGX_LOG_WARN, worker->log, 0, "Command pipe full, slot_id: %ui", slot_id);
        }
        else
        {
            ngx_log_error(NGX_LOG_ERR, worker->log, ngx_errno, "Failed to submit to tokenizer worker");
        }
        return NGX_ERROR;
    }

    return NGX_OK;
}