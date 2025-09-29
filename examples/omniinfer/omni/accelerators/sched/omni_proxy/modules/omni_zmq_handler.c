// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "omni_zmq_handler.h"
#include <string.h>

static void omni_zmq_dummy_write_handler(ngx_event_t *ev)
{
}

static void omni_zmq_event_handler(ngx_event_t *ev)
{
    omni_zmq_handler_t *handler = ev->data;

    int events;
    size_t events_size = sizeof(events);
    if (zmq_getsockopt(handler->zmq_socket, ZMQ_EVENTS, &events, &events_size) != 0)
    {
        ngx_log_error(NGX_LOG_ERR, handler->log, 0,
                      "ZMQ get events failed: %s", zmq_strerror(errno));
        handler->active = 0;
        return;
    }

    if (events & ZMQ_POLLERR)
    {
        ngx_log_error(NGX_LOG_ERR, handler->log, 0,
                      "ZMQ connection error detected");
        handler->active = 0;
        return;
    }

    while (handler->active)
    {
        zmq_msg_t topic_msg, seq_msg, payload_msg;
        int more;
        size_t more_size = sizeof(more);

        if (zmq_msg_init(&topic_msg) != 0)
            break;
        if (zmq_msg_recv(&topic_msg, handler->zmq_socket, ZMQ_DONTWAIT) == -1)
        {
            zmq_msg_close(&topic_msg);
            if (errno == EAGAIN)
                break;
            ngx_log_error(NGX_LOG_ERR, handler->log, errno,
                          "ZMQ recv topic failed: %s", zmq_strerror(errno));
            handler->active = 0;
            break;
        }
        if (zmq_getsockopt(handler->zmq_socket, ZMQ_RCVMORE, &more, &more_size) != 0 || !more)
        {
            zmq_msg_close(&topic_msg);
            continue;
        }

        if (zmq_msg_init(&seq_msg) != 0)
        {
            zmq_msg_close(&topic_msg);
            break;
        }
        if (zmq_msg_recv(&seq_msg, handler->zmq_socket, 0) == -1)
        {
            zmq_msg_close(&topic_msg);
            zmq_msg_close(&seq_msg);
            ngx_log_error(NGX_LOG_ERR, handler->log, errno,
                          "ZMQ recv seq failed: %s", zmq_strerror(errno));
            break;
        }
        if (zmq_getsockopt(handler->zmq_socket, ZMQ_RCVMORE, &more, &more_size) != 0 || !more)
        {
            zmq_msg_close(&topic_msg);
            zmq_msg_close(&seq_msg);
            continue;
        }

        if (zmq_msg_init(&payload_msg) != 0)
        {
            zmq_msg_close(&topic_msg);
            zmq_msg_close(&seq_msg);
            break;
        }
        if (zmq_msg_recv(&payload_msg, handler->zmq_socket, 0) == -1)
        {
            zmq_msg_close(&topic_msg);
            zmq_msg_close(&seq_msg);
            zmq_msg_close(&payload_msg);
            ngx_log_error(NGX_LOG_ERR, handler->log, errno,
                          "ZMQ recv payload failed: %s", zmq_strerror(errno));
            break;
        }

        if (zmq_getsockopt(handler->zmq_socket, ZMQ_RCVMORE, &more, &more_size) == 0 && more)
        {
            zmq_msg_close(&topic_msg);
            zmq_msg_close(&seq_msg);
            zmq_msg_close(&payload_msg);
            continue;
        }

        if (handler->message_callback)
        {
            const char *topic = zmq_msg_data(&topic_msg);
            const void *payload = zmq_msg_data(&payload_msg);
            size_t length = zmq_msg_size(&payload_msg);
            handler->message_callback(handler, topic, payload, length);
        }

        zmq_msg_close(&topic_msg);
        zmq_msg_close(&seq_msg);
        zmq_msg_close(&payload_msg);
    }
}
static void omni_zmq_cleanup_resources(omni_zmq_handler_t *handler)
{
    handler->active = 0;

    if (handler->zmq_connection)
    {
        ngx_del_conn(handler->zmq_connection, 0);
        ngx_free_connection(handler->zmq_connection);
        handler->zmq_connection = NULL;
        handler->zmq_event = NULL;
    }

    if (handler->zmq_socket)
    {
        zmq_close(handler->zmq_socket);
        handler->zmq_socket = NULL;
    }

    if (handler->zmq_context)
    {
        zmq_ctx_destroy(handler->zmq_context);
        handler->zmq_context = NULL;
    }
}

ngx_int_t omni_zmq_handler_reinit(omni_zmq_handler_t *handler)
{
    if (handler->active)
    {
        return NGX_OK;
    }

    omni_zmq_cleanup_resources(handler);

    handler->zmq_context = zmq_ctx_new();
    if (!handler->zmq_context)
    {
        ngx_log_error(NGX_LOG_ERR, handler->log, 0,
                      "ZMQ context creation failed");
        return NGX_ERROR;
    }

    handler->zmq_socket = zmq_socket(handler->zmq_context, ZMQ_SUB);
    if (!handler->zmq_socket)
    {
        ngx_log_error(NGX_LOG_ERR, handler->log, 0,
                      "ZMQ socket creation failed");
        zmq_ctx_destroy(handler->zmq_context);
        handler->zmq_context = NULL;
        return NGX_ERROR;
    }

    int timeout = 100;
    zmq_setsockopt(handler->zmq_socket, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));

    char addr_str[256];
    ngx_snprintf((u_char *)addr_str, sizeof(addr_str), "tcp://%V", &handler->zmq_address);
    addr_str[handler->zmq_address.len + 6] = 0;

    if (zmq_connect(handler->zmq_socket, addr_str) != 0)
    {
        ngx_log_error(NGX_LOG_ERR, handler->log, 0,
                      "ZMQ connect to %s failed: %s", addr_str, zmq_strerror(errno));
        omni_zmq_cleanup_resources(handler);
        return NGX_ERROR;
    }

    char topic_str[256];
    ngx_snprintf((u_char *)topic_str, sizeof(topic_str), "%V", &handler->subscribe_topic);
    topic_str[handler->subscribe_topic.len] = 0;

    if (zmq_setsockopt(handler->zmq_socket, ZMQ_SUBSCRIBE, topic_str, strlen(topic_str)) != 0)
    {
        ngx_log_error(NGX_LOG_ERR, handler->log, 0,
                      "ZMQ subscribe to %s failed: %s", topic_str, zmq_strerror(errno));
        omni_zmq_cleanup_resources(handler);
        return NGX_ERROR;
    }

    int zmq_fd;
    size_t fd_size = sizeof(zmq_fd);
    if (zmq_getsockopt(handler->zmq_socket, ZMQ_FD, &zmq_fd, &fd_size) != 0)
    {
        ngx_log_error(NGX_LOG_ERR, handler->log, 0,
                      "ZMQ get FD failed: %s", zmq_strerror(errno));
        omni_zmq_cleanup_resources(handler);
        return NGX_ERROR;
    }

    handler->zmq_connection = ngx_get_connection(zmq_fd, handler->log);
    if (!handler->zmq_connection)
    {
        ngx_log_error(NGX_LOG_ERR, handler->log, 0,
                      "ZMQ get connection failed");
        omni_zmq_cleanup_resources(handler);
        return NGX_ERROR;
    }

    if (ngx_nonblocking(zmq_fd) == -1)
    {
        ngx_log_error(NGX_LOG_ERR, handler->log, 0,
                      "ZMQ set nonblocking failed");
        omni_zmq_cleanup_resources(handler);
        return NGX_ERROR;
    }

    handler->zmq_connection->read->handler = omni_zmq_event_handler;
    handler->zmq_connection->read->data = handler;
    handler->zmq_connection->read->log = handler->log;

    handler->zmq_connection->write->handler = omni_zmq_dummy_write_handler;
    handler->zmq_connection->write->data = handler;
    handler->zmq_connection->write->log = handler->log;
    handler->zmq_connection->write->ready = 1;

    handler->zmq_event = handler->zmq_connection->read;

    if (ngx_add_conn(handler->zmq_connection) != NGX_OK)
    {
        ngx_log_error(NGX_LOG_ERR, handler->log, 0,
                      "ZMQ add connection failed");
        omni_zmq_cleanup_resources(handler);
        return NGX_ERROR;
    }

    handler->active = 1;
    ngx_log_error(NGX_LOG_INFO, handler->log, 0,
                  "ZMQ handler initialized for %s, topic: %s", addr_str, topic_str);

    return NGX_OK;
}

ngx_int_t omni_zmq_handler_init(ngx_cycle_t *cycle,
                                omni_zmq_handler_t *handler,
                                ngx_str_t zmq_address,
                                ngx_str_t subscribe_topic,
                                omni_zmq_msg_callback_t callback)
{
    ngx_int_t saved_index = handler->index;
    ngx_memzero(handler, sizeof(omni_zmq_handler_t));
    handler->index = saved_index;
    handler->log = cycle->log;
    handler->cycle = cycle;
    handler->zmq_address = zmq_address;
    handler->subscribe_topic = subscribe_topic;
    handler->message_callback = callback;
    handler->active = 0;

    return omni_zmq_handler_reinit(handler);
}

void omni_zmq_handler_exit(omni_zmq_handler_t *handler)
{
    omni_zmq_cleanup_resources(handler);
}