// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once
#include <zmq.hpp>
#include <boost/asio.hpp>
#include <boost/asio/experimental/awaitable_operators.hpp>
#include <optional>

namespace asio = boost::asio;
using asio::awaitable;
using asio::use_awaitable;

class ZmqCoroutineSocket
{
public:
    ZmqCoroutineSocket(int type, asio::io_context &io_ctx)
        : context(1),
          socket(context, type),
          zmq_fd(get_socket_fd()),
          zmq_read_event(io_ctx, zmq_fd)
    {
        socket.set(zmq::sockopt::linger, 0);
    }

    ~ZmqCoroutineSocket()
    {
        if (zmq_fd != -1)
        {
            zmq_read_event.release();
        }
    }

    inline void bind(std::string address)
    {
        socket.bind(address);
    }

    awaitable<std::optional<std::vector<zmq::message_t>>> async_recv_multipart()
    {
        std::vector<zmq::message_t> messages;

        while (true)
        {
            bool more = true;
            bool received_any = false;

            while (more)
            {
                zmq::message_t message;
                auto result = socket.recv(message, zmq::recv_flags::dontwait);

                if (result.has_value())
                {
                    messages.push_back(std::move(message));
                    more = socket.get(zmq::sockopt::rcvmore);
                    received_any = true;
                }
                else
                {
                    if (errno == EAGAIN)
                    {
                        if (!received_any)
                        {
                            co_await wait_for_read();
                            break;
                        }
                        else
                        {
                            co_await wait_for_read();
                            continue;
                        }
                    }
                    co_return std::nullopt;
                }
            }

            if (received_any && !more)
            {
                co_return messages;
            }
        }
    }

    awaitable<bool> async_send_multipart(std::vector<zmq::message_t> messages)
    {
        for (size_t i = 0; i < messages.size(); ++i)
        {
            zmq::send_flags flags = (i < messages.size() - 1) ? zmq::send_flags::sndmore : zmq::send_flags::none;

            while (true)
            {
                auto result = socket.send(messages[i], flags | zmq::send_flags::dontwait);

                if (result.has_value())
                {
                    break;
                }
                else if (errno == EAGAIN)
                {
                    co_await wait_for_write();
                }
                else
                {
                    co_return false;
                }
            }
        }
        co_return true;
    }

private:
    int get_socket_fd()
    {
        try
        {
            return socket.get(zmq::sockopt::fd);
        }
        catch (const zmq::error_t &e)
        {
            return -1;
        }
    }

    awaitable<void> wait_for_read()
    {
        while (!read_ready)
        {
            co_await zmq_read_event.async_wait(
                asio::posix::stream_descriptor::wait_read,
                asio::use_awaitable);

            int events = socket.get(zmq::sockopt::events);

            if (events & ZMQ_POLLIN)
            {
                read_ready = true;
            }

            if (events & ZMQ_POLLOUT)
            {
                write_ready = true;
            }
        }
        read_ready = false;
    }

    awaitable<void> wait_for_write()
    {
        while (!write_ready)
        {
            co_await asio::post(asio::use_awaitable);
        }
        write_ready = false;
    }

    zmq::context_t context;
    zmq::socket_t socket;
    int zmq_fd;
    asio::posix::stream_descriptor zmq_read_event;
    bool read_ready = false;
    bool write_ready = false;
};