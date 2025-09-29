// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <boost/asio.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/write.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/experimental/concurrent_channel.hpp>
#include <boost/asio/experimental/awaitable_operators.hpp>

#include <iostream>
#include <vector>
#include <cstdint>
#include <array>
#include <ox_block_table.hpp>

namespace asio = boost::asio;
using asio::co_spawn;
using asio::detached;
using asio::use_awaitable;
using asio::experimental::concurrent_channel;
using asio::ip::tcp;
using namespace boost::asio::experimental::awaitable_operators;

inline void optimize_tcp_socket(tcp::socket &socket)
{
    boost::system::error_code ec;

    socket.set_option(tcp::no_delay(true), ec);
    if (ec)
        std::cerr << "Failed to set TCP_NODELAY: " << ec.message() << std::endl;

#ifdef TCP_QUICKACK
    int quickack = 1;
    setsockopt(socket.native_handle(), IPPROTO_TCP, TCP_QUICKACK, &quickack, sizeof(quickack));
#endif
}

class Session : public std::enable_shared_from_this<Session>
{
public:
    Session(tcp::socket socket, BlockTable &bt)
        : socket_(std::move(socket)), bt(bt)
    {
    }

    void start()
    {
        asio::co_spawn(socket_.get_executor(), [self = shared_from_this()]() -> asio::awaitable<void>
                       { co_await self->process_connection(); }, asio::detached);
    }

private:
    asio::awaitable<void> process_connection()
    {
        try
        {
            std::array<uint64_t, 8192> recv_buffer;

            while (true)
            {
                std::size_t bytes_read = co_await socket_.async_read_some(
                    asio::buffer(recv_buffer),
                    use_awaitable);

                size_t num_blocks = bytes_read / sizeof(int64_t);

                if (num_blocks > 0)
                {
                    block_list_t block_list;
                    block_list.assign(recv_buffer.begin(), recv_buffer.begin() + num_blocks);

#ifdef CONTENT_CHECK
                    auto bufs = bt.get_buffers(0, block_list, 0);
                    for (int i = 0; i < block_list.size(); i++)
                    {
                        int64_t *data = static_cast<int64_t *>(bufs[i].data());
                        *data = block_list[i];
                    }
#endif

                    co_await asio::async_write(socket_,
                                               bt.get_buffers(0, block_list, 0),
                                               use_awaitable);
                }
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Connection closed: " << e.what() << std::endl;
        }
    }

    tcp::socket socket_;
    BlockTable &bt;
    int rank = 0;
};

class Server
{
public:
    Server(asio::io_context &io_context, boost::asio::ip::tcp::endpoint &ep, BlockTable &bt)
        : acceptor_(io_context, ep), bt(bt)
    {
        std::cout << "OX Listening Port:" << ep.port() << std::endl;
    }

    asio::awaitable<void> run()
    {
        while (true)
        {
            try
            {
                tcp::socket socket = co_await acceptor_.async_accept(asio::use_awaitable);

                std::cout << "New connection from: "
                          << socket.remote_endpoint().address().to_string()
                          << ":" << socket.remote_endpoint().port() << std::endl;

                std::make_shared<Session>(std::move(socket), bt)->start();
            }
            catch (const boost::system::system_error &e)
            {
                std::cerr << "Accept error: " << e.what() << std::endl;
                if (e.code() == boost::asio::error::operation_aborted)
                {
                    break;
                }
            }
        }
        co_return;
    }

private:
    tcp::acceptor acceptor_;
    BlockTable &bt;
};