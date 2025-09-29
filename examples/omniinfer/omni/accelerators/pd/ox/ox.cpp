// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <set>
#include <atomic>
#include <mutex>
#include <exception>
#include <msgpack.hpp>
#include <sys/mman.h>
#include <shared_mutex>

#include <ox_config.hpp>
#include <zmq_coroutine.hpp>
#include <ox_metrics.hpp>
#include <ox_server.hpp>
#include <ox_log.hpp>

namespace asio = boost::asio;
using asio::co_spawn;
using asio::detached;
using asio::experimental::concurrent_channel;
using asio::ip::tcp;
using namespace boost::asio::experimental::awaitable_operators;

struct RequestMessage
{
    request_id_t request_id;
    table_id_t table_id;
    block_list_t block_ids;
    MSGPACK_DEFINE_MAP(request_id, table_id, block_ids);
};

struct ResponseMessage
{
    request_id_t request_id;
    bool success;
    MSGPACK_DEFINE_MAP(request_id, success);
};

using ResponseTask = std::tuple<client_id_t, request_id_t, bool>;
using ZMQChannel = concurrent_channel<asio::any_io_executor,
                                      void(boost::system::error_code, ResponseTask)>;

using ConnectionMessage = std::tuple<request_id_t, table_id_t, block_list_t>;
using ConnectionChannel = concurrent_channel<asio::any_io_executor, void(boost::system::error_code, ConnectionMessage)>;

using ShardMessage = std::tuple<std::string, block_list_t>;
using ShardChannel = concurrent_channel<asio::any_io_executor, void(boost::system::error_code, ShardMessage)>;

using GroupMessage = std::tuple<client_id_t, int>;
using GroupChannel = concurrent_channel<asio::any_io_executor, void(boost::system::error_code, GroupMessage)>;

class CoroutineConnection
{
public:
    CoroutineConnection(asio::io_context &io_context, boost::asio::ip::tcp::endpoint addr,
                        Config &config, int rank, BlockTable &bt,
                        ShardChannel &response)
        : socket(io_context),
          config(config),
          rank(rank),
          addr(addr),
          bt(bt),
          request(config.get_io_context(), 128),
          upstream(response)
    {
    }

    void start()
    {
        asio::co_spawn(socket.get_executor(), run(), asio::detached);
    }

    asio::awaitable<void> run()
    {
        try
        {
            co_await socket.async_connect(
                addr,
                asio::use_awaitable);
            optimize_tcp_socket(socket);

            while (true)
            {
                auto [request_id, table_id, ids] = co_await request.async_receive(asio::use_awaitable);
                auto bufs = bt.get_buffers(table_id, ids, rank);

                co_await (asio::async_write(socket,
                                            asio::buffer(ids.data(), ids.size() * sizeof(block_id_t)),
                                            asio::use_awaitable) &&
                          asio::async_read(socket,
                                           bufs,
                                           asio::use_awaitable));

#ifdef CONTENT_CHECK
                for (int i = 0; i < ids.size(); i++)
                {
                    int64_t *data = static_cast<int64_t *>(bufs[i].data());
                    assert(*data == ids[i]);
                }
#endif

                global_stats_update(ids.size() * bt.block_tp_size());

                co_await upstream.async_send(boost::system::error_code{}, std::make_tuple(request_id, ids), asio::use_awaitable);
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Connection " << addr.address() << ":" << addr.port() << " error: " << e.what() << "\n";
        }
    }

    asio::awaitable<void> submit_request(std::string &request_id, table_id_t table_id, block_list_t &block_list)
    {
        co_await request.async_send(boost::system::error_code{},
                                    std::make_tuple(request_id, table_id, block_list), asio::use_awaitable);
    }

private:
    tcp::socket socket;
    const Config &config;
    int rank;
    boost::asio::ip::tcp::endpoint addr;

    BlockTable &bt;
    ConnectionChannel request;
    ShardChannel &upstream;
};

class TPShard
{
public:
    TPShard(boost::asio::ip::tcp::endpoint addr, int rank, Config &config, BlockTable &bt, GroupChannel &channel)
        : ip(addr), rank(rank), downstream(config.get_io_context(), 128), upstream(channel)
    {
        for (int i = 0; i < config.connections_per_shard; i++)
        {
            connections.emplace_back(std::make_shared<CoroutineConnection>(config.get_io_context(), ip, config, rank, bt, downstream));
            connections[i]->start();
        }
    }

    asio::awaitable<void> gather(RequestMessage &req)
    {
        task_status[req.request_id] = std::set<block_id_t>(req.block_ids.begin(), req.block_ids.end());
        assert(task_status[req.request_id].size() == req.block_ids.size());

        size_t total_ids = req.block_ids.size();
        size_t num_conns = connections.size();

        assert(total_ids != 0 || num_conns != 0);

        size_t base_count = total_ids / num_conns;
        size_t remainder = total_ids % num_conns;
        auto it = req.block_ids.begin();

        for (size_t i = 0; i < num_conns; i++)
        {
            size_t count = base_count + (i < remainder ? 1 : 0);

            block_list_t ids(it, std::next(it, count));
            std::advance(it, count);

            size_t conn_index = last;
            asio::co_spawn(
                co_await asio::this_coro::executor,
                connections[conn_index]->submit_request(req.request_id, req.table_id, ids),
                asio::detached);

            last = (last + 1) % num_conns;
        }
    }

    asio::awaitable<void> run()
    {
        while (true)
        {
            auto [request_id, ids] = co_await downstream.async_receive(use_awaitable);

            for (auto id : ids)
            {
                task_status[request_id].erase(id);
            }

            if (task_status[request_id].empty())
            {
                task_status.erase(request_id);
                co_await upstream.async_send(boost::system::error_code{}, std::make_tuple(request_id, rank), asio::use_awaitable);
            }
        }
        co_return;
    }

private:
    int last = 0;
    boost::asio::ip::tcp::endpoint ip;
    int rank;
    std::vector<std::shared_ptr<CoroutineConnection>> connections;
    std::unordered_map<std::string, std::set<block_id_t>> task_status;

    ShardChannel downstream;
    GroupChannel &upstream;
};

class TPGroup
{
public:
    TPGroup(Config &config, BlockTable &bt, ZMQChannel &channel)
        : downstream(config.get_io_context(), 128), upstream(channel)
    {
        for (int rank = 0; rank < config.shard_list.size(); rank++)
        {
            auto &ip = config.shard_list[rank];
            auto shard = std::make_shared<TPShard>(ip, rank, config, bt, downstream);
            shards.push_back(shard);
            co_spawn(config.get_io_context(), shard->run(), detached);
        }
    }

    asio::awaitable<void> run()
    {
        try
        {
            while (true)
            {
                auto [request_id, rank] = co_await downstream.async_receive(use_awaitable);

                requests_mutex.lock();
                auto &[client_id, rank_finished] = requests_status[request_id];
                requests_mutex.unlock();

                rank_finished[rank] = true;
                if (std::all_of(rank_finished.begin(), rank_finished.end(),
                                [](bool b)
                                { return b; }))
                {
                    client_id_t cid = client_id;
                    requests_status.erase(request_id);
                    co_await upstream.async_send(boost::system::error_code{},
                                                 std::make_tuple(cid, request_id, true),
                                                 use_awaitable);
                }
            }
        }
        catch (const std::exception &e)
        {
            std::cout << "Response sender stopped: " << e.what() << std::endl;
        }
        co_return;
    }

    asio::awaitable<void> gather(client_id_t client_id, RequestMessage &req)
    {
        {
            std::unique_lock<std::shared_mutex> lock(requests_mutex);
            requests_status[req.request_id] = std::make_tuple(client_id, std::vector<bool>(shards.size(), false));
        }

        for (auto &shard : shards)
        {
            asio::co_spawn(
                co_await asio::this_coro::executor,
                shard->gather(req),
                asio::detached);
        }
    }

public:
    mutable std::shared_mutex requests_mutex;
    std::vector<std::shared_ptr<TPShard>> shards;
    std::unordered_map<request_id_t, std::tuple<client_id_t, std::vector<bool>>> requests_status;
    GroupChannel downstream;
    ZMQChannel &upstream;
};

awaitable<void> response_sender(ZmqCoroutineSocket &router_socket, ZMQChannel &response_channel)
{
    try
    {
        while (true)
        {
            auto [client_id, request_id, success] = co_await response_channel.async_receive(use_awaitable);

            ResponseMessage response = {request_id, success};
            std::stringstream buffer;
            msgpack::pack(buffer, response);
            std::string response_data = buffer.str();

            std::vector<zmq::message_t> response_messages;
            response_messages.emplace_back(client_id.data(), client_id.size());
            response_messages.emplace_back(response_data.data(), response_data.size());

            co_await router_socket.async_send_multipart(std::move(response_messages));
        }
    }
    catch (const std::exception &e)
    {
        std::cout << "Response sender stopped: " << e.what() << std::endl;
    }
}

awaitable<void> router_receiver(ZmqCoroutineSocket &router_socket,
                                ZMQChannel &response_channel, TPGroup &group)
{
    while (true)
    {
        try
        {
            auto msg = co_await router_socket.async_recv_multipart();
            if (msg && msg->size() == 2)
            {
                std::vector<zmq::message_t> messages = std::move(*msg);
                const auto *data0 = static_cast<const uint8_t *>(messages[0].data());
                client_id_t client_id(data0, data0 + messages[0].size());

                const char *data1 = static_cast<const char *>(messages[1].data());
                std::string request_data(data1, data1 + messages[1].size());

                msgpack::object_handle handle = msgpack::unpack(request_data.data(), request_data.size());
                RequestMessage request;
                handle.get().convert(request);

                co_spawn(co_await asio::this_coro::executor,
                         group.gather(client_id, request),
                         detached);
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Receiver error: " << e.what() << std::endl;
        }
    }
}

int main(int argc, char *argv[])
{
    try
    {
        Config config = parse_arguments(argc, argv);
        BlockTable bt(config);

        asio::io_context &io_context = config.get_io_context();

        g_program_start_time = std::chrono::steady_clock::now();

        std::vector<std::shared_ptr<Server>> server_list;
        for (auto &endpoint : config.server_list)
        {
            server_list.emplace_back(std::make_shared<Server>(io_context, endpoint, bt));
        }

        for (auto &server : server_list)
        {
            co_spawn(io_context,
                     server->run(),
                     detached);
        }

        std::shared_ptr<TPGroup> tp_group;
        ZMQChannel response_channel(io_context, 128);
        ZmqCoroutineSocket zmq_router(ZMQ_ROUTER, io_context);

        if (config.shard_list.size() > 0)
        {
            tp_group = std::make_shared<TPGroup>(config, bt, response_channel);

            std::string address = "tcp://*:" + std::to_string(config.zmq_port);
            zmq_router.bind(address);

            co_spawn(io_context,
                     tp_group->run(),
                     detached);

            co_spawn(io_context,
                     router_receiver(zmq_router, response_channel, *tp_group),
                     detached);

            co_spawn(io_context,
                     response_sender(zmq_router, response_channel),
                     detached);

            co_spawn(io_context,
                     print_statistics(),
                     detached);

            std::cout << "Omni Xfer started. ZMQ: " << address << std::endl;
        }

        std::vector<std::thread> threads;
        for (size_t i = 0; i < config.num_threads; ++i)
        {
            threads.emplace_back([&io_context]()
                                 { io_context.run(); });
        }

        io_context.run();

        for (auto &thread : threads)
        {
            thread.join();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

// g++ -std=c++20 -DNDEBUG -fcoroutines -I./  -g -march=native  ox.cpp -o ox -lzmq -lpthread