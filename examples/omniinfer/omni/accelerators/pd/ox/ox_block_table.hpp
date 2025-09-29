// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#include <boost/asio.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/write.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/experimental/concurrent_channel.hpp>

#include <iostream>
#include <vector>
#include <cstdint>
#include <memory>
#include <thread>
#include <set>
#include <atomic>
#include <mutex>
#include <string>
#include <exception>
#include <msgpack.hpp>
#include <sys/mman.h>

namespace asio = boost::asio;
using asio::co_spawn;
using asio::detached;
using asio::experimental::concurrent_channel;
using asio::ip::tcp;

using request_id_t = std::string;
using client_id_t = std::string;
using block_id_t = int64_t;
using block_list_t = std::vector<block_id_t>;
using table_id_t = int64_t;

class BlockTable : public std::enable_shared_from_this<BlockTable>
{
public:
    BlockTable(Config &config) : num_blocks(config.num_blocks), config(config)
    {
        fd = shm_open(config.block_table_shm.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        if (fd >= 0)
        {
            if (ftruncate(fd, config.shm_mem_size()) != 0)
            {
                throw std::system_error(errno, std::generic_category(), "ftruncate failed");
            }
        }
        else
        {
            if (errno == EEXIST)
            {
                fd = shm_open(config.block_table_shm.c_str(), O_RDWR, 0);
            }
            else
            {
                throw std::system_error(errno, std::generic_category(), "shm_open create failed");
            }
        }

        base = mmap(nullptr, config.shm_mem_size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (base == nullptr)
        {
            throw std::system_error(errno, std::generic_category(), "mmap failed");
        }

        if (mlock(base, config.shm_mem_size()) != 0)
        {
            throw std::system_error(errno, std::generic_category(), "pin memory failed");
        }
    }

    inline void *block_addr(table_id_t table_id, size_t block_id)
    {
        assert(block_id < num_blocks);
        return ((char *)base) + table_id * config.block_table_size() + block_id * config.block_size;
    }

    inline size_t block_tp_size()
    {
        return config.block_size / config.tp_size();
    }

    inline void *block_tp_addr(table_id_t table_id, size_t block_id, int rank)
    {
        return static_cast<char *>(block_addr(table_id, block_id)) + block_tp_size() * rank;
    }

    std::vector<asio::mutable_buffer> get_buffers(table_id_t table_id, block_list_t &block_ids, int rank)
    {
        std::vector<boost::asio::mutable_buffer> buffers;
        buffers.reserve(block_ids.size());

        for (std::size_t id : block_ids)
        {
            buffers.emplace_back(block_tp_addr(table_id, id, rank), block_tp_size());
        }

        return buffers;
    }

private:
    int num_blocks;
    int fd;
    void *base;
    Config &config;
};