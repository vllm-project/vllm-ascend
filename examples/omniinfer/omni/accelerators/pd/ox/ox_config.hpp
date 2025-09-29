// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <boost/asio.hpp>
#include <boost/algorithm/string.hpp>

#pragma once

namespace asio = boost::asio;

using address_list_t = std::vector<boost::asio::ip::tcp::endpoint>;

struct Config
{
    std::shared_ptr<asio::io_context> io_context;

    asio::io_context &get_io_context()
    {
        return *io_context;
    }

    address_list_t server_list;
    address_list_t shard_list;

    std::string block_table_shm;
    size_t num_block_tables = 1;
    size_t block_size = 576 * 2 * 61 * 128; // Deepseek v3 MLA size with BF16
    size_t num_blocks = 1024;
    size_t num_threads = 16;
    size_t connections_per_shard = 16;
    int zmq_port = 5555;

    inline size_t block_table_size() const
    {
        return num_blocks * block_size;
    }

    inline size_t shm_mem_size() const
    {
        return num_block_tables * block_table_size();
    }

    inline size_t tp_size() const
    {
        if (shard_list.size() > 0)
        {
            return shard_list.size();
        }
        return 1;
    }
};

void print_usage(const char *program_name)
{
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --addr <ip:port,ip:port>          Server IP list (7.150.8.141:9000,7.150.8.141:9001)\n"
              << "  --shard-list <ip:port,ip:port>    Shard server IP list\n"
              << "  --zmq-port <port>                 ZMQ port\n"
              << "  --block-table-shm <name>          Shared memory name for block table\n"
              << "  --num-block-tables <num>          Number of block tables (default: 1)\n"
              << "  --block-size <size>               Block size with unit (e.g., 8784KB, 2196KB)\n"
              << "  --num-blocks <num>                Number of blocks in each block table\n"
              << "  --num-threads <threads>           Number of threads (default: 16)\n"
              << "  --num-connections <conn>          Number of connections per shard (default: 16)\n"
              << "  -h                                Show this help message\n";
}

address_list_t parse_address_list(const std::string &address_str)
{
    address_list_t endpoints;
    std::vector<std::string> address_pairs;
    boost::split(address_pairs, address_str, boost::is_any_of(","), boost::token_compress_on);

    for (const auto &addr_pair : address_pairs)
    {
        std::vector<std::string> parts;
        boost::split(parts, addr_pair, boost::is_any_of(":"), boost::token_compress_on);

        if (parts.size() != 2)
        {
            throw std::runtime_error("Invalid address format: " + addr_pair);
        }

        std::string ip = parts[0];
        int port = std::stoi(parts[1]);

        boost::asio::ip::address address = boost::asio::ip::address::from_string(ip);
        endpoints.emplace_back(address, port);
    }

    return endpoints;
}

size_t parse_size_with_unit(const std::string &size_str)
{
    if (size_str.empty())
        return 0;

    std::string num_str = size_str;
    size_t multiplier = 1;

    std::string upper_str = size_str;
    boost::to_upper(upper_str);

    if (upper_str.find("KB") != std::string::npos)
    {
        multiplier = 1024;
        num_str = upper_str.substr(0, upper_str.find("KB"));
    }

    num_str.erase(std::remove_if(num_str.begin(), num_str.end(),
                                 [](char c)
                                 { return !std::isdigit(c); }),
                  num_str.end());

    if (num_str.empty())
    {
        throw std::runtime_error("Invalid size format: " + size_str);
    }

    return std::stoull(num_str) * multiplier;
}

Config parse_arguments(int argc, char *argv[])
{
    Config config;
    config.io_context = std::make_shared<asio::io_context>();

    try
    {
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            if (arg == "--addr" && i + 1 < argc)
            {
                config.server_list = parse_address_list(argv[++i]);
            }
            else if ((arg == "--shard-list" || arg == "--shard_list") && i + 1 < argc)
            {
                config.shard_list = parse_address_list(argv[++i]);
            }
            else if (arg == "--block-table-shm" || arg == "--block_table_shm")
            {
                config.block_table_shm = argv[++i];
            }
            else if ((arg == "--num-block-tables" || arg == "--num_block_tables") && i + 1 < argc)
            {
                config.num_block_tables = std::stoul(argv[++i]);
            }
            else if ((arg == "--block-size" || arg == "--block_size") && i + 1 < argc)
            {
                config.block_size = parse_size_with_unit(argv[++i]);
            }
            else if ((arg == "--num-blocks" || arg == "--num_blocks") && i + 1 < argc)
            {
                config.num_blocks = std::stoul(argv[++i]);
            }
            else if ((arg == "--num-threads" || arg == "--num_threads") && i + 1 < argc)
            {
                config.num_threads = std::stoul(argv[++i]);
            }
            else if ((arg == "--num-connections" || arg == "--num_connections") && i + 1 < argc)
            {
                config.connections_per_shard = std::stoul(argv[++i]);
            }
            else if ((arg == "--zmq-port" || arg == "--zmq_port") && i + 1 < argc)
            {
                config.zmq_port = std::stoul(argv[++i]);
            }
            else if (arg == "-h" || arg == "--help")
            {
                print_usage(argv[0]);
                exit(0);
            }
            else
            {
                std::cerr << "Unknown argument: " << arg << std::endl;
                print_usage(argv[0]);
                exit(1);
            }
        }

        if (config.server_list.empty() && config.shard_list.empty())
        {
            throw std::runtime_error("Either --addr or --shard-list must be specified");
        }

        if (config.block_table_shm.empty())
        {
            throw std::runtime_error("--block-table-shm must be specified");
        }

        if (config.num_blocks == 0)
        {
            throw std::runtime_error("--num-blocks must be specified and greater than 0");
        }

        if (config.block_size == 0)
        {
            throw std::runtime_error("--block-size must be specified and greater than 0");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        print_usage(argv[0]);
        exit(1);
    }

    return config;
}