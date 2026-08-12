/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file root_info_exchanger.h
 * \brief 基于 TCP 的 HcclRootInfo 交换工具，封装 rank 间建链 + rootInfo 广播 + barrier
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "utils.h"

class RootInfoExchanger {
public:
    RootInfoExchanger(uint32_t rank, uint32_t rankNum, const std::string &endpoint)
        : rank_(rank), rankNum_(rankNum)
    {
        if (!ParseEndpoint(endpoint, ip_, port_)) {
            ERROR_LOG("invalid endpoint: %s", endpoint.c_str());
            valid_ = false;
        }
    }

    ~RootInfoExchanger()
    {
        CloseSocks();
    }

    bool Exchange(HcclRootInfo &rootInfo)
    {
        if (!valid_) {
            return false;
        }
        if (ConnectPeers() != 0) {
            ERROR_LOG("rank %u connect peers failed", rank_);
            return false;
        }
        if (ExchangeRootInfo(rootInfo) != 0) {
            ERROR_LOG("rank %u exchange rootInfo failed", rank_);
            return false;
        }
        TcpBarrier();
        return true;
    }

    void Barrier()
    {
        if (valid_ && !socks_.empty()) {
            TcpBarrier();
        }
    }

    void Close()
    {
        CloseSocks();
    }

private:
    static bool ParseEndpoint(const std::string &endpoint, std::string &ip, uint16_t &port)
    {
        const std::string prefix = "tcp://";
        if (endpoint.find(prefix) != 0) { return false; }
        std::string addr = endpoint.substr(prefix.size());
        auto colonPos = addr.rfind(':');
        if (colonPos == std::string::npos) { return false; }
        ip = addr.substr(0, colonPos);
        port = static_cast<uint16_t>(atoi(addr.substr(colonPos + 1).c_str()));
        return !ip.empty() && port > 0;
    }

    static int32_t SendAll(int32_t sock, const void *buf, size_t len)
    {
        size_t sent = 0;
        while (sent < len) {
            ssize_t n = send(sock, static_cast<const char *>(buf) + sent, len - sent, 0);
            if (n <= 0) { return -1; }
            sent += static_cast<size_t>(n);
        }
        return 0;
    }

    static int32_t RecvAll(int32_t sock, void *buf, size_t len)
    {
        size_t received = 0;
        while (received < len) {
            ssize_t n = recv(sock, static_cast<char *>(buf) + received, len - received, 0);
            if (n <= 0) { return -1; }
            received += static_cast<size_t>(n);
        }
        return 0;
    }

    int32_t ExchangeRootInfo(HcclRootInfo &rootInfo)
    {
        if (rank_ == 0) {
            auto ret = HcclGetRootInfo(&rootInfo);
            if (ret != HCCL_SUCCESS) {
                return -1;
            }
            for (uint32_t i = 1; i < rankNum_; i++) {
                if (SendAll(socks_[i - 1], &rootInfo, sizeof(HcclRootInfo)) != 0) { return -1; }
            }
        } else {
            if (RecvAll(socks_[0], &rootInfo, sizeof(HcclRootInfo)) != 0) { return -1; }
        }
        return 0;
    }

    int32_t TcpBarrier()
    {
        int32_t flag = 1;
        if (rank_ == 0) {
            for (uint32_t i = 1; i < rankNum_; i++) {
                int32_t recvFlag = 0;
                if (RecvAll(socks_[i - 1], &recvFlag, sizeof(recvFlag)) != 0) { return -1; }
            }
            for (uint32_t i = 1; i < rankNum_; i++) {
                if (SendAll(socks_[i - 1], &flag, sizeof(flag)) != 0) { return -1; }
            }
        } else {
            if (SendAll(socks_[0], &flag, sizeof(flag)) != 0) { return -1; }
            int32_t recvFlag = 0;
            if (RecvAll(socks_[0], &recvFlag, sizeof(recvFlag)) != 0) { return -1; }
        }
        return 0;
    }

    int32_t ConnectPeers()
    {
        if (rank_ == 0) {
            int32_t listenSock = socket(AF_INET, SOCK_STREAM, 0);
            if (listenSock < 0) { return -1; }
            int32_t opt = 1;
            setsockopt(listenSock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
            struct sockaddr_in addr = {};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port_);
            inet_pton(AF_INET, ip_.c_str(), &addr.sin_addr);
            if (bind(listenSock, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0 ||
                listen(listenSock, static_cast<int>(rankNum_)) < 0) {
                close(listenSock);
                return -1;
            }
            socks_.resize(rankNum_ - 1, -1);
            for (uint32_t i = 0; i < rankNum_ - 1; i++) {
                socks_[i] = accept(listenSock, nullptr, nullptr);
                if (socks_[i] < 0) {
                    close(listenSock);
                    return -1;
                }
            }
            close(listenSock);
        } else {
            socks_.resize(1, -1);
            int32_t sock = socket(AF_INET, SOCK_STREAM, 0);
            if (sock < 0) { return -1; }
            struct sockaddr_in addr = {};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port_);
            inet_pton(AF_INET, ip_.c_str(), &addr.sin_addr);
            for (int32_t retry = 0; retry < 100; retry++) {
                if (connect(sock, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) == 0) {
                    socks_[0] = sock;
                    return 0;
                }
                usleep(100000);
            }
            close(sock);
            return -1;
        }
        return 0;
    }

    void CloseSocks()
    {
        for (int32_t sock : socks_) {
            if (sock >= 0) { close(sock); }
        }
        socks_.clear();
    }

    uint32_t rank_;
    uint32_t rankNum_;
    std::string ip_;
    uint16_t port_{0};
    bool valid_{true};
    std::vector<int32_t> socks_;
};
