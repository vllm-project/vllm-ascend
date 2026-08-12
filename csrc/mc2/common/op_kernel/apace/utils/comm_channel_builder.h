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
 * \file comm_channel_builder.h
 * \brief Host 侧 HCCL channel 创建工具，封装内存注册 + channel 建链逻辑
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "hccl/hccl.h"
#include "hccl/hccl_res.h"
#include "apace/block/aiv_comm/collective_comm_context.h"

constexpr uint8_t BUILDER_COMM_ENGINE_AIV = 4;
constexpr uint32_t BUILDER_CHANNEL_NOTIFY_NUM = 3U;

#define ASC_CHECK(call) do { \
    if (!(call)) { \
        return false; \
    } \
} while (0)

enum class ChannelMode {
    URMA,
    UBMEM
};

template <ChannelMode Mode> struct CommCtxTrait;
template <> struct CommCtxTrait<ChannelMode::URMA>  { using type = Apace::AivComm::CommUdmaContext; };
template <> struct CommCtxTrait<ChannelMode::UBMEM> { using type = Apace::AivComm::CommUbmemContext; };

template <ChannelMode DataChannelMode = ChannelMode::URMA,
          ChannelMode BarrierChannelMode = ChannelMode::UBMEM>
class CommChannelBuilder {
public:
    explicit CommChannelBuilder(HcclComm comm) : comm_(comm) {}

    /*!
     * \brief 初始化 rankId 和 rankNum（使用 HcclGetRankId/HcclGetRankSize）
     * \return true 成功, false 失败
     */
    bool Init()
    {
        if (HcclGetRankId(comm_, &rankId_) != HCCL_SUCCESS) { return false; }
        if (HcclGetRankSize(comm_, &rankNum_) != HCCL_SUCCESS) { return false; }
        return true;
    }

    /*!
     * \brief 获取 rankId
     */
    uint32_t GetRankId() const { return rankId_; }

    /*!
     * \brief 获取 rankNum
     */
    uint32_t GetRankSize() const { return rankNum_; }

    /*!
     * \brief 使用 HCCL 内置 buffer + 建链，填充外部 context 对应字段
     * \param [in]  mode         通信方式 (URMA / UBMEM)
     * \param [in]  memTag       内存标签
     * \param [out] channels     context 中 channel 数组字段的指针 (不需要则传 nullptr)
     * \param [out] remoteAddrs  context 中远端地址数组字段
     * \return true 成功, false 失败
     */
    bool AllocRegAndBuildChannels(
        ChannelMode mode, const char* memTag,
        uint64_t* channels, uint64_t* remoteAddrs)
    {
        ::CommProtocol protocol = (mode == ChannelMode::URMA) ? ::CommProtocol::COMM_PROTOCOL_UBC_CTP
                                                           : ::CommProtocol::COMM_PROTOCOL_UB_MEM;

        void* buf = nullptr;
        uint64_t hcclBufSize = 0;
        if (HcclGetHcclBuffer(comm_, &buf, &hcclBufSize) != ::HCCL_SUCCESS || buf == nullptr) {
            return false;
        }
        if (aclrtMemset(buf, hcclBufSize, 0, hcclBufSize) != ACL_SUCCESS) {
            return false;
        }
        remoteAddrs[rankId_] = reinterpret_cast<uint64_t>(buf);

        return BuildChannels(mode, true, protocol, nullptr, memTag, channels, remoteAddrs);
    }

    /*!
     * \brief 由 HCCL engine 申请 device 内存，建立 data/barrier channel 并拷贝 context，返回 device 地址
     * \param [in] hostCtx     host 端 context 起始地址（用于 HcclEngineCtxCopy）
     * \param [in] ctxSize     context 大小（字节）
     * \param [in] ctxTag      引擎标签（同一 tag 可复用已创建的 context）
     * \param [in] dataCtx     data channel context 指针（类型由 DataChannelMode 决定）
     * \param [in] barrierCtx  barrier channel context 指针（类型由 BarrierChannelMode 决定，可为 nullptr）
     * \return device 端 context 地址，失败返回 nullptr
     *
     * \note HcclEngineCtxGet 命中已存在 context 时直接复用，跳过建链与字段填充；
     *       rankId/rankSize 赋值及 channel 建立仅在新建路径中执行。
     *       调用方应在本函数返回后对 rank 间做一次 barrier，确保所有 channel 握手完成。
     */
    void* CreateDeviceContext(void* hostCtx, uint64_t ctxSize, const char* ctxTag,
                              typename CommCtxTrait<DataChannelMode>::type* dataCtx,
                              typename CommCtxTrait<BarrierChannelMode>::type* barrierCtx = nullptr)
    {
        constexpr uint64_t BARRIER_BUF_SIZE = 2UL * 1024 * 1024;
        uint64_t totalSize = (barrierCtx != nullptr) ? ctxSize + BARRIER_BUF_SIZE : ctxSize;

        void* devCtx = nullptr;
        CommEngine engine = static_cast<CommEngine>(BUILDER_COMM_ENGINE_AIV);
        uint64_t existingSize = 0;
        if (HcclEngineCtxGet(comm_, ctxTag, engine, &devCtx, &existingSize) == HCCL_SUCCESS) {
            return devCtx;
        }

        if (HcclEngineCtxCreate(comm_, ctxTag, engine, totalSize, &devCtx) != HCCL_SUCCESS) {
            return nullptr;
        }

        if (HcclGetRankId(comm_, &rankId_) != HCCL_SUCCESS) {
            return nullptr;
        }
        if (HcclGetRankSize(comm_, &rankNum_) != HCCL_SUCCESS) {
            return nullptr;
        }

        dataCtx->rankId = rankId_;
        dataCtx->rankSize = rankNum_;
        std::string dataBufTag = std::string(ctxTag) + "dataBuf";
        uint64_t* channelHandles = nullptr;
        if constexpr (DataChannelMode == ChannelMode::URMA) {
            channelHandles = dataCtx->channelHandles;
        }
        if (!AllocRegAndBuildChannels(DataChannelMode, dataBufTag.c_str(),
                                      channelHandles, dataCtx->commBufferAddrs)) {
            return nullptr;
        }

        if (barrierCtx != nullptr) {
            barrierCtx->rankId = rankId_;
            barrierCtx->rankSize = rankNum_;
            void* barrierBuf = reinterpret_cast<void*>(
                reinterpret_cast<uintptr_t>(devCtx) + ctxSize);
            CommMem regMem = {};
            regMem.type = COMM_MEM_TYPE_DEVICE;
            regMem.addr = barrierBuf;
            regMem.size = BARRIER_BUF_SIZE;
            std::string syncBufTag = std::string(ctxTag) + "syncBuf";
            HcclMemHandle memHandle = nullptr;
            if (HcclCommMemReg(comm_, syncBufTag.c_str(), &regMem, &memHandle) != HCCL_SUCCESS) {
                return nullptr;
            }
            barrierCtx->commBufferAddrs[barrierCtx->rankId] = reinterpret_cast<uint64_t>(barrierBuf);
            if (!BuildChannels(BarrierChannelMode, false,
                               (BarrierChannelMode == ChannelMode::URMA)
                                   ? ::CommProtocol::COMM_PROTOCOL_UBC_CTP
                                   : ::CommProtocol::COMM_PROTOCOL_UB_MEM,
                               memHandle, syncBufTag.c_str(),
                               nullptr, barrierCtx->commBufferAddrs)) {
                return nullptr;
            }
        }

        if (HcclEngineCtxCopy(comm_, engine, ctxTag, hostCtx, ctxSize, 0) != HCCL_SUCCESS) {
            return nullptr;
        }
        return devCtx;
    }

private:

    bool BuildChannels(ChannelMode mode, bool useHcclBuf,
                       ::CommProtocol protocol, HcclMemHandle memHandle,
                       const char* memTag, uint64_t* channels, uint64_t* remoteAddrs)
    {
        uint32_t *netLayers = nullptr;
        uint32_t netLayerNum = 0;
        if (HcclRankGraphGetLayers(comm_, &netLayers, &netLayerNum) != ::HCCL_SUCCESS || netLayerNum == 0) {
            return false;
        }
        uint32_t layerId = netLayers[0];

        for (uint32_t peer = 0; peer < rankNum_; peer++) {
            if (peer == rankId_) { continue; }

            ChannelHandle channel = 0;
            uint64_t remoteAddr = 0;
            if (useHcclBuf) {
                ASC_CHECK(AcquireChannelAndRemoteMemHcclBuf(layerId, protocol,
                                                            peer, channel, remoteAddr));
            } else {
                ASC_CHECK(AcquireChannelAndRemoteMem(layerId, protocol, memHandle, memTag,
                                                         peer, channel, remoteAddr));
            }
            if (channels != nullptr) {
                channels[peer] = static_cast<uint64_t>(channel);
            }
            remoteAddrs[peer] = remoteAddr;
        }
        return true;
    }

    bool AcquireChannelAndRemoteMem(uint32_t layerId, ::CommProtocol protocol,
                                        HcclMemHandle memHandle, const char* memTag,
                                        uint32_t peer,
                                        ChannelHandle& channel, uint64_t& remoteAddr)
    {
        channel = 0;
        remoteAddr = 0;

        ASC_CHECK(AcquireChannel(layerId, protocol, peer, channel, memHandle));

        uint32_t memNum = 0;
        CommMem *remoteMems = nullptr;
        char **memTags = nullptr;
        if (HcclChannelGetRemoteMems(comm_, channel, &memNum, &remoteMems, &memTags) != ::HCCL_SUCCESS ||
            memNum == 0 || remoteMems == nullptr) {
            return false;
        }

        for (uint32_t i = 0; i < memNum; i++) {
            if (memTags != nullptr && memTags[i] != nullptr && std::strcmp(memTags[i], memTag) == 0) {
                remoteAddr = reinterpret_cast<uint64_t>(remoteMems[i].addr);
                break;
            }
        }
        if (remoteAddr == 0 && memNum == 1) {
            remoteAddr = reinterpret_cast<uint64_t>(remoteMems[0].addr);
        }

        return remoteAddr != 0;
    }

    bool AcquireChannelAndRemoteMemHcclBuf(uint32_t layerId, ::CommProtocol protocol,
                                           uint32_t peer,
                                           ChannelHandle& channel, uint64_t& remoteAddr)
    {
        channel = 0;
        remoteAddr = 0;

        ASC_CHECK(AcquireChannel(layerId, protocol, peer, channel));

        void *remoteBuf = nullptr;
        uint64_t remoteBufSize = 0;
        if (HcclChannelGetHcclBuffer(comm_, channel, &remoteBuf, &remoteBufSize) != ::HCCL_SUCCESS ||
            remoteBuf == nullptr) {
            return false;
        }
        remoteAddr = reinterpret_cast<uint64_t>(remoteBuf);

        return true;
    }

    bool AcquireChannel(uint32_t layerId, ::CommProtocol protocol,
                        uint32_t peer, ChannelHandle& channel,
                        HcclMemHandle memHandle = nullptr)
    {
        channel = 0;

        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        if (HcclRankGraphGetLinks(comm_, layerId, rankId_, peer, &links, &linkNum) != ::HCCL_SUCCESS ||
            linkNum == 0 || links == nullptr) {
            return false;
        }

        HcclChannelDesc channelDesc;
        HcclChannelDescInit(&channelDesc, 1);
        channelDesc.remoteRank = peer;
        channelDesc.channelProtocol = protocol;
        channelDesc.notifyNum = BUILDER_CHANNEL_NOTIFY_NUM;
        if (memHandle != nullptr) {
            channelDesc.memHandles = &memHandle;
            channelDesc.memHandleNum = 1;
        }

        bool linkFound = false;
        for (uint32_t i = 0; i < linkNum; i++) {
            if (links[i].linkAttr.linkProtocol == protocol) {
                channelDesc.localEndpoint = links[i].srcEndpointDesc;
                channelDesc.remoteEndpoint = links[i].dstEndpointDesc;
                linkFound = true;
                break;
            }
        }
        if (!linkFound) {
            return false;
        }

        if (HcclChannelAcquire(comm_, static_cast<CommEngine>(BUILDER_COMM_ENGINE_AIV), &channelDesc, 1, &channel) !=
            ::HCCL_SUCCESS || channel == 0) {
            return false;
        }

        return true;
    }

    HcclComm comm_;
    uint32_t rankId_{0};
    uint32_t rankNum_{0};
};
