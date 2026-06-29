/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file comm_context.cpp
 * \brief comm_context implementation supporting both KFC and HCCL Channel modes.
 *
 */

#include <torch/extension.h>
#include <acl/acl_rt.h>
#include <cstring>
#include <type_traits>
#include "comm_context.h"
#include "hccl_common.h"

namespace vllm_ascend {

// ======================== Common constants ========================

constexpr static uint8_t COMM_ENGINE_AIV = 4;
constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1;
constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
constexpr uint32_t GET_LOCAL_SERVER_RANK_SIZE_LAYER = 0;

const char *GetSocName()
{
    static const char *socName = aclrtGetSocName();
    return socName;
}

static BackendMode ResolveBackend(const std::string &backend)
{
    if (backend == "channel") return BackendMode::CHANNEL;
    if (backend == "kfc")     return BackendMode::KFC;
    if (backend == "auto") {
        const char *socName = GetSocName();
        TORCH_CHECK(socName != nullptr, "aclrtGetSocName returned nullptr");
        if (std::strstr(socName, "Ascend950") != nullptr) {
            return BackendMode::CHANNEL;
        }
        if (std::strstr(socName, "Ascend910B") != nullptr ||
            std::strstr(socName, "Ascend910_93") != nullptr) {
            return BackendMode::KFC;
        }
        TORCH_CHECK(false, "No mega_moe HCCL backend mapping for SoC: ", socName);
    }
    TORCH_CHECK(false, "backend string must be 'auto', 'kfc' or 'channel', got '", backend, "'");
}

static void CopyContextToTensor(const CommContext &context, at::Tensor &tensor)
{
    at::Tensor hostContext = at::from_blob(const_cast<CommContext *>(&context),
                                           {sizeof(CommContext) / sizeof(int32_t)}, at::kInt);
    tensor.copy_(hostContext);
}

template <typename Desc, typename ChannelNum>
static void InitHcclChannelDescChecked(Desc *desc, ChannelNum channelNum)
{
    using InitRet = decltype(HcclChannelDescInit(desc, channelNum));
    if constexpr (std::is_void_v<InitRet>) {
        HcclChannelDescInit(desc, channelNum);
    } else {
        InitRet hcclRet = HcclChannelDescInit(desc, channelNum);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HCCL channel init failed, ret: ", hcclRet);
    }
}

// ======================== KFC Mode ========================
class KfcContextBuilder {
public:
    void Build(const std::string &group, int64_t worldSize, int64_t &cclBufferSize, at::Tensor &contextTensor)
    {
        CommContext mc2ContextHost;
        GetMc2Context(mc2ContextHost, worldSize, cclBufferSize, group.c_str());

        CopyContextToTensor(mc2ContextHost, contextTensor);
    }

private:
    void CollectRankBuffers(HcclComm &comm, int64_t worldSize, int64_t &cclBufferSize, CommContext &mc2Context)
    {
        uint32_t rankId;
        (void)HcclGetRankIdFunc(comm, &rankId);
        mc2Context.epRankId = rankId;
        const char *socName = GetSocName();
        if (socName != nullptr && std::strstr(socName, "Ascend910B") != nullptr && worldSize > 8) {
            return;
        }
        for (uint64_t remoteRankId = 0; remoteRankId < static_cast<uint64_t>(worldSize); remoteRankId++) {
            void *remoteAddr = nullptr;
            uint64_t commSize = 0;
            HcclResult ret;
            if (rankId == remoteRankId) {
                ret = static_cast<HcclResult>(HcclGetHcclBufferFunc(comm, &remoteAddr, &commSize));
                cclBufferSize = commSize;
            } else {
                ret = static_cast<HcclResult>(HcclGetRemoteIpcHcclBufFunc(comm, remoteRankId, &remoteAddr, &commSize));
            }
            TORCH_CHECK((ret == HCCL_SUCCESS), "Get HcclBufferSize failed, ret=", ret);
            mc2Context.epHcclBuffer_[remoteRankId] = reinterpret_cast<uint64_t>(remoteAddr);
        }
    }

    void CreateHcclContext(HcclComm &commHandle, void *opArgs, int64_t worldSize, const char *groupName,
        std::string algConfig, uint32_t opType, CommContext &mc2Context)
    {
        HcclResult ret =
            static_cast<HcclResult>(HcclKfcOpArgsSetAlgConfigFunc(opArgs, const_cast<char *>(algConfig.c_str())));
        TORCH_CHECK(ret == 0, "HcclKfcOpArgsSetAlgConfig failed, ret:", ret);
        ret = static_cast<HcclResult>(HcclCommGetHandleWithNameFunc(groupName, &commHandle));
        TORCH_CHECK(ret == 0, "HcclGetCommHandle failed, ret:", ret);
        void *opsResCtx;
        ret = static_cast<HcclResult>(HcclCreateOpResCtxFunc(commHandle, opType, opArgs, &opsResCtx));
        TORCH_CHECK(ret == 0, "HcclCreateOpResCtx failed, ret:", ret);
        mc2Context.kfcContextAddr = reinterpret_cast<uint64_t>(opsResCtx);

        uint32_t rankId = 0;
        uint32_t worldSizeHccl = 0;
        ret = static_cast<HcclResult>(HcclGetRankSizeFunc(commHandle, &worldSizeHccl));
        TORCH_CHECK(ret == HCCL_SUCCESS, "HcclGetRankSize failed, ret:", ret);
        ret = static_cast<HcclResult>(HcclGetRankIdFunc(commHandle, &rankId));
        TORCH_CHECK(ret == HCCL_SUCCESS, "HcclGetRankId failed, ret:", ret);
        TORCH_CHECK(rankId < worldSizeHccl, "rankId:", rankId, " worldSizeHccl:", worldSizeHccl,
                    " worldSize:", worldSize);
        TORCH_CHECK(static_cast<uint32_t>(worldSize) == worldSizeHccl,
                    "worldSize:", worldSize, " != worldSizeHccl:", worldSizeHccl);
    }

    void GetMc2Context(CommContext &mc2ContextHost, int64_t worldSize, int64_t &cclBufferSize, const char *groupStr)
    {
        InitHcclFunctions();
        void *opArgs = nullptr;
        HcclResult ret = static_cast<HcclResult>(HcclKfcAllocOpArgsFunc(&opArgs));
        TORCH_CHECK(ret == 0, "HcclKfcAllocOpArgs failed, ret:", ret);
        uint8_t commEngine = COMM_ENGINE_AIV;
        ret = static_cast<HcclResult>(HcclKfcOpArgsSetCommEngineFunc(opArgs, commEngine));
        TORCH_CHECK(ret == 0, "HcclKfcOpArgsSetCommEngine failed, ret:", ret);
        HcclComm commHandle;
        const char *socName = GetSocName();
        const bool is910B = (socName != nullptr && std::strstr(socName, "Ascend910B") != nullptr);
        const bool isMultiServer = worldSize > 8;
        const std::string algConfig =
            is910B && isMultiServer ? "BatchWrite=level1:hierarchy" : "AlltoAll=level0:fullmesh;level1:pairwise";
        const uint32_t opType = is910B && isMultiServer ? 18 : 8; // 18: BatchWrite, 8: AllToAll
        CreateHcclContext(commHandle, opArgs, worldSize, groupStr, algConfig, opType, mc2ContextHost);
        ret = static_cast<HcclResult>(HcclKfcFreeOpArgsFunc(opArgs));
        TORCH_CHECK(ret == 0, "getHcclKfcFreeOpArgs failed, ret:", ret);
        CollectRankBuffers(commHandle, worldSize, cclBufferSize, mc2ContextHost);
    }
};

// ======================== HCCL Channel Mode ========================

class HcclChannelContextBuilder {
public:
    void Build(const std::string &group, int64_t worldSize, int64_t &cclBufferSize, at::Tensor &contextTensor)
    {
        ASCEND_LOGI("Start to get CommContext Tensor, group: %s", group.c_str());
        InitHcclEngineCtxFunctions();

        HcclComm hcclHandle;
        AcquireHcclHandle(group, hcclHandle);

        CommProtocol protocol;
        GetCommProtocol(hcclHandle, protocol);

        CommContext commContextStruct;
        BuildContext(hcclHandle, group, protocol, commContextStruct, cclBufferSize);

        CopyContextToTensor(commContextStruct, contextTensor);
        ASCEND_LOGI("Get CommContext Tensor Success, group: %s, ccl_buffer_size: %ld",
                    group.c_str(), cclBufferSize);
    }

    void AcquireHcclHandle(const std::string &group, HcclComm &hcclHandle)
    {
        auto aclnnRet = HcomGetCommHandleByGroupFunc(group.c_str(), &hcclHandle);
        TORCH_CHECK(aclnnRet == HCCL_SUCCESS, "Get HCCL handle failed, group: ", group.c_str(),
                    ", ret: ", aclnnRet);
        ASCEND_LOGI("Get HCCL communication handle success hcclHandle is: %p", hcclHandle);
    }

    void BuildContext(const HcclComm &hcclHandle, const std::string &group,
                      const CommProtocol &protocol, CommContext &commContextStruct,
                      int64_t &cclBufferSize)
    {
        std::string mc2ContextTag = std::string(group) + "moe_dispatch_ffn_combine";
        TORCH_CHECK(mc2ContextTag.size() <= 255, "Mc2ContextTag is too long, max size is 255, got ",
                    mc2ContextTag.size());

        CommEngine engine = CommEngine::COMM_ENGINE_AIV;
        void *ctx = nullptr;
        uint64_t hcclBuffSize = 0;

        GetOrCreateContext(hcclHandle, mc2ContextTag, engine, protocol,
                           ctx, hcclBuffSize, commContextStruct);

        cclBufferSize = hcclBuffSize;
    }

    void GetCommProtocol(const HcclComm &commHandle, CommProtocol &protocol)
    {
        ASCEND_LOGI("Start to get HCCL communication protocol");
        uint32_t layerNum = 0;
        uint32_t *layerList = nullptr;
        auto ret = HcclRankGraphGetLayersFunc(commHandle, &layerList, &layerNum);
        TORCH_CHECK(ret == HCCL_SUCCESS, "Get HCCL layers failed, ret: ", ret);

        if (layerNum == HCCL_COMM_LAYERS_MTE_CCU) {
            protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
            return;
        }

        ASCEND_LOGI("start CheckProtocolSupport, layerNum is %d", layerNum);
        CheckProtocolSupport(commHandle, layerList, layerNum);

        ASCEND_LOGI("CheckProtocolSupport success!");
        protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
    }

    void CheckProtocolSupport(const HcclComm &commHandle, uint32_t *&layerList, uint32_t &layerNum)
    {
        uint32_t srcRankId = 0;
        uint32_t dstRankId = 0;
        uint32_t netLinkNum = 0;
        uint32_t rankNumInLayer = 0;
        uint32_t *rankIdLists = nullptr;
        CommLink *linksList = nullptr;

        auto hcclRet = HcclGetRankIdFunc(commHandle, &srcRankId);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "CheckProtocolSupport Get rank ID failed, ret: ", hcclRet);
        ASCEND_LOGI("CheckProtocolSupport Get rank ID success, rankId is: %d", srcRankId);

        for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
            ASCEND_LOGI("CheckProtocolSupport Check layer %d", layerList[layerIndex]);
            hcclRet =
                HcclRankGraphGetRanksByLayerFunc(commHandle, layerList[layerIndex], &rankIdLists, &rankNumInLayer);
            TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank IDs by layer failed, ret: ", hcclRet);
            for (uint32_t rankId = 0; rankId < rankNumInLayer; ++rankId) {
                if (rankIdLists[rankId] == srcRankId || layerMap_.find(rankIdLists[rankId]) != layerMap_.end()) {
                    continue;
                }
                hcclRet = HcclRankGraphGetLinksFunc(commHandle, layerList[layerIndex], srcRankId, rankIdLists[rankId],
                                                    &linksList, &netLinkNum);
                TORCH_CHECK(hcclRet == HCCL_SUCCESS,
                            "Get HCCL links failed when checking protocol support, ret: ", hcclRet);
                TORCH_CHECK(netLinkNum > 0, "No available HCCL links found");
                TORCH_CHECK(CheckLinks(netLinkNum, linksList), "No HCCL links support UB_MEM srcRankID ", srcRankId,
                            ", dstRankID ", dstRankId, ", layer ", layerList[layerIndex]);
                layerMap_[rankIdLists[rankId]] = layerList[layerIndex];
            }
        }
    }

    static bool CheckLinks(uint32_t &netLinkNum, CommLink *linksList)
    {
        for (uint32_t j = 0; j < netLinkNum; ++j) {
            if (linksList[j].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UB_MEM) {
                return true;
            }
        }
        return false;
    }

    // ---- Channel management helpers ----

    void InitHcclChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId,
                          const CommProtocol &protocol, std::vector<HcclChannelDesc> &channelDesc)
    {
        uint32_t channelNum = channelDesc.size();
        InitHcclChannelDescChecked(channelDesc.data(), channelNum);
        ASCEND_LOGI("HCCL channel init success");

        uint32_t netLayerNum = 0;
        uint32_t layerId = 0;
        uint32_t *netLayerList = nullptr;
        GetNetLayers(commHandle, netLayerList, netLayerNum);
        TORCH_CHECK(netLayerNum > 0, "Get HCCL net layers failed, netLayerNum is ", netLayerNum);

        for (uint32_t i = 0; i < rankDim; ++i) {
            if (i == srcRankId) {
                continue;
            }
            uint32_t dstRank = i;
            uint32_t channelId = (i > srcRankId) ? (i - 1) : i;
            CommLink *links = nullptr;
            layerId = netLayerNum == 1 ?
                    netLayerList[HCCL_COMM_LAYERS_UB_MEM] :
                    layerMap_[dstRank];
            GetHcclCommLink(commHandle, layerId, srcRankId, dstRank, protocol, links);
            channelDesc[channelId].channelProtocol = protocol;
            channelDesc[channelId].remoteRank = dstRank;
            channelDesc[channelId].notifyNum = channelNum;
            channelDesc[channelId].localEndpoint = links->srcEndpointDesc;
            channelDesc[channelId].remoteEndpoint = links->dstEndpointDesc;
        }
    }

    void GetHcclCommChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId,
                             const CommProtocol &protocol, const CommEngine &engine,
                             std::vector<ChannelHandle> &channels, uint32_t &rankSizePerServer)
    {
        ASCEND_LOGI("Start to get HCCL communication channel");
        uint32_t channelNum = rankDim - 1;
        std::vector<HcclChannelDesc> channelDesc(channelNum);
        channels.resize(channelNum);

        uint32_t *netLayerList = nullptr;
        uint32_t netLayerNum = 0;
        GetNetLayers(commHandle, netLayerList, netLayerNum);

        uint32_t netLayers = netLayerList[GET_LOCAL_SERVER_RANK_SIZE_LAYER];

        GetRankSizePerServer(commHandle, netLayers, rankSizePerServer);

        InitHcclChannel(commHandle, rankDim, srcRankId, protocol, channelDesc);

        auto hcclRet = HcclChannelAcquireFunc(commHandle, engine, channelDesc.data(), channelNum, channels.data());
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Acquire HCCL channel failed, ret: ", hcclRet);
    }

    // ---- Resource management helpers ----

    void GetHcclCommResource(const HcclComm &commHandle, const CommEngine &engine, const CommProtocol &protocol,
                              CommContext *commContextStruct, uint32_t rankSize, uint64_t &hcclBuffSize)
    {
        ASCEND_LOGI("Start to get HCCL communication resource");
        uint32_t rankId = commContextStruct->epRankId;
        std::vector<ChannelHandle> channels;

        uint32_t rankSizePerServer = 0;
        GetHcclCommChannel(commHandle, rankSize, rankId, protocol, engine, channels, rankSizePerServer);
        commContextStruct->rankSizePerServer = rankSizePerServer;
        ASCEND_LOGI("Get HCCL communication channel success, channel num is: %zu", channels.size());

        for (uint32_t i = 0; i < rankSize; ++i) {
            void *tempBuffer = nullptr;
            uint64_t bufSize = 0;
            HcclResult hcclRet;

            if (i == rankId) {
                hcclRet = HcclGetHcclBufferFunc(commHandle, &tempBuffer, &hcclBuffSize);
            } else {
                uint32_t idx = (i < rankId) ? i : (i - 1);
                hcclRet = HcclChannelGetHcclBufferFunc(commHandle, channels[idx], &tempBuffer, &bufSize);
            }

            TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL buffer failed, src: ", rankId, ", dst: ", i,
                        ", ret: ", hcclRet);

            commContextStruct->epHcclBuffer_[i] = reinterpret_cast<uint64_t>(tempBuffer);
        }
        ASCEND_LOGI("Get HCCL CommResource success");
    }

    // ---- Context lifecycle helpers ----

    void CreateContext(const HcclComm &commHandle, const std::string &mc2ContextTag,
                        const CommEngine &engine, const CommProtocol &protocol, void *&ctx,
                        CommContext *commContextStruct, uint64_t &hcclBuffSize)
    {
        ASCEND_LOGI("Start to create HCCL context");
        uint64_t commContextSize = sizeof(CommContext);
        auto hcclRet = HcclEngineCtxCreateFunc(commHandle, mc2ContextTag.c_str(), engine, commContextSize, &ctx);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Create HCCL context memory failed, ret: ", hcclRet);
        ASCEND_LOGI("Create HCCL context success, ctx: %p", ctx);

        hcclRet = HcclGetRankIdFunc(commHandle, &commContextStruct->epRankId);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank ID failed, ret: ", hcclRet);
        ASCEND_LOGI("Get rank ID success, rankId is: %u", commContextStruct->epRankId);

        uint32_t rankSize = 0;
        hcclRet = HcclGetRankSizeFunc(commHandle, &rankSize);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank size failed, ret: ", hcclRet);
        ASCEND_LOGI("Get rank size success, rankSize is: %u", rankSize);

        GetHcclCommResource(commHandle, engine, protocol, commContextStruct, rankSize, hcclBuffSize);

        hcclRet =
            HcclEngineCtxCopyFunc(commHandle, engine, mc2ContextTag.c_str(), commContextStruct, commContextSize, 0);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Copy context from host to device failed, ret: ", hcclRet);
        ASCEND_LOGI("Copy context from host to device success");
    }

    void GetOrCreateContext(const HcclComm &commHandle, const std::string &mc2ContextTag,
                             const CommEngine &engine, const CommProtocol &protocol, void *&ctx,
                             uint64_t &hcclBuffSize, CommContext &commContextStruct)
    {
        uint64_t ctxSize = 0;
        auto hcclRet = HcclEngineCtxGetFunc(commHandle, mc2ContextTag.c_str(), engine, &ctx, &ctxSize);
        if (hcclRet != HCCL_SUCCESS) {
            CreateContext(commHandle, mc2ContextTag, engine, protocol, ctx,
                          &commContextStruct, hcclBuffSize);
        } else {
            GetHcclBufferSize(commHandle, hcclBuffSize);
        }
    }

    // ---- Static HCCL query helpers ----

    static void GetHcclBufferSize(const HcclComm &commHandle, uint64_t &hcclBuffSize)
    {
        void *tempBuffer = nullptr;
        auto hcclRet = HcclGetHcclBufferFunc(commHandle, &tempBuffer, &hcclBuffSize);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL Buffer Size failed, ret: ", hcclRet);
    }

    static void GetNetLayers(const HcclComm &commHandle, uint32_t *&netLayerList, uint32_t &netLayerNum)
    {
        auto hcclRet = HcclRankGraphGetLayersFunc(commHandle, &netLayerList, &netLayerNum);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL layers failed, ret: ", hcclRet);
        ASCEND_LOGI("Get HCCL layers success, netLayerNum is: %u", netLayerNum);
    }

    static void GetRankSizePerServer(const HcclComm &commHandle, uint32_t netLayers, uint32_t &rankSizePerServer)
    {
        auto hcclRet = HcclRankGraphGetRankSizeByLayerFunc(commHandle, netLayers, &rankSizePerServer);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL rank size per server failed, ret: ", hcclRet);
        ASCEND_LOGI("Get HCCL rank size per server success, rankSizePerServer is: %u", rankSizePerServer);
    }

    static void GetHcclCommLink(const HcclComm &commHandle, uint32_t netLayerId, uint32_t srcRankId,
                                  uint32_t dstRankId, const CommProtocol &protocol, CommLink *&links)
    {
        CommLink *linksList = nullptr;
        uint32_t netLinkNum = 0;
        auto hcclRet = HcclRankGraphGetLinksFunc(commHandle, netLayerId, srcRankId, dstRankId, &linksList, &netLinkNum);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL Communication link failed, ret: ", hcclRet);
        TORCH_CHECK(netLinkNum > 0, "The Net Link Is nullptr. srcRankId is ", srcRankId,
                    ", dstRankId is ", dstRankId, ", layerId is ", netLayerId);
        ASCEND_LOGI("Get HCCL Rank Links Success Links Num is: %u", netLinkNum);
        uint32_t index = 0;
        for (; index < netLinkNum; ++index) {
            if (linksList[index].linkAttr.linkProtocol == protocol) {
                links = &linksList[index];
                break;
            }
        }
        TORCH_CHECK(index < netLinkNum, "No matching communication protocol found in HCCL links, protocol is ",
                    static_cast<int>(protocol));
    }

    // 记录本卡与其他卡的通信层数，key为其他卡的rankId，value为通信层数
    std::unordered_map<uint32_t, uint32_t> layerMap_;
};

// ======================== CommContextManager implementation ========================

CommContextManager::CommContextManager(const std::string &group, int64_t worldSize,
                                       const std::string &backend)
    : group_(group), worldSize_(worldSize),
      backend_(backend), mode_(BackendMode::UNINITIALIZED), cclBufferSize_(0) {}

at::Tensor CommContextManager::create_context()
{
    EnsureResolved();
    at::Tensor context = at::empty({context_tensor_size()}, at::TensorOptions().dtype(at::kInt)
        .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    DispatchBuild(context);
    return context;
}

void CommContextManager::update_group(const std::string &group, at::Tensor &contextTensor)
{
    group_ = group;
    cclBufferSize_ = 0;
    EnsureResolved();
    DispatchBuild(contextTensor);
}

int64_t CommContextManager::context_tensor_size()
{
    return (sizeof(CommContext) + sizeof(int32_t) - 1) / sizeof(int32_t);
}

void CommContextManager::EnsureResolved()
{
    if (mode_ == BackendMode::UNINITIALIZED) {
        mode_ = ResolveBackend(backend_);
    }
}

void CommContextManager::DispatchBuild(at::Tensor &tensor)
{
    switch (mode_) {
        case BackendMode::KFC: {
            KfcContextBuilder builder;
            builder.Build(group_, worldSize_, cclBufferSize_, tensor);
            return;
        }
        case BackendMode::CHANNEL: {
            HcclChannelContextBuilder builder;
            builder.Build(group_, worldSize_, cclBufferSize_, tensor);
            return;
        }
        default:
            TORCH_CHECK(false, "Unknown backend mode: ", static_cast<int>(mode_));
    }
}

} // namespace vllm_ascend
