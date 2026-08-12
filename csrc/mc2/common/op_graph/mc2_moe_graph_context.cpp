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
 * \file mc2_moe_graph_context.cpp
 * \brief MC2 MoeContext builder for graph fusion pass (A5).
 */

#include "mc2_moe_graph_context.h"
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include "mc2_common_log.h"
#include "mc2_moe_context.h"
#include "graph/operator_factory.h"

namespace {
constexpr const char *CONTEXT_NAME = "Mc2MoeGraphContext";
constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1;
constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
constexpr uint32_t GET_LOCAL_SERVER_RANK_SIZE_LAYER = 0;
constexpr uint32_t EP_RANK_OFFSET_STEP = 1024;
} // namespace

namespace ops {
Mc2MoeGraphContext::~Mc2MoeGraphContext()
{
    if (hcclLibHandle_ != nullptr) {
        dlclose(hcclLibHandle_);
        hcclLibHandle_ = nullptr;
    }
}

Mc2MoeGraphContext &Mc2MoeGraphContext::GetInstance()
{
    static Mc2MoeGraphContext instance;
    return instance;
}

const std::string Mc2MoeGraphContext::GetLibPath()
{
    const char *ascendPath = std::getenv("ASCEND_HOME_PATH");
    if (ascendPath == nullptr) {
        OPS_LOG_E(CONTEXT_NAME, "Ascend home path doesn't exist.");
        return "";
    }
#if defined(__x86_64__)
    std::string hcclPathPostfix = "/x86_64-linux/lib64/libhccl_fwk.so";
#elif defined(__aarch64__)
    std::string hcclPathPostfix = "/aarch64-linux/lib64/libhccl_fwk.so";
#endif
    return std::string(ascendPath) + hcclPathPostfix;
}

template <typename T>
T Mc2MoeGraphContext::GetHcclLibFunc(void *handle, const std::string &funcName)
{
    T func = reinterpret_cast<T>(dlsym(handle, funcName.c_str()));
    if (func == nullptr) {
        OPS_LOG_E(CONTEXT_NAME, "Load func=%s error=%s in lib hccl failed.", funcName.c_str(), dlerror());
    }
    return func;
}

bool Mc2MoeGraphContext::LoadHcclSymbols()
{
    if (hcclLibHandle_ != nullptr) {
        return true;
    }
    const std::string libPath = GetLibPath();
    if (libPath.empty()) {
        OPS_LOG_E(CONTEXT_NAME, "Failed to get HCCL library path.");
        return false;
    }
    hcclLibHandle_ = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (hcclLibHandle_ == nullptr) {
        OPS_LOG_E(CONTEXT_NAME, "dlopen HCCL library failed: %s, error: %s", libPath.c_str(), dlerror());
        return false;
    }

    struct SymbolInfo {
        void **ptr;
        const char *name;
    };
    SymbolInfo symbols[] = {
        {reinterpret_cast<void **>(&HcomGetCommHandleByGroup), "HcomGetCommHandleByGroup"},
        {reinterpret_cast<void **>(&HcclRankGraphGetLinks), "HcclRankGraphGetLinks"},
        {reinterpret_cast<void **>(&HcclRankGraphGetLayers), "HcclRankGraphGetLayers"},
        {reinterpret_cast<void **>(&HcclRankGraphGetRankSizeByLayer), "HcclRankGraphGetRankSizeByLayer"},
        {reinterpret_cast<void **>(&HcclChannelAcquire), "HcclChannelAcquire"},
        {reinterpret_cast<void **>(&HcclGetHcclBuffer), "HcclGetHcclBuffer"},
        {reinterpret_cast<void **>(&HcclChannelGetHcclBuffer), "HcclChannelGetHcclBuffer"},
        {reinterpret_cast<void **>(&HcclGetRankId), "HcclGetRankId"},
        {reinterpret_cast<void **>(&HcclGetRankSize), "HcclGetRankSize"},
        {reinterpret_cast<void **>(&HcclRankGraphGetRanksByLayer), "HcclRankGraphGetRanksByLayer"},
    };
    for (auto &sym : symbols) {
        *(sym.ptr) = GetHcclLibFunc<void *>(hcclLibHandle_, sym.name);
        if (*(sym.ptr) == nullptr) {
            OPS_LOG_E(CONTEXT_NAME, "Failed to load %s symbol.", sym.name);
            dlclose(hcclLibHandle_);
            hcclLibHandle_ = nullptr;
            return false;
        }
    }
    OPS_LOG_D(CONTEXT_NAME, "All HCCL symbols loaded successfully.");
    return true;
}

bool Mc2MoeGraphContext::GetCommHandle(const char *groupEp, HcclComm &hcclHandle)
{
    auto ret = HcomGetCommHandleByGroup(groupEp, &hcclHandle);
    if (ret != HCCL_SUCCESS) {
        OPS_LOG_E(CONTEXT_NAME, "Get HCCL handle failed, groupEp: %s.", groupEp);
        return false;
    }
    return true;
}

bool Mc2MoeGraphContext::GetHcclCommLink(const HcclComm &hcclHandle, uint32_t netLayerId, uint32_t srcRankId,
                                         uint32_t dstRankId, const CommProtocol &protocol, CommLink *&links)
{
    CommLink *linksList = nullptr;
    uint32_t netLinkNum = 0;
    auto hcclRet = HcclRankGraphGetLinks(hcclHandle, netLayerId, srcRankId, dstRankId, &linksList, &netLinkNum);
    if (hcclRet != HCCL_SUCCESS || netLinkNum == 0) {
        OPS_LOG_E(CONTEXT_NAME, "Get HCCL link failed, srcRankId %u, dstRankId %u, layerId %u.", srcRankId,
                  dstRankId, netLayerId);
        return false;
    }
    for (uint32_t index = 0; index < netLinkNum; ++index) {
        if (linksList[index].linkAttr.linkProtocol == protocol) {
            links = &linksList[index];
            return true;
        }
    }
    OPS_LOG_E(CONTEXT_NAME, "No matching communication protocol found, protocol is %d.", protocol);
    return false;
}

bool Mc2MoeGraphContext::GetNetLayers(const HcclComm &hcclHandle, uint32_t *&netLayerList, uint32_t &netLayerNum)
{
    auto hcclRet = HcclRankGraphGetLayers(hcclHandle, &netLayerList, &netLayerNum);
    if (hcclRet != HCCL_SUCCESS || netLayerNum == 0) {
        OPS_LOG_E(CONTEXT_NAME, "Get HCCL layers failed.");
        return false;
    }
    return true;
}

bool Mc2MoeGraphContext::GetRankSizePerServer(const HcclComm &hcclHandle, uint32_t netLayers)
{
    auto hcclRet = HcclRankGraphGetRankSizeByLayer(hcclHandle, netLayers, &rankSizePerServer_);
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E(CONTEXT_NAME, "Get HCCL rank size per server failed.");
        return false;
    }
    return true;
}

bool Mc2MoeGraphContext::InitHcclChannel(const HcclComm &hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                         const CommProtocol &protocol, std::vector<HcclChannelDesc> &channelDesc)
{
    uint32_t channelNum = channelDesc.size();
    auto hcclRet = HcclChannelDescInit(channelDesc.data(), channelNum);
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E(CONTEXT_NAME, "HCCL channel init failed.");
        return false;
    }

    uint32_t *netLayerList = nullptr;
    uint32_t netLayerNum = 0;
    if (!GetNetLayers(hcclHandle, netLayerList, netLayerNum)) {
        return false;
    }

    for (uint32_t i = 0; i < rankDim; ++i) {
        if (i == srcRankId) {
            continue;
        }
        uint32_t dstRank = i;
        uint32_t channelId = (i > srcRankId) ? (i - 1) : i;
        // 如果只有一层通信，直接使用该层；如果有多层通信，使用之前记录的通信层
        uint32_t layerId = (netLayerNum == 1) ? netLayerList[HCCL_COMM_LAYERS_UB_MEM] : layerMap[dstRank];
        CommLink *links = nullptr;
        if (!GetHcclCommLink(hcclHandle, layerId, srcRankId, dstRank, protocol, links)) {
            return false;
        }
        channelDesc[channelId].channelProtocol = protocol;
        channelDesc[channelId].remoteRank = dstRank;
        channelDesc[channelId].localEndpoint = links->srcEndpointDesc;
        channelDesc[channelId].remoteEndpoint = links->dstEndpointDesc;
    }
    return true;
}

bool Mc2MoeGraphContext::GetHcclCommChannel(const HcclComm &hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                            const CommProtocol &protocol, const CommEngine &engine,
                                            std::vector<ChannelHandle> &channels)
{
    uint32_t channelNum = rankDim - 1;
    std::vector<HcclChannelDesc> channelDesc(channelNum);
    channels.resize(channelNum);

    uint32_t *netLayerList = nullptr;
    uint32_t netLayerNum = 0;
    if (!GetNetLayers(hcclHandle, netLayerList, netLayerNum)) {
        return false;
    }
    uint32_t netLayers = netLayerList[GET_LOCAL_SERVER_RANK_SIZE_LAYER]; // 获取本server卡数
    if (!GetRankSizePerServer(hcclHandle, netLayers)) {
        return false;
    }
    if (!InitHcclChannel(hcclHandle, rankDim, srcRankId, protocol, channelDesc)) {
        return false;
    }
    auto hcclRet = HcclChannelAcquire(hcclHandle, engine, channelDesc.data(), channelNum, channels.data());
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E(CONTEXT_NAME, "Acquire HCCL channel failed.");
        return false;
    }
    return true;
}

bool Mc2MoeGraphContext::CheckLinks(uint32_t &netLinkNum, CommLink *linksList)
{
    for (uint32_t j = 0; j < netLinkNum; ++j) {
        if (linksList[j].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UB_MEM) {
            return true;
        }
    }
    return false;
}

bool Mc2MoeGraphContext::CheckProtocolSupport(const HcclComm &hcclHandle, uint32_t *&layerList, uint32_t &layerNum)
{
    uint32_t srcRankId = 0;
    uint32_t netLinkNum = 0;
    uint32_t rankNumInLayer = 0;
    uint32_t *rankIdLists = nullptr;
    CommLink *linksList = nullptr;

    auto hcclRet = HcclGetRankId(hcclHandle, &srcRankId);
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E(CONTEXT_NAME, "CheckProtocolSupport get rank ID failed.");
        return false;
    }

    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        hcclRet = HcclRankGraphGetRanksByLayer(hcclHandle, layerList[layerIndex], &rankIdLists, &rankNumInLayer);
        if (hcclRet != HCCL_SUCCESS) {
            OPS_LOG_E(CONTEXT_NAME, "Get rank IDs by layer failed.");
            return false;
        }
        for (uint32_t rankId = 0; rankId < rankNumInLayer; ++rankId) {
            // 本卡或者已经校验过的卡跳过
            if (rankIdLists[rankId] == srcRankId || layerMap.find(rankIdLists[rankId]) != layerMap.end()) {
                continue;
            }
            hcclRet = HcclRankGraphGetLinks(hcclHandle, layerList[layerIndex], srcRankId, rankIdLists[rankId],
                                            &linksList, &netLinkNum);
            if (hcclRet != HCCL_SUCCESS || netLinkNum == 0 || !CheckLinks(netLinkNum, linksList)) {
                OPS_LOG_E(CONTEXT_NAME, "No HCCL links support UB_MEM, srcRankID %u, dstRankID %u, layer %u.",
                          srcRankId, rankIdLists[rankId], layerList[layerIndex]);
                return false;
            }
            layerMap[rankIdLists[rankId]] = layerList[layerIndex];
        }
    }
    return true;
}

bool Mc2MoeGraphContext::GetCommProtocol(const HcclComm &hcclHandle, CommProtocol &protocol)
{
    uint32_t layerNum = 0;
    uint32_t *layerList = nullptr;
    if (!GetNetLayers(hcclHandle, layerList, layerNum)) {
        return false;
    }
    if (layerNum == HCCL_COMM_LAYERS_MTE_CCU) {
        OPS_LOG_I(CONTEXT_NAME, "HCCL communication layerNum is %u, set protocol to UB_MEM.", layerNum);
        protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
        return true;
    }
    if (!CheckProtocolSupport(hcclHandle, layerList, layerNum)) {
        return false;
    }
    protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
    return true;
}

bool Mc2MoeGraphContext::BuildContextData(const std::string &groupEp, std::vector<int32_t> &contextData,
                                          int64_t &hcclBuffSize)
{
    if (!LoadHcclSymbols()) {
        return false;
    }
    HcclComm hcclHandle;
    if (!GetCommHandle(groupEp.c_str(), hcclHandle)) {
        return false;
    }
    CommProtocol protocol;
    if (!GetCommProtocol(hcclHandle, protocol)) {
        return false;
    }

    Mc2Aclnn::Mc2MoeContext context{};
    auto hcclRet = HcclGetRankId(hcclHandle, &context.epRankId);
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E(CONTEXT_NAME, "Get rank ID failed.");
        return false;
    }
    hcclRet = HcclGetRankSize(hcclHandle, &epRankSize_);
    if (hcclRet != HCCL_SUCCESS || epRankSize_ == 0 || epRankSize_ > Mc2Aclnn::HCCL_MAX_RANK_SIZE) {
        OPS_LOG_E(CONTEXT_NAME, "Get rank size failed or rank size %u is invalid.", epRankSize_);
        return false;
    }

    std::vector<ChannelHandle> channels;
    if (!GetHcclCommChannel(hcclHandle, epRankSize_, context.epRankId, protocol, CommEngine::COMM_ENGINE_AIV,
                            channels)) {
        return false;
    }
    context.rankSizePerServer = rankSizePerServer_;

    for (uint32_t i = 0; i < epRankSize_; ++i) {
        void *tempBuffer = nullptr;
        uint64_t bufSize = 0;
        if (i == context.epRankId) {
            hcclRet = HcclGetHcclBuffer(hcclHandle, &tempBuffer, &hcclBuffSize_);
        } else {
            uint32_t idx = (i < context.epRankId) ? i : (i - 1);
            hcclRet = HcclChannelGetHcclBuffer(hcclHandle, channels[idx], &tempBuffer, &bufSize);
        }
        if (hcclRet != HCCL_SUCCESS || tempBuffer == nullptr) {
            OPS_LOG_E(CONTEXT_NAME, "Get HCCL buffer failed, src %u, dst %u.", context.epRankId, i);
            return false;
        }
        context.epHcclBuffer_[i] = reinterpret_cast<uint64_t>(tempBuffer) + i * EP_RANK_OFFSET_STEP;
    }

    contextData.resize(sizeof(Mc2Aclnn::Mc2MoeContext) / sizeof(int32_t));
    memcpy(contextData.data(), &context, sizeof(Mc2Aclnn::Mc2MoeContext));
    hcclBuffSize = static_cast<int64_t>(hcclBuffSize_);
    OPS_LOG_I(CONTEXT_NAME, "Build Mc2MoeContext data success, groupEp %s, hcclBuffSize %ld.", groupEp.c_str(),
              hcclBuffSize);
    return true;
}

bool Mc2MoeGraphContext::GetContextData(const std::string &groupEp, std::vector<int32_t> &contextData,
                                        int64_t &hcclBuffSize)
{
    if (groupEp.empty()) {
        OPS_LOG_E(CONTEXT_NAME, "groupEp is empty.");
        return false;
    }
    Mc2MoeGraphContext &manager = GetInstance();
    auto iter = manager.contextCache_.find(groupEp);
    if (iter != manager.contextCache_.end()) {
        contextData = iter->second.first;
        hcclBuffSize = iter->second.second;
        OPS_LOG_D(CONTEXT_NAME, "Found cached Mc2MoeContext data, groupEp %s.", groupEp.c_str());
        return true;
    }
    // 与 aclnn 场景（mc2_context.cpp 每次调用新建实例）对齐：
    // 每次构建使用全新实例，layerMap 等中间状态不跨通信域复用，缓存仅保存最终产物
    Mc2MoeGraphContext builder;
    if (!builder.BuildContextData(groupEp, contextData, hcclBuffSize)) {
        return false;
    }
    manager.contextCache_.emplace(groupEp, std::make_pair(contextData, hcclBuffSize));
    return true;
}

// 模板函数显式实例化
template void *Mc2MoeGraphContext::GetHcclLibFunc<void *>(void *handle, const std::string &funcName);

ge::graphStatus Mc2MoeGraphContext::CreateContextConstNode(ge::Graph &graph, const std::string &groupEp,
                                                           const std::vector<int32_t> &contextData,
                                                           ge::GNode &constNode)
{
    const std::string constName = kContextConstNamePrefix + groupEp;
    const size_t dataLen = contextData.size() * sizeof(int32_t);
    ge::Shape shape({static_cast<int64_t>(contextData.size())});
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_INT32);
    ge::Tensor tensorValue(tensorDesc, reinterpret_cast<const uint8_t *>(contextData.data()), dataLen);

    auto constOp = ge::OperatorFactory::CreateOperator(constName, "Const");
    if (constOp.IsEmpty()) {
        OPS_LOG_E(CONTEXT_NAME, "Const op not found in factory.");
        return ge::GRAPH_FAILED;
    }
    constOp.UpdateOutputDesc(0U, tensorDesc);
    constOp.SetAttr("value", tensorValue);

    constNode = graph.AddNodeByOp(constOp);
    OPS_LOG_I(CONTEXT_NAME, "Context Const node %s added, data len %zu.", constName.c_str(), dataLen);
    return ge::GRAPH_SUCCESS;
}

ge::GNode Mc2MoeGraphContext::FindContextConstNode(ge::Graph &graph, const std::string &groupEp)
{
    const std::string constName = kContextConstNamePrefix + groupEp;
    for (auto &gNode : graph.GetDirectNode()) {
        ge::AscendString name;
        ge::AscendString type;
        if (gNode.GetName(name) == ge::GRAPH_SUCCESS && name.GetString() != nullptr &&
            constName == name.GetString() && gNode.GetType(type) == ge::GRAPH_SUCCESS &&
            type.GetString() != nullptr && std::string(type.GetString()) == "Const") {
            return gNode;
        }
    }
    return ge::GNode();
}
} // namespace ops
#endif // HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
