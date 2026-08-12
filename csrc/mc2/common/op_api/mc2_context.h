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
 * \file mc2_context.h
 * \brief MC2 Context management for HCCL communication
 */

#ifndef MC2_CONTEXT_H
#define MC2_CONTEXT_H

#define HCCL_CHANNEL_SUPPORT_VERSION 89999700
#if __has_include("version/hcomm_version.h")
#include "version/hcomm_version.h"
#else
#define HCOMM_VERSION_NUM HCCL_CHANNEL_SUPPORT_VERSION
#endif
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "hccl/hccl_rank_graph.h"
#include "aclnn/aclnn_base.h"
#include "mc2_log_compat.h"
#include "opdev/common_types.h"
#include "mc2_moe_context.h"

namespace Mc2Aclnn {

class Mc2Context {
public:
    static aclnnStatus GetMc2ContextTensor(const char *groupEp, const char *opName, uint64_t &hcclBuffSize,
                                           aclTensor *&mc2Context);
    static aclnnStatus GetMc2RankSize(const char *groupEp, uint32_t &rankSize);

private:
    explicit Mc2Context();
    ~Mc2Context();

    aclnnStatus LoadHcclSymbols();
    aclnnStatus GetCommHandle(const char *groupEp, HcclComm &hcclHandle);
    aclnnStatus GetHcclCommLink(const HcclComm &hcclHandle, uint32_t netLayers, uint32_t srcRankId, uint32_t dstRankId,
                                const CommProtocol &protocol, CommLink *&links);
    aclnnStatus GetNetLayers(const HcclComm &hcclHandle, uint32_t *&netLayerList, uint32_t &netLayerNum);
    aclnnStatus GetRankSizePerServer(const HcclComm &hcclHandle, uint32_t netLayers);
    aclnnStatus InitHcclChannel(const HcclComm &hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                const CommProtocol &protocol, std::vector<HcclChannelDesc> &channelDesc);
    aclnnStatus GetHcclCommChannel(const HcclComm &hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                   const CommProtocol &protocol, const CommEngine &engine,
                                   std::vector<ChannelHandle> &channels);
    aclnnStatus GetHcclCommResource(const HcclComm &hcclHandle, const CommEngine &engine, const CommProtocol &protocol,
                                    Mc2MoeContext *mc2ContextStruct);
    aclnnStatus CreatMc2Context(const HcclComm &hcclHandle, const std::string &mc2ContextTag, const CommEngine &engine,
                                const CommProtocol &protocol, Mc2MoeContext *mc2ContextStruct, void *&ctx,
                                uint64_t &hcclBuffSize);
    aclnnStatus CreatMc2ContextTensor(void *ctx, aclTensor *&mc2Context);
    aclnnStatus GetHcclBufferSize(const HcclComm &hcclHandle, uint64_t &hcclBuffSize);
    aclnnStatus CheckProtocolSupport(const HcclComm &hcclHandle, uint32_t *&layerList, uint32_t &layerNum);
    aclnnStatus GetCommProtocol(const HcclComm &hcclHandle, CommProtocol &protocol);
    aclnnStatus ValidateContextTag(const std::string &mc2ContextTag);
    aclnnStatus CheckLinks(uint32_t &netLinkNum, CommLink *linksList);
    aclnnStatus CheckContextCache(const HcclComm &hcclHandle, const std::string &mc2ContextTag,
                                  const CommEngine &engine, void *&ctx, uint64_t &hcclBuffSize);
    const std::string GetLibPath();
    template <typename T>
    T GetHcclLibFunc(void *handle, const std::string &funcName);

    void *hcclLibHandle_ = nullptr;
    uint64_t hcclBuffSize_ = 0;
    uint32_t epRankSize_ = 0;
    uint32_t rankSizePerServer_ = 0;
    std::unordered_map<uint32_t, uint32_t> layerMap; // 记录本卡与其他卡的通信层数，key为其他卡的rankId，value为通信层数

    HcclResult (*HcomGetCommHandleByGroup)(const char *, HcclComm *) = nullptr;
    HcclResult (*HcclRankGraphGetLinks)(HcclComm, uint32_t, uint32_t, uint32_t, CommLink **, uint32_t *) = nullptr;
    HcclResult (*HcclRankGraphGetLayers)(HcclComm, uint32_t **, uint32_t *) = nullptr;
    HcclResult (*HcclRankGraphGetRankSizeByLayer)(HcclComm, uint32_t, uint32_t *) = nullptr;
    HcclResult (*HcclChannelAcquire)(HcclComm, CommEngine, HcclChannelDesc *, uint32_t, ChannelHandle *) = nullptr;
    HcclResult (*HcclGetHcclBuffer)(HcclComm, void **, uint64_t *) = nullptr;
    HcclResult (*HcclChannelGetHcclBuffer)(HcclComm, ChannelHandle, void **, uint64_t *) = nullptr;
    HcclResult (*HcclEngineCtxCreate)(HcclComm, const char *, CommEngine, uint64_t, void **) = nullptr;
    HcclResult (*HcclEngineCtxGet)(HcclComm, const char *, CommEngine, void **, uint64_t *) = nullptr;
    HcclResult (*HcclEngineCtxCopy)(HcclComm, CommEngine, const char *, void *, uint64_t, uint64_t) = nullptr;
    HcclResult (*HcclGetRankId)(HcclComm, uint32_t *) = nullptr;
    HcclResult (*HcclGetRankSize)(HcclComm, uint32_t *) = nullptr;
    HcclResult (*HcclRankGraphGetRanksByLayer)(HcclComm, uint32_t, uint32_t **, uint32_t *) = nullptr;
};
} // namespace Mc2Aclnn

#endif
#endif // MC2_CONTEXT_H
