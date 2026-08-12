/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MEGA_MOE_TORCH_ADPT_H
#define MEGA_MOE_TORCH_ADPT_H

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <hccl/hccl_res.h>
#include <hccl/hcomm_res_defs.h>

namespace vllm_ascend {
namespace mega_moe_adpt {

constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;
constexpr uint8_t COMM_ENGINE_AIV = 4;
constexpr int64_t ACL_DTYPE_ENUM_OFFSET = 256;

struct CommContext {
    uint32_t epRankId = 0;
    uint32_t rankSizePerServer = 0;
    uint64_t kfcContextAddr = 0;
    uint64_t epHcclBuffer[HCCL_MAX_RANK_SIZE] = {};
    ChannelHandle hcommHandle[HCCL_MAX_RANK_SIZE] = {};
};

using HcclKfcAllocOpArgs = HcclResult (*)(void **);
using HcclKfcOpArgsSetAlgConfig = HcclResult (*)(void *, char *);
using HcclKfcOpArgsSetCommEngine = HcclResult (*)(void *, uint8_t);
using HcclCreateOpResCtx = HcclResult (*)(HcclComm, uint8_t, void *, void **);
using HcclGetRemoteIpcHcclBuf = HcclResult (*)(HcclComm, uint64_t, void **, uint64_t *);
using HcclKfcFreeOpArgs = HcclResult (*)(void *);
using HcclCommGetHandleWithName = HcclResult (*)(const char *, HcclComm *);
using HcclGetRankSize = HcclResult (*)(HcclComm, uint32_t *);
using HcclGetRankId = HcclResult (*)(HcclComm, uint32_t *);
using HcclGetHcclBuffer = HcclResult (*)(HcclComm, void **, uint64_t *);

template <typename T>
inline T LoadHcclSymbol(const char *library, const char *symbol)
{
    static std::unordered_map<std::string, void *> handlers;
    static std::mutex handlersMutex;
    std::lock_guard<std::mutex> lock(handlersMutex);
    void *&handler = handlers[library];
    if (handler == nullptr) {
        handler = dlopen(library, RTLD_LAZY);
    }
    TORCH_CHECK(handler != nullptr, "Failed to load ", library, ": ", dlerror());
    auto function = reinterpret_cast<T>(dlsym(handler, symbol));
    TORCH_CHECK(function != nullptr, "Failed to load ", symbol, " from ", library, ": ", dlerror());
    return function;
}

struct CachedContext {
    at::Tensor tensor;
    int64_t cclBufferSize = 0;
};

inline CachedContext BuildContext(const std::string &group, int64_t worldSize)
{
    auto allocArgs = LoadHcclSymbol<HcclKfcAllocOpArgs>("libhccl.so", "HcclKfcAllocOpArgs");
    auto setAlg = LoadHcclSymbol<HcclKfcOpArgsSetAlgConfig>("libhccl.so", "HcclKfcOpArgsSetAlgConfig");
    auto setEngine = LoadHcclSymbol<HcclKfcOpArgsSetCommEngine>("libhccl.so", "HcclKfcOpArgsSetCommEngine");
    auto createContext = LoadHcclSymbol<HcclCreateOpResCtx>("libhccl.so", "HcclCreateOpResCtx");
    auto freeArgs = LoadHcclSymbol<HcclKfcFreeOpArgs>("libhccl.so", "HcclKfcFreeOpArgs");
    auto getRankSize = LoadHcclSymbol<HcclGetRankSize>("libhccl.so", "HcclGetRankSize");
    auto getRankId = LoadHcclSymbol<HcclGetRankId>("libhccl.so", "HcclGetRankId");
    auto getLocalBuffer = LoadHcclSymbol<HcclGetHcclBuffer>("libhccl.so", "HcclGetHcclBuffer");
    auto getComm = LoadHcclSymbol<HcclCommGetHandleWithName>("libhccl_fwk.so", "HcclCommGetHandleWithName");
    auto getRemoteBuffer =
        LoadHcclSymbol<HcclGetRemoteIpcHcclBuf>("libhccl_fwk.so", "HcclGetRemoteIpcHcclBuf");

    void *opArgs = nullptr;
    TORCH_CHECK(allocArgs(&opArgs) == HCCL_SUCCESS, "HcclKfcAllocOpArgs failed");
    TORCH_CHECK(setEngine(opArgs, COMM_ENGINE_AIV) == HCCL_SUCCESS, "HcclKfcOpArgsSetCommEngine failed");

    const char *socName = aclrtGetSocName();
    const bool isAscend910B = socName != nullptr && std::strstr(socName, "Ascend910B") != nullptr;
    const bool isMultiServer = worldSize > 8;
    std::string algorithm =
        isAscend910B && isMultiServer ? "BatchWrite=level1:hierarchy" : "AlltoAll=level0:fullmesh;level1:pairwise";
    TORCH_CHECK(setAlg(opArgs, algorithm.data()) == HCCL_SUCCESS, "HcclKfcOpArgsSetAlgConfig failed");

    HcclComm comm = nullptr;
    TORCH_CHECK(getComm(group.c_str(), &comm) == HCCL_SUCCESS, "HcclCommGetHandleWithName failed for ", group);

    CommContext hostContext;
    void *resourceContext = nullptr;
    const uint8_t opType = isAscend910B && isMultiServer ? 18 : 8;
    TORCH_CHECK(createContext(comm, opType, opArgs, &resourceContext) == HCCL_SUCCESS,
                "HcclCreateOpResCtx failed for ", group);
    hostContext.kfcContextAddr = reinterpret_cast<uint64_t>(resourceContext);
    TORCH_CHECK(freeArgs(opArgs) == HCCL_SUCCESS, "HcclKfcFreeOpArgs failed");

    uint32_t actualWorldSize = 0;
    TORCH_CHECK(getRankSize(comm, &actualWorldSize) == HCCL_SUCCESS, "HcclGetRankSize failed");
    TORCH_CHECK(actualWorldSize == static_cast<uint32_t>(worldSize),
                "MegaMoe EP world size mismatch: expected ", worldSize, ", got ", actualWorldSize);
    TORCH_CHECK(getRankId(comm, &hostContext.epRankId) == HCCL_SUCCESS, "HcclGetRankId failed");

    uint64_t cclBufferSize = 0;
    if (isAscend910B && isMultiServer) {
        void *address = nullptr;
        TORCH_CHECK(getLocalBuffer(comm, &address, &cclBufferSize) == HCCL_SUCCESS, "HcclGetHcclBuffer failed");
        hostContext.epHcclBuffer[hostContext.epRankId] = reinterpret_cast<uint64_t>(address);
    } else {
        for (uint64_t peer = 0; peer < static_cast<uint64_t>(worldSize); ++peer) {
            void *address = nullptr;
            uint64_t peerBufferSize = 0;
            HcclResult result = peer == hostContext.epRankId
                ? getLocalBuffer(comm, &address, &peerBufferSize)
                : getRemoteBuffer(comm, peer, &address, &peerBufferSize);
            TORCH_CHECK(result == HCCL_SUCCESS, "Failed to get HCCL buffer for peer ", peer);
            if (peer == hostContext.epRankId) {
                cclBufferSize = peerBufferSize;
            }
            hostContext.epHcclBuffer[peer] = reinterpret_cast<uint64_t>(address);
        }
    }

    at::Tensor context = at::empty(
        {static_cast<int64_t>((sizeof(CommContext) + sizeof(int32_t) - 1) / sizeof(int32_t))},
        at::TensorOptions().dtype(at::kInt).device(c10::DeviceType::PrivateUse1));
    at::Tensor hostTensor = at::from_blob(&hostContext, {static_cast<int64_t>(sizeof(CommContext) / sizeof(int32_t))},
                                          at::TensorOptions().dtype(at::kInt));
    context.copy_(hostTensor);
    return {context, static_cast<int64_t>(cclBufferSize)};
}

inline const CachedContext &GetContext(const std::string &group, int64_t worldSize)
{
    static std::unordered_map<std::string, CachedContext> contexts;
    static std::mutex contextsMutex;
    std::lock_guard<std::mutex> lock(contextsMutex);
    const std::string key = group + ":" + std::to_string(worldSize);
    auto iterator = contexts.find(key);
    if (iterator == contexts.end()) {
        iterator = contexts.emplace(key, BuildContext(group, worldSize)).first;
    }
    return iterator->second;
}

}  // namespace mega_moe_adpt

inline std::tuple<at::Tensor, at::Tensor> mega_moe(
    const at::Tensor &x,
    const at::Tensor &topkIds,
    const at::Tensor &topkWeights,
    const at::TensorList &weight1,
    const at::TensorList &weight2,
    const at::TensorList &weightScales1,
    const at::TensorList &weightScales2,
    const c10::optional<at::Tensor> &xActiveMask,
    c10::string_view group,
    int64_t moeExpertNum,
    int64_t epWorldSize,
    int64_t maxRecvTokenNum,
    int64_t numMaxTokensPerRank,
    c10::string_view activation,
    double activationClamp,
    double activationAlpha,
    double activationBeta)
{
    TORCH_CHECK(x.dim() == 2 && topkIds.dim() == 2, "MegaMoe x and topk_ids must be 2-D");
    TORCH_CHECK(!weight1.empty() && !weight2.empty(), "MegaMoe weight lists must not be empty");
    TORCH_CHECK(weight1[0].scalar_type() == at::kChar && weight2[0].scalar_type() == at::kChar,
                "vLLM Ascend local MegaMoe adapter only supports W8A8 weights");
    TORCH_CHECK(moeExpertNum % epWorldSize == 0, "MegaMoe expert count must be divisible by EP world size");

    std::string groupString(group);
    const auto &context = mega_moe_adpt::GetContext(groupString, epWorldSize);
    at::Tensor output = at::empty_like(x);
    at::Tensor expertTokenNums = at::empty({moeExpertNum / epWorldSize}, x.options().dtype(at::kInt));

    c10::optional<at::TensorList> noTensorList = c10::nullopt;
    std::string commAlgorithm;
    std::string activationString(activation);
    std::vector<float> activationParams;
    if (activationString == "swigluoai") {
        activationParams = {
            static_cast<float>(activationClamp),
            static_cast<float>(activationAlpha),
            static_cast<float>(activationBeta),
        };
    } else if (activationClamp > 0) {
        activationParams = {static_cast<float>(activationClamp)};
    }
    char *commAlgorithmPtr = commAlgorithm.data();
    char *activationPtr = activationString.data();
    int64_t dispatchQuantMode = 2;
    int64_t dispatchQuantOutDtype = ACL_INT8;
    int64_t combineQuantMode = 0;
    int64_t topoType = 0;
    int64_t rankNumPerServer = 2;
    int64_t topkWeightsType = 0;

    EXEC_NPU_CMD(
        aclnnMegaMoe,
        context.tensor,
        x,
        topkIds,
        topkWeights,
        weight1,
        weight2,
        weightScales1,
        weightScales2,
        noTensorList,
        noTensorList,
        xActiveMask,
        noTensorList,
        noTensorList,
        noTensorList,
        noTensorList,
        noTensorList,
        noTensorList,
        moeExpertNum,
        epWorldSize,
        context.cclBufferSize,
        maxRecvTokenNum,
        dispatchQuantMode,
        dispatchQuantOutDtype,
        combineQuantMode,
        commAlgorithmPtr,
        numMaxTokensPerRank,
        activationPtr,
        activationParams,
        topoType,
        rankNumPerServer,
        topkWeightsType,
        output,
        expertTokenNums);

    return {output, expertTokenNums};
}

}  // namespace vllm_ascend
#endif
