/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

 #include <acl/acl.h>
 #include <pybind11/pybind11.h>
 #include <torch/extension.h>
 #include <unordered_set>

constexpr uint32_t BYTES_LEN = 20;

#define ERROR_LOG(...) \
   do { \
       std::time_t now = std::time(nullptr); \
       char timeBuf[BYTES_LEN]; \
       std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", std::localtime(&now)); \
       std::fprintf(stderr, "[%s] [%s:%d] ", timeBuf, __FILE__, __LINE__); \
       std::fprintf(stderr, __VA_ARGS__); \
       std::fflush(stderr); \
   } while (0)

#define OPS_LOG_STUB_IF(COND, LOG_FUNC, EXPR)                                                               \
    static_assert(std::is_same<bool, std::decay<decltype(COND)>::type>::value, "condition should be bool"); \
    do {                                                                                                    \
        if (__builtin_expect((COND), 0)) {                                                                  \
            LOG_FUNC;                                                                                       \
            EXPR;                                                                                           \
        }                                                                                                   \
    } while (0)                                                                                             \

int64_t getKeyValueOffset(const int kOrV, const int layerIdx, const int tokenIdx,
                          const int scalarOffset, const int scalarsPerToken,
                          const int numTokens, const int numLayers)
{
    return kOrV * numLayers * numTokens * scalarsPerToken +
           layerIdx * numTokens * scalarsPerToken +
           tokenIdx * scalarsPerToken + scalarOffset;
}

int64_t getPageBufferOffset(
    const int kOrV, const int tokenIdx, const int scalarOffset,
    const int scalarsPerToken, const int pageBufferSize)
{
    return kOrV * pageBufferSize * scalarsPerToken +
           tokenIdx * scalarsPerToken + scalarOffset;
}

struct AclrtStreamGuard {
    AclrtStreamGuard()
    {
        auto ret = aclrtCreateStream(&stream);
        OPS_LOG_STUB_IF(ret != ACL_SUCCESS,
            ERROR_LOG("Create stream failed! Error code: %d", static_cast<int32_t>(ret)),
            throw std::runtime_error("create stream failed!"));
    }
    ~AclrtStreamGuard()
    {
        auto ret = aclrtSynchronizeStreamWithTimeout(stream, 5000);
        OPS_LOG_STUB_IF(ret != ACL_SUCCESS, 
            ERROR_LOG("Synchronize stream failed! Error code: %d", static_cast<int32_t>(ret)), 
            throw std::runtime_error("synchronize stream failed!"));
        (void)aclrtDestroyStream(stream);
    }
    aclrtStream stream{};
};

int64_t getBlockKeyValueOffset(const int layerIdx, const int64_t blockIdx, const int scalarsPerBlock, const int sizePerBlock,
                        const int numTokens, const int numBlock, const int blockSize)
{
    return layerIdx * numBlock * blockSize * scalarsPerBlock + blockIdx * sizePerBlock;
}

int64_t getBlockPageBufferOffset(const int64_t blockIdx, const int scalarsPerToken)
{
    return blockIdx * scalarsPerToken;
}

void multiLayerBlockKvTransfer(at::Tensor &keyValue,
                            const at::Tensor &pagedBufferLoraPtrs,
                            const at::Tensor &pagedBufferRopePtrs,
                            const at::Tensor &slotMapping,
                            const at::Device &pagedMemoryDevice,
                            const int loraRank,
                            const int ropeDim,
                            const bool direction)
{
    int numLayers = keyValue.size(0);
    int numBlock = keyValue.size(1);
    int blockSize = keyValue.size(2);
    int numTokens = slotMapping.size(0);
    OPS_LOG_STUB_IF(numTokens == 0,
        ERROR_LOG("numTokens: %d", numTokens), throw std::runtime_error("numTokens is 0!"));

    int scalarsPerToken = keyValue.size(3);
    OPS_LOG_STUB_IF((loraRank + ropeDim) != scalarsPerToken,
        ERROR_LOG("loraRank: %d, ropeDim: %d, scalarsPerToken: %d", loraRank, ropeDim, scalarsPerToken), throw std::runtime_error("Wrong loraRank of ropeDim"));
    OPS_LOG_STUB_IF(blockSize == 0,
        ERROR_LOG("blockSize: %d", blockSize), throw std::runtime_error("blockSize is 0!"));
    aclError ret = 0;
    aclrtMemcpyKind memcpyKind{};
    if (keyValue.device().is_cpu()) {
        memcpyKind = direction ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_HOST_TO_DEVICE;
    } else {
        memcpyKind = ACL_MEMCPY_DEVICE_TO_HOST;
    }
    AclrtStreamGuard sg;

    int64_t* slotMappingPtr = slotMapping.data_ptr<int64_t>();
    uint64_t* pagedBufferLoraPtrsCpu = reinterpret_cast<uint64_t *>(pagedBufferLoraPtrs.data_ptr<int64_t>());
    uint64_t* pagedBufferRopePtrsCpu = reinterpret_cast<uint64_t *>(pagedBufferRopePtrs.data_ptr<int64_t>());
    const int64_t totalLoraLength = numBlock * blockSize * loraRank;
    for (int layerID = 0; layerID < numLayers; layerID++) {
        void *pagedBufferLoraPtr = reinterpret_cast<void *>(pagedBufferLoraPtrsCpu[layerID]);
        void *pagedBufferRopePtr = reinterpret_cast<void *>(pagedBufferRopePtrsCpu[layerID]);
        std::unordered_set<int64_t> copiedSet{};
        int tokenIdx = 0;
        for (int tokenID = 0; tokenID < numTokens; tokenID++) {
            const int64_t slotIdx = slotMappingPtr[tokenID];
            if (slotIdx < 0) {
                continue;
            }
            int64_t blockId = slotIdx / blockSize;
            if (copiedSet.find(blockId) == copiedSet.end()) {
                int64_t previousRealSlotIdx = (blockId + 1) * blockSize;
                int64_t previousRealTokenIdx = (tokenIdx + 1) * blockSize;
                int64_t realSlotIdx = blockId * blockSize;
                int64_t realTokenIdx = tokenIdx * blockSize;
                const int64_t startLoraLmcacheOffset = getBlockKeyValueOffset(layerID, realTokenIdx, scalarsPerToken, loraRank, numTokens, numBlock, blockSize);
                const int64_t endLoraLmcacheOffset = getBlockKeyValueOffset(layerID, previousRealTokenIdx, scalarsPerToken, loraRank, numTokens, numBlock, blockSize);
                const int64_t startLoraVllmOffset = getBlockPageBufferOffset(realSlotIdx, loraRank);
                const int64_t endLoraVllmOffset = getBlockPageBufferOffset(previousRealSlotIdx, loraRank);
                uint32_t loraLength = (endLoraVllmOffset - startLoraVllmOffset) * sizeof(int16_t);
                void* fromLora = reinterpret_cast<void *>(reinterpret_cast<int8_t *>(keyValue.data_ptr()) + startLoraLmcacheOffset * sizeof(int16_t));
                void* toLora = reinterpret_cast<void *>(reinterpret_cast<int8_t *>(pagedBufferLoraPtr) + startLoraVllmOffset * sizeof(int16_t));
                OPS_LOG_STUB_IF(startLoraLmcacheOffset * sizeof(int16_t) + loraLength > keyValue.numel() * keyValue.element_size(),
                    ERROR_LOG("Wrong offset, startLoraLmcacheOffset: %d, loraLength: %d", startLoraLmcacheOffset, loraLength), throw std::runtime_error("Wrong offset!"));

                const int64_t startRopeLmcacheOffset = totalLoraLength + getBlockKeyValueOffset(layerID, realTokenIdx, scalarsPerToken, ropeDim, numTokens, numBlock, blockSize);
                const int64_t endRopeLmcacheOffset = totalLoraLength + getBlockKeyValueOffset(layerID, previousRealTokenIdx, scalarsPerToken, ropeDim, numTokens, numBlock, blockSize);
                const int64_t startRopeVllmOffset = getBlockPageBufferOffset(realSlotIdx, ropeDim);
                const int64_t endRopeVllmOffset = getBlockPageBufferOffset(previousRealSlotIdx, ropeDim);
                uint32_t ropeLength = (endRopeVllmOffset - startRopeVllmOffset) * sizeof(int16_t);
                void* fromRope = reinterpret_cast<void *>(reinterpret_cast<int8_t *>(keyValue.data_ptr()) + startRopeLmcacheOffset * sizeof(int16_t));
                void* toRope = reinterpret_cast<void *>(reinterpret_cast<int8_t *>(pagedBufferRopePtr) + startRopeVllmOffset * sizeof(int16_t));
                OPS_LOG_STUB_IF(startRopeLmcacheOffset * sizeof(int16_t) + ropeLength > keyValue.numel() * keyValue.element_size(),
                    ERROR_LOG("Wrong offset, startRopeLmcacheOffset: %d, ropeLength: %d", startRopeLmcacheOffset, ropeLength), throw std::runtime_error("Wrong offset!"));
                if (direction) {
                    ret = aclrtMemcpyAsync(fromLora, loraLength, toLora, loraLength, memcpyKind, sg.stream);
                    OPS_LOG_STUB_IF(ret != ACL_SUCCESS,
                        ERROR_LOG("Send aclrtMemcpyAsync failed! Error code: %d", static_cast<int32_t>(ret)),
                        throw std::runtime_error("Send aclrtMemcpyAsync failed!"));
                    ret = aclrtMemcpyAsync(fromRope, ropeLength, toRope, ropeLength, memcpyKind, sg.stream);
                    OPS_LOG_STUB_IF(ret != ACL_SUCCESS,
                        ERROR_LOG("Send aclrtMemcpyAsync failed! Error code: %d", static_cast<int32_t>(ret)),
                        throw std::runtime_error("Send aclrtMemcpyAsync failed!"));
                } else {
                    ret = aclrtMemcpyAsync(toLora, loraLength, fromLora, loraLength, memcpyKind, sg.stream);
                    OPS_LOG_STUB_IF(ret != ACL_SUCCESS,
                        ERROR_LOG("Send aclrtMemcpyAsync failed! Error code: %d", static_cast<int32_t>(ret)),
                        throw std::runtime_error("Send aclrtMemcpyAsync failed!"));
                    ret = aclrtMemcpyAsync(toRope, ropeLength, fromRope, ropeLength, memcpyKind, sg.stream);
                    OPS_LOG_STUB_IF(ret != ACL_SUCCESS,
                        ERROR_LOG("Send aclrtMemcpyAsync failed! Error code: %d", static_cast<int32_t>(ret)),
                        throw std::runtime_error("Send aclrtMemcpyAsync failed!"));
                }
                tokenIdx++;
                copiedSet.insert(blockId);
            }
        }
    }
}

void CopyKV(at::Tensor &keyValue, void *pagedBufferPtr, int kOrV, int layerID, int startSrc,
            int scalarsPerToken, int numTokens, int numLayers, int startDst, int pageBufferSize,
            int previousSrcIdx, int previousDstIdx, aclrtMemcpyKind memcpyKind, int direction, aclrtStream &stream)
{
    const int64_t startLmcacheOffset = getKeyValueOffset(kOrV, layerID, startSrc, 0, scalarsPerToken, numTokens, numLayers);
    const int64_t startVllmOffset = getPageBufferOffset(kOrV, startDst, 0, scalarsPerToken, pageBufferSize);
    const int64_t endLmcacheOffset = getKeyValueOffset(kOrV, layerID, previousSrcIdx, scalarsPerToken, scalarsPerToken, numTokens, numLayers);
    const int64_t endVllmOffset = getPageBufferOffset(kOrV, previousDstIdx, scalarsPerToken, scalarsPerToken, pageBufferSize);

    OPS_LOG_STUB_IF(endVllmOffset < startVllmOffset,
                    ERROR_LOG("Wrong offset"), throw std::runtime_error("Wrong offset!"));

    uint32_t length = (endVllmOffset - startVllmOffset) * sizeof(int16_t);
    void* from = reinterpret_cast<void*>(reinterpret_cast<int8_t*>(keyValue.data_ptr()) + startLmcacheOffset * sizeof(int16_t));
    void* to = reinterpret_cast<void*>(reinterpret_cast<int8_t*>(pagedBufferPtr) + startVllmOffset * sizeof(int16_t));

     OPS_LOG_STUB_IF(startLmcacheOffset * sizeof(int16_t) + length > keyValue.numel() * keyValue.element_size(),
                    ERROR_LOG("Wrong offset"), throw std::runtime_error("Wrong offset!"));

    if (direction) {
        auto ret = aclrtMemcpyAsync(from, length, to, length, memcpyKind, stream);
        OPS_LOG_STUB_IF(ret != ACL_SUCCESS,
                        ERROR_LOG("Send aclrtMemcpyAsync failed! Error code: %d", static_cast<int32_t>(ret)),
                        throw std::runtime_error("Send aclrtMemcpyAsync failed!"));
    } else {
        auto ret = aclrtMemcpyAsync(to, length, from, length, memcpyKind, stream);
        OPS_LOG_STUB_IF(ret != ACL_SUCCESS,
                        ERROR_LOG("Send aclrtMemcpyAsync failed! Error code: %d", static_cast<int32_t>(ret)),
                        throw std::runtime_error("Send aclrtMemcpyAsync failed!"));
    }
}

void multiLayerKvTransfer(at::Tensor &keyValue,
                          const at::Tensor &pagedBufferPtrs,
                          const at::Tensor &slotMapping,
                          const at::Device &pagedMemoryDevice,
                          const int pageBufferSize,
                          const bool direction,
                          const bool useMLA)
{
    int numLayers = keyValue.size(1);
    int numTokens = slotMapping.size(0);
    int scalarsPerToken = keyValue.size(3);
    int kOrVSize = useMLA ? 1 : 2;
    aclError ret = 0;
    aclrtMemcpyKind memcpyKind{};
    if (keyValue.device().is_cpu()) {
        memcpyKind = direction ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_HOST_TO_DEVICE;
    } else {
        memcpyKind = ACL_MEMCPY_DEVICE_TO_DEVICE;
    }
    AclrtStreamGuard sg;

    int64_t* slotMappingPtr = slotMapping.data_ptr<int64_t>();
    uint64_t* pagedBufferPtrsCpu = reinterpret_cast<uint64_t*>(pagedBufferPtrs.data_ptr<int64_t>());
    for (int layerID = 0; layerID < numLayers; layerID++) {
        void *pagedBufferPtr = reinterpret_cast<void*>(pagedBufferPtrsCpu[layerID]);
        for (int kOrV = 0; kOrV < kOrVSize; kOrV++) {
            int startSrc = 0;
            int startDst = slotMappingPtr[0];
            int previousSrcIdx = 0;
            int previousDstIdx = slotMappingPtr[0];
            for (int tokenID = 1; tokenID < numTokens; tokenID++) {
                const int64_t slotIdx = slotMappingPtr[tokenID];
                if (slotIdx < 0) {
                    continue;
                }
                if (startDst < 0 || previousDstIdx < 0) {
                    startSrc = tokenID;
                    startDst = slotIdx;
                    previousSrcIdx = tokenID;
                    previousDstIdx = slotIdx;
                    continue;
                }
                if (tokenID != previousSrcIdx + 1 || slotIdx != previousDstIdx + 1) {
                    CopyKV(keyValue, pagedBufferPtr, kOrV, layerID, startSrc, scalarsPerToken, numTokens, numLayers, startDst,
                           pageBufferSize, previousSrcIdx, previousDstIdx, memcpyKind, direction, sg.stream);
                    startSrc = tokenID;
                    startDst = slotIdx;
                }
                previousSrcIdx = tokenID;
                previousDstIdx = slotIdx;
            }
            if (startDst >= 0) {
                CopyKV(keyValue, pagedBufferPtr, kOrV, layerID, startSrc, scalarsPerToken, numTokens, numLayers, startDst,
                       pageBufferSize, previousSrcIdx, previousDstIdx, memcpyKind, direction, sg.stream);
            }
        }
    }
}

namespace py = pybind11;

PYBIND11_MODULE(c_ops, m) {
    m.def("multi_layer_block_kv_transfer", &multiLayerBlockKvTransfer);
    m.def("multi_layer_kv_transfer", &multiLayerKvTransfer);
}



