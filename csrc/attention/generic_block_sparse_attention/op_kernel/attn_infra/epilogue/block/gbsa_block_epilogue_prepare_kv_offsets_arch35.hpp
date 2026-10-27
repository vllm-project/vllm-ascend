/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_EPILOGUE_PREPARE_KV_OFFSETS_ARCH35_HPP
#define GBSA_BLOCK_EPILOGUE_PREPARE_KV_OFFSETS_ARCH35_HPP

#include "../../../attn_infra/arch/gbsa_resource.hpp"
#include "../../../attn_infra/epilogue/gbsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/block/gbsa_block_epilogue_arch35_utils.hpp"
#include "../../../arch35/gbsa_layout_utils.hpp"

namespace NpuArch::Epilogue::Block {

template <bool PAGED_CACHE_FLAG_, bool RESIDUAL_BLOCK_FLAG_>
class BlockEpilogue<EpilogueAtlasA5PrepareKvOffsets<PAGED_CACHE_FLAG_, RESIDUAL_BLOCK_FLAG_>> {
public:
    using DispatchPolicy = EpilogueAtlasA5PrepareKvOffsets<PAGED_CACHE_FLAG_, RESIDUAL_BLOCK_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;

    static constexpr bool PAGED_CACHE_FLAG = DispatchPolicy::PAGED_CACHE_FLAG;
    static constexpr bool RESIDUAL_BLOCK_FLAG = DispatchPolicy::RESIDUAL_BLOCK_FLAG;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource, UBufTileHelper &uBufTileHelper,
                                    uint32_t tileSize, uint32_t l1TileN, uint32_t blockY, uint32_t pageSize,
                                    uint64_t tokenStride, uint64_t kPageStride, uint64_t vPageStride)
        : tileSize_(tileSize), l1TileN_(l1TileN), blockY_(blockY), pageSize_(pageSize),
          tokenStride_(tokenStride), kPageStride_(kPageStride), vPageStride_(vPageStride)
    {
        // The mask reservation is unused by the current sparse softmax path.
        groupWords_ = GbsaKernelArch35::CalcKvOffsetGroupWords<PAGED_CACHE_FLAG>(l1TileN_, blockY_);
        recordsBufMask_ = uBufTileHelper.kvOffsetUbBufNum - 1U;
        for (uint32_t i = 0; i < uBufTileHelper.kvOffsetUbBufNum; ++i) {
            recordsBuf_[i] = resource.ubBuf.template GetBufferByByte<uint64_t>(
                uBufTileHelper.maskStartOffset + i * groupWords_ * sizeof(uint64_t));
        }
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<uint64_t> info,
                                      AscendC::GlobalTensor<int32_t> sparseIdx,
                                      AscendC::GlobalTensor<int32_t> blockTable,
                                      uint32_t tileIdx, uint32_t tileTokens,
                                      uint32_t explicitCount, uint32_t residualId)
    {
        // Preserve the 64B header and allocated group capacity; only dense payloads shrink.
        // Dense records: [offset64, (sourceStride << 32) | (length << 1) | (ndNum - 1)].
        constexpr uint32_t recordWords = PAGED_CACHE_FLAG ? 3U : 2U;
        const uint32_t groupNum = (tileTokens + l1TileN_ - 1U) / l1TileN_;
        for (uint32_t groupIdx = AscendC::GetSubBlockIdx(); groupIdx < groupNum; groupIdx += 2U) {
            // Alternate across actual groups, including groups in successive tiles/tasks.
            // Only wait when reusing this half; the other half may still be copying.
            const auto eventId = static_cast<event_t>(EVENT_ID5 + recordsBufIdx_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
            auto records_ = recordsBuf_[recordsBufIdx_];
            auto groupInfo = info[groupIdx * groupWords_];
            const uint32_t groupBegin = groupIdx * l1TileN_;
            const uint32_t groupTokens = min(l1TileN_, tileTokens - groupBegin);
            const uint32_t compactBegin = tileIdx * tileSize_ + groupBegin;
            uint32_t inner = compactBegin % blockY_;
            uint32_t count = 0;
            uint32_t dealtLenAccum = 0;
            uint32_t previousLen = 0;
            uint64_t previousOffset = 0;
            for (uint32_t blockIdx = compactBegin / blockY_; dealtLenAccum < groupTokens; ++blockIdx) {
                const uint32_t rows = min(blockY_ - inner, groupTokens - dealtLenAccum);
                const int32_t blockId = GbsaKernelArch35::GetSparseBlockId<RESIDUAL_BLOCK_FLAG>(
                    sparseIdx, blockIdx, explicitCount, residualId);
                const uint64_t sourceToken = static_cast<uint64_t>(blockId) * blockY_ + inner;
                // A page split produces ordinary entries; Cube never parses pages.
                for (uint32_t copied = 0; copied < rows;) {
                    uint32_t len = rows - copied;
                    uint64_t token = sourceToken + copied;
                    uint64_t kOffset = 0;
                    uint64_t vOffset = 0;
                    if constexpr (PAGED_CACHE_FLAG) {
                        const uint32_t pageInner = token % pageSize_;
                        const uint32_t page = blockTable.GetValue(token / pageSize_);
                        len = min(len, pageSize_ - pageInner);
                        kOffset = static_cast<uint64_t>(page) * kPageStride_;
                        vOffset = static_cast<uint64_t>(page) * vPageStride_;
                        token = pageInner;
                    }
                    kOffset += token * tokenStride_;
                    vOffset += token * tokenStride_;
                    const uint32_t word = 8U + count * recordWords;
                    if constexpr (PAGED_CACHE_FLAG) {
                        records_.SetValue(word, kOffset);
                        records_.SetValue(word + 1U, vOffset);
                        records_.SetValue(word + 2U, len);
                        ++count;
                    } else {
                        const uint64_t stride = kOffset - previousOffset;
                        if (previousLen != 0U && kOffset == previousOffset + previousLen * tokenStride_) {
                            // Adjacent blocks share the token stride and need only one copy.
                            previousLen += len;
                            records_.SetValue(word - 1U, static_cast<uint64_t>(previousLen) << 1U);
                        } else if (previousLen == blockY_ && len == blockY_ &&
                            stride <= 0xFFFFFFFFULL && stride % 16U == 0U) {
                            records_.SetValue(word - 1U,
                                (stride << 32U) | (static_cast<uint64_t>(len) << 1U) | 1U);
                            previousLen = 0;
                        } else {
                            records_.SetValue(word, kOffset);
                            records_.SetValue(word + 1U, static_cast<uint64_t>(len) << 1U);
                            ++count;
                            previousLen = len;
                            previousOffset = kOffset;
                        }
                    }
                    copied += len;
                }
                dealtLenAccum += rows;
                inner = 0;
            }
            records_.SetValue(0, count);
            const uint32_t copyWords = (8U + count * recordWords + 3U) / 4U * 4U;
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventId);
            AscendC::DataCopy(groupInfo, records_, copyWords);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventId);
            recordsBufIdx_ ^= recordsBufMask_;
        }
    }

private:
    AscendC::LocalTensor<uint64_t> recordsBuf_[2];
    uint32_t recordsBufIdx_ = 0;
    uint32_t recordsBufMask_;
    uint32_t tileSize_;
    uint32_t l1TileN_;
    uint32_t blockY_;
    uint32_t pageSize_;
    uint32_t groupWords_;
    uint64_t tokenStride_;
    uint64_t kPageStride_;
    uint64_t vPageStride_;
};

} // namespace NpuArch::Epilogue::Block

#endif // GBSA_BLOCK_EPILOGUE_PREPARE_KV_OFFSETS_ARCH35_HPP
