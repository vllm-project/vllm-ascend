// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

// The consumer owns the signed-int16 slot/control ABI. Working-set sizes are
// derived from its input dimensions and the device UB, not a model tuple.
struct FsaLruGeometry {
    // FP32 Compare/CompareScalar calCount must cover complete 256-byte repeats.
    static constexpr uint32_t kCompareElements = 256 / sizeof(float);
    __host__ __aicore__ static constexpr uint32_t AlignWords(uint32_t n) { return (n + 7) / 8 * 8; }
    __host__ __aicore__ static constexpr uint32_t Width(uint32_t n) {
        uint32_t w = kCompareElements;
        while (w < n) w *= 2;
        return w;
    }
    uint32_t topk, capacity, topkWidth, tile, wireWords, payloadWords, workWords, lookupMaskWords;
    __host__ __aicore__ static constexpr uint32_t MaskWords(uint32_t n) { return AlignWords((n + 31) / 32); }
    __host__ __aicore__ static constexpr uint32_t ScratchWords(uint32_t k, uint32_t t) {
        return 5 * t + 16 > 6 * k ? 5 * t + 16 : 6 * k;
    }
    __host__ __aicore__ static constexpr uint32_t Words(uint32_t k, uint32_t t) {
        const uint32_t lookup = 4 * k + 3 * t + 8 * MaskWords(t) + ScratchWords(k,t);
        const uint32_t partition = 6 * k + 8 * t + 16 + 8 * MaskWords(t);
        const uint32_t plan = 14 * k + 16;
        const uint32_t maximum = lookup > partition ? lookup : partition;
        return 64 + 4 * MaskWords(k) + (maximum > plan ? maximum : plan);
    }
    // Reconstruct the layout already resolved and validated by the host entry.
    __host__ __aicore__ FsaLruGeometry(uint32_t k, uint32_t c, uint32_t paddedK,
                                     uint32_t tileWidth, uint32_t words, uint32_t masks)
        : topk(k), capacity(c), topkWidth(paddedK), tile(tileWidth),
          wireWords(AlignWords((k + 17) / 2)), payloadWords(wireWords + 8),
          workWords(words), lookupMaskWords(masks) {}
    __host__ __aicore__ FsaLruGeometry(uint32_t k, uint32_t c, uint32_t ubBytes)
        : topk(k), capacity(c), topkWidth(Width(k)), tile(kCompareElements),
          wireWords(AlignWords((k + 17) / 2)), payloadWords(wireWords + 8), workWords(0), lookupMaskWords(0) {
        const uint32_t target = Width(c);
        while (tile < target && Words(topkWidth, tile * 2) * 4 <= ubBytes &&
               uint64_t(topkWidth + 1) * tile * 2 <= (1U << 24)) tile *= 2;
        workWords = Words(topkWidth, tile);
        // Retain only hit/invalidation bits, including DMA-aligned tile padding.
        // This preserves lookup results across both phases without staging in GM.
        const uint32_t maskWords = MaskWords(tile) * ((capacity - 1 + tile - 1) / tile);
        if ((workWords + 2 * maskWords) * sizeof(int32_t) <= ubBytes) {
            lookupMaskWords = maskWords;
            workWords += 2 * maskWords;
        }
    }
};
