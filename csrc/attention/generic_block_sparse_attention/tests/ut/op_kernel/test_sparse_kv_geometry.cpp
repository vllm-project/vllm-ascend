/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
// Standalone CPU test: g++ -std=c++17 -O2 test_sparse_kv_geometry.cpp -o test && ./test
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#define __aicore__
#include "../../../op_kernel/arch35/gbsa_layout_utils.hpp"
#include "../../../op_kernel/gbsa_sparse_kv_geometry.hpp"
#undef __aicore__

using namespace GbsaKernelArch35;

struct Indices {
    std::vector<int32_t> values;
    int32_t GetValue(uint64_t i) const
    {
        return values.at(i);
    }
};

// Independent token oracle for the production length helper on valid inputs.
// Actual L1 split/copy behavior is exercised by test_blocky_device.py.
void CheckRange(Indices &indices, uint32_t count, uint32_t y, uint32_t visible)
{
    uint32_t validTokens = 0;
    CalcSparseKvSize<false>(indices, count, 0, count, y, visible, validTokens);
    std::vector<uint64_t> expected;
    for (uint32_t i = 0; i < count; ++i) {
        const uint64_t start = static_cast<uint64_t>(indices.GetValue(i)) * y;
        for (uint32_t t = 0; t < y && start + t < visible; ++t)
            expected.push_back(start + t);
    }
    assert(validTokens == expected.size());
    const uint32_t tileCount = (validTokens + 127) / 128;
    uint32_t tiledTokens = 0;
    for (uint32_t tile = 0; tile < tileCount; ++tile) {
        const uint32_t length = std::min(128U, validTokens - tile * 128);
        assert(length > 0 && length <= 128);
        tiledTokens += length;
    }
    assert(tiledTokens == expected.size());

}

void CheckResidualRanges(Indices &indices, uint32_t explicitCount, uint32_t y, uint32_t visible)
{
    const uint32_t residualSize = visible % y;
    const uint32_t virtualCount = explicitCount + static_cast<uint32_t>(residualSize != 0);
    for (uint32_t begin = 0; begin <= virtualCount; ++begin) {
        for (uint32_t end = begin; end <= virtualCount; ++end) {
            uint32_t validTokens = 0;
            CalcSparseKvSize<true>(indices, explicitCount, begin, end, y, visible, validTokens);
            const uint32_t explicitBegin = std::min(begin, explicitCount);
            const uint32_t explicitEnd = std::min(end, explicitCount);
            uint32_t expected = (explicitEnd - explicitBegin) * y;
            if (residualSize != 0 && begin <= explicitCount && end > explicitCount)
                expected += residualSize;
            assert(validTokens == expected);

            Indices local{{indices.values.begin() + explicitBegin, indices.values.begin() + explicitEnd}};
            const uint32_t localExplicitCount = explicitEnd - explicitBegin;
            for (uint32_t localIdx = 0; localIdx < end - begin; ++localIdx) {
                const int32_t expectedBlockId = localIdx == localExplicitCount ?
                    static_cast<int32_t>(visible / y) : indices.GetValue(explicitBegin + localIdx);
                assert(GetSparseBlockId<true>(local, localIdx, localExplicitCount, visible / y) == expectedBlockId);
            }
        }
    }
    const uint32_t residualId = visible / y;
    for (uint32_t i = 0; i < explicitCount; ++i)
        assert(GetSparseBlockId<true>(indices, i, explicitCount, residualId) == indices.GetValue(i));
    if (residualSize != 0)
        assert(GetSparseBlockId<true>(indices, explicitCount, explicitCount, residualId) ==
               static_cast<int32_t>(residualId));
}

void ResolveOracle(Indices &indices, uint32_t bandOffset, uint32_t selectNum, uint32_t y, uint32_t visible,
                   uint32_t &selectIdx, uint32_t &yOff, uint32_t &logicalId)
{
    GbsaSparseGeometry::ResolveSparseBandOffset<false>(indices, bandOffset, selectNum, y, visible, selectNum, 0U,
                                                       selectIdx, yOff, logicalId);
}

void CheckLocateMatchesResolve(Indices &indices, uint32_t selectNum, uint32_t y, uint32_t visible, bool expectFastDivY)
{
    uint32_t validTokens = 0;
    bool hasClipped = false;
    uint32_t fullPrefixSlots = 0;
    GbsaSparseGeometry::CalcSparseKvSize(indices, selectNum, y, visible, validTokens, hasClipped, fullPrefixSlots);
    if (expectFastDivY) {
        assert(!hasClipped);
        assert(validTokens == selectNum * y);
        assert(fullPrefixSlots == selectNum);
    } else {
        assert(hasClipped);
    }

    for (uint32_t bandOff = 0; bandOff < validTokens; ++bandOff) {
        uint32_t locSel = 0, locOff = 0, locId = 0;
        uint32_t resSel = 0, resOff = 0, resId = 0;
        GbsaSparseGeometry::LocateSparseBand<false>(indices, bandOff, selectNum, y, visible, selectNum, 0U, hasClipped,
                                                    fullPrefixSlots, locSel, locOff, locId);
        ResolveOracle(indices, bandOff, selectNum, y, visible, resSel, resOff, resId);
        assert(locSel == resSel && locOff == resOff && locId == resId);

        if (expectFastDivY || bandOff < fullPrefixSlots * y) {
            assert(locSel == bandOff / y);
            assert(locOff == bandOff % y);
        }
    }
}

void CheckLocateSparseBandCases()
{
    // All-full ordered: Locate must use /Y (and match Resolve).
    {
        Indices full{{0, 1, 2, 3}};
        const uint32_t y = 8;
        CheckLocateMatchesResolve(full, 4, y, 1000, true);
    }
    // Shuffled all-full: disorder must not force Resolve; still /Y.
    {
        Indices shuffled{{7, 2, 5, 0, 3}};
        const uint32_t y = 16;
        CheckLocateMatchesResolve(shuffled, 5, y, 10000, true);
    }
    // Causal clip on last selected block: hasClipped, prefix /Y then Resolve.
    {
        Indices clipped{{0, 1, 2}};
        const uint32_t y = 8;
        const uint32_t visible = 2 * y + 3; // last slot contributes 3
        uint32_t validTokens = 0;
        bool hasClipped = false;
        uint32_t fullPrefixSlots = 0;
        GbsaSparseGeometry::CalcSparseKvSize(clipped, 3, y, visible, validTokens, hasClipped, fullPrefixSlots);
        assert(hasClipped && fullPrefixSlots == 2 && validTokens == 2 * y + 3);
        CheckLocateMatchesResolve(clipped, 3, y, visible, false);
        uint32_t sel = 0, off = 0, id = 0;
        GbsaSparseGeometry::LocateSparseBand<false>(clipped, y + 3, 3, y, visible, 3, 0U, hasClipped, fullPrefixSlots,
                                                    sel, off, id);
        assert(sel == 1 && off == 3 && id == 1); // still in full prefix → /Y
    }
    // Mid-list clip: slot1 short → later band offsets diverge from naive /Y.
    {
        Indices midClip{{0, 10, 1}}; // block 10 starts at 10*y; visible cuts it short
        const uint32_t y = 8;
        const uint32_t visible = 10 * y + 2;
        CheckLocateMatchesResolve(midClip, 3, y, visible, false);
        uint32_t validTokens = 0;
        bool hasClipped = false;
        uint32_t fullPrefixSlots = 0;
        GbsaSparseGeometry::CalcSparseKvSize(midClip, 3, y, visible, validTokens, hasClipped, fullPrefixSlots);
        assert(hasClipped && fullPrefixSlots == 1 && validTokens == y + 2 + y);
        const uint32_t bandOff = y + 2; // start of third slot after short middle
        uint32_t sel = 0, off = 0, id = 0;
        GbsaSparseGeometry::LocateSparseBand<false>(midClip, bandOff, 3, y, visible, 3, 0U, hasClipped, fullPrefixSlots,
                                                    sel, off, id);
        assert(sel == 2 && off == 0 && id == 1);
        assert(bandOff / y != sel); // naive /Y would be wrong
    }
}

int main()
{
    uint32_t cases = 0;
    for (uint32_t y = 1; y <= 256; ++y) {
        Indices indices{{0, 1, 3, 4, 7, 8, 11, 12, -1, -1}};
        for (uint32_t count = 0; count <= 8; ++count) {
            for (uint32_t visible = 0; visible <= 14 * y; ++visible) {
                if (count != 0 && static_cast<uint64_t>(indices.GetValue(count - 1)) * y >= visible)
                    continue;
                CheckRange(indices, count, y, visible);
                ++cases;
            }
        }
    }
    // FD uses rebased subranges, including empty and causal-invisible ranges.
    const std::vector<int32_t> selected{0, 2, 3, 6, 8, 11, 13, 15};
    uint32_t rangeCases = 0;
    for (uint32_t begin = 0; begin <= selected.size(); ++begin) {
        for (uint32_t end = begin; end <= selected.size(); ++end) {
            Indices local{{selected.begin() + begin, selected.begin() + end}};
            for (uint32_t visible = 0; visible <= 16 * 128; ++visible) {
                if (end != begin && static_cast<uint64_t>(local.GetValue(end - begin - 1)) * 128 >= visible)
                    continue;
                CheckRange(local, end - begin, 128, visible);
                ++rangeCases;
            }
        }
    }
    Indices full;
    for (int i = 0; i < 256; ++i)
        full.values.push_back(i * 2);
    uint32_t tokens = 0;
    CalcSparseKvSize<false>(full, 256, 0, 256, 256, 200000, tokens);
    assert(tokens == 65536 && (tokens + 127) / 128 == 512);
    uint32_t residualCases = 0;
    for (uint32_t y = 1; y <= 256; ++y) {
        for (uint32_t explicitCount = 0; explicitCount <= 8; ++explicitCount) {
            Indices indices;
            for (uint32_t i = 0; i < explicitCount; ++i)
                indices.values.push_back(static_cast<int32_t>(i * 2));
            // Start below one full block to cover the valid implicit block id 0
            // with explicitCount == 0, as well as exact block boundaries.
            for (uint32_t visible = 1; visible <= 12 * y; ++visible) {
                CheckResidualRanges(indices, explicitCount, y, visible);
                ++residualCases;
            }
        }
    }
    for (uint32_t y = 1; y <= 128; ++y) {
        for (uint32_t n : {128U, 256U}) {
            const uint32_t maxBlocks = (n + 2U * y - 2U) / y;
            const uint32_t paBlocks = std::min(n, 2U * maxBlocks);
            assert(CalcKvOffsetGroupWords<false>(n, y) == (8U + maxBlocks * 4U + 7U) / 8U * 8U);
            assert(CalcKvOffsetGroupWords<true>(n, y) == (8U + paBlocks * 3U + 7U) / 8U * 8U);
        }
    }
    CheckLocateSparseBandCases();
    std::cout << "PASS: offset group capacity matches host formula for Y=1..128\n";
    std::cout << "PASS: " << cases << " valid length cases (all Y=1..256), tile capacity\n";
    std::cout << "PASS: " << rangeCases << " rebased FD subrange cases\n";
    std::cout << "PASS: " << residualCases << " virtual residual range cases\n";
    std::cout << "PASS: LocateSparseBand fast /Y, clipped Resolve, shuffled full /Y\n";
}
