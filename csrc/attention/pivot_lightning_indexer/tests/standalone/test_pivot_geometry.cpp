/* Copyright (c) 2026. Licensed under CANN Open Software License Agreement v2.0. */
#include <cassert>
#include <iostream>
#include <vector>
#include "../../op_kernel/pivot_lightning_indexer_pivot_geometry.h"

using namespace LICommon;

int main()
{
    uint64_t cases = 0;
    for (uint32_t rows = 0; rows <= 160; ++rows) {
        for (uint32_t prefix : {0U, 2047U, 4090U, 4094U, 4095U, 4096U, 65536U}) {
            uint32_t keys = prefix + rows;
            PivotGeometry geometry(rows, keys);
            std::vector<PivotGroup> expected;
            for (uint32_t first = 0; first < rows; first += 4) {
                uint32_t count = rows - first < 4 ? rows - first : 4;
                if (prefix + first + 1 < 4096) {
                    for (uint32_t r = 0; r < count; ++r) {
                        expected.push_back({37 + first + r, 1, prefix + first + r + 1});
                    }
                } else {
                    expected.push_back({37 + first, count, prefix + first + 1});
                }
            }
            assert(geometry.Groups() == expected.size());
            for (uint32_t g = 0; g < expected.size(); ++g) {
                auto actual = geometry.Group(37, keys, g);
                assert(actual.sourceRow == expected[g].sourceRow);
                assert(actual.copies == expected[g].copies);
                assert(actual.visibleKeys == expected[g].visibleKeys);
                assert(actual.visibleKeys + actual.copies - 1 <= keys);
            }
            ++cases;
        }
        PivotGeometry padding(rows, 0);
        uint32_t copied = 0;
        for (uint32_t g = 0; g < padding.Groups(); ++g) {
            auto group = padding.Group(19, 0, g);
            assert(group.sourceRow == 19 + copied && group.visibleKeys == 0);
            copied += group.copies;
        }
        assert(copied == rows);
    }

    // Ragged requests, empty batches and padding share a compact row space.
    for (uint32_t cores : {2U, 16U, 48U}) {
        uint32_t base = 0;
        std::vector<uint32_t> owners;
        for (uint32_t rows : {6U, 0U, 4U, 1U, 8192U, 3U}) {
            PivotGeometry geometry(rows, 16384);
            uint32_t count = geometry.Groups();
            owners.resize(base + count);
            for (uint32_t id = 0; id < cores; ++id) {
                for (uint32_t g = (id + cores - base % cores) % cores; g < count; g += cores) {
                    ++owners[base + g];
                    assert((base + g) % cores == id);
                }
            }
            base += count;
        }
        for (auto count : owners) {
            assert(count == 1);
        }
    }

    for (uint32_t rows : {1U, 3U, 4U, 8196U}) {
        PivotWorkspace reserved(rows, 17);
        for (uint32_t groups = 0; groups <= rows; ++groups) {
            PivotWorkspace actual(groups, 17);
            assert(actual.weights >= uint64_t(groups) * 8192);
            assert(actual.indices >= actual.weights + uint64_t(groups) * 128);
            assert(actual.groups >= actual.indices + uint64_t(groups) * 8192);
            assert(actual.lengths >= actual.groups + uint64_t(groups) * sizeof(PivotGroup));
            assert(actual.native >= actual.lengths + 17 * sizeof(uint32_t));
            assert(actual.native <= reserved.native);
            assert(actual.native % 512 == 0);
        }
    }
    std::cout << "PASS: " << cases << " geometries, padding, ownership and workspace bounds\n";
}
