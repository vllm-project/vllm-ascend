# SPDX-License-Identifier: Apache-2.0
"""A3 MTP5 QLI with exact early pruning of each tile's TopK candidates."""

from qli_tilelang import _SORT_TILE
from qli_tilelang_v17 import build_tilelang_call as build_v17_call

_PRUNED_TOPK = """
{
    constexpr uint32_t k = 2048;
    const int32_t row_offset = score_mask_values.GetValue(2);
    const int32_t unordered_slot = 3 + row_offset / k;
    auto prior = row_topk[2 * row_offset];
    auto mask = merged[2 * k].ReinterpretCast<uint32_t>();
    mask.SetSize(k / 32);
    auto mask_bytes = mask.ReinterpretCast<uint8_t>();
    auto values = tile_pairs.ReinterpretCast<uint32_t>();
    auto compact_values = merged.ReinterpretCast<uint32_t>();
    auto compact_ids = merged[k].ReinterpretCast<uint32_t>();
    const AscendC::GatherMaskParams gather_config{1, 1, 8, 0};
    bool unordered = score_mask_values.GetValue(unordered_slot) != 0;
    uint64_t kept = k;

    if (!unordered) {
        // A NaN disables pruning for all later tiles in this row fragment.
        uint64_t unordered_count = 0;
        AscendC::Compare(mask_bytes, tile_pairs, tile_pairs, AscendC::CMPMODE::NE, k);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::GatherMask(compact_values, values, mask, true, k, gather_config, unordered_count);
        AscendC::PipeBarrier<PIPE_ALL>();
        unordered = unordered_count != 0;
        if (unordered) {
            score_mask_values.SetValue(unordered_slot, 1);
        }
    }

    if (!unordered) {
        const uint32_t rank_lane = 2 * (k - 1);
        const uint32_t rank_bits = prior.ReinterpretCast<uint32_t>().GetValue(rank_lane);
        if ((rank_bits & 0x7f800000U) != 0x7f800000U) {
            // A finite Kth score certifies k prior candidates. Incoming
            // scores equal to this threshold retain priority over prior ties.
            const float cutoff = prior.GetValue(rank_lane);
            AscendC::CompareScalar(mask_bytes, tile_pairs, cutoff, AscendC::CMPMODE::GE, k);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::GatherMask(compact_values, values, mask, true, k, gather_config, kept);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    if (kept > 512) {
        $original_sort
        tl::ascend::MergeSort(merged, tile_pairs, prior, k, k);
        AscendC::Adds(prior.ReinterpretCast<int32_t>(), merged.ReinterpretCast<int32_t>(), 0, 2 * k);
        AscendC::PipeBarrier<PIPE_V>();
    } else if (kept != 0) {
        // GatherMask preserves physical lane order. Sort32 and MrgSort
        // prioritize the earlier lane or run on equal scores.
        uint64_t kept_ids = 0;
        AscendC::GatherMask(compact_ids, values[k], mask, true, k, gather_config, kept_ids);
        AscendC::PipeBarrier<PIPE_V>();

        const uint32_t count = static_cast<uint32_t>(kept);
        const uint32_t capacity = count <= 32 ? 32 : (count <= 128 ? 128 : 512);
        auto value_bits = compact_values.ReinterpretCast<int32_t>();
        auto index_bits = compact_ids.ReinterpretCast<int32_t>();
        if (count < capacity) {
            const uint32_t repeat_base = count & ~63U;
            const uint32_t partial = count & 63U;
            const uint32_t full_base = repeat_base + (partial == 0 ? 0 : 64);
            if (partial != 0) {
                const uint64_t valid_span = capacity == 32 ? 0xffffffffULL : ~0ULL;
                uint64_t pad[2] = {(valid_span << partial) & valid_span, 0};
                AscendC::Duplicate(value_bits[repeat_base], static_cast<int32_t>(0xff800000U),
                                   pad, 1, 1, 8);
                AscendC::Duplicate(index_bits[repeat_base], -1, pad, 1, 1, 8);
            }
            if (full_base < capacity) {
                AscendC::Duplicate(value_bits[full_base], static_cast<int32_t>(0xff800000U),
                                   capacity - full_base);
                AscendC::Duplicate(index_bits[full_base], -1, capacity - full_base);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::Sort32(tmp_pairs, merged, compact_ids, capacity / 32);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t width = 32; width < capacity; width *= 4) {
            auto source = width == 32 ? tmp_pairs : tile_pairs;
            auto dest = width == 32 ? tile_pairs : tmp_pairs;
            AscendC::MrgSort4Info config;
            for (int run = 0; run < 4; ++run) {
                config.elementLengths[run] = width;
            }
            config.ifExhaustedSuspension = false;
            config.validBit = 15;
            config.repeatTimes = capacity / (4 * width);
            AscendC::MrgSortSrcList<float> runs(
                source, source[2 * width], source[4 * width], source[6 * width]);
            AscendC::MrgSort<float>(dest, runs, config);
            AscendC::PipeBarrier<PIPE_V>();
        }
        auto ordered = capacity == 128 ? tile_pairs : tmp_pairs;
        tl::ascend::MergeSort(merged, ordered, prior, count, k);
        AscendC::Adds(prior.ReinterpretCast<int32_t>(), merged.ReinterpretCast<int32_t>(), 0, 2 * k);
        AscendC::PipeBarrier<PIPE_V>();
    }
}
""".replace("$original_sort", _SORT_TILE)


def build_tilelang_call(
    query,
    key,
    weights,
    query_scale,
    key_scale,
    actual_seq_lengths_query,
    actual_seq_lengths_key,
    block_table,
    *,
    sparse_count=2048,
    sparse_mode=3,
    core_count=None,
    verified_lengths=None,
):
    """Use V17's MTP5 schedule with an ascending-lane threshold compaction."""
    return build_v17_call(
        query,
        key,
        weights,
        query_scale,
        key_scale,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
        core_count=core_count,
        verified_lengths=verified_lengths,
        topk_source=_PRUNED_TOPK,
    )
