/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "kernel_operator.h"
#include "lib/sort/sort.h"
#include "qsa_union_inputs_tiling_data.h"

using namespace AscendC;

namespace {
constexpr uint32_t UNION_SIZE = 8;
constexpr uint32_t BLOCK_Y = 4;
constexpr int32_t INVALID_KEY = 1 << 30;
constexpr uint32_t FILL_ELEMENTS = 512;
constexpr uint32_t OUTPUT_TILE = 512;
#if __CCE_AICORE__ == 310
constexpr uint32_t B32_VECTOR_LANES =
    AscendC::VECTOR_REG_WIDTH / sizeof(int32_t);

__aicore__ inline void UnpackPackedMasks(
    const LocalTensor<int32_t>& output,
    const LocalTensor<int32_t>& packed_masks, uint32_t packed_count)
{
    __local_mem__ int32_t* output_addr =
        (__local_mem__ int32_t*)output.GetPhyAddr();
    __local_mem__ int32_t* packed_addr =
        (__local_mem__ int32_t*)packed_masks.GetPhyAddr();
    const uint32_t output_count = packed_count * BLOCK_Y;
    uint32_t remaining = output_count;
    const uint16_t repeats = static_cast<uint16_t>(
        (output_count + B32_VECTOR_LANES - 1) / B32_VECTOR_LANES);

    // Produce the external [rank, 4] int32 layout directly in vector
    // registers.  Every 64-lane iteration gathers sixteen packed words four
    // times, shifts by [0, 8, 16, 24], masks the low byte, and stores the
    // expanded values.  This avoids the per-rank GetValue + four SetValue
    // scalar loop while preserving the exact output ABI.
    __VEC_SCOPE__ {
        AscendC::MicroAPI::RegTensor<int32_t> lanes;
        AscendC::MicroAPI::RegTensor<int32_t> source_indices;
        AscendC::MicroAPI::RegTensor<int32_t> byte_lanes;
        AscendC::MicroAPI::RegTensor<int32_t> shifts;
        AscendC::MicroAPI::RegTensor<int32_t> packed;
        AscendC::MicroAPI::RegTensor<int32_t> unpacked;
        AscendC::MicroAPI::RegTensor<int32_t> constant;
        AscendC::MicroAPI::MaskReg mask;
        AscendC::MicroAPI::Arange(lanes, 0);
        for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
            mask = AscendC::MicroAPI::UpdateMask<int32_t>(remaining);
            AscendC::MicroAPI::Duplicate(constant, 2, mask);
            AscendC::MicroAPI::ShiftRight(source_indices, lanes, constant,
                                          mask);
            AscendC::MicroAPI::Adds(
                source_indices, source_indices,
                static_cast<int32_t>(repeat * (B32_VECTOR_LANES / BLOCK_Y)),
                mask);
            AscendC::MicroAPI::Duplicate(constant,
                                         static_cast<int32_t>(BLOCK_Y - 1),
                                         mask);
            AscendC::MicroAPI::And(byte_lanes, lanes, constant, mask);
            AscendC::MicroAPI::Muls(shifts, byte_lanes, 8, mask);
            AscendC::MicroAPI::DataCopyGather(
                packed, packed_addr,
                (AscendC::MicroAPI::RegTensor<uint32_t>&)source_indices,
                mask);
            AscendC::MicroAPI::ShiftRight(unpacked, packed, shifts, mask);
            AscendC::MicroAPI::Duplicate(constant, 0xff, mask);
            AscendC::MicroAPI::And(unpacked, unpacked, constant, mask);
            AscendC::MicroAPI::DataCopy(
                output_addr + repeat * B32_VECTOR_LANES, unpacked, mask);
        }
    }
}

#else
__aicore__ inline void UnpackPackedMasks(
    const LocalTensor<int32_t>& output,
    const LocalTensor<int32_t>& packed_masks, uint32_t packed_count,
    const LocalTensor<int32_t>& offsets)
{
    constexpr uint32_t PACKED_TILE = FILL_ELEMENTS / BLOCK_Y;
    for (uint32_t base = 0; base < packed_count; base += PACKED_TILE) {
        const uint32_t count = packed_count - base < PACKED_TILE
            ? packed_count - base : PACKED_TILE;
        const uint32_t elements = count * BLOCK_Y;
        LocalTensor<int32_t> tile = output[base * BLOCK_Y];
        ArithProgression(offsets, 0, 1, elements);
        PipeBarrier<PIPE_ALL>();
        ShiftRight(offsets.ReinterpretCast<uint32_t>(),
                   offsets.ReinterpretCast<uint32_t>(), 2U, elements);
        PipeBarrier<PIPE_ALL>();
        Muls(offsets, offsets, static_cast<int32_t>(sizeof(int32_t)), elements);
        PipeBarrier<PIPE_ALL>();
        Gather(tile.ReinterpretCast<float>(),
               packed_masks[base].ReinterpretCast<float>(),
               offsets.ReinterpretCast<uint32_t>(), 0U, elements);
        PipeBarrier<PIPE_ALL>();
        const uint8_t repeats = (elements + 63U) / 64U;
        for (uint32_t byte = 1; byte < BLOCK_Y; ++byte) {
            uint64_t mask[2] = {0x1111111111111111ULL << byte, 0};
            ShiftRight(tile.ReinterpretCast<uint32_t>(),
                       tile.ReinterpretCast<uint32_t>(), byte * 8U,
                       mask, repeats, {1, 1, 8, 8});
            PipeBarrier<PIPE_ALL>();
        }
        ShiftLeft(tile.ReinterpretCast<uint32_t>(),
                  tile.ReinterpretCast<uint32_t>(), 24U, elements);
        PipeBarrier<PIPE_ALL>();
        ShiftRight(tile.ReinterpretCast<uint32_t>(),
                   tile.ReinterpretCast<uint32_t>(), 24U, elements);
        PipeBarrier<PIPE_ALL>();
    }
}
#endif

class QsaUnionInputsKernel {
public:
    __aicore__ inline explicit QsaUnionInputsKernel(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(
        GM_ADDR compact_indices, GM_ADDR visible_lengths, GM_ADDR group_row_map,
        GM_ADDR union_idx, GM_ADDR union_count, GM_ADDR exact_mask,
        GM_ADDR workspace, const QsaUnionInputsTilingData* tiling)
    {
        rows_ = tiling->rows;
        groups_ = tiling->groups;
        input_groups_ = tiling->input_groups;
        topk_ = tiling->topk;
        width_ = tiling->width;
        compact_row_stride_ = tiling->compact_row_stride;
        sort_tmp_bytes_ = tiling->sort_tmp_bytes;
        groups_per_core_ = tiling->groups_per_core;

        compact_gm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(compact_indices),
            rows_ * compact_row_stride_);
        visible_gm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(visible_lengths), rows_);
        row_map_gm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(group_row_map),
            groups_ * UNION_SIZE);
        union_idx_gm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(union_idx), groups_ * width_);
        union_count_gm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(union_count), groups_);
        exact_mask_gm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(exact_mask),
            groups_ * width_ * BLOCK_Y);
#if __CCE_AICORE__ != 220
        GM_ADDR user_workspace = GetUserWorkspace(workspace);
        count_workspace_gm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(user_workspace), groups_ * 8);
#endif

#if __CCE_AICORE__ == 220
        pipe_->InitBuffer(compare_buf_, width_ * sizeof(float) * 2);
#endif
        pipe_->InitBuffer(keys_buf_, width_ * sizeof(int32_t));
        pipe_->InitBuffer(sorted_buf_, width_ * sizeof(int32_t));
        pipe_->InitBuffer(sort_tmp_buf_, sort_tmp_bytes_);
        pipe_->InitBuffer(compact_buf_, width_ * sizeof(int32_t) * 2);
        pipe_->InitBuffer(row_buf_, AlignUp(input_groups_ * sizeof(int32_t), 32));
        const uint32_t prepare_vector_bytes =
            AlignUp(input_groups_ * sizeof(int32_t), 32);
        const uint32_t prepare_mask_bytes =
            AlignUp((input_groups_ + 7) / 8, 32);
        pipe_->InitBuffer(prepare_buf_, prepare_vector_bytes * 4 +
                                           prepare_mask_bytes * 2);
        pipe_->InitBuffer(output_tile_buf_,
                          OUTPUT_TILE * (1 + BLOCK_Y) * sizeof(int32_t));
        pipe_->InitBuffer(meta_buf_, 32 * sizeof(int32_t));
        pipe_->InitBuffer(fill_buf_, FILL_ELEMENTS * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = GetBlockIdx();
        const uint32_t launched_cores = GetBlockNum();
        for (uint32_t group = core; group < groups_; group += launched_cores) {
            PublishCount(group, ProcessGroup(group));
        }
#if __CCE_AICORE__ != 220
        SyncAll();
        GatherCounts(core, launched_cores);
#endif
    }

private:
    __aicore__ inline void CompareKeys(const LocalTensor<uint8_t>& mask,
        const LocalTensor<int32_t>& left, const LocalTensor<int32_t>& right,
        CMPMODE mode, uint32_t count)
    {
#if __CCE_AICORE__ == 220
        LocalTensor<float> a = compare_buf_.Get<float>();
        LocalTensor<float> b = a[width_];
        Cast(a, left, RoundMode::CAST_NONE, count);
        Cast(b, right, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Compare(mask, a, b, mode, AlignUp(count, 64));
        PipeBarrier<PIPE_V>();
#else
        Compare(mask, left.ReinterpretCast<float>(), right.ReinterpretCast<float>(), mode, count);
#endif
    }

    __aicore__ inline void CompareKeyScalar(const LocalTensor<uint8_t>& mask,
        const LocalTensor<int32_t>& left, int32_t right,
        CMPMODE mode, uint32_t count)
    {
#if __CCE_AICORE__ == 220
        LocalTensor<float> a = compare_buf_.Get<float>();
        Cast(a, left, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        CompareScalar(mask, a, static_cast<float>(right), mode, AlignUp(count, 64));
        PipeBarrier<PIPE_V>();
#else
        CompareScalar(mask, left, right, mode, count);
#endif
    }
    __aicore__ static inline uint32_t AlignUp(uint32_t value, uint32_t alignment)
    {
        return (value + alignment - 1) / alignment * alignment;
    }

    __aicore__ inline void PublishCount(uint32_t group, uint32_t rank)
    {
        LocalTensor<int32_t> slot = fill_buf_.Get<int32_t>();
        slot.SetValue(0, static_cast<int32_t>(rank));
        event_t ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(ready);
        WaitFlag<HardEvent::S_MTE3>(ready);
#if __CCE_AICORE__ == 220
        // An exact-width DMA store avoids scalar GM cache-line writeback races.
        DataCopyPad(union_count_gm_[group], slot,
                    DataCopyExtParams(1, sizeof(int32_t), 0, 0, 0));
#else
        DataCopy(count_workspace_gm_[group * 8], slot, 8);
#endif
        event_t complete = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(complete);
        WaitFlag<HardEvent::MTE3_S>(complete);
    }

    __aicore__ inline void GatherCounts(uint32_t core,
                                        uint32_t launched_cores)
    {
        // Each producer publishes into an independent 32-byte workspace slot
        // to avoid false sharing.  Compact those slots in parallel after the
        // global barrier instead of making core 0 serialize all groups.
        constexpr uint32_t COUNTS_PER_BLOCK = 8;
        LocalTensor<int32_t> compact = meta_buf_.Get<int32_t>();
        LocalTensor<int32_t> slot = compact[COUNTS_PER_BLOCK];
        const uint32_t first = core * COUNTS_PER_BLOCK;
        const uint32_t stride = launched_cores * COUNTS_PER_BLOCK;
        for (uint32_t base = first; base < groups_; base += stride) {
            const uint32_t current =
                Min(COUNTS_PER_BLOCK, groups_ - base);
            for (uint32_t index = 0; index < current; ++index) {
                DataCopy(slot,
                         count_workspace_gm_[(base + index) *
                                             COUNTS_PER_BLOCK],
                         COUNTS_PER_BLOCK);
                event_t ready = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
                SetFlag<HardEvent::MTE2_S>(ready);
                WaitFlag<HardEvent::MTE2_S>(ready);
                compact.SetValue(index, slot.GetValue(0));
            }
            event_t compact_ready = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(compact_ready);
            WaitFlag<HardEvent::S_MTE3>(compact_ready);
            if (current == COUNTS_PER_BLOCK) {
                DataCopy(union_count_gm_[base], compact, COUNTS_PER_BLOCK);
            } else {
                // Keep tail counts on the DMA path without writing past output.
                DataCopyPad(union_count_gm_[base], compact,
                            DataCopyExtParams(1, current * sizeof(int32_t),
                                              0, 0, 0));
            }
            event_t complete = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(complete);
            WaitFlag<HardEvent::MTE3_S>(complete);
        }
    }

    template <typename T>
    __aicore__ static inline T Min(T left, T right)
    {
        return left < right ? left : right;
    }

    __aicore__ inline void FillOutput(GlobalTensor<int32_t>& output,
                                      uint32_t offset, uint32_t count,
                                      int32_t value)
    {
        LocalTensor<int32_t> fill = fill_buf_.Get<int32_t>();
        Duplicate(fill, value, FILL_ELEMENTS);
        event_t ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(ready);
        WaitFlag<HardEvent::V_MTE3>(ready);
        for (uint32_t done = 0; done < count; done += FILL_ELEMENTS) {
            const uint32_t current = Min(FILL_ELEMENTS, count - done);
            DataCopy(output[offset + done], fill, current);
        }
        event_t complete = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(complete);
        WaitFlag<HardEvent::MTE3_S>(complete);
    }

    __aicore__ inline void PrepareKeys(uint32_t group,
                                       LocalTensor<int32_t>& keys,
                                       LocalTensor<int32_t>& meta)
    {
        Duplicate(keys, INVALID_KEY, width_);
        DataCopy(meta, row_map_gm_[group * UNION_SIZE], UNION_SIZE);
        event_t ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(ready);
        WaitFlag<HardEvent::MTE2_S>(ready);
        event_t keys_ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(keys_ready);
        WaitFlag<HardEvent::V_S>(keys_ready);

        LocalTensor<int32_t> row_data = row_buf_.Get<int32_t>();
        LocalTensor<int32_t> scratch = prepare_buf_.Get<int32_t>();
        const uint32_t scratch_stride =
            AlignUp(input_groups_ * sizeof(int32_t), 32) / sizeof(int32_t);
        LocalTensor<int32_t> encoded = scratch;
        LocalTensor<int32_t> filtered = scratch[scratch_stride];
        LocalTensor<int32_t> invalid = scratch[scratch_stride * 2];
        LocalTensor<int32_t> sign = scratch[scratch_stride * 3];
        LocalTensor<uint8_t> mask_storage = prepare_buf_.Get<uint8_t>()[
            scratch_stride * sizeof(int32_t) * 4];
        const uint32_t mask_stride = AlignUp((input_groups_ + 7) / 8, 32);
        LocalTensor<uint8_t> negative_mask = mask_storage;
        LocalTensor<uint8_t> valid_mask = mask_storage[mask_stride];
        Duplicate(invalid, INVALID_KEY, input_groups_);
        PipeBarrier<PIPE_V>();
        for (uint32_t lane = 0; lane < UNION_SIZE; ++lane) {
            const int32_t row = meta.GetValue(lane);
            int32_t visible = 0;
            if (row >= 0 && static_cast<uint32_t>(row) < rows_) {
                visible = visible_gm_.GetValue(row);
                DataCopyPad(row_data, compact_gm_[row * compact_row_stride_],
                            DataCopyExtParams(1, input_groups_ * sizeof(int32_t),
                                              0, 0, 0),
                            DataCopyPadExtParams<int32_t>(false, 0, 0, 0));
                event_t row_ready = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(row_ready);
                WaitFlag<HardEvent::MTE2_V>(row_ready);
            }
            meta.SetValue(UNION_SIZE + lane, visible);
            const bool has_tail = visible > 0 && visible % BLOCK_Y != 0;
            const uint32_t capacity = topk_ - static_cast<uint32_t>(has_tail);
            if (row >= 0 && static_cast<uint32_t>(row) < rows_) {
                const int32_t visible_group = visible / BLOCK_Y;
                // Convert the scalar filter/compact walk into vector work:
                // mask negative indices, mask indices beyond visibility,
                // encode (block, lane), then compact with GatherMask.
                ShiftRight(sign.ReinterpretCast<uint32_t>(),
                           row_data.ReinterpretCast<uint32_t>(), 31U,
                           input_groups_);
                PipeBarrier<PIPE_V>();
                CompareKeyScalar(negative_mask, sign, 1, CMPMODE::EQ,
                              input_groups_);
                PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 220
                Select(filtered.ReinterpretCast<float>(), negative_mask,
                       invalid.ReinterpretCast<float>(), row_data.ReinterpretCast<float>(),
                       SELMODE::VSEL_TENSOR_TENSOR_MODE, input_groups_);
#else
                Select(filtered, negative_mask, invalid, row_data,
                       SELMODE::VSEL_TENSOR_TENSOR_MODE, input_groups_);
#endif
                PipeBarrier<PIPE_V>();
                CompareKeyScalar(valid_mask, filtered, visible_group,
                              CMPMODE::LT, input_groups_);
                PipeBarrier<PIPE_V>();
                Muls(encoded, filtered, static_cast<int32_t>(UNION_SIZE),
                     input_groups_);
                PipeBarrier<PIPE_V>();
                Adds(encoded, encoded, static_cast<int32_t>(lane),
                     input_groups_);
                PipeBarrier<PIPE_V>();

                GatherMaskParams compact_params;
                compact_params.repeatTimes = 1;
                compact_params.src0BlockStride = 1;
                compact_params.src0RepeatStride = 8;
                compact_params.src1RepeatStride = 8;
                uint64_t kept_count = 0;
                GatherMask(keys[lane * topk_].ReinterpretCast<uint32_t>(),
                           encoded.ReinterpretCast<uint32_t>(),
                           valid_mask.ReinterpretCast<uint32_t>(), true,
                           input_groups_, compact_params, kept_count);
                PipeBarrier<PIPE_V>();
                event_t compact_ready = static_cast<event_t>(
                    GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(compact_ready);
                WaitFlag<HardEvent::V_S>(compact_ready);
                if (has_tail) {
                    const uint32_t kept = Min(
                        capacity, static_cast<uint32_t>(kept_count));
                    keys.SetValue(lane * topk_ + kept,
                                  visible_group * static_cast<int32_t>(UNION_SIZE) +
                                      static_cast<int32_t>(lane));
                }
            }
        }
        event_t scalar_done = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(scalar_done);
        WaitFlag<HardEvent::S_V>(scalar_done);
    }

    __aicore__ inline uint32_t CompactSorted(uint32_t group,
                                              LocalTensor<int32_t>& sorted,
                                              LocalTensor<int32_t>& meta)
    {
        LocalTensor<int32_t> unique_keys = keys_buf_.Get<int32_t>();
        LocalTensor<int32_t> scratch0 = compact_buf_.Get<int32_t>();
        LocalTensor<int32_t> scratch1 = scratch0[width_];
        LocalTensor<uint8_t> compact_mask = prepare_buf_.Get<uint8_t>();

        // Build byte offsets [0, 0, 4, 8, ...] and gather the predecessor of
        // every sorted key.  The first predecessor is made different so the
        // first valid key is retained.  width_ is always a multiple of the
        // 64-lane B32 vector repeat.
        ArithProgression(scratch0, 0, 1, width_);
        PipeBarrier<PIPE_V>();
        Adds(scratch0, scratch0, -1, width_);
        PipeBarrier<PIPE_V>();
        Relu(scratch0, scratch0, width_);
        PipeBarrier<PIPE_V>();
        Muls(scratch0, scratch0, static_cast<int32_t>(sizeof(int32_t)),
             width_);
        PipeBarrier<PIPE_V>();
        Gather(unique_keys.ReinterpretCast<float>(),
               sorted.ReinterpretCast<float>(),
               scratch0.ReinterpretCast<uint32_t>(), 0U, width_);
        PipeBarrier<PIPE_V>();
        event_t predecessor_ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(predecessor_ready);
        WaitFlag<HardEvent::V_S>(predecessor_ready);
        unique_keys.SetValue(0, unique_keys.GetValue(0) + 1);
        event_t sentinel_ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(sentinel_ready);
        WaitFlag<HardEvent::S_V>(sentinel_ready);
        CompareKeys(compact_mask, sorted, unique_keys, CMPMODE::NE, width_);
        PipeBarrier<PIPE_V>();

        GatherMaskParams compact_params;
        compact_params.repeatTimes = 1;
        compact_params.src0BlockStride = 1;
        compact_params.src0RepeatStride = 8;
        compact_params.src1RepeatStride = 8;
        uint64_t exact_count_u64 = 0;
        GatherMask(unique_keys.ReinterpretCast<uint32_t>(),
                   sorted.ReinterpretCast<uint32_t>(),
                   compact_mask.ReinterpretCast<uint32_t>(), true, width_,
                   compact_params, exact_count_u64);
        PipeBarrier<PIPE_V>();
        event_t exact_ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(exact_ready);
        WaitFlag<HardEvent::V_S>(exact_ready);
        uint32_t exact_count = static_cast<uint32_t>(exact_count_u64);
        // The padded INVALID_KEY suffix is one sorted run and therefore
        // contributes exactly one gathered sentinel.  Drop it here.
        if (exact_count > 0 &&
            unique_keys.GetValue(exact_count - 1) >= INVALID_KEY) {
            --exact_count;
        }
        if (exact_count == 0) {
            return 0;
        }

        // Decode exact keys into block/lane vectors.  Keep blocks in sorted
        // and pack the four causal lane masks into the bytes of unique_keys:
        // [mask3 | mask2 | mask1 | mask0].
        ShiftRight(sorted.ReinterpretCast<uint32_t>(),
                   unique_keys.ReinterpretCast<uint32_t>(), 3U, exact_count);
        PipeBarrier<PIPE_V>();
        Muls(scratch0, sorted, static_cast<int32_t>(UNION_SIZE), exact_count);
        PipeBarrier<PIPE_V>();
        Sub(scratch0, unique_keys, scratch0, exact_count);
        PipeBarrier<PIPE_V>();
        Muls(scratch0, scratch0, static_cast<int32_t>(sizeof(int32_t)),
             exact_count);
        PipeBarrier<PIPE_V>();

        // Small UB lookup tables. Gather indices are byte offsets.
        event_t tables_ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(tables_ready);
        WaitFlag<HardEvent::V_S>(tables_ready);
        for (uint32_t lane = 0; lane < UNION_SIZE; ++lane) {
            meta.SetValue(16 + lane, 1 << lane);
        }
        meta.SetValue(24, 0x00000000);
        meta.SetValue(25, 0x00000001);
        meta.SetValue(26, 0x00000101);
        meta.SetValue(27, 0x00010101);
        meta.SetValue(28, 0x01010101);
        event_t tables_visible = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(tables_visible);
        WaitFlag<HardEvent::S_V>(tables_visible);

        Gather(scratch1.ReinterpretCast<float>(),
               meta[16].ReinterpretCast<float>(),
               scratch0.ReinterpretCast<uint32_t>(), 0U, exact_count);
        PipeBarrier<PIPE_V>();
        Gather(unique_keys.ReinterpretCast<float>(),
               meta[UNION_SIZE].ReinterpretCast<float>(),
               scratch0.ReinterpretCast<uint32_t>(), 0U, exact_count);
        PipeBarrier<PIPE_V>();
        Muls(scratch0, sorted, static_cast<int32_t>(BLOCK_Y), exact_count);
        PipeBarrier<PIPE_V>();
        Sub(unique_keys, unique_keys, scratch0, exact_count);
        PipeBarrier<PIPE_V>();
        Maxs(unique_keys, unique_keys, 0, exact_count);
        PipeBarrier<PIPE_V>();
        Mins(unique_keys, unique_keys, static_cast<int32_t>(BLOCK_Y),
             exact_count);
        PipeBarrier<PIPE_V>();
        Muls(unique_keys, unique_keys,
             static_cast<int32_t>(sizeof(int32_t)), exact_count);
        PipeBarrier<PIPE_V>();
        Gather(scratch0.ReinterpretCast<float>(),
               meta[24].ReinterpretCast<float>(),
               unique_keys.ReinterpretCast<uint32_t>(), 0U, exact_count);
        PipeBarrier<PIPE_V>();
        Mul(unique_keys, scratch1, scratch0, exact_count);
        PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 220
        PipeBarrier<PIPE_ALL>();
#endif

        // Segmented inclusive reduction.  Exact-key deduplication guarantees
        // at most eight distinct lanes per block, so shifts 1/2/4 propagate
        // every packed contribution to the last key of each block.  Lane bits
        // are disjoint, making integer addition equivalent to bitwise OR.
        for (uint32_t shift = 1; shift < UNION_SIZE; shift <<= 1) {
            ArithProgression(scratch0, 0, 1, exact_count);
            PipeBarrier<PIPE_V>();
            Adds(scratch0, scratch0, -static_cast<int32_t>(shift),
                 exact_count);
            PipeBarrier<PIPE_V>();
            Maxs(scratch0, scratch0, 0, exact_count);
            PipeBarrier<PIPE_V>();
            Muls(scratch0, scratch0, static_cast<int32_t>(sizeof(int32_t)),
                 exact_count);
            PipeBarrier<PIPE_V>();
            Gather(scratch1.ReinterpretCast<float>(),
                   sorted.ReinterpretCast<float>(),
                   scratch0.ReinterpretCast<uint32_t>(), 0U, exact_count);
            PipeBarrier<PIPE_V>();
            event_t block_predecessor_ready = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(block_predecessor_ready);
            WaitFlag<HardEvent::V_S>(block_predecessor_ready);
            for (uint32_t index = 0; index < shift; ++index) {
                scratch1.SetValue(index, -1);
            }
            event_t block_sentinel_ready = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(block_sentinel_ready);
            WaitFlag<HardEvent::S_V>(block_sentinel_ready);
            CompareKeys(compact_mask, sorted, scratch1, CMPMODE::EQ,
                    exact_count);
            PipeBarrier<PIPE_V>();
            Gather(scratch1.ReinterpretCast<float>(),
                   unique_keys.ReinterpretCast<float>(),
                   scratch0.ReinterpretCast<uint32_t>(), 0U, exact_count);
            PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 220
            LocalTensor<float> zero = compare_buf_.Get<float>();
            Duplicate(zero, 0.0f, exact_count);
            PipeBarrier<PIPE_V>();
            Select(scratch1.ReinterpretCast<float>(), compact_mask,
                   scratch1.ReinterpretCast<float>(), zero,
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, exact_count);
#else
            Select(scratch1, compact_mask, scratch1, 0,
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, exact_count);
#endif
            PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 220
            PipeBarrier<PIPE_ALL>();
#endif
            Add(unique_keys, unique_keys, scratch1, exact_count);
            PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 220
            PipeBarrier<PIPE_ALL>();
#endif
        }

        // Retain only the last exact key of every block.  It now carries the
        // fully reduced packed causal mask for that union block.
        ArithProgression(scratch0, 0, 1, exact_count);
        PipeBarrier<PIPE_V>();
        Adds(scratch0, scratch0, 1, exact_count);
        PipeBarrier<PIPE_V>();
        Mins(scratch0, scratch0, static_cast<int32_t>(exact_count - 1),
             exact_count);
        PipeBarrier<PIPE_V>();
        Muls(scratch0, scratch0, static_cast<int32_t>(sizeof(int32_t)),
             exact_count);
        PipeBarrier<PIPE_V>();
        Gather(scratch1.ReinterpretCast<float>(),
               sorted.ReinterpretCast<float>(),
               scratch0.ReinterpretCast<uint32_t>(), 0U, exact_count);
        PipeBarrier<PIPE_V>();
        event_t successor_ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(successor_ready);
        WaitFlag<HardEvent::V_S>(successor_ready);
        scratch1.SetValue(exact_count - 1, -1);
        event_t successor_sentinel_ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(successor_sentinel_ready);
        WaitFlag<HardEvent::S_V>(successor_sentinel_ready);
        CompareKeys(compact_mask, sorted, scratch1, CMPMODE::NE, exact_count);
        PipeBarrier<PIPE_V>();

        uint64_t union_count_u64 = 0;
        GatherMask(scratch0.ReinterpretCast<uint32_t>(),
                   sorted.ReinterpretCast<uint32_t>(),
                   compact_mask.ReinterpretCast<uint32_t>(), true, exact_count,
                   compact_params, union_count_u64);
        PipeBarrier<PIPE_V>();
        uint64_t packed_count_u64 = 0;
        GatherMask(scratch1.ReinterpretCast<uint32_t>(),
                   unique_keys.ReinterpretCast<uint32_t>(),
                   compact_mask.ReinterpretCast<uint32_t>(), true, exact_count,
                   compact_params, packed_count_u64);
        PipeBarrier<PIPE_V>();
        event_t union_ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(union_ready);
        WaitFlag<HardEvent::V_S>(union_ready);

        const uint32_t rank = static_cast<uint32_t>(union_count_u64);
        if (rank == 0 || packed_count_u64 != union_count_u64) {
            return 0;
        }
        WritePackedOutput(group, scratch0, scratch1, rank);
        return rank;
    }

    __aicore__ inline void WritePackedOutput(
        uint32_t group, LocalTensor<int32_t>& union_blocks,
        LocalTensor<int32_t>& packed_masks, uint32_t rank)
    {
        const uint32_t tile_capacity = Min(OUTPUT_TILE, width_);
        LocalTensor<int32_t> union_tile = output_tile_buf_.Get<int32_t>();
        LocalTensor<int32_t> mask_tile = union_tile[OUTPUT_TILE];
        for (uint32_t base = 0; base < rank; base += tile_capacity) {
            const uint32_t current = Min(tile_capacity, rank - base);
            ResetOutputTile(union_tile, mask_tile, tile_capacity);
            Adds(union_tile, union_blocks[base], 0, current);
            PipeBarrier<PIPE_V>();
            event_t packed_ready = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(packed_ready);
            WaitFlag<HardEvent::V_S>(packed_ready);
#if __CCE_AICORE__ == 310
            UnpackPackedMasks(mask_tile, packed_masks[base], current);
#else
            UnpackPackedMasks(mask_tile, packed_masks[base], current,
                              fill_buf_.Get<int32_t>());
#endif
            event_t unpack_ready = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(unpack_ready);
            WaitFlag<HardEvent::V_MTE3>(unpack_ready);
            FlushOutputTile(group, base, current, union_tile, mask_tile);
        }
    }

    __aicore__ inline void ResetOutputTile(
        LocalTensor<int32_t>& union_tile, LocalTensor<int32_t>& mask_tile,
        uint32_t capacity)
    {
        Duplicate(union_tile, static_cast<int32_t>(-1), capacity);
        Duplicate(mask_tile, static_cast<int32_t>(0), capacity * BLOCK_Y);
        event_t ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(ready);
        WaitFlag<HardEvent::V_S>(ready);
    }

    __aicore__ inline void FlushOutputTile(
        uint32_t group, uint32_t base, uint32_t count,
        LocalTensor<int32_t>& union_tile, LocalTensor<int32_t>& mask_tile)
    {
        event_t ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(ready);
        WaitFlag<HardEvent::S_MTE3>(ready);
        const uint32_t union_copy = AlignUp(count, 8);
        const uint32_t mask_copy = AlignUp(count * BLOCK_Y, 8);
        DataCopy(union_idx_gm_[group * width_ + base], union_tile,
                 union_copy);
        DataCopy(exact_mask_gm_[group * width_ * BLOCK_Y + base * BLOCK_Y],
                 mask_tile, mask_copy);
        event_t complete = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(complete);
        WaitFlag<HardEvent::MTE3_S>(complete);
    }

    __aicore__ inline uint32_t ProcessGroup(uint32_t group)
    {
        FillOutput(union_idx_gm_, group * width_, width_, -1);
        FillOutput(exact_mask_gm_, group * width_ * BLOCK_Y,
                   width_ * BLOCK_Y, 0);

        LocalTensor<int32_t> keys = keys_buf_.Get<int32_t>();
        LocalTensor<int32_t> sorted = sorted_buf_.Get<int32_t>();
        LocalTensor<uint8_t> sort_tmp = sort_tmp_buf_.Get<uint8_t>();
        LocalTensor<int32_t> meta = meta_buf_.Get<int32_t>();
        PrepareKeys(group, keys, meta);
#if __CCE_AICORE__ == 220
        // Hardware sort is descending. Negation preserves ascending integer keys.
        // Valid encoded keys are exactly representable under the existing 8M-token bound.
        LocalTensor<float> values = keys.ReinterpretCast<float>();
        Cast(values, keys, RoundMode::CAST_NONE, width_);
        PipeBarrier<PIPE_V>();
        Muls(values, values, -1.0f, width_);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> pairs = sort_tmp.ReinterpretCast<float>();
        LocalTensor<uint32_t> indices = pairs[width_ * 2].ReinterpretCast<uint32_t>();
        LocalTensor<float> temporary = compact_buf_.Get<float>();
        ArithProgression(indices.ReinterpretCast<int32_t>(), 0, 1, width_);
        PipeBarrier<PIPE_V>();
        Sort<float, true>(pairs, values, indices, temporary, width_ / 32);
        PipeBarrier<PIPE_V>();
        Extract(sorted.ReinterpretCast<float>(), indices, pairs, width_ / 32);
        PipeBarrier<PIPE_V>();
        Muls(sorted.ReinterpretCast<float>(), sorted.ReinterpretCast<float>(), -1.0f, width_);
        PipeBarrier<PIPE_V>();
        Cast(sorted, sorted.ReinterpretCast<float>(), RoundMode::CAST_RINT, width_);
        PipeBarrier<PIPE_V>();
#else
        Sort<int32_t, false>(sorted, keys, sort_tmp, width_);
#endif
        event_t sorted_ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(sorted_ready);
        WaitFlag<HardEvent::V_S>(sorted_ready);
        return CompactSorted(group, sorted, meta);
    }

    TBuf<TPosition::VECCALC> compare_buf_;
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> keys_buf_;
    TBuf<TPosition::VECCALC> sorted_buf_;
    TBuf<TPosition::VECCALC> sort_tmp_buf_;
    TBuf<TPosition::VECCALC> compact_buf_;
    TBuf<TPosition::VECCALC> row_buf_;
    TBuf<TPosition::VECCALC> prepare_buf_;
    TBuf<TPosition::VECCALC> output_tile_buf_;
    TBuf<TPosition::VECCALC> meta_buf_;
    TBuf<TPosition::VECCALC> fill_buf_;
    GlobalTensor<int32_t> compact_gm_;
    GlobalTensor<int32_t> visible_gm_;
    GlobalTensor<int32_t> row_map_gm_;
    GlobalTensor<int32_t> union_idx_gm_;
    GlobalTensor<int32_t> union_count_gm_;
    GlobalTensor<int32_t> exact_mask_gm_;
    GlobalTensor<int32_t> count_workspace_gm_;
    uint32_t rows_;
    uint32_t groups_;
    uint32_t input_groups_;
    uint32_t topk_;
    uint32_t width_;
    uint32_t compact_row_stride_;
    uint32_t sort_tmp_bytes_;
    uint32_t groups_per_core_;
};
}  // namespace

extern "C" __global__ __aicore__ void qsa_union_inputs(
    GM_ADDR compact_indices, GM_ADDR visible_lengths, GM_ADDR group_row_map,
    GM_ADDR union_idx, GM_ADDR union_count, GM_ADDR exact_mask,
    GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    REGISTER_TILING_DEFAULT(QsaUnionInputsTilingData);
    GET_TILING_DATA(tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    QsaUnionInputsKernel op(&pipe);
    op.Init(compact_indices, visible_lengths, group_row_map, union_idx,
            union_count, exact_mask, workspace, &tiling_data);
    op.Process();
}
