/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kernel_operator.h"

#if __has_include("smem/device/smem_shm_aicore_base_api.h")
#include "smem/device/smem_shm_aicore_base_api.h"

namespace {
constexpr uint32_t KVPP_MTE_TILE_BYTES = 64 * 1024;
constexpr uint32_t KVPP_MTE_MAX_CORES = 32;
constexpr int32_t KVPP_MTE_EVENT_ID = 0;
}  // namespace

extern "C" __global__ __aicore__ void kvpp_mte_batch_copy_pages(
    __gm__ uint8_t* local_base, __gm__ int64_t* local_offsets,
    __gm__ int64_t* staging_offsets, __gm__ int64_t* lengths,
    uint64_t descriptor_count, __gm__ uint8_t* staging_base,
    int32_t source_rank, int32_t destination_rank, uint32_t shm_id)
{
    const uint64_t symmetric_size = smem_shm_get_symmetric_size(shm_id);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> buffer;
    pipe.InitBuffer(buffer, 1, KVPP_MTE_TILE_BYTES);
    AscendC::LocalTensor<uint8_t> local = buffer.AllocTensor<uint8_t>();
    __ubuf__ uint8_t* ub_address =
        reinterpret_cast<__ubuf__ uint8_t*>(local.address_.bufferAddr);
    AscendC::GlobalTensor<int64_t> local_offset_descriptor;
    AscendC::GlobalTensor<int64_t> staging_offset_descriptor;
    AscendC::GlobalTensor<int64_t> length_descriptor;
    local_offset_descriptor.SetGlobalBuffer(local_offsets, descriptor_count);
    staging_offset_descriptor.SetGlobalBuffer(staging_offsets,
                                              descriptor_count);
    length_descriptor.SetGlobalBuffer(lengths, descriptor_count);

    const uint64_t core_index = AscendC::GetBlockIdx();
    const uint64_t core_count = AscendC::GetBlockNum();
    for (uint64_t descriptor = core_index; descriptor < descriptor_count;
         descriptor += core_count) {
        const uint64_t length = static_cast<uint64_t>(
            length_descriptor.GetValue(descriptor));
        if (length == 0) {
            continue;
        }
        const uint64_t local_offset = static_cast<uint64_t>(
            local_offset_descriptor.GetValue(descriptor));
        const uint64_t staging_offset = static_cast<uint64_t>(
            staging_offset_descriptor.GetValue(descriptor));
        __gm__ uint8_t* local_address = local_base + local_offset;
        __gm__ uint8_t* staging_address = staging_base + staging_offset;
        const int32_t staging_rank =
            source_rank >= 0 ? source_rank : destination_rank;
        staging_address += symmetric_size *
            static_cast<uint64_t>(staging_rank);

        uint64_t offset = 0;
        while (offset < length) {
            const uint32_t bytes = static_cast<uint32_t>(
                (length - offset) > KVPP_MTE_TILE_BYTES
                    ? KVPP_MTE_TILE_BYTES
                    : (length - offset));
            __gm__ uint8_t* source = source_rank >= 0
                ? staging_address + offset
                : local_address + offset;
            smem_shm_copy_gm2ub<uint8_t>(ub_address, source, bytes, false);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(
                KVPP_MTE_EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(
                KVPP_MTE_EVENT_ID);
            __gm__ uint8_t* destination = destination_rank >= 0
                ? staging_address + offset
                : local_address + offset;
            smem_shm_copy_ub2gm<uint8_t>(destination, ub_address, bytes,
                                         false);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
                KVPP_MTE_EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                KVPP_MTE_EVENT_ID);
            offset += bytes;
        }
    }
    buffer.FreeTensor(local);
}

namespace vllm_ascend {
void kvpp_mte_batch_copy_pages_impl(
    void* stream, void* local_base, void* local_offsets,
    void* staging_offsets, void* lengths, uint64_t descriptor_count,
    void* staging_base, int32_t source_rank, int32_t destination_rank,
    uint32_t shm_id)
{
    const uint32_t block_dim = descriptor_count < KVPP_MTE_MAX_CORES
        ? static_cast<uint32_t>(descriptor_count)
        : KVPP_MTE_MAX_CORES;
    kvpp_mte_batch_copy_pages<<<block_dim, nullptr, stream>>>(
        local_base, local_offsets, staging_offsets, lengths,
        descriptor_count, staging_base, source_rank, destination_rank,
        shm_id);
}
}  // namespace vllm_ascend
#endif
