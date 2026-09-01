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
} // namespace

extern "C" __global__ __aicore__ void kvpp_mte_batch_copy_pages(
    __gm__ uint8_t* local_base_address, __gm__ int64_t* physical_page_ids,
    __gm__ uint8_t* valid_page_mask, __gm__ int64_t* staging_page_indices,
    uint64_t page_descriptor_count, uint64_t page_stride_bytes,
    uint64_t page_length_bytes, uint64_t staging_region_offset_bytes,
    __gm__ uint8_t* staging_base_address,
    bool staging_is_source, int32_t staging_group_rank, uint32_t shm_id)
{
    const uint64_t symmetric_size = smem_shm_get_symmetric_size(shm_id);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> buffer;
    pipe.InitBuffer(buffer, 1, KVPP_MTE_TILE_BYTES);
    AscendC::LocalTensor<uint8_t> local = buffer.AllocTensor<uint8_t>();
    __ubuf__ uint8_t* ub_address =
        reinterpret_cast<__ubuf__ uint8_t*>(local.address_.bufferAddr);
    AscendC::GlobalTensor<int64_t> page_id_descriptor;
    AscendC::GlobalTensor<uint8_t> valid_descriptor;
    AscendC::GlobalTensor<int64_t> staging_page_index_descriptor;
    page_id_descriptor.SetGlobalBuffer(physical_page_ids,
                                       page_descriptor_count);
    valid_descriptor.SetGlobalBuffer(valid_page_mask,
                                    page_descriptor_count);
    staging_page_index_descriptor.SetGlobalBuffer(staging_page_indices,
                                                  page_descriptor_count);

    const uint64_t core_index = AscendC::GetBlockIdx();
    const uint64_t core_count = AscendC::GetBlockNum();
    for (uint64_t descriptor = core_index;
         descriptor < page_descriptor_count;
         descriptor += core_count) {
        if (valid_descriptor.GetValue(descriptor) == 0) {
            continue;
        }
        const uint64_t page_id = static_cast<uint64_t>(
            page_id_descriptor.GetValue(descriptor));
        const uint64_t staging_page_index = static_cast<uint64_t>(
            staging_page_index_descriptor.GetValue(descriptor));
        const uint64_t local_offset = page_id * page_stride_bytes;
        const uint64_t staging_offset = staging_region_offset_bytes +
            staging_page_index * page_length_bytes;
        __gm__ uint8_t* local_gm = local_base_address + local_offset;
        __gm__ uint8_t* staging_address =
            staging_base_address + staging_offset;
        staging_address +=
            symmetric_size * static_cast<uint64_t>(staging_group_rank);

        uint64_t offset = 0;
        while (offset < page_length_bytes) {
            const uint32_t bytes = static_cast<uint32_t>(
                (page_length_bytes - offset) > KVPP_MTE_TILE_BYTES
                    ? KVPP_MTE_TILE_BYTES
                    : (page_length_bytes - offset));
            __gm__ uint8_t* source = staging_is_source
                ? staging_address + offset
                : local_gm + offset;
            smem_shm_copy_gm2ub<uint8_t>(
                ub_address, source, bytes, false);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(
                KVPP_MTE_EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(
                KVPP_MTE_EVENT_ID);
            __gm__ uint8_t* destination = staging_is_source
                ? local_gm + offset
                : staging_address + offset;
            smem_shm_copy_ub2gm<uint8_t>(
                destination, ub_address, bytes, false);
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
    void* stream, void* local_base_address, void* physical_page_ids,
    void* valid_page_mask, void* staging_page_indices,
    uint64_t page_descriptor_count, uint64_t page_stride_bytes,
    uint64_t page_length_bytes, uint64_t staging_region_offset_bytes,
    void* staging_base_address,
    bool staging_is_source, int32_t staging_group_rank, uint32_t shm_id)
{
    const uint32_t block_dim = page_descriptor_count < KVPP_MTE_MAX_CORES
        ? static_cast<uint32_t>(page_descriptor_count)
        : KVPP_MTE_MAX_CORES;
    kvpp_mte_batch_copy_pages<<<block_dim, nullptr, stream>>>(
        local_base_address, physical_page_ids, valid_page_mask,
        staging_page_indices, page_descriptor_count, page_stride_bytes,
        page_length_bytes, staging_region_offset_bytes,
        staging_base_address,
        staging_is_source, staging_group_rank, shm_id);
}
} // namespace vllm_ascend
#endif
