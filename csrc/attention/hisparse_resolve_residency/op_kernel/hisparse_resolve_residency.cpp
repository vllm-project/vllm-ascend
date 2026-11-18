// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220

// arch22 (A2/A3) entry: the Resolver consumes the legacy pointer/geometry ABI
// directly, keeping the classic AscendC UB path used on these SoCs. One block
// per logical row, exactly like the former csrc/hisparse launcher. Stats
// accumulate in the adapter, so this entry only serves the resolve launch
// (invalidation runs as the shared Triton kernel on every SoC).
#include "arch22/hisparse_resolver_kernel.h"

extern "C" __global__ __aicore__ void hisparse_resolve_residency(
    GM_ADDR logical,
    GM_ADDR hot_table,
    GM_ADDR requests,
    GM_ADDR states,
    GM_ADDR source,
    GM_ADDR resident,
    GM_ADDR valid_rows,
    GM_ADDR tags,
    GM_ADDR lru,
    GM_ADDR physical,
    GM_ADDR resolved,
    GM_ADDR errors,
    GM_ADDR attention,
    GM_ADDR counts,
    GM_ADDR swap_host,
    GM_ADDR swap_device,
    GM_ADDR swap_counts,
    GM_ADDR miss_mask,
    GM_ADDR stats,
    GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    (void)workspace;

    HiSparseArch22Descriptor descriptor{};
    descriptor.pointers[0] = reinterpret_cast<uintptr_t>(logical);
    descriptor.pointers[1] = reinterpret_cast<uintptr_t>(requests);
    descriptor.pointers[2] = reinterpret_cast<uintptr_t>(states);
    descriptor.pointers[3] = reinterpret_cast<uintptr_t>(source);
    descriptor.pointers[4] = reinterpret_cast<uintptr_t>(resident);
    descriptor.pointers[5] = reinterpret_cast<uintptr_t>(hot_table);
    descriptor.pointers[6] = reinterpret_cast<uintptr_t>(tags);
    descriptor.pointers[7] = reinterpret_cast<uintptr_t>(lru);
    descriptor.pointers[8] = reinterpret_cast<uintptr_t>(physical);
    descriptor.pointers[9] = reinterpret_cast<uintptr_t>(attention);
    descriptor.pointers[10] = reinterpret_cast<uintptr_t>(resolved);
    descriptor.pointers[11] = reinterpret_cast<uintptr_t>(counts);
    descriptor.pointers[12] = reinterpret_cast<uintptr_t>(swap_host);
    descriptor.pointers[13] = reinterpret_cast<uintptr_t>(swap_device);
    descriptor.pointers[14] = reinterpret_cast<uintptr_t>(swap_counts);
    descriptor.pointers[15] = reinterpret_cast<uintptr_t>(miss_mask);
    descriptor.pointers[16] = reinterpret_cast<uintptr_t>(stats);
    descriptor.pointers[17] = reinterpret_cast<uintptr_t>(errors);
    descriptor.pointers[18] = reinterpret_cast<uintptr_t>(valid_rows);
    descriptor.geometry[0] = tilingData.rows;
    descriptor.geometry[1] = tilingData.topk;
    descriptor.geometry[2] = tilingData.hotSlots;
    descriptor.geometry[3] = tilingData.hostRows;
    descriptor.geometry[4] = tilingData.logicalStride;
    descriptor.geometry[5] = tilingData.stateIndexCount;
    descriptor.geometry[6] = tilingData.tagRows;
    descriptor.geometry[7] = tilingData.sourceRows;
    descriptor.geometry[8] = tilingData.sourceColumns;
    descriptor.geometry[9] = tilingData.sourceStride;
    descriptor.geometry[10] = tilingData.sourceBlockSize;
    descriptor.geometry[11] = tilingData.residentRows;
    descriptor.geometry[12] = tilingData.residentColumns;
    descriptor.geometry[13] = tilingData.residentStride;
    descriptor.geometry[14] = tilingData.residentBlockSize;
    descriptor.geometry[15] = tilingData.residentNullBlock;
    descriptor.geometry[16] = tilingData.hotBlockSize;
    descriptor.geometry[17] = tilingData.hotTableStride;
    descriptor.geometry[18] = tilingData.attentionStride;
    descriptor.geometry[19] = tilingData.attentionBlockStride;
    descriptor.geometry[20] = tilingData.countsStride;
    descriptor.geometry[21] = tilingData.maxUnionRows;
    descriptor.geometry[22] = tilingData.hashSize;
    descriptor.geometry[23] = tilingData.attentionCapacity;
    descriptor.geometry[24] = tilingData.byteMissMask;
    AscendC::TPipe pipe;
    Resolver resolver(descriptor, pipe);
    resolver.Process();
}

#else

#include "hisparse_resolver_kernel.h"

extern "C" __global__ __aicore__ void hisparse_resolve_residency(
    GM_ADDR logical,
    GM_ADDR hot_table,
    GM_ADDR requests,
    GM_ADDR states,
    GM_ADDR source,
    GM_ADDR resident,
    GM_ADDR valid_rows,
    GM_ADDR tags,
    GM_ADDR lru,
    GM_ADDR physical,
    GM_ADDR resolved,
    GM_ADDR errors,
    GM_ADDR attention,
    GM_ADDR counts,
    GM_ADDR swap_host,
    GM_ADDR swap_device,
    GM_ADDR swap_counts,
    GM_ADDR miss_mask,
    GM_ADDR stats,
    GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    (void)workspace;

    // Allocate the SIMT scratch dynamically through the standard UB interface;
    // the tiling stage derives the byte budget from GetCoreMemSize. The
    // descriptor sits at the low end of the allocation and the scratch area
    // follows it, so the highest accessed byte stays within the buffer.
    using vllm_ascend::hisparse::kResolverDescriptorBytes;
    AscendC::TPipe resolverPipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scratchBuf;
    resolverPipe.InitBuffer(scratchBuf, tilingData.ubBytes);
    auto descriptor = (__ubuf__ uint64_t*)scratchBuf.Get<int32_t>().GetPhyAddr();
    auto scratch = (__ubuf__ int32_t*)(descriptor + kResolverDescriptorBytes / sizeof(uint64_t));
    descriptor[0] = reinterpret_cast<uintptr_t>(logical);
    descriptor[1] = reinterpret_cast<uintptr_t>(requests);
    descriptor[2] = reinterpret_cast<uintptr_t>(states);
    descriptor[3] = reinterpret_cast<uintptr_t>(source);
    descriptor[4] = reinterpret_cast<uintptr_t>(resident);
    descriptor[5] = reinterpret_cast<uintptr_t>(hot_table);
    descriptor[6] = reinterpret_cast<uintptr_t>(tags);
    descriptor[7] = reinterpret_cast<uintptr_t>(lru);
    descriptor[8] = reinterpret_cast<uintptr_t>(physical);
    descriptor[9] = reinterpret_cast<uintptr_t>(attention);
    descriptor[10] = reinterpret_cast<uintptr_t>(resolved);
    descriptor[11] = reinterpret_cast<uintptr_t>(counts);
    descriptor[12] = reinterpret_cast<uintptr_t>(swap_host);
    descriptor[13] = reinterpret_cast<uintptr_t>(swap_device);
    descriptor[14] = reinterpret_cast<uintptr_t>(swap_counts);
    descriptor[15] = reinterpret_cast<uintptr_t>(miss_mask);
    descriptor[16] = reinterpret_cast<uintptr_t>(stats);
    descriptor[17] = reinterpret_cast<uintptr_t>(errors);
    descriptor[18] = reinterpret_cast<uintptr_t>(valid_rows);
    // Preserve the descriptor layout consumed by the SIMT vector function.
    descriptor[32] = tilingData.rows;
    descriptor[33] = tilingData.topk;
    descriptor[34] = tilingData.hotSlots;
    descriptor[35] = tilingData.hostRows;
    descriptor[36] = tilingData.logicalStride;
    descriptor[37] = tilingData.stateIndexCount;
    descriptor[38] = tilingData.tagRows;
    descriptor[39] = tilingData.sourceRows;
    descriptor[40] = tilingData.sourceColumns;
    descriptor[41] = tilingData.sourceStride;
    descriptor[42] = tilingData.sourceBlockSize;
    descriptor[43] = tilingData.residentRows;
    descriptor[44] = tilingData.residentColumns;
    descriptor[45] = tilingData.residentStride;
    descriptor[46] = tilingData.residentBlockSize;
    descriptor[47] = tilingData.residentNullBlock;
    descriptor[48] = tilingData.hotBlockSize;
    descriptor[49] = tilingData.hotTableStride;
    descriptor[50] = tilingData.attentionStride;
    descriptor[51] = tilingData.attentionBlockStride;
    descriptor[52] = tilingData.countsStride;
    descriptor[53] = tilingData.maxUnionRows;
    descriptor[54] = tilingData.hashSize;
    descriptor[55] = tilingData.attentionCapacity;
    descriptor[56] = tilingData.byteMissMask;
    const int block = AscendC::GetBlockIdx();
    if (tilingData.topk == 2048 && tilingData.hotSlots == 10240 && tilingData.sourceBlockSize == 128 &&
        tilingData.residentBlockSize == 128 && tilingData.hotBlockSize == 128 && tilingData.hashSize == 16384) {
        asc_vf_call<resolve_vf<true>>(dim3(kThreads), descriptor, scratch, block);
    } else {
        asc_vf_call<resolve_vf<false>>(dim3(kThreads), descriptor, scratch, block);
    }
}

#endif
