// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include "hisparse_resolver_kernel.h"

extern "C" __global__ __aicore__ void hi_sparse_resolve_residency(
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

    // Reserve the SIMT shared region in the compiled kernel. ACLNN does not
    // supply the dynamic UB size carried by the direct launch configuration.
    using vllm_ascend::hisparse::kResolverDescriptorBytes;
    using vllm_ascend::hisparse::kResolverUbBytes;
    __ubuf__ int32_t scratch[kResolverUbBytes / sizeof(int32_t)];
    auto descriptor = (__ubuf__ uint64_t*)(scratch + (tilingData.ubBytes - kResolverDescriptorBytes) / sizeof(int32_t));
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
