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
    auto descriptor = (__ubuf__ uint64_t*)(scratch + (tilingData.get_ubBytes() - kResolverDescriptorBytes) / sizeof(int32_t));
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
    descriptor[32] = tilingData.get_rows();
    descriptor[33] = tilingData.get_topk();
    descriptor[34] = tilingData.get_hotSlots();
    descriptor[35] = tilingData.get_hostRows();
    descriptor[36] = tilingData.get_logicalStride();
    descriptor[37] = tilingData.get_stateIndexCount();
    descriptor[38] = tilingData.get_tagRows();
    descriptor[39] = tilingData.get_sourceRows();
    descriptor[40] = tilingData.get_sourceColumns();
    descriptor[41] = tilingData.get_sourceStride();
    descriptor[42] = tilingData.get_sourceBlockSize();
    descriptor[43] = tilingData.get_residentRows();
    descriptor[44] = tilingData.get_residentColumns();
    descriptor[45] = tilingData.get_residentStride();
    descriptor[46] = tilingData.get_residentBlockSize();
    descriptor[47] = tilingData.get_residentNullBlock();
    descriptor[48] = tilingData.get_hotBlockSize();
    descriptor[49] = tilingData.get_hotTableStride();
    descriptor[50] = tilingData.get_attentionStride();
    descriptor[51] = tilingData.get_attentionBlockStride();
    descriptor[52] = tilingData.get_countsStride();
    descriptor[53] = tilingData.get_maxUnionRows();
    descriptor[54] = tilingData.get_hashSize();
    descriptor[55] = tilingData.get_attentionCapacity();
    descriptor[56] = tilingData.get_byteMissMask();
    const int block = AscendC::GetBlockIdx();
    if (tilingData.get_topk() == 2048 && tilingData.get_hotSlots() == 10240 && tilingData.get_sourceBlockSize() == 128 &&
        tilingData.get_residentBlockSize() == 128 && tilingData.get_hotBlockSize() == 128 && tilingData.get_hashSize() == 16384) {
        asc_vf_call<resolve_vf<true>>(dim3(kThreads), descriptor, scratch, block);
    } else {
        asc_vf_call<resolve_vf<false>>(dim3(kThreads), descriptor, scratch, block);
    }
}
