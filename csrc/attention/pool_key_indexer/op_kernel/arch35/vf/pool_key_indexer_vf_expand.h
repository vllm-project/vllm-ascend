/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 * You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pool_key_indexer_vf_expand.h
 * \brief pool 索引 -> token 索引的向量化展开(arch35 AIV)。
 *
 * 展开(out[m] = ps * idx[m/ps] + m%ps)用 64 lane 寄存器 gather(vselr)方案
 * 替代逐 pool 标量循环: 每组 LoadAlign 64 个池索引进寄存器, 组内 ps 个
 * 64 lane 输出 chunk 用 vselr 按相对索引选取; pow2 ps 下选源索引
 * j*64/ps + r/ps 恒落在 64 lane 内。仅支持 pow2 ps ∈ [2,64](ps|64 保证
 * chunkBase%ps==0); ps=128 由调用方走原有 Duplicate+Add 路径。
 * 尾块(lane >= effExpand)为垃圾值, 由调用方用 -1 Duplicate 覆盖。
 */

#ifndef POOL_KEY_INDEXER_VF_EXPAND_H
#define POOL_KEY_INDEXER_VF_EXPAND_H

#include "kernel_operator.h"

namespace pkiexpand {
using namespace AscendC;

// pow2 pool_size ∈ [2,64] 的向量化展开: out[m] = ps*idx[m/ps] + m%ps, 消除逐
// pool 标量循环; 尾块(lane >= effExpand)为垃圾值, 由调用方用 -1 Duplicate 覆盖
__aicore__ inline void ExpandPow2PoolIndices(const LocalTensor<int32_t> &tokenIndices,
                                             const LocalTensor<uint32_t> &poolIdxTable, const LocalTensor<int32_t> &tpl,
                                             uint32_t effExpand, uint32_t ps)
{
    __ubuf__ uint32_t *out = (__ubuf__ uint32_t *)tokenIndices.GetPhyAddr();
    __ubuf__ uint32_t *table = (__ubuf__ uint32_t *)poolIdxTable.GetPhyAddr();
    __ubuf__ uint32_t *rIdxTpl = (__ubuf__ uint32_t *)tpl.GetPhyAddr();      // r / ps
    __ubuf__ uint32_t *rOffTpl = (__ubuf__ uint32_t *)tpl.GetPhyAddr() + 64; // r % ps

    if (effExpand == 0) {
        return;
    }
    // __VEC_SCOPE__ 内向量循环归纳变量/条件必须为 uint16_t(编译器约束);
    // chunkNum <= Align(4096,64)/64 = 64, ps <= 64, 均不溢出
    uint16_t chunkNum = static_cast<uint16_t>((effExpand + 63U) / 64U);
    const uint16_t psU16 = static_cast<uint16_t>(ps);
    // 每组: LoadAlign 64 个池索引 -> 覆盖 64*ps 个输出 = ps 个 64 lane chunk
    uint16_t groupNum = static_cast<uint16_t>((chunkNum + psU16 - 1U) / psU16);

    __VEC_SCOPE__
    {
        Reg::MaskReg maskAll = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::RegTensor<uint32_t> rIdx;
        Reg::RegTensor<uint32_t> rOff;
        Reg::RegTensor<uint32_t> psReg;
        Reg::RegTensor<uint32_t> poolReg;
        Reg::RegTensor<uint32_t> gatherIdx;
        Reg::RegTensor<uint32_t> gathered;

        Reg::LoadAlign<uint32_t>(rIdx, rIdxTpl);
        Reg::LoadAlign<uint32_t>(rOff, rOffTpl);
        Reg::Duplicate<uint32_t>(psReg, ps, maskAll);

        for (uint16_t g = 0; g < groupNum; g++) {
            // 本组 64 个池索引装入寄存器(256B 对齐读)
            Reg::LoadAlign<uint32_t>(poolReg, table + static_cast<uint32_t>(g) * 64U);
            // 本组实际输出 chunk 数: 尾组可能不足 ps 个(g < groupNum 保证 > 0)
            uint16_t chunksInGroup = static_cast<uint16_t>(chunkNum - static_cast<uint32_t>(g) * psU16);
            if (chunksInGroup > psU16) {
                chunksInGroup = psU16;
            }
            for (uint16_t j = 0; j < chunksInGroup; j++) {
                uint32_t c = static_cast<uint32_t>(g) * psU16 + j; // 全局输出 chunk 号
                // 选源索引 = j*64/ps + r/ps, 恒落在 poolReg 的 64 lane 内
                // (ps <= 64 时 ps|64, lane 偏移恒为 r%ps)
                uint32_t relBase = (static_cast<uint32_t>(j) * 64U) / ps;
                Reg::Adds(gatherIdx, rIdx, relBase, maskAll);
                Reg::Gather(gathered, poolReg, gatherIdx);
                Reg::Mul(gathered, gathered, psReg, maskAll);
                Reg::Add(gathered, gathered, rOff, maskAll);
                Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(out + c * 64U, gathered, maskAll);
            }
        }
    }
}
} // namespace pkiexpand

#endif // POOL_KEY_INDEXER_VF_EXPAND_H
