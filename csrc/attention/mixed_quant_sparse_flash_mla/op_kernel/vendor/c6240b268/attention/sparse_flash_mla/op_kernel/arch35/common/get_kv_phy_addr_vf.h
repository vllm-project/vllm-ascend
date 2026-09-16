/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file get_kv_phy_addr_vf.h
 * \brief sparse_flash_mla / mixed_quant_sparse_flash_mla / quant_sparse_flash_mla 三个算子共用的
 *        KV 物理地址计算 VF 实现（Pa / Tnd / Bsnd 三种布局）。
 */

#ifndef SPARSE_FLASH_MLA_GET_KV_PHY_ADDR_VF_H
#define SPARSE_FLASH_MLA_GET_KV_PHY_ADDR_VF_H

#include <stdint.h>
#include "static_buffer.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"

namespace AttentionCommon {

template <typename T>
__simd_vf__ void GetKVPhyAddrVFPaImpl(__ubuf__ uint32_t *kvPhyAddrUb, __ubuf__ int32_t *sparseIdxUb,
                                      __ubuf__ int32_t *blkTableUb, const uint16_t s2Loop, uint32_t s2Tail,
                                      const uint32_t blockSize, const int16_t shiftRightNum,
                                      const uint32_t sparseBlockSize, const uint32_t kvDim, const uint32_t kvStride)
{
    static const uint16_t s2_num_per_loop = 128;
    static const uint16_t s2_num_per_reg = 64;
    static const uint16_t out_offset_per_loop = 256;
    static const uint16_t out_offset_per_reg = 128;
    static const uint32_t invalid_value = 0xFFFFFFFF;
    Reg::MaskReg preg_all_b32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg add_carry_l_1;
    Reg::MaskReg add_carry_h_1;
    Reg::MaskReg add_carry_l_2;
    Reg::MaskReg add_carry_h_2;
    Reg::MaskReg preg_tail_neg_1_b32;
    Reg::MaskReg preg_tail_neg_2_b32;

    Reg::RegTensor<uint32_t> vreg_kv_stride;
    Reg::RegTensor<uint32_t> vreg_sparse_idx_1;
    Reg::RegTensor<uint32_t> vreg_sparse_idx_2;
    Reg::RegTensor<uint32_t> vreg_block_size;
    Reg::RegTensor<uint32_t> vreg_shift_rights_num;
    Reg::RegTensor<uint32_t> vreg_pa_blk_idx_1;
    Reg::RegTensor<uint32_t> vreg_pa_blk_idx_2;
    Reg::RegTensor<uint32_t> vreg_pa_tmp_1;
    Reg::RegTensor<uint32_t> vreg_pa_tmp_2;
    Reg::RegTensor<uint32_t> vreg_pa_offset_1;
    Reg::RegTensor<uint32_t> vreg_pa_offset_2;
    Reg::RegTensor<uint32_t> vreg_phy_offset_1;
    Reg::RegTensor<uint32_t> vreg_phy_offset_2;
    Reg::RegTensor<uint32_t> vreg_phy_blk_idx_1;
    Reg::RegTensor<uint32_t> vreg_phy_blk_idx_2;

    Reg::RegTensor<uint32_t> vreg_blk_id_mul_stride_h_1;
    Reg::RegTensor<uint32_t> vreg_blk_id_mul_stride_tmp_h_1;
    Reg::RegTensor<uint32_t> vreg_blk_id_mul_stride_l_1;
    Reg::RegTensor<uint32_t> vreg_mul_overflow_l_1;
    Reg::RegTensor<uint32_t> vreg_total_offset_l_1;
    Reg::RegTensor<uint32_t> vreg_total_offset_h_1;

    Reg::RegTensor<uint32_t> vreg_blk_id_mul_stride_h_2;
    Reg::RegTensor<uint32_t> vreg_blk_id_mul_stride_tmp_h_2;
    Reg::RegTensor<uint32_t> vreg_blk_id_mul_stride_l_2;
    Reg::RegTensor<uint32_t> vreg_mul_overflow_l_2;
    Reg::RegTensor<uint32_t> vreg_total_offset_l_2;
    Reg::RegTensor<uint32_t> vreg_total_offset_h_2;

    Reg::RegTensor<uint32_t> vreg_zero;
    Reg::Duplicate(vreg_zero, 0);
    Reg::Duplicate(vreg_kv_stride, kvStride);

    for (; s2Loop > 1;) {
        for (uint16_t i = 0; i < s2Loop - 1; i++) {
            Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_1,
                                                              sparseIdxUb + i * s2_num_per_loop);
            Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_2,
                                                              sparseIdxUb + s2_num_per_reg + i * s2_num_per_loop);
            // * sparseBlockSize
            Reg::Muls(vreg_sparse_idx_1, vreg_sparse_idx_1, sparseBlockSize, preg_all_b32);
            Reg::Muls(vreg_sparse_idx_2, vreg_sparse_idx_2, sparseBlockSize, preg_all_b32);
            // 计算右移位数
            // 右移 -> 除blockSize 得到paBlockIdx，vreg_sparse_idx - pa_idx * blocksize -> pa offset
            Reg::ShiftRights(vreg_pa_blk_idx_1, vreg_sparse_idx_1, shiftRightNum, preg_all_b32);
            Reg::ShiftRights(vreg_pa_blk_idx_2, vreg_sparse_idx_2, shiftRightNum, preg_all_b32);

            Reg::Muls(vreg_pa_tmp_1, vreg_pa_blk_idx_1, blockSize, preg_all_b32);
            Reg::Muls(vreg_pa_tmp_2, vreg_pa_blk_idx_2, blockSize, preg_all_b32);
            // offset
            Reg::Sub(vreg_pa_offset_1, vreg_sparse_idx_1, vreg_pa_tmp_1, preg_all_b32);
            Reg::Sub(vreg_pa_offset_2, vreg_sparse_idx_2, vreg_pa_tmp_2, preg_all_b32);
            // 物理页内offset
            Reg::Muls(vreg_phy_offset_1, vreg_pa_offset_1, kvDim, preg_all_b32);
            Reg::Muls(vreg_phy_offset_2, vreg_pa_offset_2, kvDim, preg_all_b32);

            // int32 paBlockId -> 物理id
            DataCopyGather(vreg_phy_blk_idx_1, blkTableUb, vreg_pa_blk_idx_1, preg_all_b32);
            DataCopyGather(vreg_phy_blk_idx_2, blkTableUb, vreg_pa_blk_idx_2, preg_all_b32);

            // 分高低32位计算int64物理地址 -- 乘 stride
            // 低位乘 带进位
            Reg::Mull(vreg_blk_id_mul_stride_l_1, vreg_mul_overflow_l_1, vreg_phy_blk_idx_1, vreg_kv_stride,
                      preg_all_b32);
            Reg::Mull(vreg_blk_id_mul_stride_l_2, vreg_mul_overflow_l_2, vreg_phy_blk_idx_2, vreg_kv_stride,
                      preg_all_b32);

            // 分高低32位计算int64物理地址 -- 加 offset
            Reg::Add(add_carry_l_1, vreg_total_offset_l_1, vreg_blk_id_mul_stride_l_1, vreg_phy_offset_1, preg_all_b32);
            Reg::Add(add_carry_l_2, vreg_total_offset_l_2, vreg_blk_id_mul_stride_l_2, vreg_phy_offset_2, preg_all_b32);

            Reg::AddC(add_carry_h_1, vreg_total_offset_h_1, vreg_mul_overflow_l_1, vreg_zero, add_carry_l_1,
                      preg_all_b32);
            Reg::AddC(add_carry_h_2, vreg_total_offset_h_2, vreg_mul_overflow_l_2, vreg_zero, add_carry_l_2,
                      preg_all_b32);

            // 搬出 由于拆分为了int32类型，元素个数翻倍
            Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + i * out_offset_per_loop, vreg_total_offset_l_1, vreg_total_offset_h_1, preg_all_b32);
            Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + out_offset_per_reg + i * out_offset_per_loop, vreg_total_offset_l_2,
                vreg_total_offset_h_2, preg_all_b32);
        }
        break;
    }

    for (uint16_t i = s2Loop - 1; i < s2Loop; i++) {
        Reg::MaskReg preg_tail_1_b32 = Reg::UpdateMask<int32_t>(s2Tail);
        Reg::MaskReg preg_tail_2_b32 = Reg::UpdateMask<int32_t>(s2Tail);
        Reg::Not(preg_tail_neg_1_b32, preg_tail_1_b32, preg_all_b32);
        Reg::Not(preg_tail_neg_2_b32, preg_tail_2_b32, preg_all_b32);

        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_1,
                                                          sparseIdxUb + i * s2_num_per_loop);
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_2,
                                                          sparseIdxUb + s2_num_per_reg + i * s2_num_per_loop);
        // * sparseBlockSize
        Reg::Muls(vreg_sparse_idx_1, vreg_sparse_idx_1, sparseBlockSize, preg_tail_1_b32);
        Reg::Muls(vreg_sparse_idx_2, vreg_sparse_idx_2, sparseBlockSize, preg_tail_2_b32);
        // 计算右移位数
        // 右移 -> 除blockSize 得到paBlockIdx，vreg_sparse_idx - pa_idx * blocksize -> pa offset
        Reg::ShiftRights(vreg_pa_blk_idx_1, vreg_sparse_idx_1, shiftRightNum, preg_tail_1_b32);
        Reg::ShiftRights(vreg_pa_blk_idx_2, vreg_sparse_idx_2, shiftRightNum, preg_tail_2_b32);

        Reg::Muls(vreg_pa_tmp_1, vreg_pa_blk_idx_1, blockSize, preg_tail_1_b32);
        Reg::Muls(vreg_pa_tmp_2, vreg_pa_blk_idx_2, blockSize, preg_tail_2_b32);
        // offset
        Reg::Sub(vreg_pa_offset_1, vreg_sparse_idx_1, vreg_pa_tmp_1, preg_tail_1_b32);
        Reg::Sub(vreg_pa_offset_2, vreg_sparse_idx_2, vreg_pa_tmp_2, preg_tail_2_b32);
        // 物理页内offset
        Reg::Muls(vreg_phy_offset_1, vreg_pa_offset_1, kvDim, preg_tail_1_b32);
        Reg::Muls(vreg_phy_offset_2, vreg_pa_offset_2, kvDim, preg_tail_2_b32);

        // int32 paBlockId -> 物理id
        DataCopyGather(vreg_phy_blk_idx_1, blkTableUb, vreg_pa_blk_idx_1, preg_tail_1_b32);
        DataCopyGather(vreg_phy_blk_idx_2, blkTableUb, vreg_pa_blk_idx_2, preg_tail_2_b32);

        // 分高低32位计算int64物理地址 -- 乘 stride
        // 低位乘 带进位
        Reg::Mull(vreg_blk_id_mul_stride_l_1, vreg_mul_overflow_l_1, vreg_phy_blk_idx_1, vreg_kv_stride,
                  preg_tail_1_b32);
        Reg::Mull(vreg_blk_id_mul_stride_l_2, vreg_mul_overflow_l_2, vreg_phy_blk_idx_2, vreg_kv_stride,
                  preg_tail_2_b32);

        // 分高低32位计算int64物理地址 -- 加 offset
        Reg::Add(add_carry_l_1, vreg_total_offset_l_1, vreg_blk_id_mul_stride_l_1, vreg_phy_offset_1, preg_tail_1_b32);
        Reg::Add(add_carry_l_2, vreg_total_offset_l_2, vreg_blk_id_mul_stride_l_2, vreg_phy_offset_2, preg_tail_2_b32);

        Reg::AddC(add_carry_h_1, vreg_total_offset_h_1, vreg_mul_overflow_l_1, vreg_zero, add_carry_l_1,
                  preg_tail_1_b32);
        Reg::AddC(add_carry_h_2, vreg_total_offset_h_2, vreg_mul_overflow_l_2, vreg_zero, add_carry_l_2,
                  preg_tail_2_b32);

        // 无效值填充-1(0xFFFFFFFF)
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_l_1, invalid_value,
                                                              preg_tail_neg_1_b32);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_h_1, invalid_value,
                                                              preg_tail_neg_1_b32);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_l_2, invalid_value,
                                                              preg_tail_neg_2_b32);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_h_2, invalid_value,
                                                              preg_tail_neg_2_b32);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
            kvPhyAddrUb + i * out_offset_per_loop, vreg_total_offset_l_1, vreg_total_offset_h_1, preg_all_b32);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
            kvPhyAddrUb + out_offset_per_reg + i * out_offset_per_loop, vreg_total_offset_l_2, vreg_total_offset_h_2,
            preg_all_b32);
    }
}

template <typename T>
__aicore__ inline void GetKVPhyAddrVFPa(LocalTensor<uint32_t> kvPhyAddrTensor, LocalTensor<int32_t> sparseIdxTensor,
                                        LocalTensor<int32_t> blkTableTensor, const uint16_t s2Loop,
                                        const uint32_t s2Tail, const uint32_t blockSize, const int16_t shiftRightNum,
                                        const uint32_t sparseBlockSize, const uint32_t kvDim, const uint32_t kvStride)
{
    __ubuf__ uint32_t *kv_phy_addr_ub = (__ubuf__ uint32_t *)(kvPhyAddrTensor.GetPhyAddr());
    __ubuf__ int32_t *sparse_idx_ub = (__ubuf__ int32_t *)(sparseIdxTensor.GetPhyAddr());
    __ubuf__ int32_t *blk_table_ub = (__ubuf__ int32_t *)(blkTableTensor.GetPhyAddr());
    GetKVPhyAddrVFPaImpl<uint32_t>(kv_phy_addr_ub, sparse_idx_ub, blk_table_ub, s2Loop, s2Tail, blockSize,
                                   shiftRightNum, sparseBlockSize, kvDim, kvStride);
}

template <typename T>
__simd_vf__ void GetKVPhyAddrVFTndImpl(__ubuf__ uint32_t *kvPhyAddrUb, __ubuf__ int32_t *sparseIdxUb,
                                       const uint16_t s2Loop, uint32_t s2Tail, const uint32_t sparseBlockSize,
                                       const uint32_t kvDim, const uint32_t kvPrefix)
{
    static const uint16_t s2_num_per_loop = 128;
    static const uint16_t s2_num_per_reg = 64;
    static const uint16_t out_offset_per_loop = 256;
    static const uint16_t out_offset_per_reg = 128;
    static const uint32_t invalid_value = 0xFFFFFFFF;
    Reg::MaskReg preg_all_b32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg preg_tail_neg_1_b32;
    Reg::MaskReg preg_tail_neg_2_b32;

    Reg::RegTensor<uint32_t> vreg_sparse_idx_1;
    Reg::RegTensor<uint32_t> vreg_sparse_idx_2;
    Reg::RegTensor<uint32_t> vreg_kv_prefix;
    Reg::RegTensor<uint32_t> vreg_kv_dim;
    Reg::RegTensor<uint32_t> vreg_sum_1;
    Reg::RegTensor<uint32_t> vreg_sum_2;
    Reg::RegTensor<uint32_t> vreg_mul_overflow_l_1;
    Reg::RegTensor<uint32_t> vreg_mul_overflow_l_2;
    Reg::RegTensor<uint32_t> vreg_total_offset_l_1;
    Reg::RegTensor<uint32_t> vreg_total_offset_h_1;
    Reg::RegTensor<uint32_t> vreg_total_offset_l_2;
    Reg::RegTensor<uint32_t> vreg_total_offset_h_2;

    Reg::Duplicate(vreg_kv_prefix, kvPrefix);
    Reg::Duplicate(vreg_kv_dim, kvDim);

    for (; s2Loop > 1;) {
        for (uint16_t i = 0; i < s2Loop - 1; i++) {
            Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_1,
                                                              sparseIdxUb + i * s2_num_per_loop);
            Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_2,
                                                              sparseIdxUb + s2_num_per_reg + i * s2_num_per_loop);
            // * sparseBlockSize
            Reg::Muls(vreg_sparse_idx_1, vreg_sparse_idx_1, sparseBlockSize, preg_all_b32);
            Reg::Muls(vreg_sparse_idx_2, vreg_sparse_idx_2, sparseBlockSize, preg_all_b32);
            // (kvPrefix + sparseIdx) * kvDim -> int64 物理地址
            Reg::Add(vreg_sum_1, vreg_sparse_idx_1, vreg_kv_prefix, preg_all_b32);
            Reg::Add(vreg_sum_2, vreg_sparse_idx_2, vreg_kv_prefix, preg_all_b32);
            // 带进位乘法
            Reg::Mull(vreg_total_offset_l_1, vreg_total_offset_h_1, vreg_sum_1, vreg_kv_dim, preg_all_b32);
            Reg::Mull(vreg_total_offset_l_2, vreg_total_offset_h_2, vreg_sum_2, vreg_kv_dim, preg_all_b32);
            // 搬出
            Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + i * out_offset_per_loop, vreg_total_offset_l_1, vreg_total_offset_h_1, preg_all_b32);
            Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + out_offset_per_reg + i * out_offset_per_loop, vreg_total_offset_l_2,
                vreg_total_offset_h_2, preg_all_b32);
        }
        break;
    }

    for (uint16_t i = s2Loop - 1; i < s2Loop; i++) {
        Reg::MaskReg preg_tail_1_b32 = Reg::UpdateMask<int32_t>(s2Tail);
        Reg::MaskReg preg_tail_2_b32 = Reg::UpdateMask<int32_t>(s2Tail);
        Reg::Not(preg_tail_neg_1_b32, preg_tail_1_b32, preg_all_b32);
        Reg::Not(preg_tail_neg_2_b32, preg_tail_2_b32, preg_all_b32);

        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_1,
                                                          sparseIdxUb + i * s2_num_per_loop);
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_2,
                                                          sparseIdxUb + s2_num_per_reg + i * s2_num_per_loop);
        // * sparseBlockSize
        Reg::Muls(vreg_sparse_idx_1, vreg_sparse_idx_1, sparseBlockSize, preg_tail_1_b32);
        Reg::Muls(vreg_sparse_idx_2, vreg_sparse_idx_2, sparseBlockSize, preg_tail_2_b32);
        // (kvPrefix + sparseIdx) * kvDim -> int64 物理地址
        Reg::Add(vreg_sum_1, vreg_sparse_idx_1, vreg_kv_prefix, preg_tail_1_b32);
        Reg::Add(vreg_sum_2, vreg_sparse_idx_2, vreg_kv_prefix, preg_tail_2_b32);
        // 带进位乘法
        Reg::Mull(vreg_total_offset_l_1, vreg_total_offset_h_1, vreg_sum_1, vreg_kv_dim, preg_tail_1_b32);
        Reg::Mull(vreg_total_offset_l_2, vreg_total_offset_h_2, vreg_sum_2, vreg_kv_dim, preg_tail_2_b32);
        // 无效值填充-1(0xFFFFFFFF)
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_l_1, invalid_value,
                                                              preg_tail_neg_1_b32);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_h_1, invalid_value,
                                                              preg_tail_neg_1_b32);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_l_2, invalid_value,
                                                              preg_tail_neg_2_b32);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_h_2, invalid_value,
                                                              preg_tail_neg_2_b32);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
            kvPhyAddrUb + i * out_offset_per_loop, vreg_total_offset_l_1, vreg_total_offset_h_1, preg_all_b32);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
            kvPhyAddrUb + out_offset_per_reg + i * out_offset_per_loop, vreg_total_offset_l_2, vreg_total_offset_h_2,
            preg_all_b32);
    }
}

template <typename T>
__aicore__ inline void GetKVPhyAddrVFTnd(LocalTensor<uint32_t> kvPhyAddrTensor, LocalTensor<int32_t> sparseIdxTensor,
                                         const uint16_t s2Loop, const uint32_t s2Tail, const uint32_t sparseBlockSize,
                                         const uint32_t kvDim, const uint32_t kvPrefix)
{
    __ubuf__ uint32_t *kv_phy_addr_ub = (__ubuf__ uint32_t *)(kvPhyAddrTensor.GetPhyAddr());
    __ubuf__ int32_t *sparse_idx_ub = (__ubuf__ int32_t *)(sparseIdxTensor.GetPhyAddr());
    GetKVPhyAddrVFTndImpl<uint32_t>(kv_phy_addr_ub, sparse_idx_ub, s2Loop, s2Tail, sparseBlockSize, kvDim, kvPrefix);
}

template <typename T>
__simd_vf__ void GetKVPhyAddrVFBsndImpl(__ubuf__ uint32_t *kvPhyAddrUb, __ubuf__ int32_t *sparseIdxUb,
                                        const uint16_t s2Loop, uint32_t s2Tail, const uint32_t sparseBlockSize,
                                        const uint32_t kvDim, const uint32_t bS2BaseLow, const uint32_t bS2BaseHigh)
{
    static const uint16_t s2_num_per_loop = 128;
    static const uint16_t s2_num_per_reg = 64;
    static const uint16_t out_offset_per_loop = 256;
    static const uint16_t out_offset_per_reg = 128;
    static const uint32_t invalid_value = 0xFFFFFFFF;
    Reg::MaskReg preg_all_b32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg add_carry_l_1;
    Reg::MaskReg add_carry_h_1;
    Reg::MaskReg add_carry_l_2;
    Reg::MaskReg add_carry_h_2;
    Reg::MaskReg preg_tail_neg_1_b32;
    Reg::MaskReg preg_tail_neg_2_b32;

    Reg::RegTensor<uint32_t> vreg_sparse_idx_1;
    Reg::RegTensor<uint32_t> vreg_sparse_idx_2;
    Reg::RegTensor<uint32_t> vreg_kv_dim;
    Reg::RegTensor<uint32_t> vreg_b_s2_base_low;
    Reg::RegTensor<uint32_t> vreg_b_s2_base_high;
    Reg::RegTensor<uint32_t> vreg_s2_offset_l_1;
    Reg::RegTensor<uint32_t> vreg_s2_offset_l_2;
    Reg::RegTensor<uint32_t> vreg_mul_overflow_l_1;
    Reg::RegTensor<uint32_t> vreg_mul_overflow_l_2;
    Reg::RegTensor<uint32_t> vreg_total_offset_l_1;
    Reg::RegTensor<uint32_t> vreg_total_offset_h_1;
    Reg::RegTensor<uint32_t> vreg_total_offset_l_2;
    Reg::RegTensor<uint32_t> vreg_total_offset_h_2;
    Reg::RegTensor<uint32_t> vreg_zero;

    Reg::Duplicate(vreg_zero, 0);
    Reg::Duplicate(vreg_kv_dim, kvDim);
    Reg::Duplicate(vreg_b_s2_base_low, bS2BaseLow);
    Reg::Duplicate(vreg_b_s2_base_high, bS2BaseHigh);

    for (; s2Loop > 1;) {
        for (uint16_t i = 0; i < s2Loop - 1; i++) {
            Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_1,
                                                              sparseIdxUb + i * s2_num_per_loop);
            Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_2,
                                                              sparseIdxUb + s2_num_per_reg + i * s2_num_per_loop);
            // * sparseBlockSize
            Reg::Muls(vreg_sparse_idx_1, vreg_sparse_idx_1, sparseBlockSize, preg_all_b32);
            Reg::Muls(vreg_sparse_idx_2, vreg_sparse_idx_2, sparseBlockSize, preg_all_b32);
            // sparseIdx * kvDim (带进位乘法)
            Reg::Mull(vreg_s2_offset_l_1, vreg_mul_overflow_l_1, vreg_sparse_idx_1, vreg_kv_dim, preg_all_b32);
            Reg::Mull(vreg_s2_offset_l_2, vreg_mul_overflow_l_2, vreg_sparse_idx_2, vreg_kv_dim, preg_all_b32);
            // s2_offset + bS2Base (int64 + int64)
            Reg::Add(add_carry_l_1, vreg_total_offset_l_1, vreg_s2_offset_l_1, vreg_b_s2_base_low, preg_all_b32);
            Reg::Add(add_carry_l_2, vreg_total_offset_l_2, vreg_s2_offset_l_2, vreg_b_s2_base_low, preg_all_b32);
            Reg::AddC(add_carry_h_1, vreg_total_offset_h_1, vreg_mul_overflow_l_1, vreg_b_s2_base_high, add_carry_l_1,
                      preg_all_b32);
            Reg::AddC(add_carry_h_2, vreg_total_offset_h_2, vreg_mul_overflow_l_2, vreg_b_s2_base_high, add_carry_l_2,
                      preg_all_b32);
            // 搬出
            Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + i * out_offset_per_loop, vreg_total_offset_l_1, vreg_total_offset_h_1, preg_all_b32);
            Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + out_offset_per_reg + i * out_offset_per_loop, vreg_total_offset_l_2,
                vreg_total_offset_h_2, preg_all_b32);
        }
        break;
    }

    for (uint16_t i = s2Loop - 1; i < s2Loop; i++) {
        Reg::MaskReg preg_tail_1_b32 = Reg::UpdateMask<int32_t>(s2Tail);
        Reg::MaskReg preg_tail_2_b32 = Reg::UpdateMask<int32_t>(s2Tail);
        Reg::Not(preg_tail_neg_1_b32, preg_tail_1_b32, preg_all_b32);
        Reg::Not(preg_tail_neg_2_b32, preg_tail_2_b32, preg_all_b32);

        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_1,
                                                          sparseIdxUb + i * s2_num_per_loop);
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)vreg_sparse_idx_2,
                                                          sparseIdxUb + s2_num_per_reg + i * s2_num_per_loop);
        // * sparseBlockSize
        Reg::Muls(vreg_sparse_idx_1, vreg_sparse_idx_1, sparseBlockSize, preg_tail_1_b32);
        Reg::Muls(vreg_sparse_idx_2, vreg_sparse_idx_2, sparseBlockSize, preg_tail_2_b32);
        // sparseIdx * kvDim (带进位乘法)
        Reg::Mull(vreg_s2_offset_l_1, vreg_mul_overflow_l_1, vreg_sparse_idx_1, vreg_kv_dim, preg_tail_1_b32);
        Reg::Mull(vreg_s2_offset_l_2, vreg_mul_overflow_l_2, vreg_sparse_idx_2, vreg_kv_dim, preg_tail_2_b32);
        // s2_offset + bS2Base (int64 + int64)
        Reg::Add(add_carry_l_1, vreg_total_offset_l_1, vreg_s2_offset_l_1, vreg_b_s2_base_low, preg_tail_1_b32);
        Reg::Add(add_carry_l_2, vreg_total_offset_l_2, vreg_s2_offset_l_2, vreg_b_s2_base_low, preg_tail_2_b32);
        Reg::AddC(add_carry_h_1, vreg_total_offset_h_1, vreg_mul_overflow_l_1, vreg_b_s2_base_high, add_carry_l_1,
                  preg_tail_1_b32);
        Reg::AddC(add_carry_h_2, vreg_total_offset_h_2, vreg_mul_overflow_l_2, vreg_b_s2_base_high, add_carry_l_2,
                  preg_tail_2_b32);
        // 无效值填充-1(0xFFFFFFFF)
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_l_1, invalid_value,
                                                              preg_tail_neg_1_b32);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_h_1, invalid_value,
                                                              preg_tail_neg_1_b32);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_l_2, invalid_value,
                                                              preg_tail_neg_2_b32);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(vreg_total_offset_h_2, invalid_value,
                                                              preg_tail_neg_2_b32);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
            kvPhyAddrUb + i * out_offset_per_loop, vreg_total_offset_l_1, vreg_total_offset_h_1, preg_all_b32);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(
            kvPhyAddrUb + out_offset_per_reg + i * out_offset_per_loop, vreg_total_offset_l_2, vreg_total_offset_h_2,
            preg_all_b32);
    }
}

template <typename T>
__aicore__ inline void GetKVPhyAddrVFBsnd(LocalTensor<uint32_t> kvPhyAddrTensor, LocalTensor<int32_t> sparseIdxTensor,
                                          const uint16_t s2Loop, const uint32_t s2Tail, const uint32_t sparseBlockSize,
                                          const uint32_t kvDim, const uint32_t bS2BaseLow, const uint32_t bS2BaseHigh)
{
    __ubuf__ uint32_t *kv_phy_addr_ub = (__ubuf__ uint32_t *)(kvPhyAddrTensor.GetPhyAddr());
    __ubuf__ int32_t *sparse_idx_ub = (__ubuf__ int32_t *)(sparseIdxTensor.GetPhyAddr());
    GetKVPhyAddrVFBsndImpl<uint32_t>(kv_phy_addr_ub, sparse_idx_ub, s2Loop, s2Tail, sparseBlockSize, kvDim, bS2BaseLow,
                                     bS2BaseHigh);
}

} // namespace AttentionCommon

#endif // SPARSE_FLASH_MLA_GET_KV_PHY_ADDR_VF_H
