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
 * \file msa_index_score_common.h
 * \brief host / device 共用的编译期常量与 TilingKey 定义。
 *
 * 该头文件不依赖 AscendC，可被 op_host 的 tiling 直接 include。
 */

#ifndef MSA_INDEX_SCORE_COMMON_H
#define MSA_INDEX_SCORE_COMMON_H

#include <cstdint>

namespace MsaIndexScoreNs {

// 一个 sparse block（同时也是一个 paged-cache page）的 token 数。
// 该值同时是 BlockMmad 的 L1TileShape::N，必须是编译期常量。
constexpr uint32_t MSA_BLOCK_SIZE = 128;

// 一个 M-tile 的行数。行 = 请求内 (token, head) 扁平索引，head 为低位。
constexpr uint32_t MSA_ROW_TILE_M = 128;

// Cube L0A fractal / 向量归约对齐行数。末尾短 tile 不足该值时向前重叠。
constexpr uint32_t MSA_M_ALIGN = 16;

// BlockMmad 的 L1TileShape::K。headDim <= 该值时 kLoop = 1，
// headDim 更大时 BlockMmad 内部自动多轮 K 迭代。
constexpr uint32_t MSA_K_TILE = 128;

// 一个 S workspace tile 覆盖的 sparse block 数（AIC 每产出这么多 block 同步一次 AIV）。
// 取 8 使得每行的归约结果恰为 8 个 fp32 = 32B，输出可以按 32B 对齐整块写回。
constexpr uint32_t MSA_BLOCKS_PER_STILE = 8;

// S workspace 轮转级数。AIC 领先 AIV 的未消费 tile 数 <= MSA_WORKSPACE_STAGES - 1。
// 5 -> 8：AIV 乒乓化后服务速度提升，加深缓冲让 AIC 几乎不因 AIV 停顿（wait_id2→0）。
// 非量化路径 S 为 fp16，8 级 × 256KB × 20 核 = 40MB，仍可整体驻留 L2。
constexpr uint32_t MSA_WORKSPACE_STAGES = 8;

// 一个 S workspace tile 的元素数：[MSA_ROW_TILE_M, MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE]
// 非量化路径元素为 fp16（fixpipe F322F16 直接写），int8 路径为 fp32。
constexpr uint32_t MSA_STILE_ELEM_NUM = MSA_ROW_TILE_M * MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE;
// 非量化 S tile 的字节数（fp16）。
constexpr uint32_t MSA_STILE_BYTES_FP16 = MSA_STILE_ELEM_NUM * sizeof(uint16_t);
// int8 路径 S tile 的字节数（fp32）。
constexpr uint32_t MSA_STILE_BYTES_FP32 = MSA_STILE_ELEM_NUM * sizeof(float);

// int8 路径：每个 AIC 私有一块 K cast 暂存，容纳一个 S-tile 内全部 page
// （AIV 先 cast，AIC 再 Mmad，避免逐 page 细粒度同步）。
// 单页布局 [blockSize, headDim] 行主序，按 j*P*D 排列。
constexpr uint32_t MSA_K_SCRATCH_ELEM_NUM = MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE * MSA_K_TILE;

// AIV 单次归约处理的行数。受 level-0 向量 API repeatTimes(uint8_t) 上限约束：
// MSA_ROWS_PER_PASS * MSA_BLOCKS_PER_STILE <= 255，且乘积需为 8 的倍数。
constexpr uint32_t MSA_ROWS_PER_PASS = 16;

// 归约展开后的逻辑行数：把 [rows, BLOCKS, BLOCK_SIZE] 视作 [rows*BLOCKS, BLOCK_SIZE]。
constexpr uint32_t MSA_REDUCE_ROWS = MSA_ROWS_PER_PASS * MSA_BLOCKS_PER_STILE;

// 向量单元一个 repeat 处理的 fp32 数 / 一个 32B datablock 的 fp32 数。
constexpr uint32_t MSA_FP32_PER_REPEAT = 64;
constexpr uint32_t MSA_FP32_PER_BLOCK = 8;

// score 输出末维的对齐粒度。
constexpr uint32_t MSA_SCORE_STRIDE_ALIGN = 16;

// 不可见 block 的填充值（-inf），使其在下游 topk 中必然落选。
constexpr float MSA_FILL_VALUE = -3.4028234663852886e+38F;

// local_mask 强制高分（对齐 Triton decode：init=1e30，local=1e29；local 覆盖 init）。
constexpr float MSA_LOCAL_SCORE_INIT = 1.0e30F;
constexpr float MSA_LOCAL_SCORE_LOCAL = 1.0e29F;

// sparse_mode：0=defaultMask（无因果截断）；3=rightDownCausal（与 LightningIndexer 一致）。
constexpr uint32_t MSA_SPARSE_MODE_DEFAULT = 0;
constexpr uint32_t MSA_SPARSE_MODE_RIGHT_DOWN = 3;

// MiniMax 默认强制选块数（接口文档未暴露 attr；P0 以常量对齐 HF/Triton 常见配置）。
constexpr uint32_t MSA_DEFAULT_INIT_BLOCKS = 0;
constexpr uint32_t MSA_DEFAULT_LOCAL_BLOCKS = 1;

// key 数据排布。BBND / BNBD 为 PageAttention；TND 为 packed varlen，无 block_table。
constexpr uint32_t MSA_KEY_LAYOUT_BBND = 0; // [block_num, block_size, N2, D]
constexpr uint32_t MSA_KEY_LAYOUT_BNBD = 1; // [block_num, N2, block_size, D]
constexpr uint32_t MSA_KEY_LAYOUT_TND = 2;  // [T2, N2, D]

// atten_mask 压缩下三角模板边长（sparse_mode=3 时校验）。
constexpr uint32_t MSA_ATTEN_MASK_SIZE = 2048;

// FFTS cross-core flag id，合法范围 0~7（8/9/10 为保留 barrier）。
constexpr uint16_t MSA_FLAG_S_READY = 1;
constexpr uint16_t MSA_FLAG_S_READY_REVERSE = 2;
// int8 路径：AIV cast K→scratch 完成后通知 AIC。
constexpr uint16_t MSA_FLAG_K_READY = 3;
constexpr uint16_t MSA_FLAG_K_READY_REVERSE = 4;

} // namespace MsaIndexScoreNs

// TilingKey：按 query dtype × key 是否 int8 分支。
// 必须是宏而非 constexpr —— TILING_KEY_IS() 在算子预编译阶段按文本解析，只接受数值常量或宏。
#define MSA_TILING_KEY_BF16 0
#define MSA_TILING_KEY_FP16 1
#define MSA_TILING_KEY_BF16_INT8 2
#define MSA_TILING_KEY_FP16_INT8 3

#endif // MSA_INDEX_SCORE_COMMON_H
