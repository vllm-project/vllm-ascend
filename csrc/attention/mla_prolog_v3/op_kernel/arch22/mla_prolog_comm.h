/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mla_prolog_comm.h
 * \brief
 */

#ifndef MLA_PROLOG_COMM_H
#define MLA_PROLOG_COMM_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;

namespace MlaProlog {

template <typename T>
__aicore__ inline T CeilDivT(T dividend, T divisor)
{
    if (divisor == 0) {
        return static_cast<T>(0);
    }
    return (dividend + divisor - 1) / divisor;
}

template <typename T>
__aicore__ inline T Align(T value, T alignment)
{
    return (((alignment) == 0) ? 0 : (((value) + (alignment)-1) / (alignment) * (alignment)));
}

enum class CACHE_MODE : std::uint8_t {
    ND = static_cast<std::uint8_t>(0), // BSND or TND
    PA_BSND = static_cast<std::uint8_t>(1),
    PA_NZ = static_cast<std::uint8_t>(2),
    PA_BLK_BSND = static_cast<std::uint8_t>(3),
    PA_BLK_NZ = static_cast<std::uint8_t>(4),
    PA_BS = static_cast<std::uint8_t>(5)
};

enum class SCENARIO : std::uint8_t {
    RESERVED = static_cast<std::uint8_t>(0),
    NO_QUANT = static_cast<std::uint8_t>(1),
    QUANT = static_cast<std::uint8_t>(2)
};

enum class QUANT_MODE : std::uint8_t {
    NO_QUANT = static_cast<std::uint8_t>(0),
    PARTIAL_QUANT_KV_NO_QUANT = static_cast<std::uint8_t>(1),
    PARTIAL_QUANT_KV_QUANT_PER_CHANNEL = static_cast<std::uint8_t>(2),
    FULL_QUANT_KV_NO_QUANT = static_cast<std::uint8_t>(3),
    FULL_QUANT_KV_QUANT_PER_TENSOR = static_cast<std::uint8_t>(4),
    PARTIAL_QUANT_KV_QUANT_PERTILE = static_cast<std::uint8_t>(5),
    FULL_QUANT_KV_QUANT_PERTILE = static_cast<std::uint8_t>(6)
};

enum class EMPTY_TENSOR_MODE : std::uint8_t {
    NON_EMPTY = static_cast<std::uint8_t>(0),
    EMPTY_CACHE = static_cast<std::uint8_t>(1),
    EMPTY_QUERY = static_cast<std::uint8_t>(2),
};

enum class ACTUAL_SEQ_MODE : std::uint8_t {
    DISABLED = static_cast<std::uint8_t>(0),
    EN_Q_LEN = static_cast<std::uint8_t>(1),
};

enum class SPLIT_M_MODE : std::uint8_t {
    DISABLED = static_cast<std::uint8_t>(0),
    ENABLED = static_cast<std::uint8_t>(1),
};

enum class ROPE_MODE : std::uint8_t {
    INTERLEAVE_HALF = static_cast<std::uint8_t>(0),
    HALF = static_cast<std::uint8_t>(1),
};

constexpr uint64_t BYTE_BLOCK = 32UL;
constexpr uint8_t ALIGN_BLOCK_SIZE = 32;         // 32B对齐
constexpr uint32_t BLOCK_CUBE_SIZE = 16;         // L1上m轴16对齐
constexpr uint32_t REPEAT_STRIDE_UP_BOUND = 256; // repeat stride 不能超过256
constexpr uint32_t REPEAT_BLOCK_BYTE = 256;
constexpr uint32_t FP16_BLOCK_ELEMENT_NUM = ALIGN_BLOCK_SIZE / sizeof(half);
constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = ALIGN_BLOCK_SIZE / sizeof(float);
constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
constexpr uint32_t MAX_UB_SIZE = 192 * 1024; // 最大的UB大小
constexpr uint32_t DIM_HEAD_SIZE_QCQR = 192; // 算力分组方案D + Dr = 192
constexpr uint32_t QR_CORE_NUM = 4;          // 算力分组方案QR占用4核
constexpr uint32_t QC_CORE_NUM = 8;          // 算力分组方案QC占用8核
constexpr uint32_t BF16_AFULLLOAD_MAX_MSIZE = 32; // 计算mmQcQr时，bf16类型的A矩阵在msize小于等于32可以全载L1
constexpr uint32_t INT8_AFULLLOAD_MAX_MSIZE = 64; // 计算mmQcQr时，int8类型的A矩阵在msize小于等于64可以全载L1
constexpr uint32_t ONE_BYTE_TYPE_SIZE = 1;        // 数据类型int8_t大小为1字节
constexpr uint32_t K_STEP_SIZE_32 = 32;           // for move left or right
constexpr uint32_t SHIFTS_UNIT = 4;
constexpr uint32_t UNIT_SIZE = 512;
constexpr uint32_t ROUND_UP_UNIT = 15;       // for round up
constexpr uint32_t MAX_SYNC_FLAG_COUNT = 15; // 同一个flagId的计数器最多设置15次

constexpr int SYNC_MODE_ALL_VEC = 0x0;
constexpr int SYNC_MODE_CUBE_VEC = 0x2;
constexpr int SYNC_MODE_ALL_CUBE = 0x0;

constexpr int FINISH_MM_QR = 0x8; // 算力分组场景
constexpr int FINISH_MM_CQ = 0x6;
constexpr int FINISH_MM_CKVKR = 0x6;
constexpr int FINISH_MM_QCQR = 0x6;
constexpr int FINISH_MM_QC = 0x6; // 算力分组场景
constexpr int FINISH_MM_ALL = 0x7;

constexpr int FINISH_VEC_DEQUANT_QC = 0x6;
constexpr int FINISH_VEC_RMSNORM_CQ = 0x6;
constexpr int FINISH_VEC_CKVKR = 0x9;
constexpr int FINISH_VEC_ALL = 0x7;

constexpr int FINISH_MM_QCQR_SPLIT_N = 0X6;
constexpr int FINISH_VEC_DEQUANT_QC_SPLIT_N = 0X7;
constexpr int FINISH_MM_QN_SPLIT_N = 0X8;

#ifdef ENABLE_DUMP_DATA
#define DO_DUMP_DATA(srcTensor, id, len) AscendC::DumpTensor(srcTensor, id, len)
#else
#define DO_DUMP_DATA(srcTensor, id, len)
#endif

class NoneType {};

// fix <> m
#define L0C_EVENT0 EVENT_ID3
#define L0C_EVENT1 EVENT_ID4

// m <> mte1
#define L0A_EVENT0 EVENT_ID3
#define L0A_EVENT1 EVENT_ID4
#define L0B_EVENT0 EVENT_ID5
#define L0B_EVENT1 EVENT_ID6

// mte2 <> mte1
#define SCALE_EVENT EVENT_ID3
#define A_EVENT0 EVENT_ID4
#define A_EVENT1 EVENT_ID5
#define B_EVENT0 EVENT_ID6
#define B_EVENT1 EVENT_ID7

constexpr uint32_t L1_B_SIZE = 128 * 1024; // 512 / 4
constexpr uint32_t L1_A_SIZE = 128 * 1024; // 512 / 4
constexpr uint32_t L0A_PP_SIZE = 32 * 1024;
constexpr uint32_t L0B_PP_SIZE = 32 * 1024;
constexpr uint32_t L0C_PP_SIZE = 64 * 1024;


/*
                                     非量化             半量化(kv非量化)       半量化(kv量化)       int8全量化(kv非量化)
  int8全量化(kv量化)  半量化(kv per-tile量化)   int8全量化(kv per-tile量化) cacheMode PA_BSND/PA_BLK_BSND
  PA_BSND/PA_BLK_BSND  PA_BSND/PA_BLK_BSND   PA_BSND/PA_BLK_BSND   PA_BSND/PA_BLK_BSND      PA_BSND/BSND/TND
  PA_BSND/BSND/TND /PA_NZ/PA_BLK_NZ       /PA_NZ/PA_BLK_NZ     /PA_NZ/PA_BLK_NZ      /PA_NZ/PA_BLK_NZ /PA_NZ/PA_BLK_NZ
                                  /BSND/TND             /BSND/TND             /BSND/TND            /BSND/TND /BSND/TND
  enableDequantOpt                    false               true/false           true/false             true/false
  true/false                true                       true enableGroupDequantOpt               false true/false
  true/false               false                false                   false                      false quantMode 0 1
  2                      3                    4                       5                          6
  tokenXType(复用mmInputType)       bfloat16_t            bfloat16_t            bfloat16_t              int8_t int8_t
  bfloat16_t                    int8_t WdqType(复用mmInputType)          bfloat16_t            bfloat16_t bfloat16_t
  int8_t               int8_t                bfloat16_t                    int8_t WuqqrType(复用mmQcQrInputType)
  bfloat16_t              int8_t                int8_t                int8_t               int8_t int8_t int8_t
  WukType(复用mmQnInputType)        bfloat16_t            bfloat16_t            bfloat16_t            bfloat16_t
  bfloat16_t              bfloat16_t                  bfloat16_t WdkvkrType(复用mmInputType)       bfloat16_t bfloat16_t
  bfloat16_t              int8_t               int8_t                bfloat16_t                    int8_t
  rmsNormGammaType                  bfloat16_t            bfloat16_t            bfloat16_t            bfloat16_t
  bfloat16_t              bfloat16_t                  bfloat16_t gammaCkvType(复用rmsNormGammaType)bfloat16_t bfloat16_t
  bfloat16_t            bfloat16_t           bfloat16_t              bfloat16_t                  bfloat16_t
  ropeSinCosType                    bfloat16_t            bfloat16_t            bfloat16_t            bfloat16_t
  bfloat16_t              bfloat16_t                  bfloat16_t cosType(复用ropeSinCosType)       bfloat16_t bfloat16_t
  bfloat16_t            bfloat16_t           bfloat16_t              bfloat16_t                  bfloat16_t
  cacheIndexType                      int64_t               int64_t               int64_t               int64_t int64_t
  int64_t                     int64_t kvCacheType                       bfloat16_t            bfloat16_t int8_t
  bfloat16_t             int8_t                  int8_t                      int8_t krCacheType bfloat16_t bfloat16_t
  int8_t              bfloat16_t           bfloat16_t              bfloat16_t                  bfloat16_t deqScaleXType
  /                     /                     /                   float                float                     / float
  deqScaleWdqType                       /                     /                     /                   float float /
  float deqScaleWuqqrType                     /                   float                 float                 float
  float                   float                       float deqScaleWdkvkrType                    / / / float float /
  float quantScaleCkvType                     /                     /                   float                   / float
  /                           / quantScaleCkrType                     /                     /                   float /
  /                       /                           / smoothScaleCqType                     /                   float
  float                 float                float                   float                       float queryOutputType
  bfloat16_t            bfloat16_t              int8_t              bfloat16_t             int8_t bfloat16_t bfloat16_t
  ropeOutputType                    bfloat16_t            bfloat16_t            bfloat16_t            bfloat16_t
  bfloat16_t              bfloat16_t                  bfloat16_t dequantScaleQNopeType                 / / / / float / /
  queryNormType(复用mmQcQrInputType)bfloat16_t              int8_t                int8_t                int8_t int8_t
  int8_t                      int8_t dequantScaleQNormType                 /                   float float float float
  float                       float mmInputType                       bfloat16_t            bfloat16_t bfloat16_t int8_t
  int8_t                bfloat16_t                    int8_t mmCqOutputType                    bfloat16_t bfloat16_t
  bfloat16_t              int32_t              int32_t               bfloat16_t                    int32_t
  mmCkvKrInputType(复用mmInputType) bfloat16_t            bfloat16_t            bfloat16_t              int8_t int8_t
  bfloat16_t                    int8_t mmCkvKrOutputType                 bfloat16_t            bfloat16_t bfloat16_t
  int32_t              int32_t               bfloat16_t                    int32_t mmQcQrInputType bfloat16_t int8_t
  int8_t                int8_t               int8_t                  int8_t                      int8_t mmQcQrOutputType
  bfloat16_t            bfloat16_t            bfloat16_t              int32_t              int32_t bfloat16_t int32_t
  mmQnInputType                     bfloat16_t            bfloat16_t            bfloat16_t            bfloat16_t
  bfloat16_t              bfloat16_t                  bfloat16_t mmQnOutputType                    bfloat16_t bfloat16_t
  bfloat16_t            bfloat16_t           bfloat16_t              bfloat16_t                  bfloat16_t
  rmsNormComputType                   float                 float                 float                 float float
  float                       float rmsNormCqOutputType               bfloat16_t              int8_t int8_t int8_t
  int8_t                  int8_t                      int8_t rmsNormCkvOutputType              bfloat16_t bfloat16_t
  int8_t              bfloat16_t             int8_t                  int8_t                      int8_t ropeComputType
  float                 float                 float                 float                float                   float
  float
*/

template <typename X_T, typename W_T, typename C_T, typename D_S, CACHE_MODE C_M, bool ENABLE_DEQUANT_OPT,
          bool ENABLE_GROUP_COMPUTE_OPT, EMPTY_TENSOR_MODE EMPTY_MODE, ACTUAL_SEQ_MODE SEQ_MODE,
          bool IS_PERTILE = false, uint32_t CV_RATIO = 2, bool ENABLE_ROPE = true, typename... Args>
struct MLAPType {
    using mmQcQrInputType = W_T;
    using mmInputType = X_T;          // tokenX的类型与weight的类型一致
    using mmQnInputType = bfloat16_t; // matmul计算Qn的输入类型
    using mmCqOutputType = typename std::conditional<std::is_same<X_T, int8_t>::value, int32_t,
                                                     bfloat16_t>::type; // matmul计算Cq的输出类型
    using mmQcQrOutputType = typename std::conditional<std::is_same<W_T, int8_t>::value, int32_t,
                                                       bfloat16_t>::type; // matmul计算QcQr的输出类型
    using mmQnOutputType = bfloat16_t;                                    // matmul计算Qn的输出类型
    using mmCkvKrOutputType = typename std::conditional<std::is_same<X_T, int8_t>::value, int32_t,
                                                        bfloat16_t>::type; // matmul计算CkvKr的输出类型
    using rmsNormGammaType = bfloat16_t;                                   // gamma的输入类型
    using rmsNormComputType = float;
    using rmsNormCkvOutputType = C_T;
    using rmsNormCqOutputType = W_T;
    using ropeSinCosType = bfloat16_t; // sin cos的输入类型
    using ropeComputType = float;
    using ropeOutputType = bfloat16_t;
    using dequantScaleQNopeType = float; // dequantScaleQNope的类型
    using dequantScaleQNormType = float; // dequantScaleQNorm的类型
    using dequantScaleType = float;
    using kvCacheType = C_T; // kvcache的类型
    using krCacheType =
        typename std::conditional<((std::is_same<X_T, int8_t>::value && std::is_same<C_T, int8_t>::value) ||
                                   IS_PERTILE),
                                  bfloat16_t, C_T>::type; // krcache的类型

    static constexpr CACHE_MODE cacheMode = C_M;
    static constexpr bool enableGroupComputeOpt = ENABLE_GROUP_COMPUTE_OPT;
    static constexpr bool enableRope = ENABLE_ROPE;
    static constexpr bool enableDequantOpt = ENABLE_DEQUANT_OPT;
    static constexpr uint32_t cvRatio = CV_RATIO; // 默认C:V 1:2
    static constexpr EMPTY_TENSOR_MODE emptyMode = EMPTY_MODE;
    static constexpr ACTUAL_SEQ_MODE actualSeqMode = SEQ_MODE;
    static constexpr bool isPertile = IS_PERTILE;
};

struct CastQcQrSplitNParams {
    int64_t mmQnPreCastOffset;
    int64_t mmQnPreCastResOffset;
    uint32_t inputOffset;
    uint32_t outputOffset;
    uint32_t srcStride;
    uint32_t dstStride;
};

struct DequantQcQrSplitNParams {
    int64_t mmQnPreDequantOffset;
    int64_t mmQnPreDequantResOffset;
    uint32_t inputOffset;
    uint32_t outputOffset;
    uint32_t srcStride;
    uint32_t dstStride;
};

struct RopeQrSplitNParams {
    int64_t ropeQrOffset;
    int64_t ropeQrResOffset;
    uint32_t inputOffsetRope;
    uint32_t deqScaleOffset;
    uint32_t outputOffsetRope;
    int64_t ropeStride;
    uint32_t ropeDstStride;
    uint32_t deQuantScaleCqOffset;
    uint32_t sinCosOffset;
    uint32_t ropeCnt;
};

struct CkvkrParams {
    int64_t tokenIndex;
    int64_t offset;
    int64_t curVecTokenIdx;
    int64_t rowsInCurBatch;
    int64_t cacheOffset;
    int64_t nextBatchOffset;
};

struct UsedBlockParams {
    uint32_t blockStartIdx;
    uint32_t blockEndIdx;
};

struct AivOffset {
    int64_t curVecToken = 0;
    int64_t curBlockTokenOffset = 0;
    int64_t rmsNormCqOffset = 0;
    int64_t rmsNormCqResOffset = 0;
    int64_t rmsNormCkvOffset = 0;
    int64_t ropeKrOffset = 0;
    int64_t mmQnPreDequantOffset = 0;
    int64_t mmQnPreDequantResOffset = 0;
    int64_t ropeQrOffset = 0;
    int64_t ropeQrResOffset = 0;
    int64_t ropeQrSplitNOffset = 0;
    int64_t ropeQrResSplitNOffset = 0;
    int64_t qcScaleOffsetSplitN = 0;
};

struct AicOffset {
    int64_t weightDqOffset = 0;
    int64_t dequantScaleWDqOffset = 0;
    int64_t cqResOffset = 0;
    int64_t rmsNormCqResOffset = 0;
    int64_t dequantScaleWDkvKrOffset = 0;
    int64_t weightDkvKrOffset = 0;
    int64_t ckvKrResOffset = 0;
    int64_t weightUqQrOffset = 0;
    int64_t weightUqOffset = 0;
    int64_t weightQrOffset = 0;
    int64_t qcQrResOffset = 0;
    int64_t dequantScaleCqOffset = 0;
    int64_t qCResOffset = 0;
    int64_t qRResOffset = 0;
    int64_t qcOffset = 0;
    int64_t weightUkOffset = 0;
    int64_t dequantScaleWuqqrOffset = 0;
    int64_t qnResOffset = 0;
};

struct MMBufParams {
    uint32_t aL1BufIter = 0;
    uint32_t bL1BufIter = 0;
    TBuffAddr aL1BufAddr;
    TBuffAddr bL1BufAddr;
    uint32_t aL0BufIter = 0;
    uint32_t bL0BufIter = 0;
    uint32_t cL0BufIter = 0;
    TBuffAddr aL0BufAddr;
    TBuffAddr bL0BufAddr;
    TBuffAddr cL0BufAddr;
};

struct MMParams {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t orgM;
    uint32_t orgN;
    uint32_t orgKa;
    uint32_t orgKb;
    uint32_t orgKc;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t stepK;
    uint32_t needSetOrgShape;
    uint32_t kL1StepSize;
};

template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void WaitAllCore(uint16_t flagId)
{
    AscendC::CrossCoreSetFlag<modeId, pipe>(flagId);
    AscendC::CrossCoreWaitFlag(flagId);
}
} // namespace MlaProlog
#endif
