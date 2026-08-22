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
 * \file incre_flash_attention_pub.h
 * \brief
 */

#ifndef INCRE_FLASH_ATTENTION_PUB_H
#define INCRE_FLASH_ATTENTION_PUB_H


typedef struct {
  uint32_t G;
  uint32_t D;
  uint32_t S;   // sinner
  uint32_t M1;  // baseM
  uint32_t N1;
  uint32_t K1;
  uint32_t M2;
  uint32_t N2;
  uint32_t K2;
} IFAProfile;
template <typename T>
__aicore__ inline uint64_t MAX(T x, T y)
{
    return (x > y) ? static_cast<uint64_t>(x) : static_cast<uint64_t>(y);
};
static constexpr IFAProfile IFA_PROFILE_DEFAULT = {0, 0, 0, 0, 0, 0, 0, 0, 0};
static constexpr IFAProfile IFA_PROFILE_D64 = {16, 64, 512, 16, 256, 64, 16, 64, 256};
static constexpr IFAProfile IFA_PROFILE_D128 = {16, 128, 256, 16, 128, 128, 16, 128, 128};
static constexpr IFAProfile IFA_PROFILE_D256 = {16, 256, 128, 16, 128, 128, 16, 128, 128};
static constexpr IFAProfile IFA_PROFILE_D512 = {16, 512, 64, 16, 64, 256, 16, 256, 64};
static constexpr IFAProfile IFA_PROFILE_D64_PSE_OR_BAND = {8, 64, 512, 16, 256, 64, 16, 64, 256};
// 以下涉及S2细分，预留
static constexpr IFAProfile IFA_PROFILE_D64_S256 = {16, 64, 256, 16, 256, 64, 16, 64, 256};
static constexpr IFAProfile IFA_PROFILE_D64_S128 = {16, 64, 128, 16, 128, 64, 16, 64, 128};
static constexpr IFAProfile IFA_PROFILE_D128_S128 = {16, 128, 128, 16, 128, 128, 16, 128, 128};

constexpr uint32_t SHARED_CO1_BUFFER_SIZE_KB = 64;
constexpr uint32_t BYTES_PER_KB = 1024;
constexpr uint32_t FA_SOUTER_CONST_16 = 16;
constexpr uint32_t FA_SINNER_CONST_1024  = 1024;
constexpr uint32_t FA_SINNER_CONST_512  = 512;
constexpr uint32_t FA_SINNER_CONST_256  = 256;
constexpr uint32_t FA_SINNER_CONST_128  = 128;
constexpr uint32_t FA_DSIZE_CONST_64  = 64;
constexpr uint32_t FA_DSIZE_CONST_128  = 128;
constexpr uint32_t FA_DSIZE_CONST_256  = 256;
constexpr uint32_t FA_DSIZE_CONST_512  = 512;

// 生成matmul MDL模板配置
__aicore__ static constexpr MatmulConfig GenConfMM1MDL(const IFAProfile& ifa) {
  MatmulConfig mm1Cfg{};
  mm1Cfg.doNorm = false;
  mm1Cfg.doBasicBlock = false;
  mm1Cfg.doMultiDataLoad = true;
  mm1Cfg.basicM = ifa.M1;
  mm1Cfg.basicN = ifa.N1;
  mm1Cfg.basicK = ifa.K1;
  mm1Cfg.intrinsicsCheck = false;  
  mm1Cfg.isNBatch = false;
  mm1Cfg.enVecND2NZ = false;
  mm1Cfg.doSpecialBasicBlock = false;
  mm1Cfg.doMTE2Preload = 0;
  mm1Cfg.singleCoreM = ifa.G;
  mm1Cfg.singleCoreN = ifa.S;
  mm1Cfg.singleCoreK = ifa.D;
  mm1Cfg.stepM = 0;
  mm1Cfg.stepN = 0;
  mm1Cfg.baseMN = 0;
  mm1Cfg.singleCoreMN = 0;
  mm1Cfg.enUnitFlag = true;
  mm1Cfg.isPerTensor = false;
  mm1Cfg.hasAntiQuantOffset = false;
  mm1Cfg.doIBShareNorm = false;
  mm1Cfg.doSpecialMDL = false;
  mm1Cfg.enableInit = false;
  mm1Cfg.batchMode = BatchMode::BATCH_LESS_THAN_L1;
  mm1Cfg.enableEnd = true;
  mm1Cfg.enableGetTensorC = false;
  mm1Cfg.enableSetOrgShape = false;  // 应该设置为false
  mm1Cfg.enableSetBias = false;       // 应该设置为false
  mm1Cfg.enableSetTail = true;
  mm1Cfg.enableQuantVector = false;
  mm1Cfg.enableSetDefineData = false;
  mm1Cfg.iterateMode = IterateMode::ITERATE_MODE_ALL;
  mm1Cfg.enableReuse = true;
  mm1Cfg.enableUBReuse = false;
  mm1Cfg.enableL1CacheUB = false;
  mm1Cfg.intraBlockPartSum = false;
  mm1Cfg.iterateOrder = IterateOrder::UNDEF;
  mm1Cfg.scheduleType = ScheduleType ::INNER_PRODUCT;
  mm1Cfg.enableDoubleCache =false;
  mm1Cfg.isBiasBatch = false;
  mm1Cfg.enableStaticPadZeros = false;
  mm1Cfg.isA2B2Shared = false;
  mm1Cfg.isCO1Shared = false;
  mm1Cfg.sharedCO1BufferSize = SHARED_CO1_BUFFER_SIZE_KB * BYTES_PER_KB;
  return mm1Cfg;
}

// 生成matmul配置
static constexpr MatmulConfig GenConfMM1(const IFAProfile& ifa) {
  MatmulConfig mm1Cfg{};
  mm1Cfg.doNorm = true;
  mm1Cfg.doBasicBlock = false;
  mm1Cfg.doMultiDataLoad = false;
  mm1Cfg.basicM = ifa.M1;
  mm1Cfg.basicN = ifa.N1;
  mm1Cfg.basicK = ifa.K1;
  mm1Cfg.intrinsicsCheck = false;  
  mm1Cfg.isNBatch = false;
  mm1Cfg.enVecND2NZ = false;
  mm1Cfg.doSpecialBasicBlock = false;
  mm1Cfg.doMTE2Preload = 0;
  mm1Cfg.singleCoreM = ifa.G;
  mm1Cfg.singleCoreN = ifa.S;
  mm1Cfg.singleCoreK = ifa.D;
  mm1Cfg.stepM = 0;
  mm1Cfg.stepN = 0;
  mm1Cfg.baseMN = 0;
  mm1Cfg.singleCoreMN = 0;
  mm1Cfg.enUnitFlag = true;
  mm1Cfg.isPerTensor = false;
  mm1Cfg.hasAntiQuantOffset = false;
  mm1Cfg.doIBShareNorm = false;
  mm1Cfg.doSpecialMDL = false;
  mm1Cfg.enableInit = false;
  mm1Cfg.batchMode = BatchMode::BATCH_LESS_THAN_L1;
  mm1Cfg.enableEnd = true;
  mm1Cfg.enableGetTensorC = false;
  mm1Cfg.enableSetOrgShape = false;  // 应该设置为false
  mm1Cfg.enableSetBias = false;       // 应该设置为false
  mm1Cfg.enableSetTail = true;
  mm1Cfg.enableQuantVector = false;
  mm1Cfg.enableSetDefineData = false;
  mm1Cfg.iterateMode = IterateMode::ITERATE_MODE_ALL;
  mm1Cfg.enableReuse = true;
  mm1Cfg.enableUBReuse = false;
  mm1Cfg.enableL1CacheUB = false;
  mm1Cfg.intraBlockPartSum = false;
  mm1Cfg.iterateOrder = IterateOrder::UNDEF;
  mm1Cfg.scheduleType = ScheduleType ::INNER_PRODUCT;
  mm1Cfg.enableDoubleCache =false;
  mm1Cfg.isBiasBatch = false;
  mm1Cfg.enableStaticPadZeros = false;
  mm1Cfg.isA2B2Shared = false;
  mm1Cfg.isCO1Shared = false;
  mm1Cfg.sharedCO1BufferSize = SHARED_CO1_BUFFER_SIZE_KB * BYTES_PER_KB;
  return mm1Cfg;
}

// 生成matmul配置
static constexpr MatmulConfig GenConfMM2(const IFAProfile& ifa) {
  MatmulConfig mm2Cfg{};
  mm2Cfg.doNorm = true;
  mm2Cfg.doBasicBlock = false;
  mm2Cfg.doMultiDataLoad = false;
  mm2Cfg.basicM = ifa.M2;
  mm2Cfg.basicN = ifa.N2;
  mm2Cfg.basicK = ifa.K2;
  mm2Cfg.intrinsicsCheck = false;
  mm2Cfg.isNBatch = false;
  mm2Cfg.enVecND2NZ = false;
  mm2Cfg.doSpecialBasicBlock = false;
  mm2Cfg.doMTE2Preload = 0;
  mm2Cfg.singleCoreM = ifa.G;
  mm2Cfg.singleCoreN = ifa.D;
  mm2Cfg.singleCoreK = ifa.S;
  mm2Cfg.stepM = 0;
  mm2Cfg.stepN = 0;
  mm2Cfg.baseMN = 0;
  mm2Cfg.singleCoreMN = 0;
  mm2Cfg.enUnitFlag = true;
  mm2Cfg.isPerTensor = false;
  mm2Cfg.hasAntiQuantOffset = false;
  mm2Cfg.doIBShareNorm = false;
  mm2Cfg.doSpecialMDL = false;
  mm2Cfg.enableInit = false;
  mm2Cfg.batchMode = BatchMode::BATCH_LESS_THAN_L1;
  mm2Cfg.enableEnd = true;
  mm2Cfg.enableGetTensorC = false;
  mm2Cfg.enableSetOrgShape = false; // 应该设置为false
  mm2Cfg.enableSetBias = false;     // 应该设置为false
  mm2Cfg.enableSetTail = true;
  mm2Cfg.enableQuantVector = false;
  mm2Cfg.enableSetDefineData = false;
  mm2Cfg.iterateMode = IterateMode::ITERATE_MODE_ALL;
  mm2Cfg.enableReuse = true;
  mm2Cfg.enableUBReuse = false;
  mm2Cfg.enableL1CacheUB = false;
  mm2Cfg.intraBlockPartSum = false;
  mm2Cfg.iterateOrder = IterateOrder::UNDEF;
  mm2Cfg.scheduleType = ScheduleType ::INNER_PRODUCT;
  mm2Cfg.enableDoubleCache = false;
  mm2Cfg.isBiasBatch = false;
  mm2Cfg.enableStaticPadZeros = false;
  mm2Cfg.isA2B2Shared = false;
  mm2Cfg.isCO1Shared = false;
  mm2Cfg.sharedCO1BufferSize = SHARED_CO1_BUFFER_SIZE_KB * BYTES_PER_KB;
  return mm2Cfg;
}

enum class LAYOUT { BSH = 0, BNSD };

enum class INPUTKVTYPE {
  FP16 = 0,
  BF16 = 2,
  INT8 = 3,
  INT4 = 4,
  HIF8 = 5,
  FP8E4M3 = 7,
  FP4E2M1 = 8
};

enum class ANTIQUANTMODE : uint8_t {
  PER_CHANNEL = 0, // enable per-channel antiquant mode，include per-tensor
  PER_TOKEN = 1,  // enable per-token antiquant mode
  K_PER_CHANNEL_V_PER_TOKEN = 2, // enable split antiquant mode, k per-channel and v per-token
  PER_TOKEN_HEAD = 3, // enable both per-token and per-head antiquant mode
  PER_TOKEN_PAGE_ATTENTION = 4, // enable per-token antiquant mode, and enable PA for memory management
  PER_TOKEN_HEAD_PAGE_ATTENTION = 5, // enable both per-token and per-head antiquant mode, and enable PA for memory management
};

template <typename Q_T, typename KV_T, typename OUT_T, typename ORIGIN_T, const bool PAGE_ATTENTION = false,
          const bool FLASH_DECODE = false, LAYOUT LAYOUT_T = LAYOUT::BSH, const uint8_t ANTIQUANT_MODE = 0,
          INPUTKVTYPE KVTYPE_T = INPUTKVTYPE::INT8, const IFAProfile& PROFILE = IFA_PROFILE_DEFAULT,
          const bool ATTEN_MASK = false, const bool PSE_SHIFT = false, typename... Args>
struct IFAType {
  using queryType = Q_T;
  using kvType = KV_T;
  using outputType = OUT_T;
  using orginalType = ORIGIN_T;
  static constexpr bool pageAttention = PAGE_ATTENTION;
  static constexpr bool flashDecode = FLASH_DECODE;
  static constexpr LAYOUT layout = LAYOUT_T;
  static constexpr ANTIQUANTMODE antiquantMode = static_cast<ANTIQUANTMODE>(ANTIQUANT_MODE);
  static constexpr INPUTKVTYPE inputKVType = KVTYPE_T;
  static constexpr IFAProfile profile = PROFILE;
  static constexpr bool attenMask = ATTEN_MASK;
  static constexpr bool pseShift = PSE_SHIFT;
};

constexpr uint32_t FP16_ONE_BLOCK_SIZE = 16;
constexpr uint32_t FP32_ONE_BLOCK_SIZE = 8;
constexpr float FLOAT_ZERO = 0;
constexpr float FLOAT_MAX = 3e+99;
constexpr float FLOAT_MIN_INF = -3.40E+38;
constexpr uint64_t BYTE_BLOCK = 32UL;
constexpr uint32_t REPEAT_BLOCK_BYTE = 256;

constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;

constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
constexpr uint32_t BUFFER_SIZE_BYTE_64K = 65536;
constexpr uint32_t BUFFER_SIZE_BYTE_128K = 131072;
constexpr uint32_t BUFFER_SIZE_BYTE_256K = 262144;
constexpr uint32_t MAX_UINT16 = 65535;
constexpr float SOFTMAX_MIN_NUM = -2e38;

template <typename T>
__aicore__ inline T ALIGN(T num, T rnd) {
  return ((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd));
}

#endif