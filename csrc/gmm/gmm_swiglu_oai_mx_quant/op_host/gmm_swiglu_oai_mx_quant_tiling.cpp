/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// MX TensorAPI tiling specialized from ops-transformer PR 16147 (425d79b48be4).
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tensor_api_tiling_data.h"

namespace optiling {
namespace {
using GroupedMatmulSwigluQuantV2TensorApi::GMMSwigluQuantV2TensorApiTilingData;
constexpr uint64_t kDb = 2;
constexpr uint64_t kMxGroup = 32;
constexpr uint64_t kBandwidthBytes = 64 * 1024;
constexpr uint32_t kMaxScaleFactor = 127;
constexpr uint64_t kMinFp8TokensPerExpert = 513;
constexpr size_t kWorkspace = 16 * 1024 * 1024;
constexpr uint32_t kMxPerGroupMode = 8;
constexpr uint32_t kOai = 2;
// TRANS_B is the only template argument with two choices.
constexpr uint32_t kFP4TilingKey = 1;
constexpr uint32_t kFP8TilingKey = 0;

uint64_t CeilDiv(uint64_t x, uint64_t y) { return (x + y - 1) / y; }

uint64_t Align(uint64_t x, uint64_t step) { return CeilDiv(x, step) * step; }

uint64_t SelectDepth(uint64_t free, uint64_t perDepth, uint64_t initial, uint64_t baseKBytes) {
  if (initial == 1) {
    uint64_t depth = 1;
    while (depth * 2 * perDepth < free) {
      depth *= 2;
    }
    return depth;
  }
  if (perDepth > kDb * kBandwidthBytes) {
    return initial;
  }
  uint64_t scale = free / perDepth;
  while (scale * baseKBytes > 512 && (scale * baseKBytes) % 512 != 0) {
    --scale;
  }
  if ((scale * baseKBytes) % 512 != 0 && scale * baseKBytes >= 256) {
    scale = 256 / baseKBytes;
  }
  return initial * std::max<uint64_t>(scale, 1);
}

uint64_t Fp4Depth(uint64_t tile, uint64_t k, uint64_t baseK, uint64_t baseKBytes) {
  const auto depth = Align(CeilDiv(kBandwidthBytes, tile), 256) / baseKBytes * kDb;
  uint64_t rounded = 2;
  while (rounded < depth) {
    rounded *= 2;
  }
  return std::min(rounded, CeilDiv(k, baseK) * kDb);
}

struct Input {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  uint64_t experts;
  bool fp4;
};

struct Memory {
  uint64_t l1;
  uint64_t l0c;
};

struct L1 {
  uint64_t baseM;
  uint64_t fullBaseN;
  uint64_t baseK;
  uint64_t stepA;
  uint64_t stepB;
  uint64_t depthA;
  uint64_t depthB;
  uint64_t scaleA;
  uint64_t scaleB;
  uint64_t tileA;
  uint64_t tileB;
};

uint32_t ScaleFactor(uint64_t k, uint64_t stepK, uint64_t scaleSize, bool fp4) {
  const auto factor = fp4 ? CeilDiv(k, stepK) : k / stepK;
  return static_cast<uint32_t>(
      std::min(std::max<uint64_t>(1, factor), std::min<uint64_t>(kBandwidthBytes / scaleSize, kMaxScaleFactor)));
}

bool SelectScales(const Input& input, const Memory& memory, const L1& t, uint64_t& scaleK) {
  auto a = ScaleFactor(input.k, t.stepA * t.baseK, t.scaleA, input.fp4);
  auto b = ScaleFactor(input.k, t.stepB * t.baseK, t.scaleB, input.fp4);
  const auto bytesA = t.depthA * t.scaleA;
  const auto bytesB = t.depthB * t.scaleB;
  const auto used = t.depthA * t.tileA + t.depthB * t.tileB;
  if (used >= memory.l1) {
    return false;
  }
  auto free = memory.l1 - used;
  if (input.fp4) {
    const auto init = static_cast<uint32_t>(free / (std::max(t.depthA, t.depthB) * (t.scaleA + t.scaleB)));
    if (a == 0 || b == 0 || init == 0) {
      return false;
    }
    if (a > init && b > init) {
      a = static_cast<uint32_t>(t.depthA >= t.depthB ? init : init * t.depthB / t.depthA);
      b = static_cast<uint32_t>(t.depthA >= t.depthB ? init * t.depthA / t.depthB : init);
    }
    if (a > kMaxScaleFactor || b > kMaxScaleFactor) {
      return false;
    }
  } else {
    const auto init = static_cast<uint32_t>(free / (bytesA + bytesB));
    if (a <= init && b > init) {
      b = std::min(b, static_cast<uint32_t>((free - a * bytesA) / bytesB));
    } else if (b <= init && a > init) {
      a = std::min(a, static_cast<uint32_t>((free - b * bytesB) / bytesA));
    } else if (a > init && b > init) {
      free -= init * (bytesA + bytesB);
      const auto extraA = std::min(a - init, static_cast<uint32_t>(free / bytesA));
      const auto extraB = std::min(b - init, static_cast<uint32_t>(free / bytesB));
      a = init + (extraA >= extraB ? extraA : 0);
      b = init + (extraA < extraB ? extraB : 0);
    }
  }
  // The original V3 host serializes the scale factors in the low two bytes.
  const auto packed = (1U << 24) + (1U << 16) + (b << 8) + a;
  scaleK = std::min(input.k, std::max((packed & 0xFFU) * t.stepA, ((packed >> 8) & 0xFFU) * t.stepB) * t.baseK);
  return scaleK > 0;
}

bool FillMatmul(const Input& input, const Memory& memory,
                GroupedMatmulSwigluQuantV2TensorApi::GMMTensorApiMMTiling& mm) {
  L1 t{};
  const uint64_t packing = input.fp4 ? 2 : 1;
  // V3 caps M/N before sizing L1, and sizes L1 for both gate and up.
  t.baseM = Align(std::min<uint64_t>(input.m, 256), 16);
  t.fullBaseN = Align(std::min<uint64_t>(input.n, 128), 16) * 2;
  t.baseK = Align(Align(std::min(128 * packing, input.k), 32 * packing), 64);
  const auto baseKBytes = t.baseK / packing;
  t.tileA = t.baseM * baseKBytes;
  t.tileB = t.fullBaseN * baseKBytes;
  t.scaleA = CeilDiv(t.baseK, kMxGroup) * t.baseM;
  t.scaleB = CeilDiv(t.baseK, kMxGroup) * t.fullBaseN;
  const auto tileL1 = t.tileA + t.tileB + t.scaleA + t.scaleB;
  if (memory.l1 < tileL1) {
    return false;
  }

  const auto init = SelectDepth(memory.l1, tileL1, 1, baseKBytes);
  if (input.fp4) {
    t.depthA = Fp4Depth(std::min(input.m, t.baseM), input.k, t.baseK, baseKBytes);
    t.depthB = Fp4Depth(std::min(input.n, t.fullBaseN), input.k, t.baseK, baseKBytes);
    if (t.depthA * t.tileA + t.depthB * t.tileB + std::max(t.depthA, t.depthB) * (t.scaleA + t.scaleB) > memory.l1) {
      t.depthA = init;
      t.depthB = init;
    }
  } else {
    const auto free = memory.l1 - init * tileL1;
    const auto depthA = SelectDepth(free, (t.tileA + t.scaleA) * init, init, baseKBytes);
    const auto depthB = SelectDepth(free, (t.tileB + t.scaleB) * init, init, baseKBytes);
    t.depthA = std::max(depthA, depthB);
    t.depthB = t.depthA;
    if (t.depthA * tileL1 > memory.l1) {
      t.depthA = depthA >= depthB ? depthA : init;
      t.depthB = depthA < depthB ? depthB : init;
    }
  }

  const auto kTiles = CeilDiv(input.k, t.baseK);
  t.stepA = std::min(kTiles, std::max<uint64_t>(1, t.depthA / kDb));
  t.stepB = std::min(kTiles, std::max<uint64_t>(1, t.depthB / kDb));
  if (t.stepA >= t.stepB && t.stepA * t.baseK < input.k) {
    t.stepA = t.stepA / t.stepB * t.stepB;
  }
  if (t.stepB > t.stepA && t.stepB * t.baseK < input.k) {
    t.stepB = t.stepB / t.stepA * t.stepA;
  }
  t.depthA = t.stepA * kDb;
  t.depthB = t.stepB * kDb;

  uint64_t scaleK = 0;
  if (!SelectScales(input, memory, t, scaleK)) {
    return false;
  }
  mm.m = static_cast<uint32_t>(input.m);
  mm.n = static_cast<uint32_t>(input.n);
  mm.k = static_cast<uint32_t>(input.k);
  mm.baseM = static_cast<uint32_t>(t.baseM);
  mm.baseN = static_cast<uint32_t>(t.fullBaseN / 2);
  mm.baseK = static_cast<uint32_t>(t.baseK);
  mm.kAL1 = static_cast<uint32_t>(t.stepA * t.baseK);
  mm.kBL1 = static_cast<uint32_t>(t.stepB * t.baseK);
  mm.scaleKAL1 = mm.scaleKBL1 = static_cast<uint32_t>(scaleK);
  mm.dbL0C = t.baseM * t.fullBaseN * sizeof(float) * kDb <= memory.l0c ? kDb : 1;
  return true;
}

bool GetInput(gert::TilingContext* context, Input& input) {
  const auto* x = context->GetInputShape(0);
  const auto* xScale = context->GetInputShape(1);
  const auto* groups = context->GetInputShape(2);
  const auto* w = context->GetDynamicInputShape(3, 0);
  const auto* wScale = context->GetDynamicInputShape(4, 0);
  const auto* xDesc = context->GetInputDesc(0);
  if (!x || !xScale || !groups || !w || !wScale || !xDesc || context->GetDynamicInputShape(3, 1) ||
      context->GetDynamicInputShape(4, 1)) {
    return false;
  }

  const auto& xs = x->GetOriginShape();
  const auto& ws = w->GetOriginShape();
  const auto& wNz = w->GetStorageShape();
  const auto& gs = groups->GetOriginShape();
  const auto& as = xScale->GetOriginShape();
  const auto& bs = wScale->GetOriginShape();
  if (xs.GetDimNum() != 2 || ws.GetDimNum() != 3 || wNz.GetDimNum() != 5 || gs.GetDimNum() != 1 ||
      as.GetDimNum() != 3 || bs.GetDimNum() != 4) {
    return false;
  }
  const bool fp4 = xDesc->GetDataType() == ge::DT_FLOAT4_E2M1;
  const int64_t m = xs.GetDim(0);
  const int64_t k = xs.GetDim(1);
  const int64_t n = ws.GetDim(fp4 ? 1 : 2);
  const int64_t e = gs.GetDim(0);
  constexpr auto maxDim = std::numeric_limits<uint32_t>::max();
  if (m <= 0 || n <= 0 || k <= 0 || m > maxDim || n > maxDim || k > maxDim || e <= 0 || e > 1024 || n % 128 != 0 ||
      k % 128 != 0 || (!fp4 && m / e < kMinFp8TokensPerExpert) || ws.GetDim(0) != e || ws.GetDim(fp4 ? 2 : 1) != k) {
    return false;
  }
  const auto c0 = fp4 ? 64 : 32;
  if (wNz.GetDim(0) != e || wNz.GetDim(1) != (fp4 ? k : n) / c0 || wNz.GetDim(2) != (fp4 ? n : k) / 16 ||
      wNz.GetDim(3) != 16 || wNz.GetDim(4) != c0 || as.GetDim(0) != m || as.GetDim(1) != k / 64 || as.GetDim(2) != 2 ||
      bs.GetDim(0) != e || bs.GetDim(fp4 ? 1 : 2) != n || bs.GetDim(fp4 ? 2 : 1) != k / 64 || bs.GetDim(3) != 2) {
    return false;
  }
  input = {static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), static_cast<uint64_t>(e), fp4};
  return true;
}

bool GetAttrs(const gert::TilingContext* context, const Input& input, GMMSwigluQuantV2TensorApiTilingData& data) {
  const auto* attrs = context->GetAttrs();
  if (!attrs) {
    return false;
  }
  const auto* groupType = attrs->GetAttrPointer<int64_t>(0);
  const auto* limit = attrs->GetAttrPointer<float>(1);
  const auto* alpha = attrs->GetAttrPointer<float>(2);
  const auto* bias = attrs->GetAttrPointer<float>(3);
  const auto* dstMax = attrs->GetAttrPointer<float>(4);
  if (!groupType || !limit || !alpha || !bias || !dstMax || (*groupType != 0 && *groupType != 1) ||
      !std::isfinite(*limit) || *limit <= 0 || !std::isfinite(*alpha) || !std::isfinite(*bias) ||
      (*dstMax != 0 && (!input.fp4 || !std::isfinite(*dstMax) || *dstMax < 6 || *dstMax > 12))) {
    return false;
  }

  auto& quant = data.gmmQuantParams;
  quant.groupNum = static_cast<uint32_t>(input.experts);
  quant.aQuantMode = quant.bQuantMode = kMxPerGroupMode;
  quant.singleX = quant.singleW = quant.singleY = 1;
  quant.groupListType = static_cast<uint8_t>(*groupType);
  auto& swiglu = data.swigluParams;
  swiglu.swigluMode = kOai;
  swiglu.clampLimit = *limit;
  swiglu.gluAlpha = *alpha;
  swiglu.gluBias = *bias;
  swiglu.scaleAlg = input.fp4 ? 2 : 1;
  swiglu.dstTypeMax = *dstMax;
  return true;
}

ge::graphStatus Tiling(gert::TilingContext* context) {
  Input input{};
  GMMSwigluQuantV2TensorApiTilingData data{};
  auto* raw = context->GetRawTilingData();
  auto* workspace = context->GetWorkspaceSizes(1);
  if (!raw || raw->GetCapacity() < sizeof(data) || !workspace || !context->GetPlatformInfo() ||
      !GetInput(context, input) || !GetAttrs(context, input, data)) {
    return ge::GRAPH_FAILED;
  }

  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  const auto cubes = platform.GetCoreNumAic();
  Memory memory{};
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, memory.l1);
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, memory.l0c);
  if (cubes == 0 || platform.GetCoreNumAiv() != 2 * cubes || memory.l0c == 0 ||
      !FillMatmul(input, memory, data.mmTilingData)) {
    return ge::GRAPH_FAILED;
  }
  std::memcpy(raw->GetData(), &data, sizeof(data));
  raw->SetDataSize(sizeof(data));
  workspace[0] = kWorkspace;
  context->SetBlockDim(cubes);
  context->SetScheduleMode(1);
  context->SetTilingKey(input.fp4 ? kFP4TilingKey : kFP8TilingKey);
  return ge::GRAPH_SUCCESS;
}

struct CompileInfo {};

ge::graphStatus Prepare(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
}  // namespace

IMPL_OP_OPTILING(GmmSwigluOaiMxQuant).Tiling(Tiling).TilingParse<CompileInfo>(Prepare);
}  // namespace optiling
