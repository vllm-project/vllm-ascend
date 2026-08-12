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
 * \file main.cpp
 * \brief ST host entry for AllGather + QuantMatmul fusion (prefill)
 */

#include <cmath>
#include <iostream>
#include <limits.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <sys/wait.h>

#include "kernel_basic_intf.h"
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_rank_graph.h"
#include <cstdlib>
#include "utils.h"
#include "apace/utils/constant.h"
#include "apace/utils/common_utils.h"
#include "apace/tiling/quant_matmul_tiling_swat.h"
#include "apace/kernel/all_gather_quant_matmul/all_gather_mx_matmul_udma_tiling_data.h"
#include "apace/kernel/all_gather_quant_matmul/all_gather_mx_matmul_udma_impl.h"
#include "apace/block/aiv_comm/collective_comm_context.h"
#include "apace/utils/comm_channel_builder.h"
#include "../../utils/root_info_exchanger.h"

using namespace AllGatherQuantMatmulImpl;
using namespace Apace::AivComm;

using CommContext = Apace::AivComm::CommContext;
static constexpr int32_t BENCHMARK_ITERATIONS = 10;
static constexpr uint32_t TILE_M = 512;

#define HCCL_CHECK(status) do { \
  HcclResult ret = (status); \
  if (static_cast<int>(ret) != 0) { \
    std::cerr << __FILE__ << ":" << __LINE__ << " HcclResult:" << ret << std::endl; \
  } \
} while (0)

void ParseArgs(int argc, char *argv[], int *m, int *k, int *n, int *rankNum)
{
  if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    std::cerr << "Usage: " << argv[0] << " m k n rankNum" << std::endl;
    std::cerr << "  m: row of matrix A per rank" << std::endl;
    std::cerr << "  k: col of matrix A (= row of matrix B, full K)" << std::endl;
    std::cerr << "  n: col of matrix B (total N)" << std::endl;
    std::cerr << "  rankNum: number of ranks" << std::endl;
    exit(1);
  }
  if (argc < 5) {
    throw std::invalid_argument("ERROR: Lacks Arguments");
  }
  try {
    *m = std::stoi(argv[1]);
    *k = std::stoi(argv[2]);
    *n = std::stoi(argv[3]);
    *rankNum = std::stoi(argv[4]);
  } catch (const std::invalid_argument&) {
    throw std::invalid_argument("ERROR: m k n rankNum must be Integer");
  }
  if (*m <= 0 || *k <= 0 || *n <= 0 || *rankNum <= 0) {
    throw std::invalid_argument("ERROR: m k n rankNum must be positive");
  }
  if (*k % 32 != 0) {
    throw std::invalid_argument("ERROR: K must be divisible by 32 for MXFP8 quantization");
  }
  if (CeilDiv(*k, static_cast<int>(::MXFP_DIVISOR_SIZE)) % 2 != 0) {
    throw std::invalid_argument("ERROR: CeilDiv(K, 64) must be an even number");
  }
}

int LaunchKernel(uint32_t usedCoreNum, aclrtStream stream,
         CommContext *devContext,
         GM_ADDR deviceA, GM_ADDR deviceScaleA,
         GM_ADDR deviceB, GM_ADDR deviceScaleB,
         GM_ADDR deviceOutput, AllGatherMxMatmulUdmaTilingData &tilingData)
{
  AllGatherQuantMatmulKernel<<<usedCoreNum, nullptr, stream>>>(
    devContext, deviceA, deviceScaleA, deviceB, deviceScaleB, deviceOutput, tilingData);
  return 0;
}

int RunAllGatherQuantMatmul(int rankNum, int rankId, int m, int k, int n)
{
  const char *ipport = "tcp://127.0.0.1:8998";
  INFO_LOG("rankNum=%d, rankId=%d", rankNum, rankId);

  AllGatherMxMatmulUdmaTilingData tilingData;

  uint32_t tileM = TILE_M;
  if (m < static_cast<int>(tileM)) {
    tileM = static_cast<uint32_t>(m);
  }
  uint32_t tileCnt = static_cast<uint32_t>(m) / tileM;
  uint32_t tailM = static_cast<uint32_t>(m) % tileM;
  uint32_t tailCnt = (tailM > 0) ? 1 : 0;

  uint32_t paddedTailM = (tailM > 0) ? ((tailM + 15U) / 16U * 16U) : 0U;
  uint32_t totalLogicalM = static_cast<uint32_t>(rankNum) *
    (tileCnt * tileM + tailCnt * paddedTailM);

  QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN> tilingEngine;
  tilingEngine.SetOptimizeEnable(false);
  tilingEngine.GetTilingData(totalLogicalM, static_cast<uint32_t>(n),
                  static_cast<uint32_t>(k), tilingData.mmTile);
  tilingData.commTile.splitAxisTileSize = tileM;
  tilingData.commTile.splitAxisTileCnt = tileCnt;
  tilingData.commTile.splitAxisTailSize = tailM;
  tilingData.commTile.splitAxisTailCnt = tailCnt;
  tilingData.commTile.nonSplitAxisSize = static_cast<uint32_t>(k);

  printf("[Rank %d] Tiling: tileCnt=%u, tileM=%u, tailCnt=%u, tailM=%u, paddedTailM=%u, commTurn=%u\n",
       rankId, tileCnt, tileM, tailCnt, tailM, paddedTailM, tileCnt + tailCnt);

  // ---- ACL Init ----
  ACL_CHECK(aclInit(nullptr));
  int32_t deviceId = rankId;
  ACL_CHECK(aclrtSetDevice(deviceId));
  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  // ---- TCP + HCCL Init ----
  RootInfoExchanger exchanger(static_cast<uint32_t>(rankId), static_cast<uint32_t>(rankNum), ipport);
  HcclRootInfo rootInfo;
  if (!exchanger.Exchange(rootInfo)) {
    ERROR_LOG("rank %d exchange rootInfo failed", rankId); return -1;
  }

  HcclCommConfig config;
  HcclCommConfigInit(&config);
  config.hcclWorldRankID = static_cast<uint32_t>(rankId);
  HcclComm comm = nullptr;
  HCCL_CHECK(HcclCommInitRootInfoConfig(static_cast<uint32_t>(rankNum), &rootInfo,
                                        static_cast<uint32_t>(rankId), &config, &comm));
  if (comm == nullptr) { ERROR_LOG("rank %d HcclCommInitRootInfoConfig failed", rankId); return -1; }

  // ---- URMA Channel Setup ----
  CommChannelBuilder<> builder(comm);
  CommContext hostCtx = {};

  const char *ctxTag = "all_gather_quant_matmul";

  CommContext *devContext = reinterpret_cast<CommContext *>(
      builder.CreateDeviceContext(&hostCtx, sizeof(CommContext), ctxTag,
                                  &hostCtx.udmaCtx, &hostCtx.ubmemCtx));
  if (devContext == nullptr) {
    ERROR_LOG("rank %d CreateDeviceContext failed", rankId);
    return -1;
  }

  exchanger.Barrier();
  exchanger.Close();

  // ---- Device memory ----
  auto sizeA = static_cast<size_t>(m) * k * sizeof(uint8_t);
  auto sizeB = static_cast<size_t>(k) * n * sizeof(uint8_t);
  auto sizeScaleA = static_cast<size_t>(m) * CeilDiv(k, static_cast<int>(::MXFP_DIVISOR_SIZE)) *
                    static_cast<int>(::MXFP_MULTI_BASE_SIZE) * sizeof(uint8_t);
  auto sizeScaleB = static_cast<size_t>(CeilDiv(k, static_cast<int>(::MXFP_DIVISOR_SIZE)) *
                    static_cast<int>(::MXFP_MULTI_BASE_SIZE)) * n * sizeof(uint8_t);
  auto sizeOutput = static_cast<size_t>(rankNum) * m * n * sizeof(uint16_t);

  GM_ADDR deviceA = nullptr;
  GM_ADDR deviceB = nullptr;
  GM_ADDR deviceScaleA = nullptr;
  GM_ADDR deviceScaleB = nullptr;
  GM_ADDR deviceOutput = nullptr;
  ACL_CHECK(aclrtMalloc((void**)&deviceA, sizeA, ACL_MEM_MALLOC_HUGE_ONLY));
  ACL_CHECK(aclrtMalloc((void**)&deviceB, sizeB, ACL_MEM_MALLOC_HUGE_ONLY));
  ACL_CHECK(aclrtMalloc((void**)&deviceScaleA, sizeScaleA, ACL_MEM_MALLOC_HUGE_ONLY));
  ACL_CHECK(aclrtMalloc((void**)&deviceScaleB, sizeScaleB, ACL_MEM_MALLOC_HUGE_ONLY));
  ACL_CHECK(aclrtMalloc((void**)&deviceOutput, sizeOutput, ACL_MEM_MALLOC_HUGE_ONLY));

  // ---- Load data ----
  std::vector<uint8_t> hostA(sizeA, 0), hostB(sizeB, 0);
  std::vector<uint8_t> hostScaleA(sizeScaleA, 0), hostScaleB(sizeScaleB, 0);
  std::vector<uint16_t> hostOutput(rankNum * m * n, 0);

  char exePath[PATH_MAX];
  ssize_t rlLen = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
  std::string baseDir = ".";
  if (rlLen > 0) { exePath[rlLen] = '\0'; baseDir = exePath; baseDir.resize(baseDir.find_last_of('/')); }
  std::string inputDir = baseDir + "/input/" + std::to_string(rankId);
  std::string outputDir = baseDir + "/output/" + std::to_string(rankId);

  if (ReadFile(inputDir + "/input_a.bin", hostA.data(), sizeA) &&
      ReadFile(inputDir + "/input_b.bin", hostB.data(), sizeB)) {
    INFO_LOG("Loaded input data for rank %d", rankId);
    ReadFile(inputDir + "/input_scaleA.bin", hostScaleA.data(), sizeScaleA);
    ReadFile(inputDir + "/input_scaleB.bin", hostScaleB.data(), sizeScaleB);
  } else {
    INFO_LOG("No input data for rank %d, using zeros", rankId);
  }
  fprintf(stderr, "[Rank %d] memcpy start...\n", rankId); fflush(stderr);
  ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(deviceScaleA, sizeScaleA, hostScaleA.data(), sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(deviceScaleB, sizeScaleB, hostScaleB.data(), sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE));
  fprintf(stderr, "[Rank %d] memcpy done\n", rankId); fflush(stderr);

  // ---- Launch Kernel ----
  printf("[Rank %d] Launching AllGatherQuantMatmulKernel...\n", rankId);
  fflush(stdout);

  aclrtEvent kernelStartEvent = nullptr, kernelEndEvent = nullptr;
  ACL_CHECK(aclrtCreateEvent(&kernelStartEvent));
  ACL_CHECK(aclrtCreateEvent(&kernelEndEvent));
  ACL_CHECK(aclrtRecordEvent(kernelStartEvent, stream));

  for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
    LaunchKernel(tilingData.mmTile.usedCoreNum, stream, devContext,
               deviceA, deviceScaleA, deviceB, deviceScaleB, deviceOutput, tilingData);
  }
  ACL_CHECK(aclrtRecordEvent(kernelEndEvent, stream));
  ACL_CHECK(aclrtSynchronizeStream(stream));

  float kernelElapsedMs = 0.0F;
  ACL_CHECK(aclrtEventElapsedTime(&kernelElapsedMs, kernelStartEvent, kernelEndEvent));
  double kernelElapsedUs = static_cast<double>(kernelElapsedMs) * 1000.0;
  printf("[Rank %d] Kernel completed! Elapsed time: %.3f us (avg over %d iterations)\n",
       rankId, kernelElapsedUs / BENCHMARK_ITERATIONS, BENCHMARK_ITERATIONS);
  fflush(stdout);

  size_t outputSize = static_cast<size_t>(rankNum) * m * n * sizeof(uint16_t);
  ACL_CHECK(aclrtMemcpy(hostOutput.data(), outputSize, deviceOutput, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));
  { std::string cmd = "mkdir -p " + outputDir; system(cmd.c_str()); }
  WriteFile(outputDir + "/npu_out.bin", hostOutput.data(), outputSize);

  if (kernelEndEvent) aclrtDestroyEvent(kernelEndEvent);
  if (kernelStartEvent) aclrtDestroyEvent(kernelStartEvent);
  aclrtFree(devContext);
  aclrtFree(deviceA); aclrtFree(deviceScaleA); aclrtFree(deviceB);
  aclrtFree(deviceScaleB); aclrtFree(deviceOutput);
  HcclCommDestroy(comm);
  ACL_CHECK(aclrtDestroyStream(stream));
  ACL_CHECK(aclrtResetDevice(deviceId));
  ACL_CHECK(aclFinalize());
  return 0;
}

int main(int argc, char *argv[])
{
  int m = 0, k = 0, n = 0, rankNum = 0;
  try { ParseArgs(argc, argv, &m, &k, &n, &rankNum); }
  catch (const std::invalid_argument& e) {
    std::cerr << e.what() << std::endl; return -1;
  }
  INFO_LOG("Master (PID=%d) will fork %d processes", getpid(), rankNum);

  std::vector<pid_t> pids(rankNum);
  for (int rankId = 0; rankId < rankNum; ++rankId) {
    pid_t pid = fork();
    if (pid < 0) { ERROR_LOG("Fork failed for rank %d", rankId); exit(-1); }
    else if (pid == 0) { exit(RunAllGatherQuantMatmul(rankNum, rankId, m, k, n)); }
    else { pids[rankId] = pid; INFO_LOG("Forked Rank %d -> PID %d", rankId, pid); }
  }

  int status; bool allSuccess = true;
  for (int i = 0; i < rankNum; ++i) {
    waitpid(pids[i], &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) { allSuccess = false; ERROR_LOG("Worker PID %d failed", pids[i]); }
  }
  std::cout << "All workers finished. Status: " << (allSuccess ? "SUCCESS" : "FAILURE") << std::endl;
  return allSuccess ? 0 : -1;
}
