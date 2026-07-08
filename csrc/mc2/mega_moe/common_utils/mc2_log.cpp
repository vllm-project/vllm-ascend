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
 * \file mc2_log.cpp
 * \brief
 */

#include "mc2_log.h"

// Not used by mega_moe — commented out to remove 3rd dependency.
#if 0
namespace Mc2Log {
using namespace Mc2Tiling;
static void PrintTCubeTilingDataSecondPart(const std::string &opName, ::TCubeTiling &tiling)
{
    OP_LOGD(opName, "tiling.deptchAL1CacheUB %d", tiling.depthAL1CacheUB);
    OP_LOGD(opName, "tiling.deptchBL1CacheUB %d", tiling.depthBL1CacheUB);
    OP_LOGD(opName, "tiling.get_dbL0A %d", tiling.dbL0A);
    OP_LOGD(opName, "tiling.get_dbL0B %d", tiling.dbL0B);
    OP_LOGD(opName, "tiling.get_dbL0C %d", tiling.dbL0C);
    OP_LOGD(opName, "tiling.ALayoutInfoB %d", tiling.ALayoutInfoB);
    OP_LOGD(opName, "tiling.ALayoutInfoS %d", tiling.ALayoutInfoS);
    OP_LOGD(opName, "tiling.ALayoutInfoN %d", tiling.ALayoutInfoN);
    OP_LOGD(opName, "tiling.ALayoutInfoG %d", tiling.ALayoutInfoG);
    OP_LOGD(opName, "tiling.ALayoutInfoD %d", tiling.ALayoutInfoD);
    OP_LOGD(opName, "tiling.BLayoutInfoB %d", tiling.BLayoutInfoB);
    OP_LOGD(opName, "tiling.BLayoutInfoS %d", tiling.BLayoutInfoS);
    OP_LOGD(opName, "tiling.BLayoutInfoN %d", tiling.BLayoutInfoN);
    OP_LOGD(opName, "tiling.BLayoutInfoG %d", tiling.BLayoutInfoG);
    OP_LOGD(opName, "tiling.BLayoutInfoD %d", tiling.BLayoutInfoD);
    OP_LOGD(opName, "tiling.CLayoutInfoB %d", tiling.CLayoutInfoB);
    OP_LOGD(opName, "tiling.CLayoutInfoS1 %d", tiling.CLayoutInfoS1);
    OP_LOGD(opName, "tiling.CLayoutInfoN %d", tiling.CLayoutInfoN);
    OP_LOGD(opName, "tiling.CLayoutInfoG %d", tiling.CLayoutInfoG);
    OP_LOGD(opName, "tiling.CLayoutInfoS2 %d", tiling.CLayoutInfoS2);
    OP_LOGD(opName, "tiling.BatchNum %d", tiling.BatchNum);
    OP_LOGD(opName, "tiling.get_mxTypePara %d", tiling.mxTypePara);
}

void PrintMMV3TilingData(const std::string &opName,
                         Mc2Tiling::MC2MatmulV3TilingData &tiling) {
  PrintTCubeTilingData(opName, tiling.matmulTiling);
  OP_LOGD(opName, "mTailCnt %d", tiling.mTailCnt);
  OP_LOGD(opName, "nTailCnt %d", tiling.nTailCnt);
  OP_LOGD(opName, "kTailCnt %d", tiling.kTailCnt);
  OP_LOGD(opName, "mBaseTailSplitCnt %d", tiling.mBaseTailSplitCnt);
  OP_LOGD(opName, "nBaseTailSplitCnt %d", tiling.nBaseTailSplitCnt);
  OP_LOGD(opName, "mTailMain %d", tiling.mTailMain);
  OP_LOGD(opName, "nTailMain %d", tiling.nTailMain);
  OP_LOGD(opName, "isHf32 %d", tiling.isHf32);
}

void PrintTCubeTilingData(const std::string &opName, ::TCubeTiling &tiling)
{
    OP_LOGD(opName, "tiling.usedCoreNum %d", tiling.usedCoreNum);
    OP_LOGD(opName, "tiling.M %d", tiling.M);
    OP_LOGD(opName, "tiling.N %d", tiling.N);
    OP_LOGD(opName, "tiling.Ka %d", tiling.Ka);
    OP_LOGD(opName, "tiling.Kb %d", tiling.Kb);
    OP_LOGD(opName, "tiling.singleCoreM %d", tiling.singleCoreM);
    OP_LOGD(opName, "tiling.singleCoreN %d", tiling.singleCoreN);
    OP_LOGD(opName, "tiling.singleCoreK %d", tiling.singleCoreK);
    OP_LOGD(opName, "tiling.baseM %d", tiling.baseM);
    OP_LOGD(opName, "tiling.baseN %d", tiling.baseN);
    OP_LOGD(opName, "tiling.baseK %d", tiling.baseK);
    OP_LOGD(opName, "tiling.depthA1 %d", tiling.depthA1);
    OP_LOGD(opName, "tiling.depthB1 %d", tiling.depthB1);
    OP_LOGD(opName, "tiling.stepM %d", tiling.stepM);
    OP_LOGD(opName, "tiling.stepN %d", tiling.stepN);
    OP_LOGD(opName, "tiling.isBias %d", tiling.isBias);
    OP_LOGD(opName, "tiling.transLength %d", tiling.transLength);
    OP_LOGD(opName, "tiling.iterateOrder %d", tiling.iterateOrder);
    OP_LOGD(opName, "tiling.shareMode %d", tiling.shareMode);
    OP_LOGD(opName, "tiling.usedL1Size %d", tiling.shareL1Size);
    OP_LOGD(opName, "tiling.usedL0CSize %d", tiling.shareL0CSize);
    OP_LOGD(opName, "tiling.shareUBSize %d", tiling.shareUbSize);
    OP_LOGD(opName, "tiling.batchM %d", tiling.batchM);
    OP_LOGD(opName, "tiling.batchN %d", tiling.batchN);
    OP_LOGD(opName, "tiling.singleBatchM %d", tiling.singleBatchM);
    OP_LOGD(opName, "tiling.singleBatchN %d", tiling.singleBatchN);
    OP_LOGD(opName, "tiling.stepKa %d", tiling.stepKa);
    OP_LOGD(opName, "tiling.stepKb %d", tiling.stepKb);
    // for cleancode, make sure func less than 50 lines
    PrintTCubeTilingDataSecondPart(opName,tiling);
}

void PrintRCSTilingData(const std::string &opName, Mc2Tiling::RCSTiling &rcsTiling)
{
    OP_LOGD(opName, "rcsTiling.rankDim %u", rcsTiling.rankDim);
    OP_LOGD(opName, "rcsTiling.rankID %u", rcsTiling.rankID);
    OP_LOGD(opName, "rcsTiling.commtype %u", rcsTiling.commtype);
    OP_LOGD(opName, "rcsTiling.subtype %u", rcsTiling.subtype);
    OP_LOGD(opName, "rcsTiling.tileCnt %u", rcsTiling.tileCnt);
    OP_LOGD(opName, "rcsTiling.tailM %u", rcsTiling.tailM);
    OP_LOGD(opName, "rcsTiling.tailCnt %u", rcsTiling.tailCnt);
    OP_LOGD(opName, "rcsTiling.biasLen %u", rcsTiling.biasLen);
    OP_LOGD(opName, "rcsTiling.isAdd %u", rcsTiling.isAdd);
    OP_LOGD(opName, "rcsTiling.rankM %u", rcsTiling.rankM);
    OP_LOGD(opName, "rcsTiling.rankN %u", rcsTiling.rankN);
    OP_LOGD(opName, "rcsTiling.rankK %u", rcsTiling.rankK);
    OP_LOGD(opName, "rcsTiling.gatherIndex %u", rcsTiling.gatherIndex);
    OP_LOGD(opName, "rcsTiling.isTransA %u", rcsTiling.isTransposeA);
    OP_LOGD(opName, "rcsTiling.isTransB %u", rcsTiling.isTransposeB);
    OP_LOGD(opName, "rcsTiling.storageGather %u", rcsTiling.storageGather);
    OP_LOGD(opName, "rcsTiling.nd2NzWorkLen %lu", rcsTiling.nd2NzWorkLen);
    OP_LOGD(opName, "rcsTiling.cToFloatLen %lu", rcsTiling.cToFloatLen);
    OP_LOGD(opName, "rcsTiling.gatherLen %lu", rcsTiling.gatherLen);
    OP_LOGD(opName, "rcsTiling.workspaceAddr4 %u", rcsTiling.workspaceAddr4);
    OP_LOGD(opName, "rcsTiling.aicCoreNum %u", rcsTiling.aicCoreNum);
    OP_LOGD(opName, "rcsTiling.needUbBuffer %u", rcsTiling.needUbBuffer);
    OP_LOGD(opName, "rcsTiling.addX3UbCnt %u", rcsTiling.addX3UbCnt);
}

void PrintTileL2TilingData(const std::string &opName, Mc2Tiling::TileL2Tiling &tileL2Tiling)
{
    OP_LOGD(opName, "tileL2Tiling.mL2TileCnt %u", tileL2Tiling.mL2TileCnt);
    OP_LOGD(opName, "tileL2Tiling.nL2TileCnt %u", tileL2Tiling.nL2TileCnt);
    OP_LOGD(opName, "tileL2Tiling.mTileBlocks %u", tileL2Tiling.mTileBlocks);
    OP_LOGD(opName, "tileL2Tiling.nTileBlocks %u", tileL2Tiling.nTileBlocks);
    OP_LOGD(opName, "tileL2Tiling.mTailBlocks %u", tileL2Tiling.mTailBlocks);
    OP_LOGD(opName, "tileL2Tiling.nTailBlocks %u", tileL2Tiling.nTailBlocks);
    OP_LOGD(opName, "tileL2Tiling.rankTileNum %u", tileL2Tiling.rankTileNum);
    OP_LOGD(opName, "tileL2Tiling.calcOrder %u", tileL2Tiling.calcOrder);
    OP_LOGD(opName, "tileL2Tiling.enableL2Tile %u", tileL2Tiling.enableL2Tile);
}

void PrintMc2MsgData(const std::string &opName, Mc2Tiling::Mc2Msg &msg)
{
OP_LOGD(opName, "msg.sendOff %lu", msg.sendOff);
    OP_LOGD(opName, "msg.recvOff %lu", msg.recvOff);
    OP_LOGD(opName, "msg.tailSendOff %lu", msg.tailSendOff);
    OP_LOGD(opName, "msg.tailRecvOff %lu", msg.tailRecvOff);
    OP_LOGD(opName, "msg.sendCnt %lu", msg.sendCnt);
    OP_LOGD(opName, "msg.recvCnt %lu", msg.recvCnt);
    OP_LOGD(opName, "msg.tailSendCnt %lu", msg.tailSendCnt);
    OP_LOGD(opName, "msg.tailRecvCnt %lu", msg.tailRecvCnt);
    OP_LOGD(opName, "msg.totalCnt %lu", msg.totalCnt);
    OP_LOGD(opName, "msg.turnNum %u", msg.turnNum);
    OP_LOGD(opName, "msg.stride %u", msg.stride);
    OP_LOGD(opName, "msg.workspaceOff %u", msg.workspaceOff);
    OP_LOGD(opName, "msg.notifyOff %u", msg.notifyOff);

    OP_LOGD(opName, "msg.notifyBeginCnt %u", msg.notifyBeginCnt);
    OP_LOGD(opName, "msg.notifyEndCnt %u", msg.notifyEndCnt);
    OP_LOGD(opName, "msg.useBufferType %u", msg.useBufferType);
    OP_LOGD(opName, "msg.funID %u", msg.funID);
    OP_LOGD(opName, "msg.dataType %u", msg.dataType);
    OP_LOGD(opName, "msg.groupNum %u", msg.groupNum);

    OP_LOGD(opName, "msg.reuseMode %u", msg.reuseMode);
    OP_LOGD(opName, "msg.commType %u", msg.commType);
    OP_LOGD(opName, "msg.reduceOp %u", msg.reduceOp);
    OP_LOGD(opName, "msg.commOrder %u", msg.commOrder);
    OP_LOGD(opName, "msg.waitPolicy %u", msg.waitPolicy);
    OP_LOGD(opName, "msg.rspPolicy %u", msg.rspPolicy);

    OP_LOGD(opName, "msg.exitPolicy %u", msg.exitPolicy);

    OP_LOGD(opName, "msg.commAlg %u", msg.commAlg);
    OP_LOGD(opName, "msg.taskType %u", msg.taskType);
    OP_LOGD(opName, "msg.preparePosition %u", msg.preparePosition);
}

void PrintTCubeTilingParams(const std::string &opName, DequantBmm::Mc2QuantBatchMatmulV3DataParams &tiling)
{
    OP_LOGD(opName, "tiling.batchA %d", tiling.batchA);
    OP_LOGD(opName, "tiling.batchB %d", tiling.batchB);
    OP_LOGD(opName, "tiling.batchC %d", tiling.batchC);
    OP_LOGD(opName, "tiling.batchA1 %d", tiling.batchA1);
    OP_LOGD(opName, "tiling.batchA2 %d", tiling.batchA2);
    OP_LOGD(opName, "tiling.batchA3 %d", tiling.batchA3);
    OP_LOGD(opName, "tiling.batchA4 %d", tiling.batchA4);
    OP_LOGD(opName, "tiling.batchB1 %d", tiling.batchB1);
    OP_LOGD(opName, "tiling.batchB2 %d", tiling.batchB2);
    OP_LOGD(opName, "tiling.batchB3 %d", tiling.batchB3);
    OP_LOGD(opName, "tiling.batchB4 %d", tiling.batchB4);
    OP_LOGD(opName, "tiling.batchC1 %d", tiling.batchC1);
    OP_LOGD(opName, "tiling.batchC2 %d", tiling.batchC2);
    OP_LOGD(opName, "tiling.batchC3 %d", tiling.batchC3);
    OP_LOGD(opName, "tiling.batchC4 %d", tiling.batchC4);
    OP_LOGD(opName, "tiling.singleCoreBatch %d", tiling.singleCoreBatch);
    OP_LOGD(opName, "tiling.isPerTensor %d", tiling.isPerTensor);
    OP_LOGD(opName, "tiling.isPertoken %d", tiling.isPertoken);
    OP_LOGD(opName, "tiling.isDoubleScale %d", tiling.isDoubleScale);
    OP_LOGD(opName, "tiling.biasThreeDim %d", tiling.biasThreeDim);
    OP_LOGD(opName, "tiling.ubCalcM %d", tiling.ubCalcM);
    OP_LOGD(opName, "tiling.ubCalcN %d", tiling.ubCalcN);
    OP_LOGD(opName, "tiling.needUbBuffer %d", tiling.needUbBuffer);
    OP_LOGD(opName, "tiling.realSingleCoreM %d", tiling.realSingleCoreM);
    OP_LOGD(opName, "tiling.realSingleCoreN %d", tiling.realSingleCoreN);
    OP_LOGD(opName, "tiling.biasDtype %d", tiling.biasDtype);
    OP_LOGD(opName, "tiling.ubSize %d", tiling.ubSize);
    OP_LOGD(opName, "tiling.isMClash %d", tiling.isMClash);
    OP_LOGD(opName, "tiling.isNClash %d", tiling.isNClash);
    OP_LOGD(opName, "tiling.groupSizeM %d", tiling.groupSizeM);
    OP_LOGD(opName, "tiling.groupSizeN %d", tiling.groupSizeN);
    OP_LOGD(opName, "tiling.groupSizeK %d", tiling.groupSizeK);
}

void PrintTCubeTilingL2cache(const std::string &opName, DequantBmm::Mc2L2cacheTileParams &tiling)
{
    OP_LOGD(opName, "tiling.mTileCntL2 %d", tiling.mTileCntL2);
    OP_LOGD(opName, "tiling.nTileCntL2 %d", tiling.nTileCntL2);
    OP_LOGD(opName, "tiling.mTileBlock %d", tiling.mTileBlock);
    OP_LOGD(opName, "tiling.nTileBlock %d", tiling.nTileBlock);
    OP_LOGD(opName, "tiling.calOrder %d", tiling.calOrder);
    OP_LOGD(opName, "tiling.isBasicTiling %d", tiling.isBasicTiling);
}

void PrintTCubeTilingWindowParam(const std::string &opName, DequantBmm::Mc2SlidingWindowParams &tiling)
{
    OP_LOGD(opName, "tiling.mTailTile %d", tiling.mTailTile);
    OP_LOGD(opName, "tiling.nTailTile %d", tiling.nTailTile);
}

void PrintMMV3TilingData(const std::string &opName, Mc2MatMulV3TilingData &tiling)
{
    PrintTCubeTilingData(opName, tiling.tCubeTiling);
    OP_LOGD(opName, "tiling.mTailCnt %d", tiling.mTailCnt);
    OP_LOGD(opName, "tiling.nTailCnt %d", tiling.nTailCnt);
    OP_LOGD(opName, "tiling.kTailCnt %d", tiling.kTailCnt);
    OP_LOGD(opName, "tiling.isHf32 %d", tiling.isHf32);
    OP_LOGD(opName, "tiling.mBaseTailSpiltCnt %d", tiling.mBaseTailSplitCnt);
    OP_LOGD(opName, "tiling.nBaseTailSpiltCnt %d", tiling.nBaseTailSplitCnt);
    OP_LOGD(opName, "tiling.mTailMain %d", tiling.mTailMain);
    OP_LOGD(opName, "tiling.nTailMain %d", tiling.nTailMain);
    OP_LOGD(opName, "tiling.aswWindowLen %d", tiling.aswWindowLen);
}

}  // namespace Mc2Log
#endif  // #if 0 — Print* functions not needed by mega_moe
