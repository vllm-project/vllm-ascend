/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef CAUSAL_CONV1D_FN_TASKS_H
 #define CAUSAL_CONV1D_FN_TASKS_H
 
 struct FnDirectBlockTask {
     bool valid = false;
     int32_t tokenTileId = 0;
     int32_t baseDimIdx = 0;
     int32_t tokenStart = 0;
     int32_t tokenEnd = 0;
     int32_t channelStart = 0;
     int32_t baseDimSize = 0;
 };
 
 __aicore__ inline FnDirectBlockTask ResolveFnDirectBlockTask(int32_t blockIdx, int32_t tokenBlockCnt, int32_t tokenBlockSize,
                                                              int32_t cuSeqlen, int32_t baseDimCnt, int32_t baseDim,
                                                              int32_t dim)
 {
     FnDirectBlockTask task;
     if (blockIdx < 0 || tokenBlockCnt <= 0 || tokenBlockSize <= 0 || cuSeqlen <= 0 || baseDimCnt <= 0 || baseDim <= 0 ||
         dim <= 0) {
         return task;
     }
 
     const int64_t phase1Grid = static_cast<int64_t>(tokenBlockCnt) * baseDimCnt;
     if (phase1Grid <= 0 || static_cast<int64_t>(blockIdx) >= phase1Grid) {
         return task;
     }
 
     task.tokenTileId = blockIdx / baseDimCnt;
     task.baseDimIdx = blockIdx % baseDimCnt;
     task.channelStart = task.baseDimIdx * baseDim;
     if (task.channelStart >= dim) {
         return task;
     }
 
     task.baseDimSize = (task.channelStart + baseDim <= dim) ? baseDim : (dim - task.channelStart);
     task.tokenStart = task.tokenTileId * tokenBlockSize;
     if (task.tokenStart >= cuSeqlen) {
         return task;
     }
 
     const int32_t tokenEndRaw = task.tokenStart + tokenBlockSize;
     task.tokenEnd = (tokenEndRaw <= cuSeqlen) ? tokenEndRaw : cuSeqlen;
     if (task.baseDimSize <= 0 || task.tokenEnd <= task.tokenStart) {
         return {};
     }
 
     task.valid = true;
     return task;
 }
 
 __aicore__ inline bool IsFnInitStateSnapshotOwnerBlock(const FnDirectBlockTask &task)
 {
     return task.valid && task.tokenTileId == 0;
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline int32_t CAUSAL_CONV1D_CLASS::FindVarlenSeqByToken(int32_t tokenIdx) const
 {
     int32_t left = 0;
     int32_t right = static_cast<int32_t>(tilingData_->batch);
     while (left < right) {
         const int32_t mid = left + ((right - left) >> 1);
         const int32_t endVal = static_cast<int32_t>(queryStartLocGm.GetValue(mid + 1));
         if (tokenIdx < endVal) {
             right = mid;
         } else {
             left = mid + 1;
         }
     }
     return left;
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline bool CAUSAL_CONV1D_CLASS::ResolveExplicitTokenTileSeqRange(int32_t tokenTileId, int32_t &startSeq,
                                                                               int32_t &endSeq) const
 {
     if (!HasExplicitFnTokenSeqRanges() || tokenTileId < 0 || tokenTileId >= tilingData_->explicitTokenSeqRangeCount) {
         return false;
     }
     startSeq = static_cast<int32_t>(tilingData_->tokenTileStartSeq[tokenTileId]);
     endSeq = static_cast<int32_t>(tilingData_->tokenTileEndSeq[tokenTileId]);
     return (startSeq >= 0) && (endSeq >= startSeq);
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::InitRingSeqSplit(int32_t seq, int32_t cacheIdx, bool hasInit,
                                                              int32_t seqStart, int32_t tileStart, int32_t tileLen,
                                                              int32_t channelStart, int32_t baseDim, int32_t dim)
 {
     const int32_t stateLen = tilingData_->stateLen;
     const int32_t width = static_cast<int32_t>(tilingData_->width);
     const int32_t historyCount = width - 1;
     const int32_t ringStart = MAX_WIDTH - width;
     const int32_t historyStartTok = tileStart - historyCount;
     LocalTensor<T> ring = inBuf.Get<T>();
     bool hasGmHistoryCopy = false;
     bool hasVectorInit = false;
     const int64_t stateBaseOffset = static_cast<int64_t>(cacheIdx) * stateLen * dim + channelStart;
     int64_t xHistoryOffset = static_cast<int64_t>(historyStartTok) * dim + channelStart;
 
     for (int32_t i = 0; i < ringStart; ++i) {
         Duplicate(ring[i * MAX_BLOCK_DIM], static_cast<T>(0), baseDim);
         hasVectorInit = true;
     }
 
     for (int32_t i = 0, srcTok = historyStartTok; i < historyCount; ++i, ++srcTok, xHistoryOffset += dim) {
         LocalTensor<T> histSlot = ring[(ringStart + i) * MAX_BLOCK_DIM];
         if (srcTok >= seqStart) {
             DataCopy(histSlot, xGm[xHistoryOffset], baseDim);
             hasGmHistoryCopy = true;
         } else if (hasInit) {
             const int32_t statePos = srcTok - seqStart + historyCount;
             const int64_t stateOffset = stateBaseOffset + static_cast<int64_t>(statePos) * dim;
             if (tilingData_->hasInitStateWorkspace != 0) {
                 const int64_t snapshotOffset =
                     (static_cast<int64_t>(seq) * historyCount + statePos) * dim + channelStart;
                 DataCopy(histSlot, initStateWorkspaceGm_[snapshotOffset], baseDim);
             } else {
                 DataCopy(histSlot, convStatesGm[stateOffset], baseDim);
             }
             hasGmHistoryCopy = true;
         } else {
             Duplicate(histSlot, static_cast<T>(0), baseDim);
             hasVectorInit = true;
         }
     }
 
     if (hasGmHistoryCopy) {
         SetFlag<HardEvent::MTE2_V>(stateMte2ToVEvent_);
         WaitFlag<HardEvent::MTE2_V>(stateMte2ToVEvent_);
     }
     if (hasVectorInit) {
         PipeBarrier<PIPE_V>();
     }
 
     if (tileLen > 0) {
         const int32_t slot0 = SlotCurr(0);
         const int64_t xOffset = static_cast<int64_t>(tileStart) * dim + channelStart;
         DataCopy(ring[slot0 * MAX_BLOCK_DIM], xGm[xOffset], baseDim);
         SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slot0]);
     }
 
     if (tileLen > 1) {
         SetFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
     }
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::ProcessFnChunk(int32_t seq, int32_t cacheIdx, bool hasInit,
                                                            int32_t seqStart, int32_t seqLen, int32_t chunkStart,
                                                            int32_t chunkLen, int32_t channelStart, int32_t baseDim,
                                                            int32_t dim)
 {
     LoadWeightAndBias(channelStart, baseDim);
     InitRingSeqSplit(seq, cacheIdx, hasInit, seqStart, chunkStart, chunkLen, channelStart, baseDim, dim);
 
     RunSeq(chunkStart, chunkLen, channelStart, baseDim, dim);
 
     MaybeWriteBackSeqSplitTailChunk(chunkStart, chunkLen, seqStart, seqLen, cacheIdx, channelStart, baseDim, dim);
     DrainTaskMte3();
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void
 CAUSAL_CONV1D_CLASS::MaybeWriteBackSeqSplitTailChunk(int32_t chunkStart, int32_t chunkLen, int32_t seqStart,
                                                      int32_t seqLen, int32_t cacheIdx, int32_t channelStart,
                                                      int32_t baseDim, int32_t dim)
 {
     if (chunkStart + chunkLen != seqStart + seqLen) {
         return;
     }
 
     DrainTaskMte3();
     WriteBackState(cacheIdx, chunkLen, channelStart, baseDim, dim);
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::PrefetchInitStatesToWorkspace(int32_t channelStart, int32_t baseDimSize)
 {
     if (tilingData_->hasInitStateWorkspace == 0) {
         return;
     }
 
     const int32_t dim = tilingData_->dim;
     const int32_t historyCount = static_cast<int32_t>(tilingData_->width - 1);
     const int32_t batch = tilingData_->batch;
     const bool hasCacheIndices = (tilingData_->hasCacheIndices != 0);
     const bool hasInitialStateMode = (tilingData_->hasInitialStateMode != 0);
     LocalTensor<T> tmpBuf = inBuf.Get<T>()[0 * MAX_BLOCK_DIM];
 
     for (int32_t seq = 0; seq < batch; ++seq) {
         if (!ResolveSeqHasInit(seq, hasInitialStateMode)) {
             continue;
         }
 
         int32_t cacheIdx = 0;
         if (!ResolveSeqCacheIndex(seq, hasCacheIndices, cacheIdx)) {
             continue;
         }
 
         const int64_t stateBaseOffset = static_cast<int64_t>(cacheIdx) * tilingData_->stateLen * dim + channelStart;
         const int64_t snapshotBaseOffset = static_cast<int64_t>(seq) * historyCount * dim + channelStart;
         for (int32_t statePos = 0; statePos < historyCount; ++statePos) {
             const int64_t stateOffset = stateBaseOffset + static_cast<int64_t>(statePos) * dim;
             const int64_t snapshotOffset = snapshotBaseOffset + static_cast<int64_t>(statePos) * dim;
             DataCopy(tmpBuf, convStatesGm[stateOffset], baseDimSize);
             SetFlag<HardEvent::MTE2_MTE3>(initSnapshotMte2ToMte3Event_);
             WaitFlag<HardEvent::MTE2_MTE3>(initSnapshotMte2ToMte3Event_);
             DataCopy(initStateWorkspaceGm_[snapshotOffset], tmpBuf, baseDimSize);
             SetFlag<HardEvent::MTE3_MTE2>(initSnapshotMte3ToMte2Event_);
             WaitFlag<HardEvent::MTE3_MTE2>(initSnapshotMte3ToMte2Event_);
         }
     }
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::ProcessVarlenTokenTiled()
 {
     const int32_t dim = tilingData_->dim;
     const int32_t batch = tilingData_->batch;
     const int32_t seqLen = tilingData_->seqLen;
     const int32_t cuSeqlen = tilingData_->cuSeqlen;
     const int32_t baseDim = static_cast<int32_t>(tilingData_->baseDim);
     const int32_t baseDimCnt = static_cast<int32_t>(tilingData_->baseDimCnt);
     const int32_t tokenBlockSize = static_cast<int32_t>(tilingData_->tokenBlockSize);
     const int32_t tokenBlockCnt = static_cast<int32_t>(tilingData_->tokenBlockCnt);
     const bool hasCacheIndices = (tilingData_->hasCacheIndices != 0);
     const bool hasInitialStateMode = (tilingData_->hasInitialStateMode != 0);
     const bool isVarlenMode = (tilingData_->inputMode == 0);
 
     const int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
     const auto blockTask = ResolveFnDirectBlockTask(blockIdx, tokenBlockCnt, tokenBlockSize, cuSeqlen, baseDimCnt,
                                                     baseDim, dim);
     if (tilingData_->hasInitStateWorkspace != 0) {
         if (IsFnInitStateSnapshotOwnerBlock(blockTask)) {
             PrefetchInitStatesToWorkspace(blockTask.channelStart, blockTask.baseDimSize);
         }
         SyncAll();
     }
     if (!blockTask.valid) {
         return;
     }
 
     int32_t seq = 0;
     int32_t seqUpperBound = batch;
     if (isVarlenMode) {
         if (!ResolveExplicitTokenTileSeqRange(blockTask.tokenTileId, seq, seqUpperBound)) {
             seq = FindVarlenSeqByToken(blockTask.tokenStart);
         }
     } else {
         seq = (seqLen > 0) ? (blockTask.tokenStart / seqLen) : 0;
     }
 
     int32_t cursor = blockTask.tokenStart;
     while (cursor < blockTask.tokenEnd && seq < seqUpperBound) {
         int32_t seqStart = 0;
         int32_t curSeqLen = 0;
         if (!ResolveSeqTaskWindow(seq, tilingData_->inputMode, seqLen, seqStart, curSeqLen)) {
             ++seq;
             continue;
         }
         const int32_t curSeqEnd = seqStart + curSeqLen;
         if (cursor < seqStart) {
             cursor = seqStart;
         }
         if (cursor >= curSeqEnd) {
             ++seq;
             continue;
         }
 
         const int32_t tileEnd = (blockTask.tokenEnd <= curSeqEnd) ? blockTask.tokenEnd : curSeqEnd;
         const int32_t tileLen = tileEnd - cursor;
         if (tileLen <= 0) {
             ++seq;
             continue;
         }
 
         int32_t cacheIdx = 0;
         if (!ResolveSeqCacheIndex(seq, hasCacheIndices, cacheIdx)) {
             cursor = tileEnd;
             ++seq;
             continue;
         }
 
         const bool hasInit = ResolveSeqHasInit(seq, hasInitialStateMode);
         ProcessFnChunk(seq, cacheIdx, hasInit, seqStart, curSeqLen, cursor, tileLen, blockTask.channelStart,
                        blockTask.baseDimSize, dim);
 
         cursor = tileEnd;
         ++seq;
     }
 }
 
 #endif
 