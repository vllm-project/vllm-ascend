// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "fsa-lru-vector-a3.h"

namespace {
using namespace AscendC;
using namespace FsaLruVector;

__aicore__ inline void WriteWords(GlobalTensor<int32_t> dst, I32 src, uint32_t n) {
    if (!n) return;
    SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
    DataCopyPad(dst,src,DataCopyExtParams{1,n*4,0,0,0});
    SetFlag<HardEvent::MTE3_V>(0); WaitFlag<HardEvent::MTE3_V>(0);
}

template <bool PairedColumns>
__aicore__ inline void WritePartition(GlobalTensor<int32_t> state, uint64_t base,
    uint32_t capacity, uint32_t destination, I32 tokens, uint32_t tokenBegin,
    I32 slots, uint32_t slotBegin, uint32_t count, I32 scratch, uint32_t tile) {
    if (!count) return;
    constexpr uint32_t wordsPerBlock = 32 / sizeof(int32_t);
    const bool alignTokens = tokenBegin % wordsPerBlock != 0;
    const bool alignSlots = slotBegin % wordsPerBlock != 0;
    if (alignTokens || alignSlots) {
        // Gather from an aligned UB base with an element-aligned byte offset.
        auto byteOffsets = scratch[2 * tile].ReinterpretCast<uint32_t>();
        ArithProgression<int32_t>(byteOffsets.ReinterpretCast<int32_t>(), 0, sizeof(int32_t),
                                  FsaLruGeometry::AlignWords(count));
        PipeBarrier<PIPE_V>();
        if (alignTokens) {
            Gather(scratch, tokens, byteOffsets, tokenBegin * sizeof(int32_t), count);
            tokens = scratch;
        }
        if (alignSlots) {
            Gather(scratch[tile], slots, byteOffsets, slotBegin * sizeof(int32_t), count);
            slots = scratch[tile];
        }
    }
    if (!alignTokens) tokens = tokens[tokenBegin];
    if (!alignSlots) slots = slots[slotBegin];
    SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
    const uint32_t bytes = count * sizeof(int32_t);
    if constexpr (PairedColumns) {
        // Equal slices of the paired tile are still tile words apart in UB,
        // including when both tails were shifted into the aligned scratch.
        const DataCopyExtParams copy{2, bytes, (tile - FsaLruGeometry::AlignWords(count)) / wordsPerBlock,
                                     (capacity - count) * uint32_t{sizeof(int32_t)}, 0};
        DataCopyPad(state[base + destination], tokens, copy);
    } else {
        const DataCopyExtParams copy{1, bytes, 0, 0, 0};
        DataCopyPad(state[base + destination], tokens, copy);
        DataCopyPad(state[base + capacity + destination], slots, copy);
    }
    SetFlag<HardEvent::MTE3_V>(0); WaitFlag<HardEvent::MTE3_V>(0);
}

__aicore__ inline void ReadTile(GlobalTensor<int32_t> state, uint64_t base, uint32_t capacity,
                               uint32_t begin, uint32_t count, uint32_t width, bool reset,
                               I32 tokens) {
    auto slots=tokens[width];
    if (reset) {
        Duplicate(tokens,int32_t{-1},width);
        ArithProgression<int32_t>(slots,static_cast<int32_t>(begin),1,width);
        PipeBarrier<PIPE_V>();
        return;
    }
    // Every observable lane is loaded. Lookup and all output partitions mask
    // [count,width); their padding contents need neither initialization nor repair.
    SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
    const DataCopyPadExtParams<int32_t> noPad{false,0,0,0};
    // Token and slot columns are capacity apart in GM and width apart in UB.
    // GM strides are bytes; the UB gap follows each aligned DMA block.
    constexpr uint32_t wordsPerBlock=32/sizeof(int32_t);
    const DataCopyExtParams copy{2,count*4,(capacity-count)*4,
                                 (width-FsaLruGeometry::AlignWords(count))/wordsPerBlock,0};
    DataCopyPad(tokens,state[base+begin],copy,noPad);
    SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
}

__aicore__ inline void PlanRow(
    GM_ADDR topkInput, GM_ADDR stableInput, GM_ADDR visibleInput, GM_ADDR state,
    GM_ADDR mapping, GM_ADDR output, uint32_t row, uint32_t rows,
    uint32_t physicalRows, uint32_t topkWidth, uint32_t capacity,
    uint32_t paddedTopk, uint32_t tileWidth, uint32_t workWords,
    uint32_t lookupMaskWords, int32_t maxToken, bool hasNextRow) {
    const FsaLruGeometry g(topkWidth,capacity,paddedTopk,tileWidth,workWords,lookupMaskWords);
    const uint32_t k=g.topkWidth,t=g.tile,resident=capacity-1;
    // With one tile and cached hit masks, keys/first are dead after Lookup.
    // Their existing 2*K words can retain all compacted evictions directly.
    const bool keepEvictsInUb=g.lookupMaskWords && resident<=t && resident<=2*k;
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer,g.workWords*4);
    auto header=buffer.Get<int32_t>();
    auto kmasks=header[64].ReinterpretCast<uint8_t>();
    const uint32_t km=g.MaskWords(k)*4,tm=g.MaskWords(t)*4;
    auto cachedInvalid=kmasks[4*km],cachedHit=cachedInvalid[g.lookupMaskWords*4];
    auto topk=cachedHit[g.lookupMaskWords*4].ReinterpretCast<int32_t>();
    auto keys=topk[k].ReinterpretCast<float>(),first=keys[k],winner=first[k];
    auto misses=winner[k].ReinterpretCast<int32_t>();
    auto plan=misses[k].ReinterpretCast<float>();
    auto current=plan[k],evictSlots=current[k];
    auto scanA=evictSlots[k],scanB=scanA[k+8],temp=scanB[k+8];
    auto offsets=temp[k].ReinterpretCast<int32_t>();
    auto valid=kmasks,miss=kmasks[km],aux=kmasks[2*km],assignedMask=kmasks[3*km];
    // Phase lifetimes overlap in the same UB allocation: miss/plan buffers are
    // not live during lookup; tile buffers are not live while encoding plans.
    auto tokens=topk[4*k],slots=tokens[t];
    auto hitPos=slots[t].ReinterpretCast<float>();
    auto tmasks=hitPos[t].ReinterpretCast<uint8_t>();
    auto evict=tmasks,hit=tmasks[tm],active=tmasks[2*tm],scratchMask=tmasks[5*tm];
    auto work=tmasks[8*tm].ReinterpretCast<float>();
    GlobalTensor<int32_t> topkGM,stableGM,visibleGM,stateGM,mapGM,outGM;
    topkGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(topkInput));
    stableGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(stableInput));
    visibleGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(visibleInput));
    stateGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(state));
    mapGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(mapping));
    outGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output));
    const DataCopyPadExtParams<int32_t> noPad{false,0,0,0};
    const DataCopyExtParams one{1,4,0,0,0};
    if (topkWidth < k) {
        // Only the padded tail needs initialization before the DMA.
        Duplicate(topk,int32_t{-1},k);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
    }
    DataCopyPad(topk,topkGM[uint64_t(row)*topkWidth],DataCopyExtParams{1,topkWidth*4,0,0,0},noPad);
    // Pack whole metadata arrays when they fit this phase's existing scratch;
    // otherwise read just the five row values. No geometry-specific route.
    const uint32_t mapSpan=g.AlignWords(4*physicalRows),lenSpan=g.AlignWords(rows);
    const bool batchMetadata=mapSpan+2*lenSpan<=g.ScratchWords(k,t);
    auto meta=batchMetadata?work.ReinterpretCast<int32_t>():header;
    if(batchMetadata) {
        DataCopyPad(meta,mapGM,DataCopyExtParams{1,4*physicalRows*4,0,0,0},noPad);
        DataCopyPad(meta[mapSpan],stableGM,DataCopyExtParams{1,rows*4,0,0,0},noPad);
        DataCopyPad(meta[mapSpan+lenSpan],visibleGM,DataCopyExtParams{1,rows*4,0,0,0},noPad);
    } else {
        DataCopyPad(meta,mapGM[row],one,noPad);
        DataCopyPad(meta[8],mapGM[2*physicalRows+row],one,noPad);
        DataCopyPad(meta[16],mapGM[3*physicalRows+row],one,noPad);
        DataCopyPad(meta[24],stableGM[row],one,noPad);
        DataCopyPad(meta[32],visibleGM[row],one,noPad);
    }
    SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0);
    const uint32_t physical=meta.GetValue(batchMetadata?row:0);
    const bool reset=meta.GetValue(batchMetadata?2*physicalRows+row:8)!=0;
    const uint32_t oldBank=meta.GetValue(batchMetadata?3*physicalRows+row:16);
    int32_t stable=meta.GetValue(batchMetadata?mapSpan+row:24);
    int32_t visible=meta.GetValue(batchMetadata?mapSpan+lenSpan+row:32);
    stable=stable<0?0:(stable>maxToken?maxToken:stable);
    visible=visible<0?0:(visible>maxToken?maxToken:visible);
    for(uint32_t i=topkWidth;i<g.AlignWords(topkWidth);++i) topk.SetValue(i,-1);
    SetFlag<HardEvent::S_V>(0); WaitFlag<HardEvent::S_V>(0);
    const uint64_t oldBase=(uint64_t(physical)*4+2*oldBank)*capacity;
    const uint64_t newBase=(uint64_t(physical)*4+2*(1-oldBank))*capacity;
    SortTopk(topk,keys,first,work,valid,k,maxToken);
    Duplicate(winner,-1.0f,k);
    uint32_t evictCount=0;
    for(uint32_t begin=0;begin<resident;begin+=t) {
        const uint32_t count=resident-begin<t?resident-begin:t;
        ReadTile(stateGM,oldBase,capacity,begin,count,t,reset,tokens);
        const uint32_t maskOffset=g.lookupMaskWords?(begin/t)*tm:0;
        auto lookupHit=g.lookupMaskWords?cachedHit[maskOffset]:hit;
        Lookup<false>(keys,first,tokens,hitPos,work,scratchMask,lookupHit,k,t,count,stable,visible,maxToken,
               cachedInvalid[maskOffset],g.lookupMaskWords!=0);
        CompareScalar(evict,hitPos,-1.0f,CMPMODE::EQ,t);
        ArithProgression<float>(work,0.0f,1.0f,t);
        PipeBarrier<PIPE_V>();
        CompareScalar(active,work,CountAsFloat(count),CMPMODE::LT,t);
        Intersect(evict,active,t);
        auto evictPrefix=keepEvictsInUb?keys.ReinterpretCast<int32_t>():work.ReinterpretCast<int32_t>();
        const uint32_t countEvict=Compact(evictPrefix,slots,evict,t);
        if(!keepEvictsInUb) {
            const uint32_t save=evictCount<k?(countEvict<k-evictCount?countEvict:k-evictCount):0;
            WriteWords(stateGM[newBase+capacity+evictCount],evictPrefix,save);
        }
        evictCount+=countEvict;

        // Each tile selects the last LRU hit of a token. Increasing tile order
        // then reproduces CPU overwrite order, while retaining ALL hit slots.
        Muls(work,hitPos,CountAsFloat(t),t);
        ArithProgression<float>(work[t],0.0f,1.0f,t);
        PipeBarrier<PIPE_V>();
        Add(work,work,work[t],t);
        DataCopy(work[t].ReinterpretCast<int32_t>(),slots,t);
        PipeBarrier<PIPE_V>();
        auto sorted=SortPacked(work,work[t].ReinterpretCast<int32_t>(),work,t);
        DataCopy(tokens.ReinterpretCast<float>(),sorted,t);
        Cast(slots.ReinterpretCast<float>(),sorted[t].ReinterpretCast<int32_t>(),RoundMode::CAST_ROUND,t);
        // The rank vectors are dead after inversion. Reuse them for sorted
        // lookup tables, and the whole sort workspace for K-sized queries.
        auto query=work,lo=work[2*k],candidate=work[3*k],gathered=work[4*k];
        auto queryOffsets=work[5*k].ReinterpretCast<int32_t>();
        ArithProgression<float>(query,CountAsFloat(t-1),CountAsFloat(t),k);
        PipeBarrier<PIPE_V>();
        LowerBound(query,tokens.ReinterpretCast<float>(),t,lo,candidate,
                   gathered,queryOffsets,valid,k);
        Adds(candidate,query,-CountAsFloat(t-1),k);
        PipeBarrier<PIPE_V>();
        Compare(valid,gathered,candidate,CMPMODE::GE,k);
        Compare(aux,gathered,query,CMPMODE::LE,k);
        Intersect(valid,aux,k);
        Gather(gathered,slots.ReinterpretCast<float>(),queryOffsets.ReinterpretCast<uint32_t>(),0,k);
        PipeBarrier<PIPE_V>();
        Select(winner,valid,gathered,winner,SELMODE::VSEL_TENSOR_TENSOR_MODE,k);
        PipeBarrier<PIPE_V>();
    }
    // Plan valid TopK positions, including duplicate misses, in input order.
    // No tile is live in this phase; only two K vectors of scratch are needed.
    work=offsets[k].ReinterpretCast<float>();
    if(keepEvictsInUb) {
        // Only the source changes; scanA/scanB keep their original UB addresses.
        evictSlots=keys;
    } else {
        Duplicate(evictSlots.ReinterpretCast<int32_t>(),int32_t{-1},k);
        PipeBarrier<PIPE_V>();
        if(evictCount) {
            SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
            DataCopyPad(evictSlots.ReinterpretCast<int32_t>(),stateGM[newBase+capacity],
                DataCopyExtParams{1,(evictCount<k?evictCount:k)*4,0,0,0},noPad);
            SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
        }
    }
    // Only sign and the exact current-token difference are observed here.
    // Clamp negatives first so subtraction never overflows, including INT_MIN.
    Maxs(offsets,topk,int32_t{-1},k);
    Cast(scanB,topk,RoundMode::CAST_ROUND,k);
    PipeBarrier<PIPE_V>();
    Adds(offsets,offsets,-visible,k);
    PipeBarrier<PIPE_V>();
    Cast(temp,offsets,RoundMode::CAST_ROUND,k);
    PipeBarrier<PIPE_V>();
    CompareScalar(valid,scanB,0.0f,CMPMODE::GE,k);
    CompareScalar(aux,temp,0.0f,CMPMODE::LT,k);
    Intersect(valid,aux,k);
    CompareScalar(miss,winner,0.0f,CMPMODE::LT,k);
    Intersect(miss,valid,k);
    Duplicate(scanB[8],1.0f,k);
    PipeBarrier<PIPE_V>();
    Select(scanA[8],miss,scanB[8],0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,k);
    PipeBarrier<PIPE_V>();
    auto prefix=fsa_lru_prefix::InclusiveScan(scanA,scanB,work,offsets,k);
    SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0);
    const uint32_t missCount=CountFromFloat(prefix.GetValue(k-1));
    const uint32_t assigned=missCount<evictCount?missCount:evictCount;
    CompareScalar(assignedMask,prefix,CountAsFloat(assigned),CMPMODE::LE,k);
    Intersect(assignedMask,miss,k);
    Adds(work,prefix,-1.0f,k);
    PipeBarrier<PIPE_V>();
    Maxs(work,work,0.0f,k); Mins(work,work,CountAsFloat(k-1),k);
    PipeBarrier<PIPE_V>();
    GatherAt(current,evictSlots,offsets,work,k);
    Cast(current,current.ReinterpretCast<int32_t>(),RoundMode::CAST_ROUND,k);
    PipeBarrier<PIPE_V>();
    Select(current,assignedMask,current,winner,SELMODE::VSEL_TENSOR_TENSOR_MODE,k);
    PipeBarrier<PIPE_V>();
    Compact(misses,topk,miss,k);
    Adds(plan,current,1.0f,k);
    Muls(work,plan,-1.0f,k);
    CompareScalar(aux,temp,-1.0f,CMPMODE::NE,k);
    Intersect(aux,assignedMask,k);
    PipeBarrier<PIPE_V>();
    Select(plan,aux,work,plan,SELMODE::VSEL_TENSOR_TENSOR_MODE,k);
    PipeBarrier<PIPE_V>();
    CompareScalar(aux,current,0.0f,CMPMODE::GE,k);
    Intersect(aux,valid,k);
    Select(plan,aux,plan,0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,k);
    CompareScalar(valid,temp,-1.0f,CMPMODE::EQ,k);
    Intersect(valid,aux,k);
    ArithProgression<float>(work,0.0f,1.0f,k);
    PipeBarrier<PIPE_V>();
    Select(work,valid,work,-1.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,k);
    PipeBarrier<PIPE_V>();
    ReduceMax(work[k],work,work[k+8],k);
    SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0);
    const int currentPosition=static_cast<int>(work.GetValue(k));
    const uint32_t currentSlot=currentPosition<0?resident:CountFromFloat(current.GetValue(currentPosition));
    SetFlag<HardEvent::S_V>(0); WaitFlag<HardEvent::S_V>(0);

    // Stable LRU partition: unused evictions, assigned evictions, then hits.
    // Two contiguous state passes bound UB use for any consumer-legal capacity.
    uint32_t previousEvicts=0,previousHits=0;
    tokens=topk[6*k]; slots=tokens[t]; hitPos=slots[t].ReinterpretCast<float>();
    tmasks=hitPos[t].ReinterpretCast<uint8_t>();
    evict=tmasks; hit=tmasks[tm]; active=tmasks[2*tm]; scratchMask=tmasks[5*tm];
    work=tmasks[8*tm].ReinterpretCast<float>();
    for(uint32_t begin=0;begin<resident;begin+=t) {
        const uint32_t count=resident-begin<t?resident-begin:t;
        ReadTile(stateGM,oldBase,capacity,begin,count,t,reset,tokens);
        if(g.lookupMaskWords) {
            const uint32_t maskOffset=(begin/t)*tm;
            Duplicate(work.ReinterpretCast<int32_t>(),int32_t{-1},t);
            hit=cachedHit[maskOffset];
            PipeBarrier<PIPE_V>();
            Select(tokens.ReinterpretCast<float>(),cachedInvalid[maskOffset],work,tokens.ReinterpretCast<float>(),
                   SELMODE::VSEL_TENSOR_TENSOR_MODE,t);
            Not(evict.ReinterpretCast<uint16_t>(),hit.ReinterpretCast<uint16_t>(),t/16);
            PipeBarrier<PIPE_V>();
        } else {
            Lookup<true>(keys,first,tokens,hitPos,work,scratchMask,hit,k,t,count,stable,visible,maxToken,
                   cachedInvalid,false);
            CompareScalar(evict,hitPos,-1.0f,CMPMODE::EQ,t);
        }
        ArithProgression<float>(work,0.0f,1.0f,t);
        PipeBarrier<PIPE_V>();
        CompareScalar(active,work,CountAsFloat(count),CMPMODE::LT,t);
        Intersect(evict,active,t);
        auto tokenOutput=work.ReinterpretCast<int32_t>();
        auto slotOutput=tokenOutput[t];
        auto alignScratch=slotOutput[t];
        const uint32_t tileEvicts=Compact(slotOutput,slots,evict,t);
        const uint32_t firstAssigned=previousEvicts<assigned?previousEvicts:assigned;
        const uint32_t remaining=assigned-firstAssigned;
        const uint32_t tileAssigned=remaining<tileEvicts?remaining:tileEvicts;
        const uint32_t unassigned=tileEvicts-tileAssigned;
        // GatherMask preserves LRU order. Compact once so assigned evictions
        // are a prefix and unused evictions are the remaining suffix.
        if (unassigned) {
            Compact(tokenOutput,tokens,evict,t);
            WritePartition<true>(stateGM,newBase,capacity,previousEvicts-firstAssigned,
                           tokenOutput,tileAssigned,slotOutput,tileAssigned,unassigned,alignScratch,t);
        }
        if (tileAssigned) {
            WritePartition<false>(stateGM,newBase,capacity,evictCount-assigned+firstAssigned,
                           misses,firstAssigned,slotOutput,0,tileAssigned,alignScratch,t);
        }
        const uint32_t tileHits=count-tileEvicts;
        if (tileHits) {
            Compact(tokenOutput,tokens,hit,t);
            Compact(slotOutput,slots,hit,t);
            WritePartition<true>(stateGM,newBase,capacity,evictCount+previousHits,
                           tokenOutput,0,slotOutput,0,tileHits,alignScratch,t);
        }
        previousEvicts+=tileEvicts; previousHits+=count-tileEvicts;
    }
    // The reserved token and slot are one word in each state column.
    const uint32_t reservedGap = (capacity - 1) * sizeof(int32_t);
    const DataCopyExtParams readReserved{2, sizeof(int32_t), reservedGap, 0, 0};
    const DataCopyExtParams writeReserved{2, sizeof(int32_t), 0, reservedGap, 0};
    DataCopyPad(header,stateGM[oldBase+resident],readReserved,noPad);
    SetFlag<HardEvent::MTE2_MTE3>(0); WaitFlag<HardEvent::MTE2_MTE3>(0);
    DataCopyPad(stateGM[newBase+resident],header,writeReserved);
    Cast(plan.ReinterpretCast<int32_t>(),plan,RoundMode::CAST_ROUND,k);
    ArithProgression<int32_t>(offsets,0,4,k);
    PipeBarrier<PIPE_V>();
    auto packed=work.ReinterpretCast<int16_t>();
    Gather(packed,plan.ReinterpretCast<int16_t>(),offsets.ReinterpretCast<uint32_t>(),0,k);
    SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0);
    if (resident == 0) {
        // K = C = 1. Only the current token is injected into the reserved slot.
        packed.SetValue(0, visible > 0 && topk.GetValue(0) == visible - 1 ? 1 : 0);
    }
    auto control=packed[topkWidth];
    control.SetValue(0,-1); control.SetValue(1,0x5A45); control.SetValue(2,static_cast<int16_t>(topkWidth));
    control.SetValue(3,static_cast<int16_t>(16384-topkWidth)); control.SetValue(4,static_cast<int16_t>(physical));
    control.SetValue(5,0x5A44); control.SetValue(6,static_cast<int16_t>(capacity)); control.SetValue(7,0x5A56);
    for(uint32_t i=8;i<16;++i) control.SetValue(i,-1);
    for(uint32_t i=topkWidth+16;i<g.wireWords*2;++i) packed.SetValue(i,0);
    auto packedWords=packed.ReinterpretCast<int32_t>();
    packedWords.SetValue(g.wireWords,static_cast<int32_t>(physical*capacity+currentSlot));
    for(uint32_t i=g.wireWords+1;i<g.payloadWords;++i) packedWords.SetValue(i,0);
    SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
    DataCopy(outGM[uint64_t(row)*g.payloadWords],packedWords,g.payloadWords);
    // Kernel completion drains the final writes. Wait before the next row
    // only when it can reuse this block's UB.
    if (hasNextRow) {
        SetFlag<HardEvent::MTE3_S>(0); WaitFlag<HardEvent::MTE3_S>(0);
    }
}
} // namespace
