/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file reshape_and_cache_base.h
 * \brief
 */

#ifndef ASCEND_STORE_KV_BLOCK_H
#define ASCEND_STORE_KV_BLOCK_H

#include "kernel_operator.h"

namespace StoreKVBlock {
using namespace AscendC;


#ifndef STORE_KV_BLOCK_TILING_DATA_H_
#define STORE_KV_BLOCK_TILING_DATA_H_
struct StoreKVBlockTilingData{
    uint32_t blockTableSize;    
    uint32_t typeByte;
    uint32_t tokenSize;
    uint32_t corePerNum;
    uint32_t coreTail;
    uint32_t numTokens;
    uint32_t numCache;
    uint32_t groupInfoLen;

};
#endif
template <typename T>
class StoreKVBlockBase {
public:

    uint32_t tokenSize = 0;
    uint32_t tokenByteSize = 0;
    uint32_t blockTableSize = 0;
    uint32_t typeByte = 0;
    uint32_t numTokens = 0;
    uint32_t numCache = 0;
    uint32_t groupInfoLen = 0;

    uint32_t coreId = 0;
    uint32_t coreTail = 0;
    uint32_t corePerNum = 0;
    uint32_t blockNum = 0;
    AscendC::TPipe* pipeThis;
    AscendC::LocalTensor<T> tokenLocal;
    AscendC::GlobalTensor<T> keyInputGt;
    AscendC::GlobalTensor<T> keyCacheIntputGt;
    AscendC::GlobalTensor<uint32_t> groupLenGt;
    AscendC::GlobalTensor<uint32_t> groupKeyIdxGt;
    AscendC::GlobalTensor<uint32_t> groupKeyCacheIdxGt;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tokenBuf;
    __aicore__ inline StoreKVBlockBase() {}

    __aicore__ inline uint32_t RoundUp(uint32_t x, uint32_t y = 16)
    {
        return y == 0 ? 0 : (x + y - 1) / y * y;
    }

    __aicore__ inline void Init( AscendC::TPipe *pipe, StoreKVBlockTilingData *tilingData)
    {
        pipeThis = pipe;
        typeByte = tilingData->typeByte;
        tokenSize = tilingData->tokenSize;
        tokenByteSize = tokenSize*typeByte;
        blockTableSize = tilingData->blockTableSize;
        numTokens = tilingData->numTokens;
        numCache = tilingData->numCache;
        groupInfoLen = tilingData->groupInfoLen;


        coreId = AscendC::GetBlockIdx();
        coreTail = tilingData->coreTail;
        blockNum = AscendC::GetBlockNum();
        if (coreId < coreTail){
            corePerNum = tilingData->corePerNum+1;
        }else {
            corePerNum = tilingData->corePerNum;
        }
    }
    __aicore__ inline void Process(GM_ADDR keyIn, GM_ADDR keyCacheIn, GM_ADDR groupLen, GM_ADDR groupKeyIdx, GM_ADDR groupKeyCacheIdx)
    {
        
        keyInputGt.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(keyIn));
        keyCacheIntputGt.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(keyCacheIn));
        groupLenGt.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(groupLen));
        groupKeyIdxGt.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(groupKeyIdx));
        groupKeyCacheIdxGt.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(groupKeyCacheIdx));

        pipeThis->InitBuffer(tokenBuf,  blockTableSize*tokenByteSize);
        tokenLocal = tokenBuf.Get<T>();

        AscendC::DataCopyExtParams copyParams{1, 0,  0, 0, 0};//todo完整块长度
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        for (uint32_t i = 0; i < corePerNum; i++) {
            uint32_t idx = (coreId+i*blockNum);
         
            // if( groupLenGt.GetValue(idx)<= 0 || groupKeyIdxGt.GetValue(idx)<0 || groupKeyCacheIdxGt.GetValue(idx)<0){
            //     continue;
            // }
           
            copyParams.blockLen = groupLenGt.GetValue(idx)*tokenByteSize; 
            DataCopyPad(tokenLocal, keyInputGt[ groupKeyIdxGt.GetValue(idx)*tokenSize], copyParams, padParams); 
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
            DataCopyPad(keyCacheIntputGt[groupKeyCacheIdxGt.GetValue(idx)*tokenSize], tokenLocal, copyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }

    }

};
}

#endif
