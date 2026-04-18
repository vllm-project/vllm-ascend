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
 * \file write_cache_by_group_list_common.cpp
 * \brief
 */
#include "exe_graph/runtime/extended_kernel_context.h"
#include "store_kv_block_common.h"
#include <chrono>

#include <cstdint>
namespace optiling {

void StoreKVBlockCommonTiling::PrintTilingData()
{
    std::cout<<context_->GetNodeName()<<" Start WriteCacheByGroupListTilingData priting"<<std::endl;
    std::cout<<"params.blockTableSize "<< params.blockTableSize<<std::endl;
    std::cout<<"params.typeByte "<< params.typeByte<<std::endl;
    std::cout<<"params.tokenSize "<< params.tokenSize<<std::endl;
    std::cout<<"params.corepernum "<< params.corepernum<<std::endl;
    std::cout<<"params.coretail "<< params.coretail<<std::endl;
    std::cout<<"params.numTokens "<< params.numTokens<<std::endl;
    std::cout<<"params.numCache "<< params.numCache<<std::endl;
    std::cout<<"params.groupInfoLen "<< params.groupInfoLen<<std::endl;
    std::cout<<context_->GetNodeName()<<" End WriteCacheByGroupListTilingData priting"<<std::endl;

}

void StoreKVBlockCommonTiling::SetTiling()
{
    if (params.blockTableSize<=0) printf("[ZTLOG] params.blockTableSize<=0  \n");
    else tilingData_.set_blockTableSize(params.blockTableSize);
    if (params.typeByte<=0) printf("[ZTLOG] params.typeByte<=0 numTokens\n");
    else  tilingData_.set_typeByte(params.typeByte);

    if (params.tokenSize<=0) printf("[ZTLOG] params.tokenSize<=0 \n");
    else tilingData_.set_tokenSize(params.tokenSize);
    
    if (params.corepernum<=0 && params.coretail==0) printf("[ZTLOG] params.corepernum<=0 && params.coretail==0 \n");
    else tilingData_.set_corePerNum(params.corepernum); 
    
    if (params.coretail>=48) printf("[ZTLOG] params.coretail>=48 \n");
    else tilingData_.set_coreTail(params.coretail); 
    
    if (params.numTokens<=0) printf("[ZTLOG] params.numTokens<=0 \n");
    else tilingData_.set_numTokens(params.numTokens); 
        
    if (params.numCache<=0) printf("[ZTLOG] params.numCache<=0 \n");
    else tilingData_.set_numCache(params.numCache); 

    if (params.groupInfoLen<=0) printf("[ZTLOG] params.groupInfoLen<=0 \n");
    else tilingData_.set_groupInfoLen(params.groupInfoLen); 

    size_t* workspaceSize = context_->GetWorkspaceSizes(1);
    *workspaceSize = params.workspaceSize + params.sysWorkspaceSize;
    context_->SetTilingKey(params.tilingKey);
    if (params.coreNum<=0) printf("[ZTLOG] params.coreNum<=0 \n");
    else     context_->SetBlockDim(params.coreNum);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    
}

ge::graphStatus StoreKVBlockCommonTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    params.coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF((params.coreNum == 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "Failed to get core num."),
                    return ge::GRAPH_FAILED);
    params.sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus StoreKVBlockCommonTiling::DoCommonTiling()
{
    auto kShape = context_->GetInputShape(DIM_0);
    auto kDimNum=kShape->GetStorageShape().GetDimNum();
    if (kDimNum<2||kDimNum>7){
        printf("[ERROR] StoreKVBlock Intput kDimNum dim < 2 || kDimNum>7");
        // OP_LOGE(context_->GetNodeName(), "StoreKVBlock Intput first params dim < 2 || dim_num>7");
    }else {
        for (int i = 0; i < kDimNum; i++){
            if(i==0) params.numTokens = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(i));
            else if (i==1) params.numHeads = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(i));
            else if(static_cast<uint32_t>(kShape->GetStorageShape().GetDim(i))!=0)  params.headSize[i-2]=static_cast<uint32_t>(kShape->GetStorageShape().GetDim(i));
        }
    }

    auto kCacheShape = context_->GetInputShape(DIM_1);
    auto kCacheDimNum=kCacheShape->GetStorageShape().GetDimNum();
    if (kCacheDimNum<2||kCacheDimNum>7){
        printf("[ERROR] StoreKVBlock Intput kCacheDimNum < 2 ");
        // OP_LOGE(context_->GetNodeName(), "StoreKVBlock Intput first params dim < 2 || dim_num>7");
    }else {
        params.numCache = kCacheShape->GetStorageShape().GetDim(0)* kCacheShape->GetStorageShape().GetDim(1);
    }

    const int64_t* blockSizePtr = context_->GetAttrs()->GetInt(0);
    uint32_t blockSize = static_cast<uint32_t>(* blockSizePtr);
    params.tokenSize =  params.numHeads * params.headSize[0] * params.headSize[1] * params.headSize[2] * params.headSize[3] * params.headSize[4];
    params.blockTableSize =  blockSize;

    uint32_t typeByte = 0;
    auto xDataType = context_->GetInputDesc(DIM_0)->GetDataType();
    if (xDataType == ge::DataType::DT_INT8) {
        typeByte = sizeof(int8_t);
        params.tilingKey = 1;
    } else if (xDataType == ge::DataType::DT_FLOAT16 || xDataType == ge::DataType::DT_BF16) {
        typeByte = sizeof(uint16_t);
        params.tilingKey = 2;
    } else if (xDataType == ge::DataType::DT_INT32 || xDataType == ge::DataType::DT_UINT32) {
        typeByte = sizeof(uint32_t);
        params.tilingKey = 4;
    } else {
        OP_LOGE(context_->GetNodeName(), "Unsupport type.");
        return ge::GRAPH_FAILED;
    }

    params.typeByte = typeByte;

    auto groupInfoShape = context_->GetInputShape(DIM_2);
    params.groupInfoLen =  static_cast<uint32_t>(groupInfoShape->GetStorageShape().GetDim(0));
    params.corepernum = params.groupInfoLen/params.coreNum;
    params.coretail= params.groupInfoLen%params.coreNum;
    
    uint32_t pageBlockEleSize = params.blockTableSize*params.tokenSize;
    if( pageBlockEleSize > MAX_UB_USE_SIZE){
        OP_LOGE(context_->GetNodeName(), "pageBlockEleSize > MaxUBSize");
        std::cout<<context_->GetNodeName()<< "  pageBlockEleSize > MaxUBSize"<<std::endl;
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;

}

ge::graphStatus StoreKVBlockCommonTiling::DoTiling()
{

    auto ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = DoCommonTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    SetTiling();

    // PrintTilingData();
    return ge::GRAPH_SUCCESS;
}
    

} // namespace optiling
