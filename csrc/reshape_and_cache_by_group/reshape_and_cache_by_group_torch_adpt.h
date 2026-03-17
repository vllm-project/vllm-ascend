/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
//  #include "../aclnn_torch_adapter/op_api_common.h"

#ifndef RESHAPE_AND_CACHE_BY_GROUP_TORCH_ADPT_H
#define RESHAPE_AND_CACHE_BY_GROUP_TORCH_ADPT_H

namespace vllm_ascend {
typedef struct {
    int32_t length;         // 该group长度
    int32_t keyIdx;    // 输入key的位置
    int32_t keyCacheIdx;         // keycache的位置
} GroupInfo;

at::Tensor cache_by_group_pre(
    const at::Tensor &slotMappingNpu,
    at::IntArrayRef slotMappingList,
    int64_t blockSize)
{
    int64_t slotMappingLen =  slotMappingList.size();

    std::vector<GroupInfo> allGroups(16, GroupInfo{0,0,0});
    int32_t idxSlotmap = 0;
    int32_t idxGroups = 0;
    while (idxSlotmap < slotMappingLen) {

        // 1. 获取当前起始元素的索引和所属block信息
        int32_t current_idx = slotMappingList[idxSlotmap];
        if(current_idx <0){
            while(idxSlotmap<slotMappingLen){
                if(slotMappingList[idxSlotmap++] >=0){
                    TORCH_CHECK(0>1, "cache_by_group_pre find slotMappingList err ");
                    at::Tensor group_info_npu = at::empty_like({slotMappingNpu});
                    return group_info_npu;
                }
            }
            break;
        }
        // 检测slotMapping有小于0的异常值
        // if(slotMappingList[idxSlotmap]<0){
        //     OP_LOGE(context_->GetNodeName(), "slotMappingList < 0");
        //     return ge::GRAPH_FAILED;
        // }

        int32_t blockid = current_idx / blockSize;       // 所属block编号
        int32_t y= current_idx % blockSize;
        // 检测有无重复且非连续的block,暂时不处理直接抛出错误
        // if((++blockCount[allGroups[idxGroups].quotient])>1){
        //     // OP_LOGE(context_->GetNodeName(), "Exited discontinuous and repeated Block");
        //     std::cout<<context_->GetNodeName()<<"  Exited discontinuous and repeated Block"<<current_idx<<std::endl;
        //     // return ge::GRAPH_FAILED;
        // }

        allGroups[idxGroups].keyIdx = idxSlotmap;                          // 该组起始位置
        allGroups[idxGroups].keyCacheIdx = current_idx;    
        // 计算该block理论上的最后一个元素值及其索引跳block计算
        int32_t j = idxSlotmap;
        // 如果无序
        if(j+1 < slotMappingLen &&slotMappingList[j+1]!=slotMappingList[j]+1 ) {
            j++;

        }else{//如果有序
            int32_t idx_stride = std::min(blockSize-y,slotMappingLen-idxSlotmap)-1;
            int32_t expected_last =  current_idx + idx_stride;
            int32_t expected_last_idx = idxSlotmap + (expected_last-current_idx);
            //如果是完整block
            if (expected_last == slotMappingList[expected_last_idx]){
                j = expected_last_idx+1;
            }else{
                // 3. 找该组实际的最后一个元素位置
                // 循环条件：不越界 + 当前元素属于当前block + 未超过理论最后元素; 
                while (j < slotMappingLen && slotMappingList[j] / blockSize == blockid && slotMappingList[j] <= expected_last) {
                    j++;
                }
            }
        }

        // 4. 计算该组长度
        allGroups[idxGroups].length = (j - idxSlotmap);
        // 5. 直接跳到下一组起始位置（核心：跳过中间元素，降低时间复杂度）
        idxSlotmap = j;
        idxGroups++;
        // 6. 超了扩容

        if(idxGroups>=allGroups.capacity()){
            int32_t newCapacity = allGroups.capacity() * 2;
            allGroups.reserve(newCapacity);
            for (int32_t k = idxGroups; k < newCapacity; ++k) allGroups.emplace_back(GroupInfo{0,0,0});
        }
    }

    at::Tensor group_info_npu = at::empty({idxGroups*3},
        at::TensorOptions(slotMappingNpu.options().device()).dtype(torch::kInt32)
        );
    void* devAddr = group_info_npu.data_ptr();
    uint32_t deviceSize=idxGroups*sizeof(allGroups[0]);

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("cache_by_group_pre");
    cmd.SetCustomHandler([ stream,devAddr,allGroups,deviceSize]() -> int {
        aclrtMemcpyKind memcpy_type=ACL_MEMCPY_HOST_TO_DEVICE;
        // int ret =aclrtMemcpy(devAddr, device_size, &allGroups[0], device_size,  memcpy_type);
        aclrtMemcpyAsync(devAddr, deviceSize, &allGroups[0], deviceSize, ACL_MEMCPY_HOST_TO_DEVICE, stream);  
        return 0;
    });
    cmd.Run();
    return group_info_npu;
}

void reshape_and_cache_by_group(
    const at::Tensor &keyIn,
    const at::Tensor &keyCacheIn,
    const at::Tensor &groupInfo,
    int64_t blockSize)
{
    EXEC_NPU_CMD(aclnnReshapeAndCacheByGroup, keyIn, keyCacheIn,groupInfo, blockSize);
     
} 

}
#endif