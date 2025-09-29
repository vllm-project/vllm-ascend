// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <gtest/gtest.h>
#include "dynamic_eplb_greedy.h"
#include "placement_mapping.h"
#include "tensor.h"
#include "expert_activation.h"
#include "placement_optimizer.h" 
#include "expert_load_balancer.h"

std::vector<int32_t> read_flat_array(const std::string& filename) {
    std::vector<int32_t> result;
    std::ifstream file(filename);
    std::string line;
    
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        int value;
        while (ss >> value) {
            result.push_back(value);
        }
    }

    return result;
}
std::vector<int> read_vector(const std::string& filename) {
    std::vector<int> result;
    std::ifstream file(filename);
    std::string line;
    
    if (file.is_open()) {
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            int value;
            while (ss >> value) {
                result.push_back(value);
            }
        }
        file.close();
    } else {
        std::cerr << "Unable to open file for reading: " << filename << std::endl;
    }

    return result;
}

std::vector<int64_t> read_int64_vector(const std::string& filename) {
    std::vector<int64_t> result;
    std::ifstream file(filename);
    std::string line;
    
    if (file.is_open()) {
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            int value;
            while (ss >> value) {
                result.push_back(value);
            }
        }
        file.close();
    } else {
        std::cerr << "Unable to open file for reading: " << filename << std::endl;
    }

    return result;
}
class GreedyAlgorithmTest : public ::testing::Test {
protected:
    int max_redundants_per_expert=20;
    int num_layers=58;
    int rank = 0;
    int num_devices_per_host = 16;
    int world_size=32;
    int num_logits_expert = 256;
    int deployed_experts_per_layer = num_logits_expert+world_size;
    void* redundant_expert_mapping;
    void* global_expert_mapping;
    void* redundant_count_per_expert;
    PlacementMapping *placement_mapping;
    std::vector<int32_t> placement_pattern_cpu;
    void* selector;

    // For Activation
    void* npu_count_ptr;
    ClusterActivation* activation_ptr;
    int activation_window_size = 10;
    size_t num_deploy_experts_per_rank = deployed_experts_per_layer/world_size;
    int64_t max_activation_count= 10000000000000;

    PlacementOptimizer* optimizer_;

    void SetUp() override {
        // placement_pattern_cpu.resize(world_size*num_layers*num_logits_expert,0);
        // // // g
        // int num_expert_per_rank = num_logits_expert/world_size;
        // for(int rank_id=0;rank_id<world_size;++rank_id){
        //     for (int layer_id=0;layer_id<num_layers;++layer_id){
        //         for(int idx = 0;idx<num_expert_per_rank;++idx){
        //             int local_pos_id = rank_id*num_expert_per_rank+idx;
        //             int offset = rank_id*num_layers*num_logits_expert+layer_id*num_logits_expert+local_pos_id;
        //             placement_pattern_cpu[offset] = 1;
        //         }
        //     }
        // }
        placement_pattern_cpu = read_flat_array("/data/kww/debug/20250709_2121/basepattern32.txt");

        aclInit(NULL); // 初始化 ACL
        aclrtContext context;
        aclrtCreateContext(&context, 0);
        aclrtSetCurrentContext(context);

        size_t size = num_layers*max_redundants_per_expert*num_logits_expert*sizeof(int32_t);
        aclrtMalloc(&redundant_expert_mapping, size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(redundant_expert_mapping, size, 0,size);

        size = num_layers*num_logits_expert*max_redundants_per_expert*sizeof(int32_t);
        aclrtMalloc(&global_expert_mapping, size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(global_expert_mapping, size, 0,size);

        size = num_layers*num_logits_expert*sizeof(int32_t);
        aclrtMalloc(&redundant_count_per_expert, size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(redundant_count_per_expert, size, 0,size);

        std::vector<int64_t> shape1 = {num_layers,max_redundants_per_expert,num_logits_expert};
        std::vector<int64_t> shape2 = {num_layers,num_logits_expert,max_redundants_per_expert};
        std::vector<int64_t> shape3 = {num_layers,num_logits_expert};
        std::vector<int64_t> shape4 = {world_size,num_layers,num_logits_expert};
        void* tmp = placement_pattern_cpu.data();

        size = num_layers*num_logits_expert*sizeof(int32_t);
        aclrtMalloc(&selector, size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(selector, size, 0,size);
        
        placement_mapping = new PlacementMapping("",rank,num_devices_per_host,deployed_experts_per_layer,(size_t) redundant_expert_mapping,shape1,
        (size_t)redundant_count_per_expert, shape3,
        (size_t) tmp,shape4,
        (size_t) selector);
        

        size = num_layers*num_deploy_experts_per_rank*sizeof(int64_t);
        aclrtMalloc(&npu_count_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(npu_count_ptr, size, 0,size);

        Tensor tensor((uint64_t) npu_count_ptr,num_layers*num_deploy_experts_per_rank,sizeof(int64_t),"int64_t","");
        activation_ptr = new ClusterActivation(tensor,max_activation_count,num_layers,num_deploy_experts_per_rank,activation_window_size,world_size,rank);
        activation_ptr->set_params(num_logits_expert);

        optimizer_ = new PlacementOptimizer(placement_mapping, activation_ptr);

        
    };
    void TearDown() override {

    }
};

// ./test_placement --gtest_filter=GreedyAlgorithmTest.DefaultConstructor*
TEST_F(GreedyAlgorithmTest, DefaultConstructor) {
    std::cout<<"222222222222222222222"<<std::endl;
    std::cout<<"placement_mapping->get_num_deploy_experts():" << placement_mapping->get_num_deploy_experts() <<std::endl;
    std::cout<<"activation_ptr->get_num_deploy_experts_per_rank(): "<<activation_ptr->get_num_deploy_experts_per_rank()<<std::endl;
    std::vector<ChangeInstruction> changeInstructions;

    // activation_ptr->collect_from_txt("/data/kww/activation_counts_recordstep_1_rank_0.txt");
    // std::vector<ChangeInstruction> changeInstructions = optimizer_->optimize();
    // activation_ptr->collect_from_txt("/data/kww/debug/type-1/activation_counts_recordstep_1_rank_0.txt");
    // std::vector<ChangeInstruction> changeInstructions = optimizer_->optimize();
    activation_ptr->collect_from_txt("/data/kww/debug/20250709_2121/activation_counts_recordstep_4_rank_0.txt");
    changeInstructions = optimizer_->optimize();
    for (auto& instruction : changeInstructions)
    {
        if (instruction.layer_idx!=8) continue;
        if (instruction.source_rank !=rank && instruction.target_rank!=rank) continue;
        std::cout<<"type["<<(int) instruction.type<<"] layer_idx["<<instruction.layer_idx<<"] source_rank["<<instruction.source_rank<<"] source_global_position["<<instruction.source_global_position<<"] target_rank["<<instruction.target_rank<<"] target_expert_id["<<instruction.target_expert_id<<"] target_global_position["<<instruction.target_global_position<<"]"<<std::endl;
    }
    
    std::cout<<"---------------------------"<<std::endl;

    for (auto& instruction : changeInstructions)
    {
        if (instruction.layer_idx!=8) continue;
        if (instruction.source_rank !=13 && instruction.target_rank!=13) continue;
        std::cout<<"type["<<(int) instruction.type<<"] layer_idx["<<instruction.layer_idx<<"] source_rank["<<instruction.source_rank<<"] source_global_position["<<instruction.source_global_position<<"] target_rank["<<instruction.target_rank<<"] target_expert_id["<<instruction.target_expert_id<<"] target_global_position["<<instruction.target_global_position<<"]"<<std::endl;

        int layer = instruction.layer_idx;
        OperationType type = instruction.type;
        bool need_enqueue_recv_buff=true;
        int t_rank = (instruction.source_rank == rank) ? instruction.target_rank : instruction.source_rank;
        int global_position_id_this_layer = (instruction.source_rank == rank) ? instruction.source_global_position : instruction.target_global_position;
        int expert_id = (instruction.source_rank == rank) ? instruction.target_expert_id : instruction.source_expert_id;
        int position_offset = placement_mapping->getGlobalPositionOffset(layer,global_position_id_this_layer);
        if (type==OperationType::ADD && instruction.source_rank == rank){
            need_enqueue_recv_buff=false;
            position_offset = -1; // add的source端不更新
        }

        if (type==OperationType::REMOVE && instruction.target_rank == rank){
            t_rank = rank; // 自己跟自己握手
            expert_id = -1; // 告诉该位置专家id修改为-1
        }
        std::cout<<"t_rank["<<t_rank<<"]"<<"expert_id["<<expert_id<<"]"<<"position_offset["<<position_offset<<"] \n";
    }

    // for(int i=0;i<24;++i){
    //     std::string name = "/data/kww/debug/type-1/mapping-log/log_"+std::to_string(rank)+"_"+std::to_string(i)+".txt";
    //     std::vector<int> tmp  =read_vector(name);
    //     placement_mapping->update_globalDeployedPositionToLogisticsIdMapping(tmp,3);
    // }

    
}

// ./test_placement --gtest_filter=GreedyAlgorithmTest.Optimize_UpdateMapping*
TEST_F(GreedyAlgorithmTest, Optimize_UpdateMapping) {
    std::vector<ChangeInstruction> changeInstructions;
    activation_ptr->collect_from_txt("/data/kww/debug/20250709_2121/activation_counts_recordstep_4_rank_0.txt");
    changeInstructions = optimizer_->optimize();
    std::vector<bool> layer_update(num_layers,true);
    
    size_t offset;
    for (auto& instruction : changeInstructions)
    {
        if (instruction.type==OperationType::REMOVE){
            offset = instruction.layer_idx*deployed_experts_per_layer+instruction.target_global_position;
            placement_mapping->update_globalDeployedPositionToLogisticsIdMapping(instruction.layer_idx,offset,-1);
        }
        else if (instruction.type==OperationType::ADD){
            offset = instruction.layer_idx*deployed_experts_per_layer+instruction.target_global_position;
            placement_mapping->update_globalDeployedPositionToLogisticsIdMapping(instruction.layer_idx,offset,instruction.source_expert_id);
            std::cout<<"type["<<(int) instruction.type<<"] layer_idx["<<instruction.layer_idx<<"] source_rank["<<instruction.source_rank<<"] source_global_position["<<instruction.source_global_position<<"] target_rank["<<instruction.target_rank<<"] target_expert_id["<<instruction.target_expert_id<<"] target_global_position["<<instruction.target_global_position<<"]"<<std::endl;
        }
        else {
            offset = instruction.layer_idx*deployed_experts_per_layer+instruction.target_global_position;
            placement_mapping->update_globalDeployedPositionToLogisticsIdMapping(instruction.layer_idx,offset,instruction.source_expert_id);
            offset = instruction.layer_idx*deployed_experts_per_layer+instruction.source_global_position;
            placement_mapping->update_globalDeployedPositionToLogisticsIdMapping(instruction.layer_idx,offset,instruction.target_expert_id);
        }
        // std::cout<<"type["<<(int) instruction.type<<"] layer_idx["<<instruction.layer_idx<<"] source_rank["<<instruction.source_rank<<"] source_global_position["<<instruction.source_global_position<<"] target_rank["<<instruction.target_rank<<"] target_expert_id["<<instruction.target_expert_id<<"] target_global_position["<<instruction.target_global_position<<"]"<<std::endl;
    }

    // std::cout<<"---------------------------"<<std::endl;
    // size_t size = num_layers*num_logits_expert*sizeof(int32_t);
    // std::vector<int32_t> tmp(size,0);
    // Tensor selector = placement_mapping->get_selector();
    // selector.to_host(tmp.data());
    // int layer_id =13;
    // for (int expert_id=0; expert_id<num_logits_expert;++expert_id){
    //     std::cout<<tmp[layer_id*num_logits_expert+expert_id]<<" ";
    // }
    // std::cout<<std::endl;
    
    // std::cout<<"********************************************"<<std::endl;
    // for (int rank_id =0; rank_id < world_size;rank_id++){
    //     placement_mapping->set_rank(rank_id);
    //     placement_mapping->update_selector(layer_update);
    //     std::cout<<std::endl;
    //     selector.to_host(tmp.data());
    //     // std::cout<<"rank_id["<<rank_id<<"] "<<tmp[layer_id*num_logits_expert+129]<<std::endl;
    //     for (int expert_id=0; expert_id<num_logits_expert;++expert_id){
    //         std::cout<<tmp[layer_id*num_logits_expert+expert_id]<<" ";
    //     }
    //     std::cout<<std::endl<<std::endl;;
    // }

    // std::cout<<"---------------------------"<<std::endl;
    activation_ptr->collect_from_txt("/data/kww/debug/20250709_2121/activation_counts_recordstep_5_rank_0.txt");
    changeInstructions = optimizer_->optimize();


    
}

// ./test_placement --gtest_filter=GreedyAlgorithmTest.SpecialPatternConstructor*
TEST_F(GreedyAlgorithmTest, SpecialPatternConstructor) {

    std::vector<int> placement = read_vector("/data/kww/dump_data/placement-1.txt");
    std::vector<int64_t> activation = read_int64_vector("/data/kww/dump_data/activations-1.txt");
    optimizer_->optimize(placement, activation);

    // placement = read_vector("/data/kww/vllm-09/debug/placement.txt");
    // activation = read_int64_vector("/data/kww/vllm-09/debug/activations.txt");
    // optimizer_->optimize(placement, activation);
    // activation_ptr->collect_from_txt("/data/kww/vllm-09/debug/activation_counts_recordstep_10_rank_0.txt");
    // optimizer_->optimize();
}

// TEST_F(GreedyAlgorithmTest, SpecialPatternConstructor1) {
//     for(int i=0;i<59;++i){
//         std::string name = "/data/kww/debug/log_"+std::to_string(rank)+"_"+std::to_string(i)+".txt";
//         std::vector<int> tmp  =read_vector(name);
//         placement_mapping->update_globalDeployedPositionToLogisticsIdMapping(tmp,3);
//     }
// }

class SelectorTest : public ::testing::Test {
protected:
    int max_redundants_per_expert=10;
    int num_layers=58;
    int rank = 0;
    int num_devices_per_host = 16;
    int deployed_experts_per_layer = 288;
    int world_size=32;
    int num_logits_expert = 256;
    void* redundant_expert_mapping;
    void* global_expert_mapping;
    void* redundant_count_per_expert;
    void* selector;
    PlacementMapping *placement_mapping;
    std::vector<int32_t> placement_pattern_cpu;

    // For Activation
    void* npu_count_ptr;
    ClusterActivation* activation_ptr;
    int activation_window_size = 10;
    size_t num_deploy_experts_per_rank = deployed_experts_per_layer/world_size;
    int64_t max_activation_count= 10000000000000;

    PlacementOptimizer* optimizer_;

    void SetUp() override {
        // placement_pattern_cpu.resize(world_size*num_layers*num_logits_expert,0);
        // // g
        // int num_expert_per_rank = num_logits_expert/world_size;
        // for(int rank_id=0;rank_id<world_size;++rank_id){
        //     for (int layer_id=0;layer_id<num_layers;++layer_id){
        //         for(int idx = 0;idx<num_expert_per_rank;++idx){
        //             int local_pos_id = rank_id*num_expert_per_rank+idx;
        //             int offset = rank_id*num_layers*num_logits_expert+layer_id*num_logits_expert+local_pos_id;
        //             placement_pattern_cpu[offset] = 1;
        //         }
        //     }
        // }
        // np.savetxt('/data/kww/debug/20250709_2121/basepattern32.txt', tmp.flatten(), fmt='%d', newline=' ')
        // placement_pattern_cpu = read_flat_array("./test_data/basepattern.txt");
        placement_pattern_cpu = read_flat_array("/data/kww/debug/20250709_2121/basepattern32.txt");
        

        aclInit(NULL); // 初始化 ACL
        aclrtContext context;
        aclrtCreateContext(&context, 0);
        aclrtSetCurrentContext(context);

        size_t size = num_layers*max_redundants_per_expert*num_logits_expert*sizeof(int32_t);
        aclrtMalloc(&redundant_expert_mapping, size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(redundant_expert_mapping, size, 0,size);

        size = num_layers*num_logits_expert*max_redundants_per_expert*sizeof(int32_t);
        aclrtMalloc(&global_expert_mapping, size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(global_expert_mapping, size, 0,size);

        size = num_layers*num_logits_expert*sizeof(int32_t);
        aclrtMalloc(&redundant_count_per_expert, size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(redundant_count_per_expert, size, 0,size);

        std::vector<int64_t> shape1 = {num_layers,max_redundants_per_expert,num_logits_expert};
        std::vector<int64_t> shape2 = {num_layers,num_logits_expert,max_redundants_per_expert};
        std::vector<int64_t> shape3 = {num_layers,num_logits_expert};
        std::vector<int64_t> shape4 = {world_size,num_layers,num_logits_expert};
        void* tmp = placement_pattern_cpu.data();

        size = num_layers*num_logits_expert*sizeof(int32_t);
        aclrtMalloc(&selector, size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(selector, size, 0,size);
        
        placement_mapping = new PlacementMapping("",rank,num_devices_per_host,deployed_experts_per_layer,(size_t) redundant_expert_mapping,shape1,
        (size_t) global_expert_mapping,shape2 ,
        (size_t)redundant_count_per_expert, shape3,
        (size_t) tmp,shape4,
        (size_t) selector);
        
    };
    void TearDown() override {

    }
};

// ./test_placement --gtest_filter=SelectorTest.DefaultConstructor*
TEST_F(SelectorTest, DefaultConstructor) {
    size_t size = num_layers*num_logits_expert*sizeof(int32_t);
    // std::vector<int32_t> selector_host = placement_mapping->getSelector();
    // for (int layer_id =0; layer_id<num_layers;layer_id++){
    //     for (int expert_id=0; expert_id<num_logits_expert;++expert_id){
    //         std::cout<<selector_host[layer_id*num_logits_expert+expert_id]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }

    std::vector<int32_t> tmp(size,0);
    Tensor selector = placement_mapping->get_selector();
    selector.to_host(tmp.data());
    std::cout<<"********************************************"<<std::endl;
    for (int layer_id =0; layer_id<num_layers;layer_id++){
        for (int expert_id=0; expert_id<num_logits_expert;++expert_id){
            std::cout<<tmp[layer_id*num_logits_expert+expert_id]<<" ";
        }
        std::cout<<std::endl;
    }

}
// ./test_placement --gtest_filter=SelectorTest.TimeTest*
TEST_F(SelectorTest, TimeTest) {
    std::vector<bool> layer_update(num_layers,true);
    TIME_IT_LABEL("update_placement--rank_"+std::to_string(rank),{
        placement_mapping->update_selector(layer_update);
    });
}