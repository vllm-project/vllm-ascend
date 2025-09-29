// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "placement_mapping.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include "tensor.h"

// 读取 .txt 文件到连续内存的一维数组，返回指针和维度信息
int32_t* load3DArrayFromFile(const std::string& filename, std::vector<int64_t> &shape) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "无法打开文件：" << filename << std::endl;
        return nullptr;  // 返回空指针表示失败
    }

    // 读取 shape
    file >> shape[0] >> shape[1] >> shape[2];

    // 计算总元素数量
    int totalSize = shape[0] * shape[1] * shape[2];
    if (totalSize <= 0) {
        std::cerr << "无效的数组维度：" << shape[0] << "x" << shape[1] << "x" << shape[2] << std::endl;
        return nullptr;
    }

    // 分配连续内存
    int32_t* data = new int32_t[totalSize];
    if (!data) {
        std::cerr << "内存分配失败！" << std::endl;
        return nullptr;
    }

    // 读取所有数据到连续内存
    for (int i = 0; i < totalSize; ++i) {
        if (!(file >> data[i])) {
            std::cerr << "数据读取失败，文件中的数据量不足！" << std::endl;
            delete[] data;  // 释放已分配的内存
            return nullptr;
        }
    }

    // 检查是否有多余数据
    int extra;
    if (file >> extra) {
        std::cerr << "警告：文件中包含多余数据！" << std::endl;
        // 可以选择是否因此失败
    }

    file.close();
    return data;
}
const int ScalarType_Int = 3;

// 使用模拟的aclrtMemcpy函数



aclError my_mem_fun(void* dst, size_t destMax, const void* src,
                         size_t count, aclrtMemcpyKind kind) {
    if (dst == nullptr || src == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    memcpy(dst, src, count);
    return ACL_ERROR_NONE;
}


class PlacementMappingTest : public ::testing::Test {
    protected:
        memcpy_fun_t old_fun = nullptr;
        void SetUp() override {
            aclInit(NULL); // 初始化 ACL
            aclrtContext context;
            aclrtCreateContext(&context, 0);
            aclrtSetCurrentContext(context);
            
            // 初始化测试数据
            rank_ = 1;
            max_redundant_count_ = 2;
            num_devices_per_host_ = 8;

            std::string filename = "./test_data/placement_pattern_3d_v3_indevrrori0322_58moe_4devices_SC_64eps.txt";
            placement_pattern_ptr = load3DArrayFromFile(filename, placement_pattern_shape);
            world_size_ = placement_pattern_shape[0];
            num_layers_ = placement_pattern_shape[1];
            num_experts_ = placement_pattern_shape[2];
            
            // 创建expert_mapping的mock数据
            expert_mapping_shape[0] = num_layers_;
            expert_mapping_shape[1] = max_redundant_count_;
            expert_mapping_shape[2] = num_experts_;
            count_shape_[0] = num_layers_;
            count_shape_[1] = num_experts_;
            
            // 分配模拟 HBM 内存
            size_t size = num_layers_ * max_redundant_count_ * num_experts_ * sizeof(int32_t);
            ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc((void**)&redundant_expert_mapping_, size, ACL_MEM_MALLOC_HUGE_FIRST));
            aclrtMemset((void*)redundant_expert_mapping_, size, 0, size);
            size = num_layers_ * num_experts_ * sizeof(int32_t);
            ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc((void**)&redundant_count_per_expert_, size, ACL_MEM_MALLOC_HUGE_FIRST));
            aclrtMemset((void*)redundant_count_per_expert_, size, 0, size);

            // old_fun = get_memcpy_fun();
            // set_memcpy_fun(&my_mem_fun);

            // 注意：这里创建了一个局部对象PMINI，它会在SetUp结束时被销毁
            // 如果需要在测试中使用，应该将其作为类成员或在每个测试函数中创建
            // PlacementMapping PMINI(rank, num_devices_per_host,
            //     redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
            //     expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
            //     placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);
        }

        void TearDown() override {
            // 使用free()而不是delete释放malloc()分配的内存
            if (redundant_expert_mapping_) {
                aclrtFree(redundant_expert_mapping_);
                redundant_expert_mapping_ = nullptr;
            }
            if (redundant_count_per_expert_) {
                aclrtFree(redundant_count_per_expert_);
                redundant_count_per_expert_ = nullptr;
            }
            if (placement_pattern_ptr) {
                free(placement_pattern_ptr);
                placement_pattern_ptr = nullptr;
            }       
            // set_memcpy_fun(old_fun);
        }

        int32_t rank_;
        int32_t world_size_;
        int32_t num_layers_;
        int32_t num_experts_;
        int32_t max_redundant_count_;
        int32_t num_devices_per_host_;

        int32_t* redundant_expert_mapping_;
        int32_t* redundant_count_per_expert_;
        std::vector<int64_t> expert_mapping_shape = std::vector<std::int64_t>(3, 0);
        std::vector<int64_t> count_shape_ = std::vector<std::int64_t>(2, 0);

        int32_t * placement_pattern_ptr = nullptr;  // 初始化为nullptr以防止野指针
        std::vector<int64_t> placement_pattern_shape = std::vector<std::int64_t>(3, 0);
    };

// 测试构造函数
TEST_F(PlacementMappingTest, Constructor) {
    EXPECT_NO_THROW({
        const std::string filename = "./test_data/placement_pattern_3d_v3_indevrrori0322_58moe_4devices_SC_64eps.txt";
        PlacementMapping pm(filename, rank_, num_devices_per_host_, (int)placement_pattern_shape[2], 
                          (size_t)redundant_expert_mapping_, expert_mapping_shape,
                          (size_t)redundant_count_per_expert_, count_shape_,
                          (size_t)placement_pattern_ptr, placement_pattern_shape,
                          (size_t)redundant_count_per_expert_);

        // 验证基本属性
        EXPECT_EQ(pm.get_rank(), rank_);
        EXPECT_EQ(pm.get_world_size(), placement_pattern_shape[0]);
        EXPECT_EQ(pm.get_num_layers(), placement_pattern_shape[1]);
        EXPECT_EQ(pm.get_num_experts(), placement_pattern_shape[2]);
        EXPECT_EQ(pm.get_num_devices_per_host(), num_devices_per_host_);
    });
}

TEST_F(PlacementMappingTest, TestPositionToEpidNaive) {
    const std::string filename = "./test_data/placement_pattern_3d_v3_indevrrori0322_58moe_4devices_SC_64eps.txt";
    PlacementMapping pm(filename, rank_, num_devices_per_host_, (int)placement_pattern_shape[2], 
                        (size_t)redundant_expert_mapping_, expert_mapping_shape,
                        (size_t)redundant_count_per_expert_, count_shape_,
                        (size_t)placement_pattern_ptr, placement_pattern_shape,
                        (size_t)redundant_count_per_expert_);

    auto position_to_epid = pm.get_global_deployed_position_to_logistics_id_mapping();

    // 1. 验证行数
    EXPECT_EQ(position_to_epid.size(), pm.get_num_layers() * pm.get_num_deploy_experts())
        << "position_to_epid_ size mismatch with layers * experts";

    // 2. 验证列数 & epid 合法性
    for (size_t i = 0; i < position_to_epid.size(); ++i) {
        std::cout << " " << position_to_epid[i];
        EXPECT_GE(position_to_epid[i], 0)
            << "Invalid epid at position_to_epid_[" << i << "]: "
            << position_to_epid[i];
        EXPECT_GE(pm.get_num_experts() - 1, position_to_epid[i])
            << "Invalid epid at position_to_epid_[" << i << "]: "
            << position_to_epid[i];
    }
    std::cout  << std::endl;
}

// 检查这个 get_position_to_epid() 是否满足基本要求： epid合法以及全覆盖性；
TEST_F(PlacementMappingTest, TestPositionToEpidBasicLogic)  {
    const std::string filename = "./test_data/placement_pattern_3d_v3_indevrrori0322_58moe_4devices_SC_64eps.txt";
    PlacementMapping pm(filename, rank_, num_devices_per_host_, (int)placement_pattern_shape[2], 
                        (size_t)redundant_expert_mapping_, expert_mapping_shape,
                        (size_t)redundant_count_per_expert_, count_shape_,
                        (size_t)placement_pattern_ptr, placement_pattern_shape,
                        (size_t)redundant_count_per_expert_);

    auto position_to_epid = pm.get_global_deployed_position_to_logistics_id_mapping();
    int num_experts = pm.get_num_experts();
    pm.printDeployedPositionToLogisticsIdMapping();
    // 验证 position_to_epid 的合法性
    for (size_t i = 0; i < pm.get_num_layers(); ++i) {

        std::unordered_set<int> seen_experts; // 记录出现过的 expert ID

        for (size_t j = 0; j < pm.get_num_deploy_experts(); ++j) {
            auto index = i * pm.get_num_deploy_experts() + j;
            int epid = position_to_epid[index];

            // 1. 检查 epid 是否非负
            EXPECT_GE(epid, 0)
                << "Invalid epid at position_to_epid_[" << i << "][" << j << "]: " << epid;

            // 2. 检查 epid 是否小于 num_experts（不越界）
            EXPECT_LT(epid, num_experts)
                << "Out-of-bounds epid at position_to_epid_[" << i << "][" << j << "]: " << epid
                << " (num_experts=" << num_experts << ")";

            // 记录出现的 expert ID
            seen_experts.insert(epid);
        }

        // 3. 检查 0 ~ num_experts-1 是否都出现过
        EXPECT_EQ(seen_experts.size(), static_cast<size_t>(num_experts))
            << "Missing experts in position_to_epid_[" << i << "]: expected " << num_experts
            << " unique expert IDs, but found " << seen_experts.size();

        for (int expert_id = 0; expert_id < num_experts; ++expert_id) {
            EXPECT_TRUE(seen_experts.count(expert_id) > 0)
                << "Expert ID " << expert_id << " is missing in position_to_epid_[" << i << "]";
        }
    }
}

/*
TEST_F(PlacementMappingTest, TestRedundantEpidThisHostBasic) {

    PlacementMapping pm(rank, num_devices_per_host,
                        redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                        expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                        placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    auto redundant_epid_this_host = pm.get_redundant_epid_this_host();
    // auto& redundant_epid_this_host = pm.get_position_to_epid();
    // std::cout << "333333337777777777  pm.get_redundant_epid_this_host(): " << redundant_epid_this_host.size() << std::endl;

    // 1. 验证 redundant_epid_this_host 层数
    EXPECT_EQ(redundant_epid_this_host.size(), pm.get_num_layers())
        << "redundant_epid_this_host_ size mismatch with num_layers_";

    // 1.1 验证 redundant_epid_this_host 列数
    int ww = std::min(pm.get_num_devices_per_host(), pm.get_world_size() );
    EXPECT_EQ(redundant_epid_this_host[0].size(), (pm.get_num_deploy_experts() - pm.get_num_experts()) / pm.get_world_size() * ww )
        << "position_to_epid_ size mismatch with placement_pattern_shape[1]";


    // 2. 验证每层的 epid 是否合法
    for (size_t i = 0; i < redundant_epid_this_host.size(); ++i) {
        for (size_t j = 0; j < redundant_epid_this_host[i].size(); ++j) {
            EXPECT_GE(redundant_epid_this_host[i][j], 0)
                << "Invalid epid at redundant_epid_this_host_[" << i << "][" << j << "]: "
                << redundant_epid_this_host[i][j];

            EXPECT_GE(pm.get_num_deploy_experts() - 1 , redundant_epid_this_host[i][j])
            << "Invalid epid at redundant_epid_this_host_[" << i << "][" << j << "]: "
            << redundant_epid_this_host[i][j];

        }
    }
}


TEST_F(PlacementMappingTest, TestRedundantEpidThisHostBasicLogic) {

    PlacementMapping pm(rank, num_devices_per_host,
                        redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                        expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                        placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    auto redundant_epid_this_host = pm.get_redundant_epid_this_host();
    // auto& redundant_epid_this_host = pm.get_position_to_epid();
    // std::cout << "333333337777777777  pm.get_redundant_epid_this_host(): " << redundant_epid_this_host.size() << std::endl;


    //  验证  epid 是否合法
    for (size_t i = 0; i < redundant_epid_this_host.size(); ++i) {
        for (size_t j = 0; j < redundant_epid_this_host[i].size(); ++j) {
            EXPECT_GE(redundant_epid_this_host[i][j], 0)
                << "Invalid epid at redundant_epid_this_host_[" << i << "][" << j << "]: "
                << redundant_epid_this_host[i][j];

            EXPECT_GE(pm.get_num_experts() - 1 , redundant_epid_this_host[i][j])
            << "Invalid epid at redundant_epid_this_host_[" << i << "][" << j << "]: "
            << redundant_epid_this_host[i][j];
        }
    }
}

// //////////////////////////////////////////////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////////////////////////////////////////////

// //////////////////////////////////////////////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////////////////////////////////////////////


TEST_F(PlacementMappingTest, TestRedundantPosThisHostBasic) {

    PlacementMapping pm(rank, num_devices_per_host,
                        redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                        expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                        placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    auto redundant_epid_this_host = pm.get_redundant_epid_this_host();
    // auto& redundant_epid_this_host = pm.get_position_to_epid();
    // std::cout << "333333337777777777  pm.get_redundant_epid_this_host(): " << redundant_epid_this_host.size() << std::endl;

    // 1. 验证 redundant_epid_this_host 层数
    EXPECT_EQ(redundant_epid_this_host.size(), pm.get_num_layers())
        << "redundant_epid_this_host_ size mismatch with num_layers_";

    // 1.1 验证 redundant_epid_this_host 列数
    int ww = std::min(pm.get_num_devices_per_host(), pm.get_world_size() );
    EXPECT_EQ(redundant_epid_this_host[0].size(), (pm.get_num_deploy_experts() - pm.get_num_experts()) / pm.get_world_size() * ww )
        << "position_to_epid_ size mismatch with placement_pattern_shape[1]";


    // 2. 验证每层的 epid 是否合法
    for (size_t i = 0; i < redundant_epid_this_host.size(); ++i) {
        for (size_t j = 0; j < redundant_epid_this_host[i].size(); ++j) {
            EXPECT_GE(redundant_epid_this_host[i][j], 0)
                << "Invalid epid at redundant_epid_this_host_[" << i << "][" << j << "]: "
                << redundant_epid_this_host[i][j];

            EXPECT_GE(pm.get_num_deploy_experts() - 1 , redundant_epid_this_host[i][j])
            << "Invalid epid at redundant_epid_this_host_[" << i << "][" << j << "]: "
            << redundant_epid_this_host[i][j];

        }
    }
}


TEST_F(PlacementMappingTest, TestRedundantPosThisHostBasicLogic) {

    PlacementMapping pm(rank, num_devices_per_host,
                        redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                        expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                        placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    auto get_redundant_positionid_this_host = pm.get_redundant_positionid_this_host();
    // auto& redundant_epid_this_host = pm.get_position_to_epid();
    // std::cout << "333333337777777777  pm.get_redundant_positionid_this_host(): " << redundant_epid_this_host.size() << std::endl;


    //  验证  epid 是否合法
    for (size_t i = 0; i < get_redundant_positionid_this_host.size(); ++i) {
        for (size_t j = 0; j < get_redundant_positionid_this_host[i].size(); ++j) {
            EXPECT_GE(get_redundant_positionid_this_host[i][j], 0)
                << "Invalid epid at get_redundant_positionid_this_host[" << i << "][" << j << "]: "
                << get_redundant_positionid_this_host[i][j];

            EXPECT_GE(pm.get_num_deploy_experts() - 1 , get_redundant_positionid_this_host[i][j])
            << "Invalid epid at get_redundant_positionid_this_host[" << i << "][" << j << "]: "
            << get_redundant_positionid_this_host[i][j];
        }
    }
}
// //////////////////////////////////////////////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_F(PlacementMappingTest, TestUpdateRedundantExpertMappingBasic111) {
    PlacementMapping pm(rank, num_devices_per_host,
                        redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                        expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                        placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    auto redundant_epid_this_host = pm.get_redundant_epid_this_host();
    auto redundant_positionid_this_host = pm.get_redundant_positionid_this_host();

    // 1. 先确保 redundant_epid_this_host 的层数匹配
    EXPECT_EQ(redundant_epid_this_host.size(), pm.get_num_layers())
        << "redundant_epid_this_host size mismatch with num_layers";

    // 2. 预填充测试数据
    int layer_id = 0;
    int target_position = 1;
    int source_epid = pm.get_num_experts() / 2;


    // 3. 测试异常情况：超出 layer_id 范围
    EXPECT_THROW(pm.update_Redundant_Expert_Mapping(-1, target_position, source_epid), std::out_of_range);
    EXPECT_THROW(pm.update_Redundant_Expert_Mapping(pm.get_num_layers(), target_position, source_epid), std::out_of_range);

    // 4. 测试异常情况：target_position 不在 redundant_positionid_this_host 中

    EXPECT_THROW(pm.update_Redundant_Expert_Mapping(layer_id, -1, source_epid), std::out_of_range);
    EXPECT_THROW(pm.update_Redundant_Expert_Mapping(layer_id, pm.get_num_deploy_experts(), source_epid), std::out_of_range);

    // 5. 测试异常情况：source_epid 越界
    EXPECT_THROW(pm.update_Redundant_Expert_Mapping(layer_id, target_position, -1), std::out_of_range);
    EXPECT_THROW(pm.update_Redundant_Expert_Mapping(layer_id, target_position, pm.get_num_experts()), std::out_of_range);
}



TEST_F(PlacementMappingTest, TestUpdateRedundantExpertMappingForAll) {
    PlacementMapping pm(rank, num_devices_per_host,
                        redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                        expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                        placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    auto redundant_epid_this_host = pm.get_redundant_epid_this_host();
    auto redundant_epid_this_host_NEW = pm.get_redundant_epid_this_host();
    auto redundant_positionid_this_host = pm.get_redundant_positionid_this_host();
    int target_position;
    int layer_id;
    int source_epid = pm.get_num_experts() / 2;

    // 1. 先确保 redundant_epid_this_host 的层数匹配
    EXPECT_EQ(redundant_epid_this_host.size(), pm.get_num_layers())
        << "redundant_epid_this_host size mismatch with num_layers";


    for (int i = 0; i < pm.get_num_layers(); i++) {
        for (size_t j = 0; j < redundant_epid_this_host[0].size(); j++) {

            // 2. 预填充测试数据
            layer_id = i;
            target_position = redundant_positionid_this_host[i][j];

            // 3. 调用函数进行更新
            pm.update_Redundant_Expert_Mapping(layer_id, target_position, source_epid);

            // 获取更新后的数据
            auto redundant_epid_this_host_NEW = pm.get_redundant_epid_this_host();

            // 4. 检查 target_position 对应的 expert id 是否更新正确
            EXPECT_EQ(redundant_epid_this_host_NEW[layer_id][j], source_epid)
                << "Epid at target_position was not updated correctly";
        }
    }


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_F(PlacementMappingTest, TestUpdatePosition_To_ExpertBasic111) {
    PlacementMapping pm(rank, num_devices_per_host,
                        redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                        expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                        placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    auto redundant_epid_this_host = pm.get_redundant_epid_this_host();
    auto redundant_positionid_this_host = pm.get_redundant_positionid_this_host();

    // 1. 先确保 redundant_epid_this_host 的层数匹配
    EXPECT_EQ(redundant_epid_this_host.size(), pm.get_num_layers())
        << "redundant_epid_this_host size mismatch with num_layers";

    // 2. 预填充测试数据
    int layer_id = 0;
    int target_position = 1;
    int source_epid = pm.get_num_experts() / 2;


    // 3. 测试异常情况：超出 layer_id 范围
    EXPECT_THROW(pm.update_Position_To_Expert_Mapping(-1, target_position, source_epid), std::out_of_range);
    EXPECT_THROW(pm.update_Position_To_Expert_Mapping(pm.get_num_layers(), target_position, source_epid), std::out_of_range);

    // 4. 测试异常情况：target_position 不在 redundant_positionid_this_host 中
    EXPECT_THROW(pm.update_Position_To_Expert_Mapping(layer_id, -1, source_epid), std::out_of_range);
    EXPECT_THROW(pm.update_Position_To_Expert_Mapping(layer_id, pm.get_num_deploy_experts(), source_epid), std::out_of_range);

    // 5. 测试异常情况：source_epid 越界
    EXPECT_THROW(pm.update_Position_To_Expert_Mapping(layer_id, target_position, -1), std::out_of_range);
    EXPECT_THROW(pm.update_Position_To_Expert_Mapping(layer_id, target_position, pm.get_num_experts()), std::out_of_range);
}
// //////////////////////////////////////////////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_F(PlacementMappingTest, TestUpdatePosition_To_ExpertForAll) {
    PlacementMapping pm(rank, num_devices_per_host,
                        redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                        expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                        placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    auto redundant_epid_this_host = pm.get_redundant_epid_this_host();
    auto redundant_epid_this_host_NEW = pm.get_redundant_epid_this_host();
    auto position_to_epid_this_host = pm.get_position_to_epid();
    int source_epid = pm.get_num_experts() / 2;

    // 1. 先确保 redundant_epid_this_host 的层数匹配
    EXPECT_EQ(redundant_epid_this_host.size(), pm.get_num_layers())
        << "redundant_epid_this_host size mismatch with num_layers";


    for (int i = 0; i < pm.get_num_layers(); i++) {
        for (int j = 0; j < pm.get_num_deploy_experts(); j++) {


            // 3. 调用函数进行更新
            pm.update_Position_To_Expert_Mapping(i, j, source_epid);

            // 获取更新后的数据
            position_to_epid_this_host = pm.get_position_to_epid();

            // 4. 检查 target_position 对应的 expert id 是否更新正确
            EXPECT_EQ(position_to_epid_this_host[i][j], source_epid)
                << "Epid at target_position was not updated correctly";

            source_epid = j % pm.get_num_experts();
        }
    }


}





//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////// 超融合测试////////////

TEST_F(PlacementMappingTest, TestUltimate_ExpertForAll) {
    PlacementMapping pm(rank, num_devices_per_host,
                        redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                        expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                        placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    auto redundant_epid_this_host = pm.get_redundant_epid_this_host();
    auto redundant_posid_this_hos = pm.get_redundant_positionid_this_host();
    auto position_to_epid_this_host = pm.get_position_to_epid();

    int epid_tmp;
    int posid_tmp;

    for (int i = 0; i < pm.get_num_layers(); i++) {
        for (size_t j = 0; j < redundant_epid_this_host[0].size(); j++) {

            epid_tmp = redundant_epid_this_host[i][j];
            posid_tmp = redundant_posid_this_hos[i][j];

            //  检查 posid_tmp 对应的 epid_tmp 是否更新正确
            EXPECT_EQ(position_to_epid_this_host[i][posid_tmp], epid_tmp)
                << "Epid at target_position was not updated correctly";

        }
    }


}


// 测试正常情况 - 基本功能测试
TEST_F(PlacementMappingTest, TestChangePositionIdBasic) {
    // 使用测试夹具中已初始化的数据创建PlacementMapping对象
    PlacementMapping pm(rank, num_devices_per_host,
                      redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    // 测试正常情况
    int layer_id = 0;
    int expert_id = 0;
    int new_position = 0;

    pm.change_pos_id(layer_id, expert_id, new_position);

    int32_t actual_position = pm.read_pos_id(layer_id, expert_id);
    EXPECT_EQ(actual_position, new_position)
        << "Position ID was not updated correctly for layer " << layer_id
        << " and expert " << expert_id;
}

// 测试边界值情况 - 最大有效layer_id和expert_id
TEST_F(PlacementMappingTest, TestChangePositionIdMaxValidValues) {
    PlacementMapping pm(rank, num_devices_per_host,
                      redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    // 测试边界值
    int layer_id = pm.get_num_layers() - 1;
    int expert_id = pm.get_num_experts() - 1;
    int new_position = 42;

    pm.change_pos_id(layer_id, expert_id, new_position);

    int32_t actual_position = pm.read_pos_id(layer_id, expert_id);
    EXPECT_EQ(actual_position, new_position)
        << "Position ID was not updated correctly for max layer and expert IDs";
}



// 测试边界值情况 - 全覆盖
TEST_F(PlacementMappingTest, TestChangePositionIdMaxValidValues888) {
    PlacementMapping pm(rank, num_devices_per_host,
                      redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    // 测试边界值
    int new_position = pm.get_num_deploy_experts() - 1;

    for (int i = 0; i < pm.get_num_layers(); i++) {
        for (int j = 0; j < pm.get_num_experts(); j++) {

            pm.change_pos_id(i, j, new_position);

            //  检查 posid_tmp 对应的 epid_tmp 是否更新正确
            EXPECT_EQ(pm.read_pos_id(i, j), new_position)
                << "Epid at target_position was not updated correctly";

        }
    }

}

// 测试多个位置更新 - 确保可以正确更新多个不同位置
TEST_F(PlacementMappingTest, TestChangeMultiplePositionIds) {
    PlacementMapping pm(rank, num_devices_per_host,
                      redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    // 更新第一个位置
    int layer_id1 = 0;
    int expert_id1 = 0;
    int new_position1 = 3;

    pm.change_pos_id(layer_id1, expert_id1, new_position1);

    // 更新第二个位置
    int layer_id2 = 0;
    int expert_id2 = 2;
    int new_position2 = 4;

    pm.change_pos_id(layer_id2, expert_id2, new_position2);

    // 验证两个位置都正确更新
    int32_t actual_position1 = pm.read_pos_id(layer_id1, expert_id1);
    EXPECT_EQ(actual_position1, new_position1)
        << "First position ID was not updated correctly";

    int32_t actual_position2 = pm.read_pos_id(layer_id2, expert_id2);
    EXPECT_EQ(actual_position2, new_position2)
        << "Second position ID was not updated correctly";
}
// 测试异常情况 - layer_id越界(过大)
TEST_F(PlacementMappingTest, TestChangePositionIdLayerIdTooLarge) {
    PlacementMapping pm(rank, num_devices_per_host,
                      redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    // 测试layer_id越界(过大)
    int layer_id = pm.get_num_layers();  // 无效的layer_id
    int expert_id = 0;
    int new_position = 0;  // 确保new_position是有效的

    EXPECT_THROW({
        pm.change_pos_id(layer_id, expert_id, new_position);
    }, std::out_of_range) << "Function should throw std::out_of_range for too large layer_id";
}

// 测试异常情况 - layer_id越界(负数)
TEST_F(PlacementMappingTest, TestChangePositionIdNegativeLayerId) {
    PlacementMapping pm(rank, num_devices_per_host,
                      redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    // 测试layer_id越界(负数)
    int layer_id = -1;
    int expert_id = 0;
    int new_position = 0;  // 确保new_position是有效的

    EXPECT_THROW({
        pm.change_pos_id(layer_id, expert_id, new_position);
    }, std::out_of_range) << "Function should throw std::out_of_range for negative layer_id";
}

// 测试异常情况 - expert_id越界(过大)
TEST_F(PlacementMappingTest, TestChangePositionIdExpertIdTooLarge) {
    PlacementMapping pm(rank, num_devices_per_host,
                      redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    // 测试expert_id越界(过大)
    int layer_id = 0;
    int expert_id = pm.get_num_experts();  // 无效的expert_id
    int new_position = 0;  // 确保new_position是有效的

    EXPECT_THROW({
        pm.change_pos_id(layer_id, expert_id, new_position);
    }, std::out_of_range) << "Function should throw std::out_of_range for too large expert_id";
}

// 测试异常情况 - expert_id越界(负数)
TEST_F(PlacementMappingTest, TestChangePositionIdNegativeExpertId) {
    PlacementMapping pm(rank, num_devices_per_host,
                      redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    // 测试expert_id越界(负数)
    int layer_id = 0;
    int expert_id = -1;
    int new_position = 0;  // 确保new_position是有效的

    EXPECT_THROW({
        pm.change_pos_id(layer_id, expert_id, new_position);
    }, std::out_of_range) << "Function should throw std::out_of_range for negative expert_id";
}

// 测试异常情况 - new_position越界(过大)
TEST_F(PlacementMappingTest, TestChangePositionIdNewPositionTooLarge) {
    PlacementMapping pm(rank, num_devices_per_host,
                      redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    // 测试new_position越界(过大)
    int layer_id = 0;
    int expert_id = 0;
    int new_position = pm.get_num_deploy_experts();  // 无效的new_position

    EXPECT_THROW({
        pm.change_pos_id(layer_id, expert_id, new_position);
    }, std::out_of_range) << "Function should throw std::out_of_range for too large new_position";
}

// 测试异常情况 - new_position越界(负数)
TEST_F(PlacementMappingTest, TestChangePositionIdNegativeNewPosition) {
    PlacementMapping pm(rank, num_devices_per_host,
                      redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    // 测试new_position越界(负数)
    int layer_id = 0;
    int expert_id = 0;
    int new_position = -1;  // 无效的new_position

    EXPECT_THROW({
        pm.change_pos_id(layer_id, expert_id, new_position);
    }, std::out_of_range) << "Function should throw std::out_of_range for negative new_position";
}

// 测试正常情况 - 基本功能测试
TEST_F(PlacementMappingTest, TestChangePositionBasic) {
    // 使用测试夹具中已初始化的数据创建PlacementMapping对象
    PlacementMapping pm(rank, num_devices_per_host,
                      redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    // 测试正常情况
    int layer_id = 0;
    int expert_id = 0;
    int new_position = 5;

    pm.change_expert_position(layer_id, expert_id, new_position);

    int32_t actual_position = pm.get_expert_position(layer_id, expert_id);
    EXPECT_EQ(actual_position, new_position)
        << "Position ID was not updated correctly for layer " << layer_id
        << " and expert " << expert_id;
}


// 测试多个位置更新 - 确保可以正确更新多个不同位置
TEST_F(PlacementMappingTest, TestChangeMultiplePositions) {
    PlacementMapping pm(rank, num_devices_per_host,
                      redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    // 更新第一个位置
    int layer_id1 = 0;
    int expert_id1 = 0;
    int new_position1 = 3;

    pm.change_expert_position(layer_id1, expert_id1, new_position1);

    // 更新第二个位置
    int layer_id2 = 0;
    int expert_id2 = 2;
    int new_position2 = 4;

    pm.change_expert_position(layer_id2, expert_id2, new_position2);

    // 验证两个位置都正确更新
    int32_t actual_position1 = pm.get_expert_position(layer_id1, expert_id1);
    EXPECT_EQ(actual_position1, new_position1)
        << "First position ID was not updated correctly";

    int32_t actual_position2 = pm.get_expert_position(layer_id2, expert_id2);
    EXPECT_EQ(actual_position2, new_position2)
        << "Second position ID was not updated correctly";
}
// 测试异常情况 - layer_id越界(过大)
TEST_F(PlacementMappingTest, TestChangePositionLayerIdTooLarge) {
    PlacementMapping pm(rank, num_devices_per_host,
                      redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    // 测试layer_id越界(过大)
    int layer_id = pm.get_num_layers();  // 无效的layer_id
    int expert_id = 0;
    int new_position = 0;  // 确保new_position是有效的

    EXPECT_THROW({
        pm.change_expert_position(layer_id, expert_id, new_position);
    }, std::out_of_range) << "Function should throw std::out_of_range for too large layer_id";
}

// 测试异常情况 - layer_id越界(负数)
TEST_F(PlacementMappingTest, TestChangePositionNegativeLayerId) {
    PlacementMapping pm(rank, num_devices_per_host,
                      redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    // 测试layer_id越界(负数)
    int layer_id = -1;
    int expert_id = 0;
    int new_position = 0;  // 确保new_position是有效的

    EXPECT_THROW({
        pm.change_expert_position(layer_id, expert_id, new_position);
    }, std::out_of_range) << "Function should throw std::out_of_range for negative layer_id";
}

// 测试异常情况 - expert_id越界(过大)
TEST_F(PlacementMappingTest, TestChangePositionExpertIdTooLarge) {
    PlacementMapping pm(rank, num_devices_per_host,
                      redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    // 测试expert_id越界(过大)
    int layer_id = 0;
    int expert_id = pm.get_num_experts();  // 无效的expert_id
    int new_position = 0;  // 确保new_position是有效的

    EXPECT_THROW({
        pm.change_expert_position(layer_id, expert_id, new_position);
    }, std::out_of_range) << "Function should throw std::out_of_range for too large expert_id";
}

// 测试异常情况 - expert_id越界(负数)
TEST_F(PlacementMappingTest, TestChangePositionNegativeExpertId) {
    PlacementMapping pm(rank, num_devices_per_host,
                      redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    // 测试expert_id越界(负数)
    int layer_id = 0;
    int expert_id = -1;
    int new_position = 0;  // 确保new_position是有效的

    EXPECT_THROW({
        pm.change_expert_position(layer_id, expert_id, new_position);
    }, std::out_of_range) << "Function should throw std::out_of_range for negative expert_id";
}

// 测试从有效文件加载 PlacementMapping
TEST_F(PlacementMappingTest, TestLoadFromValidFile) {
    // 创建一个测试用的文件路径
    const char* test_file = "./test_data/placement_pattern_3d_v3_indevrrori0328_16devices_58moe_SC_TZNB.bin";

    // 从有效文件创建 PlacementMapping
    PlacementMapping pm(rank, num_devices_per_host,
                      redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      nullptr, nullptr, 0,
                      test_file);

    // 验证基本属性是否正确加载
    EXPECT_GT(pm.get_world_size(), 0) << "World size should be positive";
    EXPECT_GT(pm.get_num_layers(), 0) << "Number of layers should be positive";
    EXPECT_GT(pm.get_num_experts(), 0) << "Number of experts should be positive";

    // 验证冗余专家映射是否正确初始化
    EXPECT_TRUE(pm.is_redundant_mapping_enabled()) << "Redundant mapping should be enabled by default";
    auto redundant_epid = pm.get_redundant_epid_this_host();
    EXPECT_FALSE(redundant_epid.empty()) << "Redundant epid list should not be empty";

    // 验证位置到专家的映射是否正确加载
    auto position_to_epid = pm.get_position_to_epid();
    EXPECT_FALSE(position_to_epid.empty()) << "Position to epid mapping should not be empty";
}

// 测试从无效文件加载 PlacementMapping (文件不存在)
TEST_F(PlacementMappingTest, TestLoadFromNonExistentFile) {
    // 使用一个不存在的文件路径
    const char* non_existent_file = "./non_existent_file.bin";

    // 从不存在的文件创建 PlacementMapping 应当抛出异常
    EXPECT_THROW({
        PlacementMapping pm(rank, num_devices_per_host,
                          redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                          expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                          nullptr, nullptr, 0,
                          non_existent_file);
    }, std::runtime_error) << "Should throw exception when file does not exist";
}

// 测试从格式错误的文件加载 PlacementMapping
TEST_F(PlacementMappingTest, TestLoadFromInvalidFormatFile) {
    // 创建一个测试用的格式错误文件路径
    const char* invalid_format_file = "./test_data/invalid_format.bin";

    // 从格式错误的文件创建 PlacementMapping 应当抛出异常
    EXPECT_THROW({
        PlacementMapping pm(rank, num_devices_per_host,
                          redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                          expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                          nullptr, nullptr, 0,
                          invalid_format_file);
    }, std::runtime_error) << "Should throw exception when file format is invalid";
}

// 测试既不提供指针也不提供文件名的情况
TEST_F(PlacementMappingTest, TestNoPointerAndNoFile) {
    // 不提供指针也不提供文件名应当抛出异常
    EXPECT_THROW({
        PlacementMapping pm(rank, num_devices_per_host,
                          redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                          expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                          nullptr, nullptr, 0,
                          "");
    }, std::invalid_argument) << "Should throw exception when neither pointer nor filename is provided";
}

// 测试同时提供指针和文件名，应该优先使用指针
TEST_F(PlacementMappingTest, TestBothPointerAndFile) {
    // 创建一个测试用的文件路径
    const char* test_file = "./test_data/placement_pattern_3d_v3_indevrrori0322_58moe_4devices_SC_64eps.bin";

    // 同时提供指针和文件名，应该优先使用指针
    PlacementMapping pm(rank, num_devices_per_host,
                      redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      placement_pattern_ptr, placement_pattern_shape, ScalarType_Int,
                      test_file);

    // 验证使用的是指针版本的数据（可能需要根据你的实现调整验证方法）
    // 这里假设从文件和从指针加载的 world_size 不同，以此来区分是否使用了文件
    EXPECT_EQ(pm.get_world_size(), 4 )  // the world_size for placement_pattern_3d_v3_indevrrori0322_58moe_4devices_SC_64eps is 4;
        << "Should use pointer data when both pointer and filename are provided";
}

// 测试文件中的数据和通过指针加载的数据一致性
TEST_F(PlacementMappingTest, TestFileDataConsistency) {
    // 首先从指针创建一个参考对象
    PlacementMapping pm_from_ptr(rank, num_devices_per_host,
                               redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                               expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                               placement_pattern_ptr, placement_pattern_shape, ScalarType_Int);

    // 从文件创建第二个对象（假设文件中的数据与指针一致）
    const char* test_file = "./test_data/placement_pattern_3d_v3_indevrrori0328_16devices_58moe_SC_TZNB.bin";
    PlacementMapping pm_from_file(rank, num_devices_per_host,
                               redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                               expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                               nullptr, nullptr, 0,
                               test_file);

    // 验证两个对象的基本属性是否一致
    EXPECT_EQ(pm_from_ptr.get_world_size(), pm_from_file.get_world_size())
        << "World size should be the same";
    EXPECT_EQ(pm_from_ptr.get_num_layers(), pm_from_file.get_num_layers())
        << "Number of layers should be the same";
    EXPECT_EQ(pm_from_ptr.get_num_experts(), pm_from_file.get_num_experts())
        << "Number of experts should be the same";

    // 验证位置映射一致性
    auto pos_to_epid_ptr = pm_from_ptr.get_position_to_epid();
    auto pos_to_epid_file = pm_from_file.get_position_to_epid();

    for (int i = 0; i < pm_from_ptr.get_num_layers(); i++) {
        for (int j = 0; j < pm_from_ptr.get_num_deploy_experts(); j++) {
            EXPECT_EQ(pos_to_epid_ptr[i][j], pos_to_epid_file[i][j])
                << "Position to expert mapping should be the same at [" << i << "][" << j << "]";
        }
    }

    // 如果冗余专家映射启用，验证冗余专家列表一致性
    if (pm_from_ptr.is_redundant_mapping_enabled() && pm_from_file.is_redundant_mapping_enabled()) {
        auto redundant_epid_ptr = pm_from_ptr.get_redundant_epid_this_host();
        auto redundant_epid_file = pm_from_file.get_redundant_epid_this_host();

        EXPECT_EQ(redundant_epid_ptr.size(), redundant_epid_file.size())
            << "Redundant epid list size should be the same";

        if (redundant_epid_ptr.size() == redundant_epid_file.size()) {
            for (size_t i = 0; i < redundant_epid_ptr.size(); i++) {
                EXPECT_EQ(redundant_epid_ptr[i].size(), redundant_epid_file[i].size())
                    << "Redundant epid list inner size should be the same at index " << i;

                for (size_t j = 0; j < redundant_epid_ptr[i].size(); j++) {
                    EXPECT_EQ(redundant_epid_ptr[i][j], redundant_epid_file[i][j])
                        << "Redundant epid values should be the same at [" << i << "][" << j << "]";
                }
            }
        }
    }
}

// 测试禁用冗余专家映射的情况
TEST_F(PlacementMappingTest, TestLoadFromFileWithoutRedundantMapping) {
    // 创建一个测试用的文件路径
    const char* test_file = "./test_data/placement_pattern_3d_v3_indevrrori0328_16devices_58moe_SC_TZNB.bin";

    // 从文件创建 PlacementMapping，但禁用冗余专家映射
    PlacementMapping pm(rank, num_devices_per_host,
                      redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      nullptr, nullptr, 0,
                      test_file,
                      false);  // 禁用冗余专家映射

    // 验证冗余专家映射已禁用
    EXPECT_FALSE(pm.is_redundant_mapping_enabled()) << "Redundant mapping should be disabled";

    // 验证获取冗余专家信息会抛出异常或返回空
    EXPECT_THROW({
        pm.get_default_mapping_position(0, 0);
    }, std::runtime_error) << "Should throw exception when accessing redundant mapping functions while disabled";

    // 其他测试仍然应该正常工作
    EXPECT_GT(pm.get_world_size(), 0) << "World size should be positive";
    EXPECT_GT(pm.get_num_layers(), 0) << "Number of layers should be positive";
    EXPECT_GT(pm.get_num_experts(), 0) << "Number of experts should be positive";
}

// 测试读取大型文件的性能
TEST_F(PlacementMappingTest, TestLoadLargeFile) {
    // 创建一个测试用的大文件路径
    const char* large_file = "./test_data/placement_pattern_3d_v3_indevrrori0328_16devices_58moe_SC_TZNB.bin";

    // 测量加载时间
    auto start_time = std::chrono::high_resolution_clock::now();

    PlacementMapping pm(rank, num_devices_per_host,
                      redundant_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                      nullptr, nullptr, 0,
                      large_file);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // 输出加载时间
    std::cout << "Large file loading time: " << duration << " ms" << std::endl;

    // 验证数据是否正确加载
    EXPECT_GT(pm.get_world_size(), 0) << "World size should be positive";
    EXPECT_GT(pm.get_num_layers(), 0) << "Number of layers should be positive";
    EXPECT_GT(pm.get_num_experts(), 0) << "Number of experts should be positive";
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////


// PlacementMapping 测试夹具
class PlacementGlobalMappingTest : public ::testing::Test {
protected:
    memcpy_fun_t old_fun = nullptr;
    aclrtContext context;
    void SetUp() override {
        // 初始化测试数据
        rank_ = 0;
        world_size_ = 2;
        num_layers_ = 2;
        num_experts_ = 4;
        max_redundant_count_ = 2;
        num_devices_per_host_ = 8;

        aclrtCreateContext(&context, 0);
        aclrtSetCurrentContext(context);
        // 分配模拟 HBM 内存
        size_t size = num_layers_ * num_experts_ * max_redundant_count_ * sizeof(int32_t);
        ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc((void**)&global_expert_mapping_, size, ACL_MEM_MALLOC_HUGE_FIRST));
        aclrtMemset((void*)global_expert_mapping_, size, 0, size);
        ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc((void**)&redundant_expert_mapping_, size, ACL_MEM_MALLOC_HUGE_FIRST));
        aclrtMemset((void*)redundant_expert_mapping_, size, 0, size);
        size = num_layers_ * num_experts_ * sizeof(int32_t);
        ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc((void**)&redundant_count_per_expert_, size, ACL_MEM_MALLOC_HUGE_FIRST));
        aclrtMemset((void*)redundant_count_per_expert_, size, 0, size);

        // 初始化 placement_pattern_vector_
        // placement_pattern_vector_ = {
        //     {{1, 0, 1, 0}, {0, 1, 0, 1}}, // rank 0
        //     {{1, 1, 0, 0}, {0, 0, 1, 1}}  // rank 1
        // };
        placement_pattern_vector_ = {
            {{1, 0, 1, 1}, {0, 1, 0, 1}}, // rank 0
            {{1, 1, 0, 1}, {1, 0, 1, 0}}  // rank 1
        };

        // 初始化形状数组
        redundant_mapping_shape_[0] = num_layers_;
        redundant_mapping_shape_[1] = max_redundant_count_;
        redundant_mapping_shape_[2] = num_experts_;
        mapping_shape_[0] = num_layers_;
        mapping_shape_[1] = num_experts_;
        mapping_shape_[2] = max_redundant_count_;
        count_shape_[0] = num_layers_;
        count_shape_[1] = num_experts_;

        // 从有效文件创建 PlacementMapping
        mapping = new PlacementMapping(placement_pattern_vector_, rank_, num_devices_per_host_,
            redundant_expert_mapping_, redundant_mapping_shape_,
            redundant_count_per_expert_, count_shape_);

        // old_fun = get_memcpy_fun();
        // set_memcpy_fun(&my_mem_fun);
    }

    void TearDown() override {
        aclrtFree(redundant_expert_mapping_);
        aclrtFree(global_expert_mapping_);
        aclrtFree(redundant_count_per_expert_);
        aclrtDestroyContext(context);
        // delete[] global_expert_mapping_;
        // delete[] redundant_count_per_expert_;
        delete mapping;
    }

    // 测试用成员变量
    int32_t rank_;
    int32_t world_size_;
    int32_t num_layers_;
    int32_t num_experts_;
    int32_t max_redundant_count_;
    int32_t num_devices_per_host_;
    int32_t* redundant_expert_mapping_;
    int32_t* global_expert_mapping_;
    int32_t* redundant_count_per_expert_;
    std::vector<std::vector<std::vector<int>>> placement_pattern_vector_;
    PlacementMapping *mapping;
    int64_t redundant_mapping_shape_[NUM_DIM_OF_MAPPING];
    int64_t mapping_shape_[NUM_DIM_OF_MAPPING];
    int64_t count_shape_[NUM_DIM_OF_COUNT];
};


// 测试 construct_epid_mapping_to_position
TEST_F(PlacementGlobalMappingTest, ConstructEpidMappingToPosition) {
    // 验证 redundant_count_per_expert_
    // int32_t expected_redundant_count[] = {2, 1, 1, 2, 1, 1, 1, 1};
    int32_t expected_redundant_count[] = {2, 1, 1, 2, 1, 1, 1, 1};
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        for (int32_t expert = 0; expert < num_experts_; ++expert) {
            int32_t value = mapping->get_redundant_count(layer, expert);
            int32_t expected = expected_redundant_count[layer * num_experts_ + expert];
            EXPECT_EQ(value, expected) << "Mismatch at layer " << layer << ", expert " << expert;
        }
    }

    // 验证 global_expert_mapping_
    // int32_t expected_mapping[] = {0, 2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 3};
    int32_t expected_mapping[] = {0, 3, 4, 0, 1, 0, 2, 5, 3, 0, 0, 0, 4, 0, 1, 0};
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        for (int32_t expert = 0; expert < num_experts_; ++expert) {
            for (int32_t index = 0; index < max_redundant_count_; ++index) {
                int32_t value = mapping->get_global_expert_position_id(layer, expert, index);
                int32_t expected = expected_mapping[(layer * num_experts_ + expert) * max_redundant_count_ + index];
                EXPECT_EQ(value, expected) << "Mismatch at layer " << layer << ", expert " << expert << ", index " << index;
            }
        }
    }
}

class PlacementSimpleGlobalMappingTest : public ::testing::Test {
    protected:
        memcpy_fun_t old_fun = nullptr;
        aclrtContext context;
        void SetUp() override {
            // 初始化测试数据
            rank_ = 0;
            world_size_ = 2;
            num_layers_ = 1;
            num_experts_ = 3;
            max_redundant_count_ = 2;
            num_devices_per_host_ = 8;
            aclrtCreateContext(&context, 0);
            aclrtSetCurrentContext(context);
            // 分配模拟 HBM 内存
            size_t size = num_layers_ * num_experts_ * max_redundant_count_ * sizeof(int32_t);
            ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc((void**)&global_expert_mapping_, size, ACL_MEM_MALLOC_HUGE_FIRST));
            aclrtMemset((void*)global_expert_mapping_, size, 0, size);
            ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc((void**)&redundant_expert_mapping_, size, ACL_MEM_MALLOC_HUGE_FIRST));
            aclrtMemset((void*)redundant_expert_mapping_, size, 0, size);
            size = num_layers_ * num_experts_ * sizeof(int32_t);
            ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc((void**)&redundant_count_per_expert_, size, ACL_MEM_MALLOC_HUGE_FIRST));
            aclrtMemset((void*)redundant_count_per_expert_, size, 0, size);

            // 初始化 placement_pattern_vector_
            placement_pattern_vector_ = {
                { {0, 1, 1} }, // rank 0
                { {1, 0, 1} }  // rank 1
            };

            // 初始化形状数组
            redundant_mapping_shape_[0] = num_layers_;
            redundant_mapping_shape_[1] = max_redundant_count_;
            redundant_mapping_shape_[2] = num_experts_;
            mapping_shape_[0] = num_layers_;
            mapping_shape_[1] = num_experts_;
            mapping_shape_[2] = max_redundant_count_;
            count_shape_[0] = num_layers_;
            count_shape_[1] = num_experts_;

            // 从有效文件创建 PlacementMapping
            mapping = new PlacementMapping(placement_pattern_vector_, rank_, num_devices_per_host_,
            redundant_expert_mapping_, redundant_mapping_shape_,
            redundant_count_per_expert_, count_shape_);

            // old_fun = get_memcpy_fun();
            // set_memcpy_fun(&my_mem_fun);
        }

        void TearDown() override {
        aclrtFree(redundant_expert_mapping_);
        aclrtFree(global_expert_mapping_);
        aclrtFree(redundant_count_per_expert_);
        aclrtDestroyContext(context);
            // delete[] global_expert_mapping_;
            // delete[] redundant_count_per_expert_;
            delete mapping;
        }

        // 测试用成员变量
        int32_t rank_;
        int32_t world_size_;
        int32_t num_layers_;
        int32_t num_experts_;
        int32_t max_redundant_count_;
        int32_t num_devices_per_host_;
        int32_t* redundant_expert_mapping_;
        int32_t* global_expert_mapping_;
        int32_t* redundant_count_per_expert_;
        std::vector<std::vector<std::vector<int>>> placement_pattern_vector_;
        PlacementMapping *mapping;
        int64_t redundant_mapping_shape_[NUM_DIM_OF_MAPPING];
        int64_t mapping_shape_[NUM_DIM_OF_MAPPING];
        int64_t count_shape_[NUM_DIM_OF_COUNT];
    };


    // 测试 construct_epid_mapping_to_position
    TEST_F(PlacementSimpleGlobalMappingTest, ConstructSimpleEpidMappingToPosition) {
        // 验证 redundant_count_per_expert_
        int32_t expected_redundant_count[] = {1, 1, 2};
        for (int32_t layer = 0; layer < num_layers_; ++layer) {
            for (int32_t expert = 0; expert < num_experts_; ++expert) {
                int32_t value = mapping->get_redundant_count(layer, expert);
                int32_t expected = expected_redundant_count[layer * num_experts_ + expert];
                EXPECT_EQ(value, expected) << "Mismatch at layer " << layer << ", expert " << expert;
            }
        }

        // 验证 global_expert_mapping_
        int32_t expected_mapping[] = {2, 0, 0, 0, 1, 3};
        for (int32_t layer = 0; layer < num_layers_; ++layer) {
            for (int32_t expert = 0; expert < num_experts_; ++expert) {
                for (int32_t index = 0; index < max_redundant_count_; ++index) {
                    int32_t value = mapping->get_global_expert_position_id(layer, expert, index);
                    int32_t expected = expected_mapping[(layer * num_experts_ + expert) * max_redundant_count_ + index];
                    EXPECT_EQ(value, expected) << "Mismatch at layer " << layer << ", expert " << expert << ", index " << index;
                }
            }
        }
    }

class PlacementDynamicGlobalMappingTest : public ::testing::Test {
    protected:
        aclrtContext context;
        void SetUp() override {
            // 初始化测试数据
            rank_ = 0;
            world_size_ = 2;
            num_layers_ = 1;
            num_experts_ = 3;
            max_redundant_count_ = 3;
            num_devices_per_host_ = 8;
            aclrtSetDevice(0);
            aclrtCreateContext(&context, 0);
            aclrtSetCurrentContext(context);
            // 分配模拟 HBM 内存
            size_t size = num_layers_ * num_experts_ * max_redundant_count_ * sizeof(int32_t);
            ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc((void**)&global_expert_mapping_, size, ACL_MEM_MALLOC_HUGE_FIRST));
            aclrtMemset((void*)global_expert_mapping_, size, 0, size);
            ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc((void**)&redundant_expert_mapping_, size, ACL_MEM_MALLOC_HUGE_FIRST));
            aclrtMemset((void*)redundant_expert_mapping_, size, 0, size);
            size = num_layers_ * num_experts_ * sizeof(int32_t);
            ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc((void**)&redundant_count_per_expert_, size, ACL_MEM_MALLOC_HUGE_FIRST));
            aclrtMemset((void*)redundant_count_per_expert_, size, 0, size);

            // 初始化 placement_pattern_vector_
            placement_pattern_vector_ = {
                { {1, 1, 1} }, // rank 0
                { {1, 1, 1} } // rank 1
            };

            // 初始化形状数组
            redundant_mapping_shape_[0] = num_layers_;
            redundant_mapping_shape_[1] = max_redundant_count_;
            redundant_mapping_shape_[2] = num_experts_;
            mapping_shape_[0] = num_layers_;
            mapping_shape_[1] = num_experts_;
            mapping_shape_[2] = max_redundant_count_;
            count_shape_[0] = num_layers_;
            count_shape_[1] = num_experts_;

            // 从有效文件创建 PlacementMapping
            mapping = new PlacementMapping(placement_pattern_vector_, rank_, num_devices_per_host_,
                                    redundant_expert_mapping_, redundant_mapping_shape_,
                                    redundant_count_per_expert_, count_shape_);
        }

        void TearDown() override {
            aclrtFree(redundant_expert_mapping_);
            aclrtFree(global_expert_mapping_);
            aclrtFree(redundant_count_per_expert_);
            if (context != nullptr) {
                aclrtDestroyContext(context);
                context = nullptr;
            }
            delete mapping;
        }

        // 测试用成员变量
        int32_t rank_;
        int32_t world_size_;
        int32_t num_layers_;
        int32_t num_experts_;
        int32_t max_redundant_count_;
        int32_t num_devices_per_host_;
        int32_t* redundant_expert_mapping_;
        int32_t* global_expert_mapping_;
        int32_t* redundant_count_per_expert_;
        std::vector<std::vector<std::vector<int>>> placement_pattern_vector_;
        PlacementMapping *mapping;
        int64_t redundant_mapping_shape_[NUM_DIM_OF_MAPPING];
        int64_t mapping_shape_[NUM_DIM_OF_MAPPING];
        int64_t count_shape_[NUM_DIM_OF_COUNT];
    };

    // 测试用例 1：正常情况
    // TEST_F(PlacementDynamicGlobalMappingTest, NormalCase) {
    //     std::vector<int32_t> global_mapping = {0, 1, 0, 2, 1, 0};
    //     mapping->update_globalLogisticsIdToDeployedPositionMapping(global_mapping);

    //     // 验证结果
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 0, 0), 0) << "Expected position_id=0 for (layer=0, redundant_idx=0, expert_id=0)";
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 1, 0), 2) << "Expected position_id=2 for (layer=0, redundant_idx=1, expert_id=0)";
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 2, 0), 5) << "Expected position_id=5 for (layer=0, redundant_idx=2, expert_id=0)";
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 0, 1), 1) << "Expected position_id=1 for (layer=0, redundant_idx=0, expert_id=1)";
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 1, 1), 4) << "Expected position_id=4 for (layer=0, redundant_idx=1, expert_id=1)";
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 2, 1), 1) << "Expected position_id=1 for (layer=0, redundant_idx=2, expert_id=1)";
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 0, 2), 3) << "Expected position_id=3 for (layer=0, redundant_idx=0, expert_id=2)";
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 1, 2), 3) << "Expected position_id=3 for (layer=0, redundant_idx=1, expert_id=2)";
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 2, 2), 3) << "Expected position_id=3 for (layer=0, redundant_idx=2, expert_id=2)";
    //     // 验证未设置的位置
    //     // EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 3, 0), -1) << "Expected -1 for unset position";
    // }

    // 测试用例 2：单一 expert_id
    // TEST_F(PlacementDynamicGlobalMappingTest, SingleExpertId) {
    //     std::vector<int32_t> global_mapping = {0, 0, 0, 0, 0, 0};
    //     mapping->update_globalLogisticsIdToDeployedPositionMapping(global_mapping);

    //     // 验证结果
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 0, 0), 0) << "Expected position_id=0 for (layer=0, redundant_idx=0, expert_id=0)";
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 1, 0), 1) << "Expected position_id=1 for (layer=0, redundant_idx=1, expert_id=0)";
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 2, 0), 2) << "Expected position_id=2 for (layer=0, redundant_idx=1, expert_id=0)";
    // }

    // 测试用例 3：空输入
    // TEST_F(PlacementDynamicGlobalMappingTest, EmptyInput) {
    //     std::vector<int32_t> global_mapping = {};
    //     ASSERT_DEATH(mapping->update_globalLogisticsIdToDeployedPositionMapping(global_mapping), "Assertion `deployPosition\\.size\\(\\) == num_deploy_experts_ \\* num_layers_' failed");
    // }

// 测试用例：动态mapping修改，多层输入
class PlacementDynamicMappingMutiLayerTest : public ::testing::Test {
    protected:
        aclrtContext context;
        void SetUp() override {
            // 初始化测试数据
            rank_ = 0;
            world_size_ = 2;
            num_layers_ = 2;
            num_experts_ = 3;
            max_redundant_count_ = 3;
            num_devices_per_host_ = 8;
            aclrtSetDevice(0);
            aclrtCreateContext(&context, 0);
            aclrtSetCurrentContext(context);
            // 分配模拟 HBM 内存
            size_t size = num_layers_ * num_experts_ * max_redundant_count_ * sizeof(int32_t);
            ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc((void**)&global_expert_mapping_, size, ACL_MEM_MALLOC_HUGE_FIRST));
            aclrtMemset((void*)global_expert_mapping_, size, 0, size);
            ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc((void**)&redundant_expert_mapping_, size, ACL_MEM_MALLOC_HUGE_FIRST));
            aclrtMemset((void*)redundant_expert_mapping_, size, 0, size);
            size = num_layers_ * num_experts_ * sizeof(int32_t);
            ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc((void**)&redundant_count_per_expert_, size, ACL_MEM_MALLOC_HUGE_FIRST));
            aclrtMemset((void*)redundant_count_per_expert_, size, 0, size);

            // 初始化 placement_pattern_vector_
            placement_pattern_vector_ = {
                { {1, 1, 1}, {1, 1, 1} }, // rank 0
                { {1, 1, 1}, {1, 1, 1} }  // rank 1
            };

            // 初始化形状数组
            redundant_mapping_shape_[0] = num_layers_;
            redundant_mapping_shape_[1] = max_redundant_count_;
            redundant_mapping_shape_[2] = num_experts_;
            mapping_shape_[0] = num_layers_;
            mapping_shape_[1] = num_experts_;
            mapping_shape_[2] = max_redundant_count_;
            count_shape_[0] = num_layers_;
            count_shape_[1] = num_experts_;

            // 从有效文件创建 PlacementMapping
            mapping = new PlacementMapping(placement_pattern_vector_, rank_, num_devices_per_host_,
                                    redundant_expert_mapping_, redundant_mapping_shape_,
                                    redundant_count_per_expert_, count_shape_);
        }

        void TearDown() override {
            aclrtFree(redundant_expert_mapping_);
            aclrtFree(global_expert_mapping_);
            aclrtFree(redundant_count_per_expert_);
            if (context != nullptr) {
                aclrtDestroyContext(context);
                context = nullptr;
            }
            delete mapping;
        }

        // 测试用成员变量
        int32_t rank_;
        int32_t world_size_;
        int32_t num_layers_;
        int32_t num_experts_;
        int32_t max_redundant_count_;
        int32_t num_devices_per_host_;
        int32_t* redundant_expert_mapping_;
        int32_t* global_expert_mapping_;
        int32_t* redundant_count_per_expert_;
        std::vector<std::vector<std::vector<int>>> placement_pattern_vector_;
        PlacementMapping *mapping;
        int64_t redundant_mapping_shape_[NUM_DIM_OF_MAPPING];
        int64_t mapping_shape_[NUM_DIM_OF_MAPPING];
        int64_t count_shape_[NUM_DIM_OF_COUNT];
    };

    // // 测试用例1：多层输入
    // TEST_F(PlacementDynamicMappingMutiLayerTest, MultipleLayers) {
    //     std::vector<int32_t> global_mapping = {0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0}; // 两层，每层 [0, 1, 0, 1, 1, 0]
    //     mapping->update_globalLogisticsIdToDeployedPositionMapping(global_mapping);

    //     // 验证结果
    //     // 层0
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 0, 0), 0) << "Expected position_id=0 for (layer=0, redundant_idx=0, expert_id=0)";
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 1, 0), 2) << "Expected position_id=2 for (layer=0, redundant_idx=1, expert_id=0)";
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 2, 0), 5) << "Expected position_id=5 for (layer=0, redundant_idx=2, expert_id=0)";
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 0, 1), 1) << "Expected position_id=1 for (layer=0, redundant_idx=0, expert_id=1)";
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 1, 1), 3) << "Expected position_id=3 for (layer=0, redundant_idx=1, expert_id=1)";
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 2, 1), 4) << "Expected position_id=4 for (layer=0, redundant_idx=2, expert_id=1)";
    //     // 层1
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(1, 0, 0), 0) << "Expected position_id=0 for (layer=1, redundant_idx=0, expert_id=0)";
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(1, 1, 0), 2) << "Expected position_id=2 for (layer=1, redundant_idx=1, expert_id=0)";
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(1, 0, 1), 1) << "Expected position_id=1 for (layer=1, redundant_idx=0, expert_id=1)";
    //     EXPECT_EQ(mapping->get_redundant_expert_position_id(1, 1, 1), 3) << "Expected position_id=3 for (layer=1, redundant_idx=1, expert_id=1)";
    // }

    // 测试用例 2：max_redundant_count 较小
    TEST_F(PlacementDynamicMappingMutiLayerTest, SmallMaxRedundantCount) {
        std::vector<int32_t> global_mapping = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
        mapping->update_globalLogisticsIdToDeployedPositionMapping(global_mapping);
        // 验证结果
        EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 0, 0), 0) << "Expected position_id=0 for (layer=0, redundant_idx=0, expert_id=0)";
        EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 0, 1), 1) << "Expected position_id=1 for (layer=0, redundant_idx=0, expert_id=1)";
        EXPECT_EQ(mapping->get_redundant_expert_position_id(0, 0, 2), 2) << "Expected position_id=2 for (layer=0, redundant_idx=0, expert_id=2)";
    }
*/
// 主函数
/*
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
*/


