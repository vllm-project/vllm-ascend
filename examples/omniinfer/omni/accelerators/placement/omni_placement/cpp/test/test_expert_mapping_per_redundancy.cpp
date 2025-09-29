// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <gtest/gtest.h>
#include "placement_mapping.h"
#include "tensor.h"

// 创建一个 TestPlacementMappingPerRedundancy 类来封装测试环境
class TestPlacementMappingPerRedundancy : public ::testing::Test {
    protected:
        void SetUp() override {
            // 初始化测试数据
            rank_ = 0;
            world_size_ = 2;
            num_layers_ = 2;
            num_experts_ = 4;
            max_redundant_count_ = 2;
            num_devices_per_host_ = 8;

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
                { {1, 0, 1, 1}, {0, 1, 1, 1} }, // rank 0
                { {0, 1, 1, 1}, {1, 0, 1, 1} }  // rank 1
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
            placement_mapping = new PlacementMapping("", rank_, num_devices_per_host_, num_experts_,
                                        (size_t)redundant_expert_mapping_, redundant_mapping_shape_,
                                        (size_t)redundant_count_per_expert_, count_shape_,
                                        (size_t)global_expert_mapping_, mapping_shape_,
                                        (size_t)redundant_count_per_expert_);

            // old_fun = get_memcpy_fun();
            // set_memcpy_fun(&my_mem_fun);
        }

        void TearDown() override {
            aclrtFree(redundant_expert_mapping_);
            aclrtFree(global_expert_mapping_);
            aclrtFree(redundant_count_per_expert_);
            // delete[] global_expert_mapping_;
            // delete[] redundant_count_per_expert_;
            delete placement_mapping;
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
        PlacementMapping *placement_mapping;
        std::vector<int64_t> redundant_mapping_shape_ = std::vector<std::int64_t>(NUM_DIM_OF_MAPPING, 0);
        std::vector<int64_t> mapping_shape_ = std::vector<std::int64_t>(NUM_DIM_OF_MAPPING, 0);
        std::vector<int64_t> count_shape_ = std::vector<std::int64_t>(NUM_DIM_OF_COUNT, 0);
};

// 测试 construct_per_redundancy_epid_mapping_to_position 函数
TEST_F(TestPlacementMappingPerRedundancy, ConstructPerRedundancyEpidMappingToPosition) {

    placement_mapping->print_mapping();
    // 验证结果
    const auto tmep_mapping = placement_mapping->get_redundant_expert_mapping();
    std::vector<std::vector<std::vector<int32_t>>> mapping;
    mapping = std::vector<std::vector<std::vector<int32_t>>>(
        num_layers_,
        std::vector<std::vector<int32_t>>(
            max_redundant_count_,
            std::vector<int32_t>(num_experts_, -1)
        )
    );

    size_t expert_data_size = num_experts_ * sizeof(int32_t);

    for (int layer = 0; layer < num_layers_; ++layer) {
        for (int redundancy = 0; redundancy < max_redundant_count_; ++redundancy) {
            // 计算 tmep_mapping 中当前 layer 和 redundancy 的偏移
            size_t offset = (layer * max_redundant_count_ + redundancy) * num_experts_ * sizeof(int32_t);
            // 获取 mapping[layer][redundancy] 的连续数据指针
            int32_t* dest_ptr = mapping[layer][redundancy].data();

            // 使用 memcpy 拷贝数据
            aclError ret = (*get_memcpy_fun())(
                dest_ptr,                    // 目标：mapping[layer][redundancy] 的连续内存
                expert_data_size,            // 每次拷贝 num_experts_ 个 int32_t
                (char*)tmep_mapping + offset, // 源：tmep_mapping 的偏移地址
                expert_data_size,
                ACL_MEMCPY_DEVICE_TO_HOST
            );

            if (ret != ACL_ERROR_NONE) {
                throw std::runtime_error("aclrtMemcpy failed in get_redundant_expert_mapping, error code: " +
                                         std::to_string(ret));
            }
        }
    }

    EXPECT_EQ(mapping.size(), 2);  // 2 layers

    // 层 0 验证
    EXPECT_EQ(mapping[0].size(), 2);     // 2 redundancies
    EXPECT_EQ(mapping[0][0].size(), 4);  // 4 experts
    EXPECT_EQ(mapping[0][1].size(), 4);  // 4 experts

    // 验证层 0, 冗余 0
    EXPECT_EQ(mapping[0][0][0], 0);  // Expert 0 -> Position 0
    EXPECT_EQ(mapping[0][0][1], 3);  // Expert 1 -> Position 3
    EXPECT_EQ(mapping[0][0][2], 1);  // Expert 2 -> Position 1
    EXPECT_EQ(mapping[0][0][3], 2);  // Expert 3 -> Position 2

    // 验证层 0, 冗余 1
    EXPECT_EQ(mapping[0][1][0], 0);  // Expert 0 -> Position 0
    EXPECT_EQ(mapping[0][1][1], 3);  // Expert 1 -> Position 3
    EXPECT_EQ(mapping[0][1][2], 4);  // Expert 2 -> Position 4
    EXPECT_EQ(mapping[0][1][3], 5);  // Expert 3 -> Position 5

    // 层 1 验证
    EXPECT_EQ(mapping[1].size(), 2);     // 2 redundancies
    EXPECT_EQ(mapping[1][0].size(), 4);  // 4 experts
    EXPECT_EQ(mapping[1][1].size(), 4);  // 4 experts

    // 验证层 1, 冗余 0
    EXPECT_EQ(mapping[1][0][0], 3);  // Expert 0 -> Position 3
    EXPECT_EQ(mapping[1][0][1], 0);  // Expert 1 -> Position 0
    EXPECT_EQ(mapping[1][0][2], 1);  // Expert 2 -> Position 1
    EXPECT_EQ(mapping[1][0][3], 2);  // Expert 3 -> Position 2

    // 验证层 1, 冗余 1
    EXPECT_EQ(mapping[1][1][0], 3);  // Expert 0 -> Position 3
    EXPECT_EQ(mapping[1][1][1], 0);  // Expert 1 -> Position 0
    EXPECT_EQ(mapping[1][1][2], 4);  // Expert 2 -> Position 4
    EXPECT_EQ(mapping[1][1][3], 5);  // Expert 3 -> Position 5
}

// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }