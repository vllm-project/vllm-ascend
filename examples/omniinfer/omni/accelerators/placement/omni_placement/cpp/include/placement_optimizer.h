// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#ifndef PLACEMENT_OPTIMIZER_H
#define PLACEMENT_OPTIMIZER_H

#include "dynamic_eplb_greedy.h"
#include "expert_activation.h"
#include "expert_load_balancer.h"
#include "placement_mapping.h"
#include <stdexcept>
#include <string>
#include <vector>


/**
 * @brief 用于优化分布式系统中专家放置的类。
 *
 * 通过调用 ExpertLoadBalancer 来优化专家的分布，基于激活频率和冗余配置。
 * 所有参数从 PlacementMapping 中获取。
 */
class PlacementOptimizer {
  private:
    PlacementMapping *placement_mapping_;  ///< 放置映射数据的指针
    ClusterActivation *clusterActivation_; ///< 集群激活数据的指针
    int num_layers_;                       ///< 模型中的层数
    int rank_;                             ///< 当前进程的秩
    int world_size_;                       ///< 进程总数（世界大小）
    int num_experts_;                      ///< 模型中的逻辑专家总数
    int num_devices_per_host_;             ///< 每个主机的设备数
    int num_redundant_per_rank_;           ///< 每个秩的冗余专家数
    int num_experts_per_rank_;             ///< 每个秩的专家数
    int expert_redundant_limit_;           ///< 专家冗余上限
    ExpertLoadBalancer *load_balancer_;    ///< 负载均衡器实例
    GreedyExpertLoadBalancer *greedy_load_balancer_; ///< 负载均衡器实例
    // int record_=0;

    /**
     * @brief 从 PlacementMapping 和 ClusterActivation 中提取所有层的输入数据。
     *
     * @param[out] placement 放置向量。
     * @param[out] activations 激活向量。
     */
    void extract_input_data(std::vector<int> &placement,
                            std::vector<int64_t> &activations);

  public:
    /**
     * @brief PlacementOptimizer 的构造函数。
     *
     * 使用 PlacementMapping 和 ClusterActivation 初始化优化器，并创建
     * ExpertLoadBalancer 实例。
     *
     * @param[in] placement_mapping PlacementMapping 对象的指针。
     * @param[in] clusterActivation ClusterActivation 对象的指针。
     * @throws std::runtime_error 如果任一指针为空或参数无效。
     */
    PlacementOptimizer(PlacementMapping *placement_mapping,
                       ClusterActivation *clusterActivation);

    /**
     * @brief 默认析构函数。
     *
     * 释放 load_balancer_ 的资源。
     */
    ~PlacementOptimizer();

    /**
     * @brief 为所有层优化专家放置。
     *
     * @return std::vector<ChangeInstruction> 所有层的优化更改指令向量。
     */
    std::vector<ChangeInstruction> optimize();
    std::vector<ChangeInstruction> optimize(std::vector<int> placement,
                                            std::vector<int64_t> activations);

    /** @brief 获取模型中的层数。 @return int 层数。 */
    int get_num_layers() const { return num_layers_; }
    /** @brief 获取当前进程的秩。 @return int 秩。 */
    int get_rank() const { return rank_; }
    /** @brief 获取进程总数。 @return int 世界大小。 */
    int get_world_size() const { return world_size_; }
    /** @brief 获取模型中的逻辑专家总数。 @return int 专家数。 */
    int get_num_experts() const { return num_experts_; }
    /** @brief 获取每个主机的设备数。 @return int 每主机设备数。 */
    int get_num_devices_per_host() const { return num_devices_per_host_; }
    /** @brief 获取每个秩的专家数。 @return int 每秩专家数。 */
    int get_num_experts_per_rank() const { return num_experts_per_rank_; }
    /** @brief 获取每个秩的冗余专家数。 @return int 每秩冗余专家数。 */
    int get_num_redundant_per_rank() const { return num_redundant_per_rank_; }
    /** @brief 获取专家冗余上限。 @return int 冗余上限。 */
    int get_expert_redundant_limit() const { return expert_redundant_limit_; }
};

#endif // PLACEMENT_OPTIMIZER_H