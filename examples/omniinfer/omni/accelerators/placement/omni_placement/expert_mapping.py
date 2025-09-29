# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple
from . import omni_placement
from .config import Config

class ExpertMapping:
    def __init__(self, config: Config, device: str = "npu", rank: int = 0, world_size: int = 1, num_devices_per_host: int = 8, enable_dynamic: bool = False, num_experts: int = 256, enable_rank_round_robin: bool = False):
        self.pattern_path = config.getattr("pattern_path", None)
        self.device = device
        self.rank = rank
        self.world_size = world_size 
        self.num_devices_per_host = num_devices_per_host
        self.max_moe_layer_num = config.max_moe_layer_num
        self.num_experts = num_experts
        self.enable_rank_round_robin = enable_rank_round_robin

        self.placement_pattern = self._load_placement_pattern_with_validation()

        self.enable_dynamic = enable_dynamic
        self.max_redundant_per_expert = config.getattr('max_redundant_per_expert', None) if self.enable_dynamic else None
        self.max_redundant_per_rank = config.getattr('max_redundant_per_rank', None) if self.enable_dynamic else None

        self._init_expert_mapping()
        self.local_expert_offsets = self._calc_expert_offset_each_layer()
        self.max_num_deployed_expert_per_rank = max(max(self.get_deployed_experts_per_layer()) // self.get_world_size(), 1)


    def _init_expert_mapping(self) :

        _, num_layers, num_eps = self.placement_pattern.shape

        max_redundant_per_expert = self.get_max_redundant_per_expert()
        
        if self.enable_rank_round_robin:
            self.selector = torch.zeros(num_layers,num_eps,1,dtype=torch.int32,device=self.device)
        else:
            self.selector = torch.zeros(num_layers,num_eps,max_redundant_per_expert,dtype=torch.int32,device=self.device) 
        
        self.num_redundant_per_expert = torch.zeros(num_layers, num_eps, dtype=torch.int32, device=self.device)

        self.placement_pattern_cpu = self.placement_pattern.cpu()
        self.placement_mapping = omni_placement.PlacementMapping("",  # TODO: pattern path, parse pattern in native C++
                                                                self.rank,
                                                                self.num_devices_per_host,
                                                                max_redundant_per_expert,
                                                                max(self.get_deployed_experts_per_layer()),
                                                                self.placement_pattern_cpu.data_ptr(),
                                                                list(self.placement_pattern_cpu.size()),
                                                                self.selector.data_ptr(),
                                                                self.enable_rank_round_robin,
                                                                self.num_redundant_per_expert.data_ptr())

    def get_selector(self):
        return self.selector
    
    def get_num_redundant_per_expert(self):
        return self.num_redundant_per_expert

    def _load_placement_pattern_with_validation(self) -> Optional[torch.Tensor]:
        """Load and validate placement pattern from config."""

        def build_basepattern(world_size, layers, num_experts):
            # Calculate num_experts_per_rank
            num_experts_per_rank = num_experts // world_size
            
            # Initialize a 3D matrix with zeros
            matrix = np.zeros((world_size, layers, num_experts)).astype(np.int32)
            
            # Set specific slices to 1 based on rank
            for rank in range(world_size):
                start_idx = rank * num_experts_per_rank
                end_idx = (rank + 1) * num_experts_per_rank
                matrix[rank, :, start_idx:end_idx] = 1
            
            return matrix
         
        if self.pattern_path is None:
            print(f"[Placement-Warning]: pattern_path is None, BasePattern will be Used!")
            pattern = build_basepattern(self.world_size, self.max_moe_layer_num, self.num_experts)
        else:
            if not os.path.exists(self.pattern_path):
                raise FileNotFoundError(f"[Placement-Error]: Placement pattern file not found: {self.pattern_path}")
            else:
                pattern = np.load(self.pattern_path).astype(np.int32)
            if pattern.shape != (self.world_size, self.max_moe_layer_num, self.num_experts):
                raise ValueError(f"[Placement-Error]: pattern.shape[{pattern.shape}] is not equals to (world_size[{self.world_size}], layers[{self.max_moe_layer_num}], num_experts[{self.num_experts}])")

        pattern = torch.tensor(
            pattern,
            dtype=torch.int32,
            device=self.device
        )
        # Validate pattern shape against num_devices_per_host
        if pattern.shape[0] % self.num_devices_per_host != 0:
            print(f"Warning: Number of devices in pattern ({pattern.shape[0]}) is not "
                    f"evenly divisible by num_devices_per_host ({self.num_devices_per_host})")
        return pattern

    def is_moe_layer(self, layer_idx_moe):
        return layer_idx_moe < self.max_moe_layer_num

    # @calculate_time
    def is_expert_on_current_rank(
        self,
        layer_idx_moe: int,
        expert_id: int,
        current_rank: int,
        experts_per_rank: int
    ) -> Tuple[bool, int]:
        """
        Check if expert is deployed on current rank and get its position.

        Args:
            layer_idx_moe: ID of the MoE layer
            expert_id: Expert ID within the layer
            current_rank: Target device rank to check
            experts_per_rank: Experts per device in default deployment

        Returns:
            Tuple (exists_on_rank, local_position)
        """
        if not self.is_moe_layer(layer_idx_moe):
            return self._default_deployment_check(expert_id, current_rank, experts_per_rank)


        layer_mapping = self.placement_pattern[current_rank, layer_idx_moe]
        exists = layer_mapping[expert_id] > 0.5
        position = int(torch.sum(layer_mapping[:expert_id]).item())
        return exists, position

    def _default_deployment_check(
        self,
        expert_id: int,
        current_rank: int,
        experts_per_rank: int
    ) -> Tuple[bool, int]:
        """Check default sequential expert deployment."""
        start = current_rank * experts_per_rank
        end = (current_rank + 1) * experts_per_rank
        in_range = start <= expert_id < end
        position = expert_id - start if in_range else -1
        return in_range, position

    def get_num_of_redundant_experts(self, moe_layer_idx: int, num_expert_per_device_origin=16, rank_device=0) -> int:
        """
        Calculate the number of redundant experts for a specific device and MoE layer.

        Args:
            moe_layer_idx : int
                Index of the MoE layer to query expert distribution.
            num_expert_per_device_origin : int, optional (default=16)
                Original number of experts assigned to this device/layer.
            rank_device : int, optional (default=0)
                Rank identifier of the target device in the distributed system.

        Returns:
            int
                Number of redundant experts, calculated as: (current experts count) - (original experts count).
        """
        # dynamic redundant num from config yml
        if self.max_redundant_per_rank is not None:
            return self.max_redundant_per_rank

        # static redundant num from parttern
        experts_here = self.placement_pattern[rank_device][moe_layer_idx]
        num_redundant_experts = round(torch.sum(experts_here).item() - num_expert_per_device_origin)
        return num_redundant_experts

    def get_world_size(self) -> int:
        return self.placement_pattern.shape[0]

    def get_total_num_expert(self) -> int:
        num_expert = self.placement_pattern.shape[2]
        return num_expert

    def get_total_num_layers(self) -> int:
        num_layers = self.placement_pattern.shape[1]
        return num_layers

    def get_deployed_experts_per_layer(self) -> list:
        # dynamic redundant num from config yml
        if self.max_redundant_per_rank is not None:
            num_layers = self.get_total_num_layers()
            return [self.get_total_num_expert() +  self.max_redundant_per_rank * self.get_world_size()] * num_layers
        # static redundant num from parttern
        deployed_experts_per_layer = torch.sum(self.placement_pattern, dim=(0, 2)).tolist()
        return deployed_experts_per_layer

    def get_redundant_enable_per_layer(self) -> list:
        num_layers = self.get_total_num_layers()
        # dynamic redundant num from config yml
        if self.max_redundant_per_rank is not None:
            return [False] * num_layers if self.max_redundant_per_rank == 0 else [True] * num_layers

        # static redundant num from parttern
        deployed_experts_per_layer = self.get_deployed_experts_per_layer()
        num_logits_expert_per_rank = self.get_total_num_expert()
        redundant_enable_per_layer = [not (value==num_logits_expert_per_rank) for value in deployed_experts_per_layer]
        return redundant_enable_per_layer

    def _calc_expert_offset_each_layer(self) :
        """
        初始化时预计算每个 layer 中每个 rank 的expert offset。
        """
        # 计算每个 rank 在每个 layer 中的专家数，形状：(rank, layer)
        rank_expert_counts = self.placement_pattern.sum(dim=-1)  # 沿 expert 轴求和

        # 计算累积和，沿 rank 轴，形状仍为 (rank, layer)
        # cumsum[i, j] 表示第 j 层中，前 i个 rank 的专家总数
        cumsum_experts = torch.cumsum(rank_expert_counts, dim=0)

        # offset[i, j] 表示第 j 层中，第 i 个 rank 前的专家总数
        # 用零填充第 0 个 rank 的 offset，并取 cumsum 的前一行
        local_expert_offsets = torch.zeros_like(cumsum_experts)
        local_expert_offsets[1:] = cumsum_experts[:-1]  # 第 i 个 rank 的 offset 是前 i-1 个 rank 的和
        return local_expert_offsets

    def get_local_expert_indices_offset(self, layer_idx_moe: int, current_rank: int, default_experts_per_rank: int) -> int:
        if self.max_redundant_per_rank is not None:
            return self.rank * self.max_num_deployed_expert_per_rank

        return self.local_expert_offsets[current_rank, layer_idx_moe].item()

    def get_max_redundant_per_expert(self) :
        # max_redundant_per_expert from config yml
        if self.max_redundant_per_expert is not None:
            return self.max_redundant_per_expert

        pattern = self.placement_pattern.to(dtype=torch.int64)
        redundant_expert_num = pattern.sum(dim=0)
        max_redundant_expert_num = redundant_expert_num.max()
        return max_redundant_expert_num

    def get_default_placement_layers(self):
        """
        Vectorized check for whether each layer satisfies the default placement requirements.

        Returns:
            list[bool]: A list of length num_layers, where True indicates the layer satisfies
                        the default placement, and False indicates it does not.
        """
        placement_pattern = self.placement_pattern
        world_size, num_layers, num_experts = placement_pattern.shape

        # 创建专家 ID 和预期 rank
        expert_ids = torch.arange(num_experts, dtype=torch.long, device=placement_pattern.device)
        num_experts_per_rank = num_experts // world_size
        expected_ranks = expert_ids // num_experts_per_rank

        # 检查预期 rank 是否在 world_size 范围内
        if torch.any(expected_ranks >= world_size):
            raise RuntimeError(f"Some experts require ranks beyond world_size {world_size}")

        # 存储每层的布尔结果
        valid_layers = []

        # 为每一层检查放置情况
        for layer in range(num_layers):
            # 提取当前层的 placement
            layer_placement = placement_pattern[:, layer, :]

            # 检查每个专家在预期 rank 上的值是否为 1
            valid_expected = layer_placement[expected_ranks, expert_ids] == 1

            # 检查每个专家在非预期 rank 上的值是否为 0
            valid_non_expected = torch.ones(num_experts, dtype=torch.bool, device=placement_pattern.device)
            for rank in range(world_size):
                # 跳过预期 rank
                mask = expected_ranks != rank
                if mask.any():
                    # 检查非预期 rank 上的值是否为 0
                    valid_non_expected[mask] &= (layer_placement[rank, expert_ids[mask]] == 0)

            # 该层有效当且仅当预期 rank 值为 1 且非预期 rank 值为 0
            valid_layers.append(torch.all(valid_expected & valid_non_expected).item())

        return valid_layers

    def update_working_mapping(self):
        print("Not implement update_working_mapping.")

    def get_working_mapping(self) -> torch.tensor:
        return self.redundant_expert_mapping

    def get_max_num_deployed_expert_per_rank(self) ->int:
        return self.max_num_deployed_expert_per_rank