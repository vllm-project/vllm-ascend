# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
 
import csv
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, cast
import numpy as np
import torch
import torch_npu
import torchair as tng
import ctypes

from typing import Optional
from .cluster_status import ClusterStatus
from .placement_handler import create_cluster_activation, create_placement_manager, init_dram_weights, do_placement_optimizer
from .optim.optimizers import Optimizer
from .optim.optimizers_loader import _create_optimizers
from .config import Config
from .expert_mapping import ExpertMapping
from .utils import calculate_time
from . import omni_placement
from datetime import datetime

import time

class OmniPlannerMeta(type):
    """Metaclass to implement singleton pattern for OmniPlanner."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    @classmethod
    def cleanup(cls):
        if cls in cls._instances and not cls._cleanup_called:
            cls._instances[cls].cleanup()
            del cls._instances[cls]
            cls._cleanup_called = True

class OmniPlanner(metaclass=OmniPlannerMeta):
    """
    Optimizes token-to-expert mapping using multiple optimizers.
    Manages expert deployment across distributed systems.

    Attributes:
        config: Configuration object for planner settings
        cluster_status: Cluster status monitor
        optimizers: List of optimization algorithms
        expert_mapping: Expert deployment pattern mapping
    """
    def __init__(self, config_file: str = "/etc/omni/config.yaml", device: str = "npu",
                 rank: int = None, world_size: int = None, num_devices_per_host: int = 16,
                 num_experts = 256, num_redundancy_shared_expert_rank=0):
        """Initialize OmniPlanner with configuration and distributed settings.

        Args:
            config_file: Path to configuration YAML file
            device: Target device type (e.g., "npu", "cuda")
            rank: Process rank in distributed environment
            world_size: Total number of processes in distributed environment
            num_devices_per_host: Number of devices per host machine (default: 8)
        """
        # Load configuration
        self.config = Config(config_file)
        self.device = torch.device(device)
        self.max_moe_layer_num = self.config.getattr("max_moe_layer_num",None)
        if self.max_moe_layer_num is None:
            print(f"[Placement-Error]-max_moe_layer_num is not defined in config.yaml")
            exit(1)

        # Initialize distributed settings with fallback
        self._init_distributed(rank, world_size, num_devices_per_host, num_redundancy_shared_expert_rank)

        self.enable_dynamic = getattr(self.config, 'enable_dynamic', True)

        self.enable_rank_round_robin = self.config.getattr("enable_rank_round_robin",None)
        if self.enable_rank_round_robin is None:
            print(f"[Placement-Error]-enable_rank_round_robin is not defined in config.yaml")
            exit(1)

        # Load and validate placement pattern
        self.expert_mapping = ExpertMapping(self.config, self.device, self.rank, self.world_size, self.num_devices_per_host, self.enable_dynamic, num_experts, self.enable_rank_round_robin)
        if (self.expert_mapping.get_world_size() != self.world_size):
            print(f"[Placement-Error]-Pattern world_size is {self.expert_mapping.get_world_size()} should be {self.world_size}.")
            exit(1)
        
        # TODO: 无效代码
        """Initialize cluster status and optimizers."""
        self.cluster_status = ClusterStatus(self.config, self.expert_mapping, self.rank)
        # self.optimizers = _create_optimizers(self.config.Optimizers, self.cluster_status)
        # self.optimizer = self.optimizers[0]
        
        dump_dir = getattr(self.config, 'dump_dir', None)
        self.enable_dump = getattr(self.config, 'enable_dump', False) if dump_dir else False

        # Initialize placement manager
        self._init_placement_manager()  

        # Get selector
        if not self.enable_rank_round_robin:
            max_num_tokens = 100000
            self.redundant_bias = torch.arange(max_num_tokens, dtype=torch.int32, device = self.device).view(-1,1)

        self.selector = self.expert_mapping.get_selector()
        self.num_redundant_per_expert = self.expert_mapping.get_num_redundant_per_expert()

        if self.enable_dump and self.rank == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dump_dir = os.path.join(dump_dir,timestamp)
            os.makedirs(dump_dir, exist_ok=True)
            self.cluster_activation.setDumpDir(dump_dir)

        # redundant_enable_per_layer, True is redundant layer, False is Origin Layer
        self.redundant_enable_per_layer = self.expert_mapping.get_redundant_enable_per_layer()
        print("OmniPlanner successfully initialized.")

    @classmethod
    def cleanup(cls):
        if cls in cls._instances:
            del cls._instances[cls]

    def __del__(self):
        if hasattr(self, 'cluster_activation'):
            self.cluster_activation.stop_thread()
            del self.cluster_activation
            time.sleep(1)
        if hasattr(self, 'placement_manager'):
            del self.placement_manager
            time.sleep(1)

    def _init_distributed(self, rank: int = None, world_size: int = None, num_devices_per_host: int = 16, num_redundancy_shared_expert_rank: int = 0) -> None:
        """Initialize distributed settings with fallback to provided values.

        Args:
            rank: Process rank in distributed environment
            world_size: Total number of processes in distributed environment
            num_devices_per_host: Number of devices per host machine
        """
        # Get rank and world size from distributed environment if not provided
        if rank is None or world_size is None:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank, self.world_size = rank, world_size

        self.num_redundancy_shared_expert_rank = num_redundancy_shared_expert_rank
        self.rank -= self.num_redundancy_shared_expert_rank
        if self.rank < 0:
            self.rank += self.world_size
        self.hccl_comm_world_size = self.world_size
        self.world_size -= self.num_redundancy_shared_expert_rank

        self.num_devices_per_host = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")  # omni_placement config file
        self.num_devices_per_host = len(self.num_devices_per_host.split(",")) if self.num_devices_per_host else num_devices_per_host

        # Validate that world_size is consistent with num_devices_per_host
        if self.world_size % self.num_devices_per_host != 0:
            print(f"Warning: world_size ({self.world_size}) is not evenly divisible by "
                  f"num_devices_per_host ({self.num_devices_per_host})")

    def _init_placement_manager(self) -> None:
        """Initialize placement handler, and activation tracking."""
        num_layers = self.expert_mapping.get_total_num_layers()

        self.npu_activation_count = torch.zeros(
            (num_layers, self.get_max_num_deployed_expert_per_rank()),
            device=self.device,
            dtype=torch.int64
        )
        self.max_activation_count = int(1e16)

        self.cluster_activation = create_cluster_activation(
            self.rank,
            self.world_size,
            self.hccl_comm_world_size,
            self.expert_mapping.get_total_num_layers(),
            self.get_max_num_deployed_expert_per_rank(),
            self.npu_activation_count,
            self.max_activation_count
        )
        if self.enable_dynamic or self.enable_dump:
            self.placement_manager = create_placement_manager(
                self.rank,
                self.world_size,
                self.hccl_comm_world_size,
                self.num_devices_per_host,
                self.cluster_activation,
                self.cluster_status.expert_mapping,
                self.enable_dynamic
            )

    def is_moe_layer(self, layer_idx_moe):
        return layer_idx_moe < self.max_moe_layer_num
    
    def start_dynamic_optimize_expert_load_balance(self):
        is_thread_required = self.enable_dynamic or self.enable_dump
        if is_thread_required:
            self.placement_manager.start_thread()

    def get_max_num_deployed_expert_per_rank(self)-> int:
        return self.expert_mapping.get_max_num_deployed_expert_per_rank()
    
    def is_expert_on_current_rank(
        self,
        layer_id: int,
        expert_id: int,
        current_rank: int,
        experts_per_rank: int
    ) -> Tuple[bool, int]:
        """
        Check if expert is deployed on current rank and get its position.

        Args:
            layer_id: ID of the MoE layer
            expert_id: Expert ID within the layer
            current_rank: Target device rank to check
            experts_per_rank: Experts per device in default deployment

        Returns:
            Tuple (exists_on_rank, local_position)
        """
        return self.expert_mapping.is_expert_on_current_rank(layer_id, expert_id, current_rank, experts_per_rank)

    def expert_mapping_on_current_layer(
        self,
        layer_idx_moe: torch.tensor,
        is_prefill=False) -> torch.tensor:
        if not self.is_moe_layer(layer_idx_moe):
            return None
        return self.selector[layer_idx_moe]

    # @calculate_time
    def place_experts(self, layer_idx_moe: Optional[int] = None) :
        """Dynamically places expert weights across ranks based on activation status.

        Args:
            layer_idx_moe: Identifier for the current layer (optional)

        Returns:
            int: 0 on success, error code on failure.

        Raises:
            RuntimeError: If HCCL operations (e.g., memory allocation, communication) fail.
        """
        if self.enable_dynamic:
            omni_placement.do_placement_optimizer(self.placement_manager)

    def plan(
        self,
        layer_idx_moe: Optional[int] = None,
        tokens: Optional[torch.Tensor] = None,
        token_expert_ids: Optional[torch.Tensor] = None,
        token_expert_scores: Optional[torch.Tensor] = None,
        top_k: int = 8,
        expert_mapping: Optional[torch.Tensor] = None,
        is_prefill=True
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Optimize token-to-expert mapping using configured optimizers.

        This method takes input tokens and their initially assigned experts and scores. It computes
        expert loads, updates the cluster status accordingly, and then optimizes the assignment
        of tokens to experts by applying the configured optimization strategies.

        Args:
            layer_idx_moe: Identifier for the current layer (optional)
            tokens: Input tokens tensor with shape [num_tokens, ...]
            token_expert_ids: Initial expert assignments, shape [num_tokens, top_k], -1 indicates unassigned
            token_expert_scores: Importance scores for expert assignments, shape [num_tokens, top_k]

        Returns:
            Tuple containing (original tokens, optimized expert IDs, optimized scores)
        """
        # Input validation check
        if not self.is_moe_layer(layer_idx_moe):
            return tokens, token_expert_ids, token_expert_scores
        if self.enable_rank_round_robin:
            token_expert_ids = torch.nn.functional.embedding(token_expert_ids,expert_mapping).squeeze(-1)
        else:
            batch_size = token_expert_ids.shape[0]
            token_expert_ids = expert_mapping[token_expert_ids, self.redundant_bias[:batch_size,] % self.num_redundant_per_expert[layer_idx_moe][token_expert_ids]]
        return tokens, token_expert_ids, token_expert_scores


    @staticmethod
    def get_deepseek_v3_moe_layer_idx(prefix: str, first_k_dense_replace=3) -> int:
        """
        Calculate the adjusted DeepSeek-V3 MoE layer index from a model layer prefix.

        The function parses a prefix string of format `model.layers.{N}.mlp.experts` to extract the
        layer index `N`, then adjusts this index by subtracting a fixed offset of dense layers
        (FIRST_K_DENSE_REPLACE) as per the DeepSeek-V3 model configuration.

        Args:
            prefix: A layer path string formatted as `model.layers.{N}.mlp.experts`
                (e.g., "model.layers.5.mlp.experts" represents layer 5)

        Returns:
            int: The adjusted layer index after subtracting FIRST_K_DENSE_REPLACE.
                Formula: parsed_layer_id - FIRST_K_DENSE_REPLACE

        Note:
            - LAYER_ID_IDX (2): Indicates layer ID position after splitting the prefix by '.'
            (e.g., ["model", "layers", "5", "mlp", "experts"] -> index 2 is "5")
            - FIRST_K_DENSE_REPLACE (3): Number of initial dense layers from the model's config.json
            that should be excluded when working with MoE layers.

        Example:
            >>> get_deepseek_v3_moe_layer_idx("model.layers.5.mlp.experts")
            2   # 5 (parsed) - 3 (offset) = 2
        """
        # Parses prefix string like 'model.layers.3.mlp.experts'
        LAYER_ID_IDX = 2               # Position of the layer ID after splitting by '.'

        return int(prefix.split(sep='.')[LAYER_ID_IDX]) - first_k_dense_replace

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
        if not self.is_moe_layer(moe_layer_idx):
            return 0
        return self.expert_mapping.get_num_of_redundant_experts(moe_layer_idx, num_expert_per_device_origin, rank_device)

    def init_dram_weights(self, param_dict, first_k_dense_replace):
        if self.enable_dynamic and not self.is_redundant_share_expert_rank():
            moe_weights = self.placement_manager.get_moe_weights()
            init_dram_weights(moe_weights, param_dict, first_k_dense_replace,init_shm=False)
    
    def is_redundant_share_expert_rank(self):
        return self.rank>=self.world_size

    def record_activation(self, layer_idx_moe, expert_token_num, support_multi_stream=False):
        if  self.is_moe_layer(layer_idx_moe) and (self.enable_dynamic or self.enable_dump):
            if not support_multi_stream:
                self.npu_activation_count[layer_idx_moe:layer_idx_moe + 1] = (self.npu_activation_count[layer_idx_moe:layer_idx_moe + 1]+expert_token_num[None]) % self.max_activation_count
            else:
                with tng.scope.npu_stream_switch('21'):
                    self.npu_activation_count[layer_idx_moe:layer_idx_moe + 1] = (self.npu_activation_count[layer_idx_moe:layer_idx_moe + 1]+expert_token_num[None]) % self.max_activation_count

# Example usage
if __name__ == "__main__":
    from optimizer.ada_router_optimizer import AdaRouter
    from optimizer.token_balance_optimizer import TokenBalance

    # Example input: 3 tokens, 4 experts each, with importance scores
    input_token = torch.tensor([
        [0, 1, 2, 3],  # Token 1
        [1, 0, 3, 2],  # Token 2
        [3, 2, 1, 0]   # Token 3
    ], dtype=torch.float32).npu()

    input_expert_id = torch.tensor([
        [0, 1, 2, 3],  # Token 1 expert
        [1, 0, 3, 2],  # Token 2 expert
        [3, 2, 1, 0]   # Token 3 expert
    ], dtype=torch.long).npu()

    input_expert_score = torch.tensor([
        [0.9, 0.5, 0.3, 0.7],  # Token 1 expert score
        [0.4, 0.8, 0.6, 0.2],  # Token 2 expert score
        [0.7, 0.3, 0.9, 0.5]   # Token 3 expert score
    ], dtype=torch.float32).npu()

    planner = OmniPlanner("./config.yaml")

    token, token_expert_ids, token_scores = planner.plan(
        layer_id=0,
        tokens=input_token,
        token_expert_ids=input_expert_id,
        token_expert_scores=input_expert_score
    )

    print("Input mapping:")
    print(input_token, input_expert_id, input_expert_score)

    print("\nOptimized mapping:")
    print(token, token_expert_ids, token_scores)