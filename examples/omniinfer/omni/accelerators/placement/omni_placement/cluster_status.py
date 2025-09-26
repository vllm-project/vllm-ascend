# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
import torch_npu
from collections import deque
from .config import Config
from .expert_mapping import ExpertMapping

class ClusterStatus:
    """
    A class to represent the status of the cluster.
    """
    def __init__(self, config: Config, expert_mapping: ExpertMapping = None, rank: int = 0, **kwargs):
        """
        Initialize ClusterStatus。

        Args:
           config (Config): Configuration object containing cluster settings.
           placement_pattern (torch.Tensor): Experts deloyment mapping
           **kwargs: other dynamic status information.
        """
        self.expert_mapping = expert_mapping
        self.config = config
        self.placement_pattern = self.expert_mapping.placement_pattern
        self.current_status = kwargs  # Use a dictionary to store dynamic status tensors.
        self.state_actions_queue = deque()
        self.rank = rank
        self.device = self.placement_pattern.device




    def add_state_action(self, state, action):
        """
        add state update action which will be executed asynchronously peroiodically (e.g. every 5 seconds)?
        poentially update the current state in multiple queues
        """
        # add state update action to a queue
        self.state_actions_queue.append((state, action))

    def execute_state_actions(self):
        """
        Performs all status update action in the queue.
        This method can be called by a timer to implement periodic action.
        """
        while self.state_actions_queue:
            state, action = self.state_actions_queue.popleft()
            # do action when state matched
            if state(self):
                action()

    def add_status(self, name: str, tensor: torch.Tensor):
        """
        Add a new status tensor, such expert loads

        Args:
            name (str): status name.
            tensor (torch.Tensor): status info
        """
        self.current_status[name] = tensor


    def add_status_by_layer(self, name: str, layer_id: int, tensor: torch.Tensor):
        """
        Add a new status tensor for a specific layer
        Args:
            name (str): status name.
            layer_id (int): MoE layer id, maximum is self.config.max_moe_layer_num, default is 58
            tensor (torch.Tensor): status info, such as expert loads
        """

        token_num, expert_num, expert_metric_dim = tensor.shape  # Extract dimensions from the tensor shape
        layer_num = self.config.max_moe_layer_num  # Set layer_num to the maximum layer number

        #if the tensor is not in the current_status, create a new one
        if name not in self.current_status:
            #new a  torch tensor with shape (layer_num, token_num, expert_num, expert_metric_dim) and fill with 0
            self.current_status[name] = torch.zeros((layer_num, token_num, expert_num, expert_metric_dim))

        # if input tensor is shape (token_num, expert_num, expert_metric_dim), copy it to the corresponding layer of current_status
        if tensor.shape == (token_num, expert_num, expert_metric_dim):
            self.current_status[name][layer_id] = tensor
        else:
            raise ValueError(f"Invalid tensor shape: {tensor.shape}. Expected {(token_num, expert_num, expert_metric_dim)}.")


    def get_status(self, name: str) -> torch.Tensor:
        """
        Get a status info by name

        Args:
            name (str): status name.

        Return:
            torch.Tensor: status info
        """
        return self.current_status.get(name)

    def __repr__(self):
        return f"ClusterStatus(placement_pattern={self.placement_pattern}, current_status={self.current_status})"