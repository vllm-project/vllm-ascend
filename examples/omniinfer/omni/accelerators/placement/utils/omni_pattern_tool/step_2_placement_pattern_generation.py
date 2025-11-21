#!/usr/bin/env python
# coding: utf-8
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.


import heapq
import logging
import os
from datetime import datetime
from typing import List, Optional, Tuple, Union

import numpy as np

# Configure font to support Chinese characters
try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    logging.warning("matplotlib is not installed. Chinese font settings are skipped, but this does not affect placement pattern generation.")

def allocate_expert_deployments_improved(
    loads: Union[List[float], np.ndarray],
    expert_redundant_limit: int,
    budget_limit: int,
    load_normalization: str = None,
    is_redundant: bool = False
) -> List[int]:
    """
    Allocate expert deployments with improved strategy, supporting both rearrange-only and redundant modes.

    Args:
        loads: List or numpy array of expert loads.
        expert_redundant_limit: Maximum additional deployments per expert (total = 1 + limit).
        budget_limit: Total deployment budget.
        load_normalization: Normalization method ('log' or None).
        is_redundant: If True, allow redundant deployments; if False, each expert is deployed once.

    Returns:
        List of deployment counts for each expert.
    """
    logger = logging.getLogger(__name__)
    is_numpy = isinstance(loads, np.ndarray)
    num_experts = loads.size if is_numpy else len(loads)
    loads_list = loads.tolist() if is_numpy else list(loads)
    
    if load_normalization == 'log':
        normalized_loads = [np.log1p(load) for load in loads_list]
    else:
        normalized_loads = loads_list

    deployments = [1] * num_experts
    if not is_redundant:
        return deployments  # Rearrange-only mode: each expert deployed once

    # Redundant mode: allow multiple deployments
    remaining_budget = budget_limit
    max_deployments_per_expert = 1 + expert_redundant_limit

    if remaining_budget == 0:
        return deployments

    heap = []
    for i in range(num_experts):
        original_load = normalized_loads[i]
        current_deploy_count = deployments[i]
        priority = -original_load / current_deploy_count if original_load > 0 else 0.0
        if current_deploy_count < max_deployments_per_expert:
            heap.append((priority, original_load, i))
    heapq.heapify(heap)

    deployments_added = 0
    while deployments_added < remaining_budget and heap:
        neg_load_per_instance, original_load, index = heapq.heappop(heap)
        deployments[index] += 1
        deployments_added += 1
        new_deploy_count = deployments[index]

        if new_deploy_count < max_deployments_per_expert:
            new_priority = -original_load / new_deploy_count if original_load > 0 else 0.0
            heapq.heappush(heap, (new_priority, original_load, index))

    if deployments_added < remaining_budget:
        logger.warning(f"Allocated {deployments_added} / {remaining_budget} of the budget.")
    
    return deployments

def distribute_experts_sequentially(
    num_experts: int,
    num_ranks_target_pattern: int
) -> Tuple[float, np.ndarray]:
    """
    Distribute experts sequentially across ranks.

    Args:
        num_experts: Total number of experts.
        num_ranks_target_pattern: Number of target ranks.

    Returns:
        Tuple of maximum device load and placement matrix.
    """
    if num_experts % num_ranks_target_pattern != 0:
        raise ValueError(f"Total number of experts ({num_experts}) must be divisible by target rank count ({num_ranks_target_pattern}).")

    experts_per_device = num_experts // num_ranks_target_pattern
    placement_matrix = np.zeros((num_ranks_target_pattern, num_experts), dtype=int)

    for rank_id in range(num_ranks_target_pattern):
        start_idx = rank_id * experts_per_device
        end_idx = (rank_id + 1) * experts_per_device
        placement_matrix[rank_id, start_idx:end_idx] = 1

    return 0.0, placement_matrix

def distribute_experts_to_ranks(
    initial_loads: Union[List[float], np.ndarray],
    deployments: List[int],
    num_ranks_target_pattern: int,
    layer_idx: int = 0
) -> Tuple[float, np.ndarray]:
    """
    Distribute experts to ranks based on loads and deployments.

    Args:
        initial_loads: List or numpy array of initial expert loads.
        deployments: List of deployment counts for each expert.
        num_ranks_target_pattern: Number of target ranks.

    Returns:
        Tuple of maximum device load and placement matrix.
    """
    logger = logging.getLogger(__name__)
    if isinstance(initial_loads, list):
        loads_np = np.array(initial_loads, dtype=float)
    elif isinstance(initial_loads, np.ndarray):
        if initial_loads.ndim != 1:
            raise ValueError("Input initial_loads must be one-dimensional.")
        loads_np = initial_loads.astype(float)
    else:
        raise TypeError("initial_loads must be a list or numpy.ndarray.")

    if not isinstance(deployments, list) or not all(isinstance(d, int) for d in deployments):
        raise TypeError("deployments must be a list of integers.")
    if len(loads_np) != len(deployments):
        raise ValueError("initial_loads and deployments must have the same length.")
    if num_ranks_target_pattern <= 0:
        raise ValueError("num_ranks_target_pattern must be a positive integer.")

    num_experts = len(loads_np)
    total_deployments = sum(deployments)
    if total_deployments == 0:
        return 0.0, np.zeros((num_ranks_target_pattern, num_experts), dtype=int)

    if total_deployments % num_ranks_target_pattern != 0:
        raise ValueError(f"Total deployments ({total_deployments}) must be divisible by target rank count ({num_ranks_target_pattern}).")

    experts_per_rank = total_deployments // num_ranks_target_pattern
    if experts_per_rank == 0 and total_deployments > 0:
        raise ValueError("Calculated experts per rank is 0, but total deployments is greater than 0.")

    if experts_per_rank > num_experts:
        raise ValueError(f"Each rank requires ({experts_per_rank}) expert instances, but only ({num_experts}) unique expert types are available.")

    max_deployments = max(deployments)
    if max_deployments > num_ranks_target_pattern:
        max_req_expert_idx = np.argmax(deployments)
        raise ValueError(f"Expert {max_req_expert_idx} requires {deployments[max_req_expert_idx]} deployments, "
                         f"exceeding the total number of target ranks {num_ranks_target_pattern}.")

    expert_instances = []
    for expert_idx, count in enumerate(deployments):
        if count > 0:
            load = loads_np[expert_idx] / count
            for _ in range(count):
                expert_instances.append((load, expert_idx))

    expert_instances.sort(key=lambda x: x[0], reverse=True)

    device_loads = np.zeros(num_ranks_target_pattern, dtype=float)
    placement_matrix = np.zeros((num_ranks_target_pattern, num_experts), dtype=int)
    device_expert_counts = np.zeros(num_ranks_target_pattern, dtype=int)
    
    start_rank = 0
    logger.info(f"Layer {layer_idx}: Starting expert placement from rank {start_rank}")

    for load, expert_idx in expert_instances:
        best_device = -1
        min_load_for_candidate = float('inf')

        possible_devices = []
        for i in range(num_ranks_target_pattern):
            rank_id = (start_rank + i) % num_ranks_target_pattern
            can_place_expert = (placement_matrix[rank_id, expert_idx] == 0)
            has_space = (device_expert_counts[rank_id] < experts_per_rank)
            if can_place_expert and has_space:
                possible_devices.append(rank_id)

        if not possible_devices:
            raise RuntimeError(f"Unable to find a suitable rank for expert {expert_idx} (load {load}).")

        best_device = possible_devices[0]
        for i in range(1, len(possible_devices)) :
            if (device_loads[possible_devices[i]] < device_loads[best_device]) :
                best_device = possible_devices[i]

        placement_matrix[best_device, expert_idx] = 1
        device_loads[best_device] += load
        device_expert_counts[best_device] += 1

    new_placement_matrix = placement_matrix.copy()
    num_ranks_per_host = 16
    num_hosts = max(int((num_ranks_target_pattern + 1) / num_ranks_per_host), 1)
    host_cur_rank = [0] * num_hosts
    for i in range(num_ranks_target_pattern) :
        new_host_id = i % num_hosts
        new_placement_matrix[new_host_id * num_ranks_per_host + host_cur_rank[new_host_id]] = placement_matrix[i]
        host_cur_rank[new_host_id] += 1
    placement_matrix = new_placement_matrix
    if not np.all(device_expert_counts == experts_per_rank):
        logger.warning(f"Expert counts per rank after allocation do not all equal the expected value {experts_per_rank}.")
        logger.warning(f"Actual counts: {device_expert_counts}")

    max_device_load = np.max(device_loads) if total_deployments > 0 else 0.0
    return max_device_load, placement_matrix

def process_expert_deployments(
    input_file: str,
    output_dir: str,
    num_ranks_target_pattern: int,
    num_special_layers: int = None,
    expert_redundant_limit: int = 11,
    num_layers_target_pattern: int = 58,
    num_eps_target_pattern: int = 256,
    output_file: str = None,
    is_redundant: bool = False,
    collecting_modes: str = 'all',
    log_timestamp: Optional[str] = None,
    recordstep_range: Optional[str] = None
) -> np.ndarray:
    """
    Process expert deployments and generate a placement pattern, supporting both rearrange-only and redundant modes.

    Args:
        input_file: Path to input CSV file.
        output_dir: Directory for output placement pattern.
        num_ranks_target_pattern: Number of target ranks.
        num_special_layers: Number of layers to apply special allocation (rearrange or redundant).
        expert_redundant_limit: Maximum additional deployments per expert.
        num_layers_target_pattern: Number of target layers.
        num_eps_target_pattern: Number of experts per layer.
        output_file: Output filename (optional).
        is_redundant: If True, allow redundant deployments; if False, rearrange only.
        collecting_modes: Data collection mode ('prefill', 'decode', or 'all') for filename.
        log_timestamp: Timestamp for log file naming (optional).
        recordstep_range: Range of recordstep or step values (format: start:end, optional).

    Returns:
        Placement pattern as a numpy array.
    """
    # Set up logging
    if log_timestamp is None:
        raise ValueError("log_timestamp must be provided.")
    log_file = f"pattern_generation_pipeline_{log_timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8', mode='a')  # Append mode to share log file
        ]
    )
    logger = logging.getLogger(__name__)

    if not os.path.exists(input_file):
        raise ValueError(f"Input file {input_file} does not exist.")

    if num_special_layers is None:
        num_special_layers = num_layers_target_pattern

    data = np.genfromtxt(input_file, delimiter=',', skip_header=1)
    ep_activation_counts = data[:, 1:] + 3
    logger.info(f"Shape of ep_activation_counts: {ep_activation_counts.shape}")
    logger.info(f"Maximum activation counts per layer: {ep_activation_counts.max(1)}")

    budget_limit = [0 for _ in range(num_layers_target_pattern)]
    placement_pattern = np.zeros((num_ranks_target_pattern, num_layers_target_pattern, num_eps_target_pattern), dtype=np.int32)

    layer_max_loads = np.zeros(num_layers_target_pattern)
    for layer_idx in range(num_layers_target_pattern):
        _, base_placement = distribute_experts_sequentially(
            num_experts=num_eps_target_pattern,
            num_ranks_target_pattern=num_ranks_target_pattern
        )
        device_loads = np.sum(ep_activation_counts[layer_idx] * base_placement, axis=1)
        layer_max_loads[layer_idx] = np.max(device_loads) if device_loads.size > 0 else 0.0

    sorted_indices = np.argsort(layer_max_loads)
    logger.info(f"Indices sorted by maximum load: {sorted_indices}")

    high_load_layers = set()
    if num_special_layers > 0:
        high_load_layers = set(sorted_indices[-num_special_layers:])
        logger.info(f"High load layer indices (top {num_special_layers} layers): {high_load_layers}")
        if is_redundant:
            for layer_idx in high_load_layers:
                budget_limit[layer_idx] = num_ranks_target_pattern
            logger.info(f"Budget limit for redundant layers: {budget_limit}")
    else:
        logger.info("No special allocation applied.")

    for layer_idx_moe in range(num_layers_target_pattern):
        if is_redundant or layer_idx_moe in high_load_layers:
            # Redundant mode or high load layers in rearrange mode: use optimized allocation
            expert_allocation_count = allocate_expert_deployments_improved(
                ep_activation_counts[layer_idx_moe],
                expert_redundant_limit=expert_redundant_limit,
                budget_limit=budget_limit[layer_idx_moe] if is_redundant else num_eps_target_pattern,
                is_redundant=is_redundant
            )
            max_load, placement_matrix = distribute_experts_to_ranks(
                initial_loads=ep_activation_counts[layer_idx_moe],
                deployments=expert_allocation_count,
                num_ranks_target_pattern=num_ranks_target_pattern,
                layer_idx=layer_idx_moe
            )
            logger.info(f"Layer {layer_idx_moe}: Optimized allocation, number of deployed experts = {sum(expert_allocation_count)}")
        else:
            # Rearrange mode, non-high load layers: use sequential allocation
            max_load, placement_matrix = distribute_experts_sequentially(
                num_experts=num_eps_target_pattern,
                num_ranks_target_pattern=num_ranks_target_pattern
            )
            logger.info(f"Layer {layer_idx_moe}: Sequential allocation, number of deployed experts = {num_eps_target_pattern}")
        placement_pattern[:, layer_idx_moe, :] += placement_matrix

    logger.info("Allocation status for high load layers:")
    for layer_idx in high_load_layers:
        total_deployments = placement_pattern[:, layer_idx, :].sum()
        logger.info(f"Layer {layer_idx}: Total deployments = {total_deployments}")
      
    if output_file is None:
        step_suffix = f"_step{recordstep_range.split(':')[0]}to{recordstep_range.split(':')[1]}" if recordstep_range else ""
        mode = 'redundant' if is_redundant else 'rearrange'
        suffix = f'epmaxdeploy_{expert_redundant_limit+1}_{collecting_modes}{step_suffix}' if is_redundant else f'{collecting_modes}{step_suffix}'
        output_file = (f"placement_pattern_{log_timestamp}_{num_special_layers}_{mode}_layers_"
                    f"{num_layers_target_pattern}_layers_{num_ranks_target_pattern}_ranks_{suffix}.npy")
    
    output_path = os.path.join(output_dir, output_file)
    output_path = os.path.normpath(output_path)
    logger.info(f"Attempting to save placement pattern to: {output_path}")
    np.save(output_path, placement_pattern)
    if not os.path.exists(output_path):
        raise OSError(f"Failed to save placement pattern file: {output_path}")
    
    # Only print final output to terminal
    print(f"Saving placement pattern to: {output_path}")
    print(f"Placement pattern shape: {placement_pattern.shape}")
    logger.info(f"Placement pattern saved successfully: {output_path}")
    return placement_pattern

if __name__ == "__main__":
    raise ValueError("Please provide input_file and num_ranks_target_pattern to run this script.")