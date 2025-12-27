import json
import random
import numpy as np
from typing import Dict, List
from typing import Optional,Union
from pathlib import Path
import torch
import torch.distributed as dist
from vllm_ascend.ops.expert_load_balancer import ExpertLoadBalancer
from vllm_ascend.eplb.core.policy.policy_abstract import DynamicConfig
from vllm_ascend.eplb.core.policy.policy_factory import PolicyFactory
from vllm_ascend.eplb.core.policy.policy_flashlb import FlashLB

def generate_expert_map(
    num_router_expert: int,
    num_shared_expert: int,
    world_size: int,
    num_moe_layers: int,
    num_of_redundant_expert: int
) -> torch.Tensor:
    
    total_expert = num_router_expert + num_of_redundant_expert
    expert_per_rank = total_expert // world_size + num_shared_expert
    if total_expert % world_size != 0:
        raise ValueError(
            f"Total base expert count ({total_expert}) must be divisible by world_size ({world_size}). "
            f"Got remainder: {total_expert % world_size}"
        )
    
    router_experts = torch.arange(num_router_expert, dtype=torch.int32)
    redundant_experts = torch.arange(num_of_redundant_expert, dtype=torch.int32)
    
    shared_experts = torch.arange(num_router_expert, num_router_expert + num_shared_expert, dtype=torch.int32)

    base_experts = torch.cat([router_experts, redundant_experts])
    base_split = base_experts.chunk(world_size)

    single_layer_map = []
    for rank_idx in range(world_size):
        rank_base = base_split[rank_idx]
        rank_experts = torch.cat([rank_base, shared_experts])
        if len(rank_experts) != expert_per_rank:
            raise ValueError(
                f"Rank {rank_idx} expert count mismatch: expected {expert_per_rank}, got {len(rank_experts)}. "
                f"Base experts: {len(rank_base)}, shared experts: {len(shared_experts)}"
            )
        
        single_layer_map.append(rank_experts)
    
    single_layer_map = torch.stack(single_layer_map)
    expert_map = single_layer_map.unsqueeze(0).repeat(num_moe_layers, 1, 1)

    return expert_map

def save_expert_map_to_json(
    expert_map: torch.Tensor, 
    file_path: str,
) -> None:
    num_moe_layers = expert_map.shape[0]
    world_size = expert_map.shape[1]
    exp_map = {
        "moe_layer_count": num_moe_layers,
        "layer_list": []
    }

    for layer_id in range(num_moe_layers):
        layer = {
            "layer_id": layer_id,
            "device_count": world_size,
            "device_list": []
        }

        for device_id in range(world_size):
            global_expert_indices = expert_map[layer_id, device_id]
            device_expert = global_expert_indices.tolist()

            layer["device_list"].append({
                "device_id": device_id,
                "device_expert": device_expert
            })

        exp_map["layer_list"].append(layer)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(exp_map, f, indent=2, ensure_ascii=False)
    print(f"Expert map saved to: {file_path}")

def get_logical_expert_hotness(
    num_of_expert: int,
    deployment: torch.Tensor,
    rank_load: torch.Tensor,
    layer_dim: Optional[int] = 0
) -> torch.Tensor:
    if deployment.dim() == 2:
        deployment = deployment.unsqueeze(layer_dim)
        rank_load = rank_load.unsqueeze(layer_dim)
    num_layers = deployment.size(layer_dim)
    
    hotness = torch.zeros(
        num_layers,
        num_of_expert,
        dtype=rank_load.dtype,
        device=rank_load.device
    )
    
    deployment_flat = deployment.flatten(start_dim=layer_dim+1).long()
    rank_load_flat = rank_load.flatten(start_dim=layer_dim+1)
    index_expanded = deployment_flat
    src_expanded = rank_load_flat
    hotness.scatter_add_(
        dim=1,
        index=index_expanded,
        src=src_expanded
    )
    return hotness

def construct_moe_load(
    file_dir: Union[str, Path],
    mix_placement: bool = False,
) -> torch.Tensor:
    """
    Load moe_load files from the specified directory and expand dimensions
    Args:
        file_dir: Directory where moe_load files are located (e.g., "moe_load/moe_load_baseline")
        ep_size: Expert parallel size (default 32, corresponding to the second dimension of the tensor)
        expert_dim: Original expert dimension (default 8, corresponding to the third dimension of the tensor)
        target_col: Target number of columns after expansion (default 9)
    Returns:
        moe_load_expanded: Expanded load tensor with shape [num_layers, ep_size, target_col]
    """

    file_dir = Path(file_dir)
    if not file_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {file_dir}")
    
    moe_files = sorted(list(file_dir.glob("moe_load_*.npy")))
    if not moe_files:
        raise FileNotFoundError(f"No moe_load_*.npy files found in directory {file_dir}")
    
    moe_load = None
    for file_path in moe_files:
        data = np.load(file_path)
        if moe_load is None:
            moe_load = data
        else:
            moe_load += data

    moe_load = torch.tensor(moe_load, dtype=torch.int32)
    ep_size=moe_load.shape[1]
    if mix_placement:
        total_sum = moe_load.sum(dim=(1, 2))
        avg_val = total_sum / (ep_size * 8)
        avg_col = avg_val.unsqueeze(1).unsqueeze(2).expand(-1, ep_size, 1)
        moe_load = torch.cat([moe_load, avg_col], dim=2)
    return moe_load

def generate_policy_expert_map(src_expert_map_path,moe_load_path,dst_expert_map_path,mix_placement,policy_type):
    num_experts = 257 if mix_placement else 256
    exp_lb = ExpertLoadBalancer(src_expert_map_path,num_experts)
    expert_map = exp_lb.expert_map_tensor
    dummy_config = DynamicConfig()
    policy = PolicyFactory.generate_policy(policy_type,dummy_config)
    moe_load = construct_moe_load(moe_load_path,mix_placement)
    expert_map = torch.tensor(policy.rebalance_experts(expert_map,moe_load)[2])
    save_expert_map_to_json(expert_map,dst_expert_map_path)

def generate_multi_stage_expert_map(src_expert_map_path,moe_load_path,dst_expert_map_path,mix_placement):
    num_experts = 257 if mix_placement else 256
    exp_lb = ExpertLoadBalancer(src_expert_map_path,num_experts)
    expert_map = exp_lb.expert_map_tensor
    ep_size=expert_map.shape[1]
    dummy_config = DynamicConfig()
    policy = FlashLB(dummy_config)
    policy.max_stage_window = 64
    policy.buffer_expert_layer_num = 59
    file_dir = Path(moe_load_path)
    moe_files = sorted(list(file_dir.glob("moe_load_*.npy")))
    if not moe_files:
        raise FileNotFoundError(f"No moe_load_*.npy files found in directory {file_dir}")
    
    for file_path in moe_files:
        moe_load = np.load(file_path)
        moe_load = torch.tensor(moe_load, dtype=torch.int32)
        total_sum = moe_load.sum(dim=(1, 2))
        avg_val = total_sum / (ep_size * 8)
        avg_col = avg_val.unsqueeze(1).unsqueeze(2).expand(-1, ep_size, 1)
        moe_load = torch.cat([moe_load, avg_col], dim=2)
        expert_map = torch.tensor(policy.rebalance_experts(expert_map,moe_load)[2])
    save_expert_map_to_json(expert_map,dst_expert_map_path)

num_router_expert = 256
num_shared_expert = 1
world_size = 32
num_moe_layers = 59
num_of_redundant_routed_expert = 0

expert_map = generate_expert_map(
    num_router_expert=num_router_expert,
    num_shared_expert=num_shared_expert,
    world_size=world_size,
    num_moe_layers=num_moe_layers,
    num_of_redundant_expert=num_of_redundant_routed_expert
)

save_expert_map_to_json(expert_map,"expert_map.json")

generate_policy_expert_map(
    src_expert_map_path="expert_map.json",
    moe_load_path="/home/r00934900/moe_load/moe_load_baseline",
    dst_expert_map_path="expert_map_policy.json",
    mix_placement=True,
    policy_type=1)
generate_multi_stage_expert_map(
    src_expert_map_path="expert_map.json",
    moe_load_path="/home/r00934900/moe_load/data",
    dst_expert_map_path="expert_map_policy.json",
    mix_placement=True,
)
