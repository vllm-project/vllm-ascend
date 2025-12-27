def generate_expert_map(global_expert_num, world_size, num_moe_layers, num_of_redundant_expert=0):
    """
    Generate expert mapping table for MoE layers with redundant experts support.

    Args:
        global_expert_num (int): Total global experts count
        world_size (int): Number of devices/processes
        num_moe_layers (int): Number of MoE layers
        num_of_redundant_expert (int, optional): Total redundant experts count. Defaults to 0.

    Returns:
        dict: Structured expert mapping table, including layer/device/expert mapping relations
    """
    base_experts = global_expert_num // world_size
    redundant_experts = num_of_redundant_expert // world_size
    remainder = global_expert_num % world_size
    local_num_experts = base_experts + remainder + redundant_experts

    expert_map_tensor = torch.full(
        (num_moe_layers, world_size, global_expert_num),
        -1,
        dtype=torch.int32
    )

    for device_id in range(world_size):
        local_ids = torch.arange(base_experts + redundant_experts, dtype=torch.int32)
        expand_ids = torch.arange(
            base_experts + redundant_experts,
            local_num_experts,
            dtype=torch.int32
        )

        if device_id < world_size - 1:
            start = device_id * base_experts
            end = start + base_experts + redundant_experts
            expert_map_tensor[:, device_id, start:end] = local_ids.unsqueeze(0).expand(num_moe_layers, -1)
        else:
            if remainder > 0:
                slice_end = -remainder
                slice_start = slice_end - (base_experts + redundant_experts)
            else:
                slice_start = -(base_experts + redundant_experts)
                slice_end = None
            expert_map_tensor[:, device_id, slice_start:slice_end] = local_ids.unsqueeze(0).expand(num_moe_layers, -1)
        if remainder > 0:
            expert_map_tensor[:, device_id, -remainder:] = expand_ids.unsqueeze(0).expand(num_moe_layers, -1)

    expert_map = {
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
            global_expert_indices = torch.where(expert_map_tensor[layer_id, device_id] != -1)[0]
            device_expert = global_expert_indices.tolist()
            layer["device_list"].append({
                "device_id": device_id,
                "device_expert": device_expert
            })

        expert_map["layer_list"].append(layer)
    return expert_map

def save_expert_map_to_json(expert_map, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(expert_map, f, indent=2, ensure_ascii=False)
    print(f"Expert map saved to: {file_path}")

if __name__ == '__main__':
    #global_expert_num设为257，num_of_redundant_expert设为0是混置
    #global_expert_num设为256，num_of_redundant_expert设为16是冗余专家（不加混置）
    #global_expert_num设为256，num_of_redundant_expert设为0是 无冗余专家（无混置）

    # 生成与示例类似的配置（可根据需要调整参数）
    expert_map = generate_expert_map(
        global_expert_num=257,  # 全局专家总数
        world_size=16,  # 设备数量
        num_moe_layers=59,  # MoE层数量
        num_of_redundant_expert=0  # 冗余专家总数
    )

    # 打印部分结果验证
    import json
    save_expert_map_to_json(expert_map, "expert_map16_mix.json")
