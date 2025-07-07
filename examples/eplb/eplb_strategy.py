# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import json
import logging
import os

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch

logger = logging.getLogger("msit_logger")


def save_matrix_to_json(output_path, file_name, deployment):
    num_layers = deployment.shape[0]
    num_cards = deployment.shape[1]

    data = {"moe_layer_count": num_layers}
    layer_list = []
    for i in range(num_layers):
        layer = {"layer_id": i, "device_count": num_cards}
        device_list = []
        for j in range(num_cards):
            device = {
                "device_id": j,
                "device_expert": deployment[i, j].tolist()
            }
            device_list.append(device)
        layer["device_list"] = device_list
        layer_list.append(layer)
    data["layer_list"] = layer_list

    file_name = f"{output_path}{file_name}.json"

    # Save as JSON file
    try:
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"write {file_name} failed: {e}")


def calculate_average(lst):
    """calculate the average of a list"""
    if not lst:
        raise ValueError("list is empty")

    total = 0.0
    count = 0

    for element in lst:
        # Check if element is numeric
        if isinstance(element, (int, float, np.int64, np.float64)):
            total += float(element)
            count += 1
        else:
            # Non-numeric elements will be ignored with a warning
            print(f"warning: element {element} is not a number, ignored")

    if count == 0:
        raise ValueError("list does not contain any number")

    return total / count


def layer_imblance_polt(y_list, label_names, device_num, output_path,
                        file_name):

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    x = [i for i in range(58)]
    for index, y in enumerate(y_list):
        plt.plot(x,
                 y,
                 label=rf'{label_names[index]}，avg={calculate_average(y)}')

    plt.legend()
    plt.title(rf'Load Distribution (num_gpus={device_num})')
    plt.xlabel('layer')
    plt.ylabel('Device Load')

    # Show grid lines
    plt.grid(True)

    plt.savefig(os.path.join(output_path, file_name), dpi=300)

    # Clear current plot
    plt.close()


def deepseek_deploy(workload, num_redundancy_expert, num_groups, num_nodes,
                    num_gpus, num_original_expert):
    from eplb_deepseek import rebalance_experts
    num_replicas = num_original_expert + num_redundancy_expert
    hy2log, log2phy, logcnt = rebalance_experts(workload, num_replicas,
                                                num_groups, num_nodes,
                                                num_gpus)

    # Convert to global_deployment
    workload = workload.cpu().numpy()
    global_deployment = []
    layer_num = log2phy.shape[0]
    num_physical_experts_local = (num_original_expert +
                                  num_redundancy_expert) // num_gpus
    for layer_idx in range(layer_num):
        layer_deployment = []
        for gpu_idx in range(num_gpus):
            local_deployment = hy2log[layer_idx][gpu_idx *
                                                 num_physical_experts_local:
                                                 (gpu_idx + 1) *
                                                 num_physical_experts_local]
            local_deployment = local_deployment.flatten()
            layer_deployment.append(local_deployment.tolist())
        global_deployment.append(layer_deployment)

    # Remap expert distribution according to log2phy
    original_weights = []
    max_weights = []
    average_weights = []
    y_list = []
    for layer_idx in range(layer_num):
        new_value = workload[layer_idx].reshape(num_gpus, -1)
        row_sum = np.sum(new_value, axis=1)
        original_weights.append(row_sum.max())
        average_weights.append((np.sum(workload[layer_idx]) / num_gpus))

        opt_workload = np.zeros((num_original_expert + num_redundancy_expert),
                                dtype=np.float64)
        for expert_idx in range(num_original_expert):
            physical_expert_idxs = log2phy[layer_idx][expert_idx]
            physical_expert_idxs = physical_expert_idxs.flatten()
            physical_expert_idxs = physical_expert_idxs[
                physical_expert_idxs != -1]
            for physical_expert_idx in physical_expert_idxs:
                opt_workload[physical_expert_idx] += workload[layer_idx][
                    expert_idx] / len(physical_expert_idxs)
        opt_workload = opt_workload.reshape(num_gpus, -1)
        row_sum = np.sum(opt_workload, axis=1)
        max_weights.append(row_sum.max())

    y_list = [original_weights, max_weights, average_weights]
    return global_deployment, y_list


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="gsm8k_temp0.0")
    parser.add_argument("--num_original_expert", type=int, default=256)
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--num_redundancy_expert", type=int, default=0)
    parser.add_argument("--num_devices", type=int, default=32)
    parser.add_argument("--num_groups", type=int, default=8)
    parser.add_argument("--num_nodes", type=int, default=4)
    args = parser.parse_args()
    exp_name = args.exp_name
    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    num_redundancy_expert = args.num_redundancy_expert
    num_devices = args.num_devices
    num_original_expert = args.num_original_expert
    num_groups = args.num_groups
    num_nodes = args.num_nodes

    # NOTE: assume input workload format: [layer_num, num_experts]
    workload = torch.load(input_path, map_location=torch.device('cpu'))
    global_deployment, y_list = deepseek_deploy(workload,
                                                num_redundancy_expert,
                                                num_groups, num_nodes,
                                                num_devices,
                                                num_original_expert)

    file_name = f"{exp_name}_{num_devices}_{num_redundancy_expert}"
    save_matrix_to_json(output_path, file_name, np.array(global_deployment))
    label_names = [
        'default deployment max load', 'balanced load max load',
        'balanced load avg load'
    ]
    new_file_name = f"{exp_name}_{num_devices}_{num_redundancy_expert}.png"
    layer_imblance_polt(y_list, label_names, num_devices, output_path,
                        new_file_name)
