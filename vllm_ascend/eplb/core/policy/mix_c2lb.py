# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
from collections import defaultdict

def save_matrix_to_csv(output_path, file_name, matrix):
    """
    保存矩阵到 CSV 文件或 Excel 文件（用于处理三维矩阵的多个 sheet）
    :param output_path: 输出文件的路径
    :param file_name: 输出文件的名字
    :param matrix: 矩阵
    """
    if matrix.ndim == 2:
        # 二维矩阵保存为普通 CSV 文件
        df = pd.DataFrame(matrix)
        file_name = f"{output_path}/{file_name}.csv"
        df.to_csv(file_name, index=False)
    elif matrix.ndim == 3:
        # 三维矩阵保存到 Excel 文件的不同 sheet 中
        file_name = f"{output_path}/{file_name}.xlsx"
        with pd.ExcelWriter(file_name) as writer:
            for i in range(matrix.shape[0]):
                slice_2d = matrix[i]
                df = pd.DataFrame(slice_2d)
                df.to_excel(writer, sheet_name=f'slice_{i}', index=False)
    else:
        print(f"矩阵的维度 {matrix.ndim} 不支持，仅支持二维和三维矩阵。")


def save_matrix_to_json(output_path, file_name, deployment):
    num_layers = len(deployment)
    num_cards = len(deployment[0])

    data = {"moe_layer_count": num_layers}
    layer_list = []
    for i in range(num_layers):
        layer = {"layer_id": i, "device_count": num_cards}
        device_list = []
        for j in range(num_cards):
            # 将 1*4 的行矩阵转换为列表
            device = {"device_id": j, "device_expert": list(deployment[i][j])}
            device_list.append(device)
        layer["device_list"] = device_list
        layer_list.append(layer)
    data["layer_list"] = layer_list

    file_name = f"{output_path}/{file_name}.json"
    # 保存为 JSON 文件
    try:
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"写入文件 {deployment} 时出错: {e}")

def plt_imbalance(c2lb_imbalance, original_imbalance, output_path):
    # 使用列表的索引作为 X 轴数据
    x_values = list(range(len(c2lb_imbalance)))
    # 绘制折线图
    c2lb_mean = np.mean(c2lb_imbalance, axis=0)
    plt.plot(x_values, c2lb_imbalance, marker='o', label=f'c2lb imbalance (mean: {c2lb_mean:.2f})')
    original_mean = np.mean(original_imbalance, axis=0)
    plt.plot(x_values, original_imbalance, marker='o', label=f'original imbalance (mean: {original_mean:.2f})')

    # 添加标题和标签
    plt.title("npu imbalance")
    plt.xlabel("layer id")
    plt.ylabel("imbalance")
    # 显示图例
    plt.legend()
    plt.ylim(bottom=0)

    # 保存图像
    output_path_name = f"{output_path}/test_layer_imbalance.png"
    plt.savefig(output_path_name, dpi=1000, bbox_inches='tight')

def calculate_initial_imbalance(global_deployment, new_layer_workloads):

    num_npus = global_deployment.shape[1]
    layer_imbalance = []
    num_expert = np.zeros_like(new_layer_workloads)
    for layer_id, layer in enumerate(global_deployment):
        for device in layer:
            for expert_id in device:
                num_expert[layer_id][expert_id] += 1

    for layer_id, layer in enumerate(global_deployment):
        cur_layer_max_workload = 0
        total_workload = 0
        for box in layer:
            box_workload = 0
            for expert_id in box:
                update_workload = new_layer_workloads[layer_id][expert_id] / num_expert[layer_id][expert_id]
                box_workload += update_workload
                total_workload += update_workload
            if cur_layer_max_workload < box_workload:
                cur_layer_max_workload = box_workload

        cur_layer_imbalance = cur_layer_max_workload / (total_workload / num_npus)
        layer_imbalance.append(cur_layer_imbalance)

    return layer_imbalance


def compute_balanced_pack_redundancy(origin_weights, card_num, num_redundancy_expert, shared_workload):
    route_expert_num = len(origin_weights)
    route_expert_redundancy = [[] for _ in range(route_expert_num)]
    for i in range(num_redundancy_expert):
        sorted_indices = np.argsort([t[1] for t in origin_weights], kind='stable')[::-1]
        weights = [origin_weights[idx] for idx in sorted_indices]
        tmp_raw_weight = weights[0][1] * (len(route_expert_redundancy[weights[0][0]]) + 1)
        route_expert_redundancy[weights[0][0]].append(route_expert_num + i)
        avg_weight = tmp_raw_weight / (len(route_expert_redundancy[weights[0][0]]) + 1)
        weights[0] = (weights[0][0], avg_weight)
        origin_weights = weights

    expert_num = route_expert_num + card_num
    shared_expert = card_num - num_redundancy_expert
    shared_expert_workload = shared_workload / shared_expert
    items_per_box = expert_num // card_num
    remaining_items = expert_num % card_num

    boxes = [[] for _ in range(card_num)]
    boxes_weights = [[] for _ in range(card_num)]
    box_weights = [0] * card_num
    box_counts = [0] * card_num

    step = card_num // shared_expert
    for i in range(shared_expert):
        i = i * step
        # print("i", i)
        boxes[i].append(256)
        boxes_weights[i].append(shared_expert_workload)
        box_weights[i] += shared_expert_workload
        box_counts[i] += 1

    all_weights = np.zeros((expert_num-shared_expert,), dtype='object')
    all_weights[: route_expert_num] = origin_weights

    index = route_expert_num
    for i in range(route_expert_num):
        redundancy_num = len(route_expert_redundancy[i])
        for _ in range(redundancy_num):
            for item, weight in origin_weights:
                if item == i:
                    all_weights[index] = (item, weight)
                    index += 1

    sorted_indices = np.argsort([t[1] for t in all_weights], kind='stable')[::-1]
    all_weights = [all_weights[idx] for idx in sorted_indices]
    for item_id, weight in all_weights:
        min_box_index = -1
        for i in range(card_num):
            if box_counts[i] < items_per_box or (box_counts[i] == items_per_box and remaining_items > 0):
                if min_box_index == -1 or box_weights[i] < box_weights[min_box_index]:
                    if item_id not in boxes[i]:
                        min_box_index = i

        boxes[min_box_index].append(item_id)
        boxes_weights[min_box_index].append(weight)
        box_weights[min_box_index] += weight
        box_counts[min_box_index] += 1

        if box_counts[min_box_index] == (items_per_box + 1) and remaining_items > 0:
            remaining_items -= 1

    # boxes = [sorted(box) for box in boxes]
    result = []
    max_weight = 0
    for i in range(card_num):
        if box_weights[i] >= max_weight:
            max_weight = box_weights[i]
        result.append({
            "box_index": i + 1,
            "items": boxes[i],
            "weight": boxes_weights[i],
            "total_weight": box_weights[i],
            "item_count": box_counts[i]
        })

    return result, boxes, max_weight

# 冗余专家部署
def lb_and_intra_layer_affinity_redundancy_deploy(
        layer_workloads,
        num_redundancy_expert,
        output_path,
        file_name,
        num_npus,
        num_original_expert):
    """
    :param layer_workloads[layer_num, expert_num] 58*256
    :return: optimized layer_deployment: [layer_num, card_num, card_expert_num] 58*64*4
    """
    # 计算负载均衡，部署冗余专家
    layer_num = layer_workloads.shape[0]
    expert_num = layer_workloads.shape[1]
    # 校验专家数量、卡数量、冗余专家数量不能超过卡数量
    if num_original_expert != expert_num:
        raise ValueError(f"原始专家数量 {num_original_expert} 必须等于 expert_num {expert_num}")

    if num_npus <= 0:
        raise ValueError("NPUs 数量必须大于 0")

    if num_npus < num_redundancy_expert:
        raise ValueError(f"NPUs 数量 {num_npus} 必须大于或等于冗余专家数量 {num_redundancy_expert}")

    # 每个卡部署的专家数量 一个冗余专家
    global_deployment = [[[] for _ in range(num_npus)] for _ in range(layer_num)]
    max_weights = []
    # 遍历获得每一层的放置策略，考虑计算均衡
    for layer in range(layer_num):
        # 获取当前层专家ID和对应负载，负载需要进行正则化处理, 每个卡加一个冗余专家
        weights = np.zeros((expert_num,), dtype='object')
        for expert_id, workload_weight in enumerate(layer_workloads[layer]):
            weights[expert_id] = (expert_id, workload_weight)

        shared_workload = np.sum(layer_workloads[layer]) / 8
        # print("shared_workload", np.sum(layer_workloads[layer]), shared_workload)
        result, layer_deployment, max_weight = compute_balanced_pack_redundancy(weights, num_npus, num_redundancy_expert, shared_workload)
        for box in result:
            print(
                f"before: Box {box['box_index']}: "
                f"Items = {box['items']}, weight = {box['weight']}, "
                f"Total Weight = {box['total_weight']}, Item Count = {box['item_count']}"
            )

        global_deployment[layer] = layer_deployment
        max_weights.append(max_weight)

    save_matrix_to_json(output_path, file_name, global_deployment)

    ave_workload = np.sum(layer_workloads[0]) / num_npus
    imbalance = max_weights / ave_workload

    return global_deployment, imbalance

def read_many_pt(file_path):
    # 分别表示卡，层，专家id
    layer_workloads = np.zeros((16, 58, 256), dtype=int)
    pt_files = [f for f in os.listdir(file_path) if f.endswith('.pt')]
    # 循环读取每个 .pt 文件
    for i in range(len(pt_files)):
        pt_file = pt_files[i]
        new_file_path = os.path.join(file_path, pt_file)
        split_str = [part for part in pt_file.split('_') if part]
        card_id = int(split_str[0])
        layer_id = int(split_str[1])
        load_tensor = torch.load(new_file_path)
        load_numpy = load_tensor.numpy()
        for expert_id in load_numpy.flat:
            if expert_id >= 0 and expert_id < 256:
                layer_workloads[card_id, layer_id, expert_id] += 1
            else:
                print("error expert_id", expert_id)

    return layer_workloads

def get_original_workload(current_expert_table, expert_workload,
                    num_original_expert):


    layer_num, npu_num, experts_per_npu = expert_workload.shape
    workload_new = np.zeros((layer_num, num_original_expert))
    for layer_idx in range(layer_num):
        workload_dict: dict[int, int] = defaultdict(int)
        placement_layer = current_expert_table[layer_idx].copy()
        workload_layer = expert_workload[layer_idx].copy()
        for npu_idx in range(npu_num):
            for expert_idx in range(experts_per_npu):
                workload_dict[placement_layer[npu_idx][
                    expert_idx]] += workload_layer[npu_idx][expert_idx]
        for expert_idx in range(num_original_expert):
            workload_new[layer_idx][expert_idx] = workload_dict[expert_idx]
    return workload_new


def expert_file_to_tensor(expert_map_path):
    with open(expert_map_path, "r") as f:
        data = json.load(f)
    layers_num = data["moe_layer_count"]
    gpus_num = data["layer_list"][0]["device_count"]
    tensor_data = []
    for layer in data["layer_list"]:
        device_data = []
        for device in layer["device_list"]:
            device_data.append(device["device_expert"])
        tensor_data.append(device_data)
    expert_map_tensor = torch.tensor(tensor_data, dtype=torch.int32)
    return expert_map_tensor, layers_num, gpus_num

if __name__ == '__main__':
    num_original_expert = 256
    output_path = f"D:/2025年学习/专家负载均衡/混合专家负载均衡"

    file_path_name = f"D:/2025年学习/专家负载均衡/混合专家负载均衡/moe_load3/moe_load_baseline/moe_load_0.npy"
    layer_workloads = np.load(file_path_name, allow_pickle=True)
    for i in range(1, 5):
        k = i * 1000 # prefill改为100
        file_path_name = f"D:/2025年学习/专家负载均衡/混合专家负载均衡/moe_load3/moe_load_baseline/moe_load_{k}.npy"
        layer_workload = np.load(file_path_name, allow_pickle=True)
        layer_workloads += layer_workload
        # print("i", i, layer_workloads, layer_workloads.shape)

    expert_map, num_layers, num_npus = expert_file_to_tensor(f"D:/2025年学习/专家负载均衡/混合专家负载均衡/moe_load3/moe_load_baseline/expert_map32_mtp.json")
    # 建议冗余专家数量配置为卡数据的一半,共享专家为另一半
    num_redundancy_expert = num_npus // 2
    placement_table = np.array(expert_map)
    # print("placement_table", placement_table.shape)
    final_workload = get_original_workload(placement_table, layer_workloads, 256)
    # print("final_workload.shape: ", final_workload.shape)

    original_imbalance = calculate_initial_imbalance(placement_table, final_workload)

    file_name = f"global_exployment"
    new_placement_table, imbalance = lb_and_intra_layer_affinity_redundancy_deploy(final_workload, num_redundancy_expert, output_path, file_name, num_npus, num_original_expert)
    # 画图不均衡性
    plt_imbalance(imbalance, original_imbalance, output_path)


