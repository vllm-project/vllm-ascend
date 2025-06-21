# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
from collections import defaultdict
import numpy as np
from abc import abstractmethod


class DynamicConfig:
    placement_policy = None

    max_transferred_expert_per_layer = 100
    # 一台机器上，一层最多搬运多少专家

    ep_worldsize = 64  # 整个集群上所有的专家分布在多少个die上
    num_die_per_host = 8  # 每台机器上有几个die


class EplbPolicy:
    def __init__(self, config: DynamicConfig):
        self.config = config

    @abstractmethod
    def rebalance_experts(self, current_expert_table, expert_workload):
        """
        传入weight并返回相关限制条件下的专家复制和放置
        INPUT:
        current_expert_table: [layerId, rankId, expert_num_i]
        expert_workload = expert_table[layer0][rankId][expert_num_i]

        RETURNED: (res, expert_table)
        res:
        1 -- table_changed
        0 -- not_changed

        expert_table: [layerId, rankId, expert_num_i]
        expert_num_i --- [0, MaxExpertPerRank]
        expertID = expert_table[layer0][rankId][expert_num_i]
        array_values:
        [0, 1, 2, 3, 248]
        [4, 5, 6, 7, 254]
        [8, 9, 10, 11, 71]
        ...
        [252, 253, 254, 255, 0]
        """
        pass

class DynamicTable:
    # workload_table:
    # 三维矩阵，[layer, gpus, experts_per_gpu_per_layer] -> value: 所在位置的热度
    # 大小为 层数 * 卡数 * 每层每卡的专家数量
    # 里面i, j, k的元素代表 第 i 层 第 j 张卡第 k 个专家的热度
    # 对于收集不到的专家，填为 -1
    workload_table = None

    # placement_table:
    # 三维矩阵，[layer, gpus, experts_per_gpu_per_layer] -> value: 所在位置的物理专家id
    # 大小为 层数 * 卡数 * 每层每卡的专家数量
    # 里面i, j, k的元素代表 第 i 层 第 j 张卡第 k 个专家的物理id
    # 对于收集不到的专家，填为 -1
    placement_table = None


class DynamicEplbV2(EplbPolicy):

    def __init__(self, config: DynamicConfig):
        super().__init__(config)

    @staticmethod
    def add_redundant(current_expert_table, expert_workload, num_original_expert):
        layer_num, npu_num, experts_per_npu = expert_workload.shape
        workload_new = np.zeros((layer_num, num_original_expert))
        for layer_idx in range(layer_num):
            workload_dict = defaultdict(int)
            placement_layer = current_expert_table[layer_idx].copy()
            workload_layer = expert_workload[layer_idx].copy()
            for npu_idx in range(npu_num):
                for expert_idx in range(experts_per_npu):
                    workload_dict[placement_layer[npu_idx][expert_idx]] += workload_layer[npu_idx][expert_idx]
            for expert_idx in range(num_original_expert):
                workload_new[layer_idx][expert_idx] = workload_dict[expert_idx]
        return workload_new

    @staticmethod
    # 热点专家拆分为冗余专家
    def original_compute_balanced_pack_redundancy(origin_weights, card_num, num_redundancy_expert):
        # Step 1: Sort the items by weight in descending order (we are sorting by weight now)
        # Sort based on the second element (the second value of each tuple)
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

        # Step 2: Calculate the number of items per box
        expert_num = route_expert_num + num_redundancy_expert
        items_per_box = expert_num // card_num  # Number of items per box
        remaining_items = expert_num % card_num  # Number of items per box

        # Step 3: Initialize card_num boxes with empty lists to store item IDs
        boxes = [[] for _ in range(card_num)]
        boxes_weights = [[] for _ in range(card_num)]
        box_weights = [0] * card_num  # To store the total weight of each box
        box_counts = [0] * card_num  # To store the number of items in each box
        index = 0
        for i in range(route_expert_num):
            redundancy_num = len(route_expert_redundancy[i])
            for _ in range(redundancy_num):
                cur_weight = 0
                for item, weight in origin_weights:
                    if item == i:
                        cur_weight = weight

                boxes[index].append(i)
                boxes_weights[index].append(cur_weight)
                box_weights[index] += cur_weight
                box_counts[index] += 1
                index += 1

        sorted_indices = np.argsort([t[1] for t in origin_weights], kind='stable')[::-1]
        origin_weights = [origin_weights[idx] for idx in sorted_indices]
        # Step 4: Distribute items into boxes based on weight
        for item_id, weight in origin_weights:
            # Find the box with the least items but not full
            min_box_index = -1
            for i in range(card_num):
                # Only choose boxes that still have space (box_counts[i] < items_per_box)
                if box_counts[i] < items_per_box or (box_counts[i] == items_per_box and remaining_items > 0):
                    if min_box_index == -1 or box_weights[i] < box_weights[min_box_index]:
                        min_box_index = i

            # Place the item (id) into the selected box
            boxes[min_box_index].append(item_id)
            boxes_weights[min_box_index].append(weight)
            box_weights[min_box_index] += weight
            box_counts[min_box_index] += 1

            # If there's an imbalance in the remaining items, reduce the "remaining_items" counter
            if box_counts[min_box_index] == (items_per_box + 1) and remaining_items > 0:
                remaining_items -= 1

        # Step 5: Output each box's contents and total weight
        result = []
        for i in range(card_num):
            result.append({
                "box_index": i + 1,
                "items": boxes[i],  # List of item IDs in the box
                "weight": boxes_weights[i],
                "total_weight": box_weights[i],  # Total weight in this box
                "item_count": box_counts[i]  # Number of items in the box
            })

        return result, boxes

    # 热点专家拆分为冗余专家
    @staticmethod
    def compute_balanced_pack_redundancy(origin_weights, card_num, num_redundancy_expert):
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

        expert_num = route_expert_num + num_redundancy_expert
        if card_num == 0:
            raise RuntimeError("card_num can not be 0.")
        items_per_box = expert_num // card_num
        remaining_items = expert_num % card_num

        boxes = [[] for _ in range(card_num)]
        boxes_weights = [[] for _ in range(card_num)]
        box_weights = [0] * card_num
        box_counts = [0] * card_num

        all_weights = np.zeros((expert_num,), dtype='object')
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

        result = []
        for i in range(card_num):
            result.append({
                "box_index": i + 1,
                "items": boxes[i],
                "weight": boxes_weights[i],
                "total_weight": box_weights[i],
                "item_count": box_counts[i]
            })

        return result, boxes

    # 无冗余专家方案
    @staticmethod
    def compute_balanced_pack(origin_weights, card_num):
        sorted_indices = np.argsort([t[1] for t in origin_weights])[::-1]
        weights = origin_weights[sorted_indices]
        expert_num = len(weights)
        if card_num == 0:
            raise RuntimeError("card_num can not be 0.")
        items_per_box = expert_num // card_num
        remaining_items = expert_num % card_num

        boxes = [[] for _ in range(card_num)]
        boxes_weights = [[] for _ in range(card_num)]
        box_weights = [0] * card_num
        box_counts = [0] * card_num

        for item_id, weight in weights:
            min_box_index = -1
            for i in range(card_num):
                if box_counts[i] < items_per_box or (box_counts[i] == items_per_box and remaining_items > 0):
                    if min_box_index == -1 or box_weights[i] < box_weights[min_box_index]:
                        min_box_index = i

            boxes[min_box_index].append(item_id)
            boxes_weights[min_box_index].append(weight)
            box_weights[min_box_index] += weight
            box_counts[min_box_index] += 1

            if box_counts[min_box_index] == (items_per_box + 1) and remaining_items > 0:
                remaining_items -= 1

        result = []
        for i in range(card_num):
            result.append({
                "box_index": i + 1,
                "items": boxes[i],
                "weight": boxes_weights[i],
                "total_weight": box_weights[i],
                "item_count": box_counts[i]
            })

        return result, boxes

    @staticmethod
    def get_redundant_num(npu_num, counts):
        redundant_num_each_npu = np.sum(counts - 1)
        return redundant_num_each_npu

    @staticmethod
    def calculate_max_heat_per_layer(workload_table, layer_num):
        max_heat_per_layer = []
        for layer_idx in range(layer_num):
            npu_heats_now = np.sum(workload_table[layer_idx], axis=1)
            max_heat_per_layer.append(np.max(npu_heats_now))
        return max_heat_per_layer

    @staticmethod
    def calculate_initial_imbalance(global_deployment, new_layer_workloads):

        device_num = global_deployment.shape[1]
        layer_imbalance = []
        expert_num = np.zeros_like(new_layer_workloads)
        # 基于部署做更新负载
        for layer_id, layer in enumerate(global_deployment):
            for device in layer:
                for expert_id in device:
                    expert_num[layer_id][expert_id] += 1

        for layer_id, layer in enumerate(global_deployment):
            cur_layer_max_workload = 0
            total_workload = 0
            for box in layer:
                box_workload = 0
                for expert_id in box:
                    update_workload = new_layer_workloads[layer_id][expert_id] / expert_num[layer_id][expert_id]
                    box_workload += update_workload
                    total_workload += update_workload
                if cur_layer_max_workload < box_workload:
                    cur_layer_max_workload = box_workload

            cur_layer_imbalance = cur_layer_max_workload / (total_workload / device_num)
            layer_imbalance.append(cur_layer_imbalance)

        return layer_imbalance

    def rebalance_experts(self, current_expert_table, expert_workload):

        info = DynamicTable()
        info.workload_table = np.array(expert_workload)
        info.placement_table = np.array(current_expert_table)
        layer_num, num_npus, experts_per_npu = info.workload_table.shape
        expert_ids, counts = np.unique(info.placement_table[0], return_counts=True)
        num_redundancy_expert = self.get_redundant_num(num_npus, counts)
        num_original_expert = len(expert_ids)
        layer_workloads = self.add_redundant(info.placement_table, info.workload_table, num_original_expert)
        max_heat_per_layer_before = self.calculate_max_heat_per_layer(info.workload_table, layer_num)
        npu_heat_all_origin = sum(max_heat_per_layer_before)

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
        # 遍历获得每一层的放置策略，考虑计算均衡
        max_heat_per_layer_after = np.zeros([layer_num])
        for layer in range(layer_num):
            # 获取当前层专家ID和对应负载，负载需要进行正则化处理, 每个卡加一个冗余专家
            weights = np.zeros((expert_num,), dtype='object')
            for expert_id, workload_weight in enumerate(layer_workloads[layer]):
                weights[expert_id] = (expert_id, workload_weight)

            # 获取每一层全局计算均衡的放置策略
            result, layer_deployment = self.original_compute_balanced_pack_redundancy(
                weights, num_npus, num_redundancy_expert
            )
            global_deployment[layer] = layer_deployment
            max_heat_per_layer_after[layer] = max(result, key=lambda x: x['total_weight'])['total_weight']

        # 获取层优先级
        layer_changed_ratio = []
        for layer_idx in range(layer_num):
            layer_changed_ratio.append(max_heat_per_layer_after[layer_idx] / max_heat_per_layer_before[layer_idx])

        per_layer_priority = np.argsort(layer_changed_ratio)
        npu_heat_all_after = sum(max_heat_per_layer_after)

        change = 0

        return change, per_layer_priority, np.array(global_deployment).tolist()

    @staticmethod
    def compute_redundant_assignments(base_experts, num_redundant_experts, num_experts):
        """
        计算每个基础专家需要分配的冗余专家，并动态调整专家权重
        返回冗余分配表和更新后的基础专家权重列表
        """
        redundant_assignments = [[] for _ in range(num_experts)]
        current_weights = base_experts.copy()

        for i in range(num_redundant_experts):
            # 按权重降序排序（使用稳定排序保持相同权重的顺序）
            sorted_indices = np.argsort([w for _, w in current_weights], kind='stable')[::-1]
            sorted_weights = [current_weights[i] for i in sorted_indices]

            # 选择当前权重最高的专家
            target_expert = sorted_weights[0]
            expert_id, original_weight = target_expert

            # 计算添加冗余后的新平均权重
            current_redundancy = len(redundant_assignments[expert_id])
            new_avg_weight = original_weight * (current_redundancy + 1) / (current_redundancy + 2)

            # 更新分配表和权重列表
            redundant_assignments[expert_id].append(num_experts + i)
            current_weights[sorted_indices[0]] = (expert_id, new_avg_weight)

        sorted_indices = np.argsort([w for _, w in current_weights], kind='stable')[::-1]
        sorted_weights = [current_weights[i] for i in sorted_indices]

        return redundant_assignments, sorted_weights

    @staticmethod
    def prepare_expert_list(base_experts, redundant_assignments, num_redundant_experts):
        """
        生产冗余专家的完整列表，并按权重降序排序
        """
        redundant_expert_list = np.empty(num_redundant_experts, dtype=object)

        # 填充冗余专家（使用对应基础专家的当前权重）
        index = 0
        num_experts = len(redundant_assignments)
        for expert_id in range(num_experts):
            for _ in redundant_assignments[expert_id]:
                redundant_expert_list[index] = (expert_id, next(w for eid, w in base_experts if eid == expert_id))
                index += 1

        # 按权重降序排序
        sorted_indices = np.argsort([w for _, w in redundant_expert_list], kind='stable')[::-1]
        return [redundant_expert_list[i] for i in sorted_indices]

    @staticmethod
    def non_redundant_expert_information(origin_deployment, updated_weights, num_radundant_experts):

        device_num = len(origin_deployment)

        device_assignments = [[] for _ in range(device_num)]
        device_weights = [[] for _ in range(device_num)]
        device_loads = [0] * device_num
        device_counts = [0] * device_num
        if num_radundant_experts:
            start_id = 1
        else:
            start_id = 0

        # 统计卡上非冗余专家信息
        for box_id, box in enumerate(origin_deployment):
            for i in range(start_id, len(box)):
                device_assignments[box_id].append(box[i])
                cur_weight = next(weight for expert_id, weight in updated_weights if expert_id == box[i])
                device_weights[box_id].append(cur_weight)
                device_loads[box_id] += cur_weight
                device_counts[box_id] += 1

        return device_assignments, device_weights, device_loads, device_counts

    @staticmethod
    def distribute_redun_experts(device_assignments, device_weights, device_loads, device_counts, redundant_expert_list,
                                 items_per_device, expert_form_device, num_experts):

        num_devices = len(device_assignments)
        com_between_devices = [{} for _ in range(num_devices)]

        for expert_id, weight in redundant_expert_list:
            # 寻找最优设备（满足容量限制且负载最小）
            candidate = -1
            for dev_id in range(num_devices):
                # 保证设备内节点不同
                if expert_id in device_assignments[dev_id]:
                    continue
                # 检查容量限制
                if device_counts[dev_id] < items_per_device:
                    # 选择负载最小的候选设备
                    if candidate == -1 or device_loads[dev_id] < device_loads[candidate]:
                        candidate = dev_id
            if candidate != -1:
                # 分配专家到选定的设备
                device_assignments[candidate].insert(0, expert_id)
                device_weights[candidate].insert(0, weight)
                device_loads[candidate] += weight
                device_counts[candidate] += 1

                communication_box_index = expert_form_device[expert_id]
                com_between_devices[candidate][communication_box_index] = expert_id
        # 极端情况下存在冗余专家没装箱 导致箱子有空位 随机填入专家 待优化
        for dev_id in range(num_devices):
            # 检查容量限制
            if device_counts[dev_id] < items_per_device:
                # 遍历合适的专家
                for expert_id in range(num_experts):
                    if expert_id not in device_assignments[dev_id]:
                        # 找到对应权重
                        weight = 0
                        for i in range(num_devices):
                            for j in range(len(device_assignments[i])):
                                if expert_id == device_assignments[i][j]:
                                    weight = device_weights[i][j]
                        # 和该专家相关的卡权重发生变化 待修改
                        device_assignments[dev_id].insert(0, expert_id)
                        device_weights[dev_id].insert(0, weight)
                        device_loads[dev_id] += weight
                        device_counts[dev_id] += 1

                        communication_box_index = expert_form_device[expert_id]
                        com_between_devices[dev_id][communication_box_index] = expert_id
                        break
        #todo 重新生成权重

        return device_assignments, device_weights, device_loads, device_counts, com_between_devices

    @staticmethod
    def redundancy_again(self, origin_weights, num_redundant_experts, origin_deployment, expert_form_device, num_node,
                         is_node_redundant):

        # 每张卡上专家数量
        expert_num_per_device = origin_deployment.shape[1]

        num_experts = len(origin_weights)
        if is_node_redundant:
            num_experts = num_experts * num_node

        # 根据新负载重新计算冗余专家
        redundant_assignments, updated_weights = self.compute_redundant_assignments(origin_weights,
                                                                                    num_redundant_experts,
                                                                                    num_experts)

        # 收集冗余专家信息并排序
        redundant_expert_list = self.prepare_expert_list(updated_weights, redundant_assignments, num_redundant_experts)

        # 收集重新计算冗余后卡上非冗余专家信息
        device_assignments, device_weights, device_loads, device_counts = self.non_redundant_expert_information(
            origin_deployment, updated_weights, num_redundant_experts)

        # 新计算的冗余专家进行分配
        device_assignments, device_weights, device_loads, device_counts, com_between_devices = self.distribute_redun_experts(
            device_assignments,
            device_weights,
            device_loads,
            device_counts,
            redundant_expert_list,
            expert_num_per_device,
            expert_form_device,
            num_experts)


        return device_assignments, device_weights, device_loads, device_counts, com_between_devices

    @staticmethod
    def generate_allocation_report(device_assignments, device_weights, device_loads, device_counts):
        """
        生成最终分配报告并计算最大负载
        """
        report = []
        max_load = 0.0

        for dev_id in range(len(device_assignments)):
            current_load = device_loads[dev_id]
            max_load = max(max_load, current_load)

            report.append({
                "device_id": dev_id + 1,
                "assigned_experts": device_assignments[dev_id],
                "expert_weights": device_weights[dev_id],
                "total_load": current_load,
                "expert_count": device_counts[dev_id]
            })

        return report, max_load

    @staticmethod
    def exchange_expert(cur_exchange_index, next_exchange_index, cur_device_id, next_device_id, cur_layer_result,
                        com_between_devices):

        cur_device_deployment = cur_layer_result[cur_device_id]['assigned_experts']
        next_device_deployment = cur_layer_result[next_device_id]['assigned_experts']

        cur_device_weight = cur_layer_result[cur_device_id]['expert_weights']
        next_device_weight = cur_layer_result[next_device_id]['expert_weights']

        # 两张卡上对应的两个专家进行交换
        cur_expert_id = cur_device_deployment[cur_exchange_index]
        next_expert_id = next_device_deployment[next_exchange_index]
        cur_device_deployment[cur_exchange_index] = next_expert_id
        next_device_deployment[next_exchange_index] = cur_expert_id

        cur_expert_weight = cur_device_weight[cur_exchange_index]
        next_expert_weight = next_device_weight[next_exchange_index]
        cur_device_weight[cur_exchange_index] = next_expert_weight
        next_device_weight[next_exchange_index] = cur_expert_weight

        cur_layer_result[cur_device_id]['total_load'] += next_expert_weight - cur_expert_weight
        cur_layer_result[next_device_id]['total_load'] += cur_expert_weight - next_expert_weight

        # 记录这两卡进行了通信
        com_between_devices[cur_device_id][next_device_id] = next_expert_id
        com_between_devices[next_device_id][cur_device_id] = cur_expert_id

    @staticmethod
    # 分层调整冗余专家
    def redundant_expert_deployment(self, layer_workloads, original_deployment, expert_form_device, node_num,
                                    is_node_redundant):
        device_num, per_device_expert_num = original_deployment.shape
        route_expert_num = layer_workloads.shape[0]
        redundancy_expert_num = per_device_expert_num * device_num - route_expert_num
        per_node_device_num = device_num // node_num
        per_node_route_expert_num = per_node_device_num * (per_device_expert_num - 1)
        per_node_redun_expert_num = redundancy_expert_num // node_num

        weights = np.zeros((route_expert_num,), dtype='object')
        for expert_id, workload_weight in enumerate(layer_workloads):
            weights[expert_id] = (expert_id, int(workload_weight))

        if is_node_redundant:

            device_assignments = []
            device_weights = []
            device_loads = []
            device_counts = []
            com_between_devices = []

            for node_id in range(node_num):
                cur_node_weights = weights[
                                   node_id * per_node_route_expert_num: (node_id + 1) * per_node_route_expert_num]
                cur_original_deployment = original_deployment[
                                          node_id * per_node_device_num: (node_id + 1) * per_node_device_num]

                cur_device_assignments, cur_device_weights, cur_device_loads, cur_device_counts, cur_com_between_devices = self.redundancy_again(
                    self,
                    cur_node_weights,
                    per_node_redun_expert_num,
                    cur_original_deployment,
                    expert_form_device,
                    node_num,
                    is_node_redundant)
                device_assignments += cur_device_assignments
                device_weights += cur_device_weights
                device_loads += cur_device_loads
                device_counts += cur_device_counts
                com_between_devices += cur_com_between_devices

        else:
            device_assignments, device_weights, device_loads, device_counts, com_between_devices = self.redundancy_again(
                self,
                weights,
                redundancy_expert_num,
                original_deployment,
                expert_form_device,
                node_num,
                is_node_redundant)
        # 生成报告
        report, max_load = self.generate_allocation_report(device_assignments, device_weights, device_loads,
                                                           device_counts)

        return report, max_load, com_between_devices

    @staticmethod
    def two_device_exchange_experts(cur_device_result, exchange_device_result, cur_exchanged_expert_id,
                                    next_exchanged_expert_id, ave_workload, increment, num_redundancy_expert):

        cur_device_weight = cur_device_result['expert_weights']
        next_device_weight = exchange_device_result['expert_weights']

        cur_device_expert_id = cur_device_result['assigned_experts']
        next_device_expert_id = exchange_device_result['assigned_experts']

        cur_device_total_weight = int(cur_device_result['total_load'])
        next_device_total_weight = int(exchange_device_result['total_load'])
        max_weight = max(cur_device_total_weight, next_device_total_weight)

        cur_exchange_index = -1
        next_exchange_index = -1

        redun = False
        if num_redundancy_expert != 0:
            redun = True

        for index, weight in enumerate(cur_device_weight):
            for next_index, next_weight in enumerate(next_device_weight):
                # 跳过冗余专家
                if (index == 0 or next_index == 0) and redun :
                    continue
                    # 交换专家限制卡内专家不同
                change_flag = True
                if cur_device_expert_id[index] in next_device_expert_id or next_device_expert_id[next_index] in cur_device_expert_id:
                    change_flag = False
                # 选择的专家不能是参与过交换的
                if (cur_device_expert_id[index] not in cur_exchanged_expert_id) and (
                        next_device_expert_id[next_index] not in next_exchanged_expert_id) and change_flag:
                    cur_total_weight_after_exchange = cur_device_total_weight - weight + next_weight
                    next_total_weight_after_exchange = next_device_total_weight - next_weight + weight
                    exchange_max_weight = max(cur_total_weight_after_exchange, next_total_weight_after_exchange)
                    if exchange_max_weight < max_weight and (max_weight - exchange_max_weight) >= (
                            ave_workload * increment):
                        max_weight = exchange_max_weight
                        cur_exchange_index = index
                        next_exchange_index = next_index

        return cur_exchange_index, next_exchange_index

    @staticmethod
    def expert_exchange_between_devices(self, ave_workload, increment, cur_layer_result, com_between_devices, num_redundancy_expert,
                                        node_idx=0,
                                        per_node_device_num=0, is_node_redundant=False):

        if is_node_redundant:
            # 拿出当前节点内设备的信息
            cur_devices_result = cur_layer_result[node_idx * per_node_device_num:(node_idx + 1) * per_node_device_num]
        else:
            # 拿取所有设备信息
            cur_devices_result = cur_layer_result

        devices_total_weight = []
        for device in cur_devices_result:
            devices_total_weight.append((int(device['total_load']), device['device_id'] - 1))

        # 当迭代次数超过100或负载最大的设备无法进行调整时退出
        exchange_frequency = 100
        while exchange_frequency > 0:
            exchange_frequency -= 1

            # 根据负载从小到大排序
            devices_total_weight.sort(key=lambda x: x[0])
            # 负载最大的设备id
            max_weight_device_id = devices_total_weight[-1][1]

            exchange = False
            # 按照负载从小到大依次取卡
            for index in range(0, len(devices_total_weight) - 1):
                min_weight_device_id = devices_total_weight[index][1]
                # 两个节点没有进行过通信
                if min_weight_device_id not in com_between_devices[max_weight_device_id]:
                    # 找到设备中交换过的专家id，（除了冗余之外通信过的id）
                    set_cur_com_expert_id = set(com_between_devices[max_weight_device_id].values())
                    set_next_com_expert_id = set(com_between_devices[min_weight_device_id].values())
                    if num_redundancy_expert != 0:
                        set_cur_device_expert_id = set(cur_layer_result[max_weight_device_id]['assigned_experts'][1:])
                        set_next_device_expert_id = set(cur_layer_result[min_weight_device_id]['assigned_experts'][1:])
                    else:
                        set_cur_device_expert_id = set(cur_layer_result[max_weight_device_id]['assigned_experts'])
                        set_next_device_expert_id = set(cur_layer_result[min_weight_device_id]['assigned_experts'])

                    cur_exchanged_expert_id = set_cur_com_expert_id & set_cur_device_expert_id
                    next_exchanged_expert_id = set_next_com_expert_id & set_next_device_expert_id

                    cur_exchange_index, next_exchange_index = self.two_device_exchange_experts(
                        cur_layer_result[max_weight_device_id],
                        cur_layer_result[min_weight_device_id],
                        cur_exchanged_expert_id,
                        next_exchanged_expert_id,
                        ave_workload,
                        increment,
                        num_redundancy_expert)

                    # 有符合条件的专家进行交换
                    if cur_exchange_index != -1:
                        self.exchange_expert(cur_exchange_index,
                                             next_exchange_index,
                                             max_weight_device_id,
                                             min_weight_device_id,
                                             cur_layer_result,
                                             com_between_devices)

                        devices_total_weight[-1] = (
                            cur_layer_result[max_weight_device_id]['total_load'], max_weight_device_id)
                        devices_total_weight[index] = (
                            cur_layer_result[min_weight_device_id]['total_load'], min_weight_device_id)
                        exchange = True
                        break

            if not exchange:
                break

    @staticmethod
    def exchange_experts(self, layer_result, layer_com_between_devices, num_nodes, device_num, is_node_redundant,
                         ave_workload, increment, num_redundancy_expert):

        global_deployment = []

        if is_node_redundant:
            per_node_device_num = device_num // num_nodes
            for node_idx in range(num_nodes):
                self.expert_exchange_between_devices(self, ave_workload, increment, layer_result,
                                                     layer_com_between_devices, num_redundancy_expert,
                                                     node_idx, per_node_device_num, is_node_redundant)
        else:
            self.expert_exchange_between_devices(self, ave_workload, increment, layer_result, layer_com_between_devices, num_redundancy_expert)

        max_workload = 0
        for box in layer_result:
            global_deployment.append(box['assigned_experts'])
            if max_workload < box['total_load']:
                max_workload = box['total_load']

        global_deployment = np.array(global_deployment)

        return global_deployment, max_workload

    @staticmethod
    def count_elements(self, lst):
        count = 0
        for item in lst:
            if isinstance(item, list):
                count += self.count_elements(self, item)
            else:
                count += 1
        return count

    def rebalance_experts(self, current_expert_table, expert_workload):

        info = DynamicTable()
        info.workload_table = np.array(expert_workload)
        info.placement_table = np.array(current_expert_table)
        layer_num, num_npus, experts_per_npu = info.workload_table.shape
        expert_ids, counts = np.unique(info.placement_table[0], return_counts=True)
        num_redundancy_expert = self.get_redundant_num(num_npus, counts)
        num_original_expert = len(expert_ids)
        layer_workloads = self.add_redundant(info.placement_table, info.workload_table, num_original_expert)
        max_heat_per_layer_before = self.calculate_max_heat_per_layer(info.workload_table, layer_num)
        npu_heat_all_origin = sum(max_heat_per_layer_before)

        # 计算负载均衡，部署冗余专家
        num_node = num_npus / 8
        layer_num = layer_workloads.shape[0]
        expert_num = layer_workloads.shape[1]
        expert_from_device = np.zeros((layer_num, num_original_expert))
        # 校验专家数量、卡数量、冗余专家数量不能超过卡数量
        if num_original_expert != expert_num:
            raise ValueError(f"原始专家数量 {num_original_expert} 必须等于 expert_num {expert_num}")

        if num_npus <= 0:
            raise ValueError("NPUs 数量必须大于 0")

        if num_npus < num_redundancy_expert:
            raise ValueError(f"NPUs 数量 {num_npus} 必须大于或等于冗余专家数量 {num_redundancy_expert}")

        # 每个卡部署的专家数量 一个冗余专家
        global_deployment = [[[] for _ in range(num_npus)] for _ in range(layer_num)]
        # 统计更换数据集后的初始58层不均衡度
        layer_initial_imbalance = self.calculate_initial_imbalance(current_expert_table, layer_workloads)
        # 遍历获得每一层的放置策略，考虑计算均衡
        max_heat_per_layer_after = np.zeros([layer_num])
        sum_num = 0
        for layer in range(layer_num):
            # 不均衡度小于特定阈值不调整
            if layer_initial_imbalance[layer] < 1.1:
                global_deployment[layer] = current_expert_table[layer]
                continue

            ave_workload = np.sum(layer_workloads[layer]) / num_npus
            for device_id, device in enumerate(current_expert_table[layer]):
                for index, expert_id in enumerate(device):
                    if index != 0:
                        expert_from_device[layer][expert_id] = device_id

            # 调整冗余专家
            result, max_workload, com_between_devices = self.redundant_expert_deployment(self, layer_workloads[layer],
                                                                                         current_expert_table[layer],
                                                                                         expert_from_device[layer],
                                                                                         num_node, False)
            # 交换专家
            global_deployment[layer], new_max_workload = self.exchange_experts(self, result, com_between_devices,
                                                                               num_node, num_npus, False, ave_workload,
                                                                               0.05, num_redundancy_expert)

            for device_id in range(num_npus):
                com_between_devices[device_id] = {int(key): int(value) for key, value in
                                                  com_between_devices[device_id].items()}
                sum_num += self.count_elements(self, com_between_devices[device_id])

            max_heat_per_layer_after[layer] = max(result, key=lambda x: x['total_load'])['total_load']

        # 获取层优先级
        layer_changed_ratio = []
        for layer_idx in range(layer_num):
            layer_changed_ratio.append(max_heat_per_layer_after[layer_idx] / max_heat_per_layer_before[layer_idx])

        per_layer_priority = np.argsort(layer_changed_ratio)
        npu_heat_all_after = sum(max_heat_per_layer_after)

        change = 0
        if npu_heat_all_after < 0.95 * npu_heat_all_origin:
            change = 1

        return change, per_layer_priority, np.array(global_deployment).tolist()




