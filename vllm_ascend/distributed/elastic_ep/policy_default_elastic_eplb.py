# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# Todo: Once https://github.com/vllm-project/vllm/pull/24069 is merged in vllm. Remove this policy.
import numpy as np

from vllm_ascend.eplb.core.policy.policy_abstract import EplbPolicy


class DefaultElasticEplb(EplbPolicy):
    def __init__(self):
        self._new_ep_size: int | None = None

    def set_new_ep_size(self, new_ep_size: int):
        self._new_ep_size = new_ep_size

    @staticmethod
    def add_redundant(current_expert_table, expert_workload, num_original_expert):
        placement = current_expert_table
        workload = expert_workload
        L, _, _ = workload.shape

        flat_placement = placement.reshape(L, -1)
        flat_workload = workload.reshape(L, -1)
        workload_new = np.zeros((L, num_original_expert))
        for layer_idx in range(L):
            np.add.at(workload_new[layer_idx], flat_placement[layer_idx], flat_workload[layer_idx])
        return workload_new

    @staticmethod
    def original_compute_balanced_pack_redundancy(origin_weights, card_num, num_redundancy_expert):
        # Step 1: Sort by weight descending; create redundant copies of hot experts
        route_expert_num = len(origin_weights)
        route_expert_redundancy: list[list[int]] = [[] for _ in range(route_expert_num)]
        for i in range(num_redundancy_expert):
            sorted_indices = np.argsort([t[1] for t in origin_weights], kind="stable")[::-1]
            weights = [origin_weights[idx] for idx in sorted_indices]
            index = 0
            while (len(route_expert_redundancy[weights[index][0]])) == card_num - 1:
                index += 1
                assert index < route_expert_num
            tmp_raw_weight = weights[index][1] * (len(route_expert_redundancy[weights[index][0]]) + 1)
            route_expert_redundancy[weights[index][0]].append(route_expert_num + i)
            avg_weight = tmp_raw_weight / (len(route_expert_redundancy[weights[index][0]]) + 1)
            weights[index] = (weights[index][0], avg_weight)
            origin_weights = weights

        # Step 2-3: Calculate box capacity; initialize empty boxes
        expert_num = route_expert_num + num_redundancy_expert
        items_per_box = expert_num // card_num
        remaining_items = expert_num % card_num

        boxes: list[list[int]] = [[] for _ in range(card_num)]
        boxes_weights: list[list[float]] = [[] for _ in range(card_num)]
        box_weights = [0] * card_num
        box_counts = [0] * card_num
        index = 0
        for i in range(route_expert_num):
            redundancy_num = len(route_expert_redundancy[i])
            for _ in range(redundancy_num):
                expert_weight = 0
                for item, weight in origin_weights:
                    if item == i:
                        expert_weight = weight
                        break

                boxes[index].append(i)
                boxes_weights[index].append(expert_weight)
                box_weights[index] += expert_weight
                box_counts[index] += 1
                index += 1
                index = index % card_num

        sorted_indices = np.argsort([t[1] for t in origin_weights], kind="stable")[::-1]
        origin_weights = [origin_weights[idx] for idx in sorted_indices]
        # Step 4: Distribute items into boxes, preferring least-loaded
        for item_id, weight in origin_weights:
            min_box_index = -1
            for i in range(card_num):
                if item_id in boxes[i]:
                    continue
                if box_counts[i] < items_per_box or (box_counts[i] == items_per_box and remaining_items > 0):
                    if min_box_index == -1 or box_weights[i] < box_weights[min_box_index]:
                        min_box_index = i

            if min_box_index == -1:
                for i in range(card_num):
                    if box_counts[i] < items_per_box or (box_counts[i] == items_per_box and remaining_items > 0):
                        min_box_index = i
                        break

            boxes[min_box_index].append(item_id)
            boxes_weights[min_box_index].append(weight)
            box_weights[min_box_index] += weight
            box_counts[min_box_index] += 1

            if box_counts[min_box_index] == (items_per_box + 1) and remaining_items > 0:
                remaining_items -= 1

        # Step 5: Replace forced duplicates with candidates that minimize weight delta
        for i in range(card_num):
            arr = np.asarray(boxes[i])
            unique, inv, cnt = np.unique(arr, return_inverse=True, return_counts=True)
            mask = cnt > 1
            for item_id, counts in zip(unique[mask], cnt[mask]):
                for _ in range(counts - 1):
                    cur_position = boxes[i].index(item_id)
                    cur_weight: float = boxes_weights[i][cur_position]

                    def score(t, cw=cur_weight):
                        before = len(route_expert_redundancy[t[0]]) + 1
                        after = len(route_expert_redundancy[t[0]]) + 2
                        adjusted = t[1] * before / after
                        return abs(adjusted - cw)

                    sorted_indices = np.argsort([score(t) for t in origin_weights], kind="stable")
                    candidate_weights = [origin_weights[idx] for idx in sorted_indices]

                    index = 0
                    while index < len(candidate_weights):
                        candidate_id = candidate_weights[index][0]
                        if (
                            len(route_expert_redundancy[candidate_id]) < card_num - 1
                            and candidate_id != item_id
                            and candidate_id not in boxes[i]
                        ):
                            break
                        index += 1

                    assert index < len(candidate_weights)
                    boxes[i][cur_position] = candidate_weights[index][0]
                    tmp_raw_weight = candidate_weights[index][1] * (len(route_expert_redundancy[candidate_weights[index][0]]) + 1)
                    route_expert_redundancy[candidate_weights[index][0]].append(0)
                    avg_weight = tmp_raw_weight / (len(route_expert_redundancy[candidate_weights[index][0]]) + 1)
                    boxes_weights[i][cur_position] = avg_weight
                    candidate_weights[index] = (candidate_weights[index][0], avg_weight)

                    tmp_raw_weight = cur_weight * (len(route_expert_redundancy[item_id]) + 1)
                    avg_weight = tmp_raw_weight / len(route_expert_redundancy[item_id])
                    route_expert_redundancy[item_id].pop()

                    for idx, (eid, ew) in enumerate(candidate_weights):
                        if eid == item_id:
                            candidate_weights[idx] = (eid, avg_weight)
                    origin_weights = candidate_weights

        # Step 6: Build result for each box
        result = []
        for i in range(card_num):
            result.append(
                {
                    "box_index": i + 1,
                    "items": boxes[i],
                    "weight": boxes_weights[i],
                    "total_weight": box_weights[i],
                    "item_count": box_counts[i],
                }
            )

        return result, boxes

    @staticmethod
    def constraint_expert_local_exchange(current_expert_table, global_deployment):
        for layer_id in range(len(global_deployment)):
            num_cards = min(len(current_expert_table[layer_id]), len(global_deployment[layer_id]))

            for card_id in range(num_cards):
                cur_expert_ids = np.array(current_expert_table[layer_id][card_id])
                new_expert_ids = np.array(global_deployment[layer_id][card_id])

                assert len(cur_expert_ids) == len(new_expert_ids), (
                    "Number of experts must match between current and new deployment."
                )

                # Experts appearing in both old and new placement (first occurrence only)
                _, first_occurrence_idx = np.unique(cur_expert_ids, return_index=True)
                mask_experts_not_move = np.zeros(len(cur_expert_ids), dtype=bool)
                mask_experts_not_move[first_occurrence_idx] = True
                mask_experts_not_move &= np.isin(cur_expert_ids, new_expert_ids)
                slot_experts_not_move = cur_expert_ids[mask_experts_not_move]

                # Experts in new placement that come from other NPUs
                mask_new_experts = ~np.isin(new_expert_ids, slot_experts_not_move)
                new_experts = new_expert_ids[mask_new_experts]

                final_expert_list = new_expert_ids.copy()
                final_expert_list[mask_experts_not_move] = slot_experts_not_move
                final_expert_list[~mask_experts_not_move] = new_experts
                global_deployment[layer_id][card_id] = final_expert_list.tolist()

        return global_deployment

    def rebalance_experts(self, current_expert_table, expert_workload):
        placement = np.array(current_expert_table)
        workload = np.array(expert_workload)
        layer_num, num_npus, experts_per_npu = workload.shape
        expert_ids, counts = np.unique(placement[0], return_counts=True)
        num_original_expert = len(expert_ids)
        assert self._new_ep_size is not None and self._new_ep_size != num_npus
        num_npus = self._new_ep_size
        num_redundancy_expert = experts_per_npu * self._new_ep_size - num_original_expert
        layer_workloads = self.add_redundant(placement, workload, num_original_expert)

        # Validate parameters
        layer_num = layer_workloads.shape[0]
        expert_num = layer_workloads.shape[1]
        if num_original_expert != expert_num:
            raise ValueError(
                f"the number of original experts {num_original_expert} must be equal to expert_num {expert_num}"
            )
        if num_npus <= 0:
            raise ValueError("the number of NPUs must be greater than 0")
        if experts_per_npu > expert_num:
            raise ValueError(
                f"the number of experts per NPU {experts_per_npu} can't be greater than expert_num {expert_num}"
            )
        if num_npus * experts_per_npu < num_original_expert:
            raise ValueError(
                f"num_npus {num_npus} * experts_per_npu {experts_per_npu} "
                f"can't be less than num_original_expert {num_original_expert}"
            )

        # Balance experts across NPUs using global redundant placement strategy
        global_deployment: list[list[list[int]]] = [[[] for _ in range(num_npus)] for _ in range(layer_num)]
        for layer in range(layer_num):
            weights = np.zeros((expert_num,), dtype="object")
            for expert_id, workload_weight in enumerate(layer_workloads[layer]):
                weights[expert_id] = (expert_id, workload_weight)

            _, layer_deployment = self.original_compute_balanced_pack_redundancy(
                weights, num_npus, num_redundancy_expert
            )
            global_deployment[layer] = layer_deployment

        new_global_deployment = self.constraint_expert_local_exchange(current_expert_table, global_deployment)

        self._new_ep_size = None

        return np.array(new_global_deployment).tolist()
