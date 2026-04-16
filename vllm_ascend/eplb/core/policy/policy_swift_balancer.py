# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict

import numpy as np
import torch

from .policy_abstract import DynamicConfig, EplbPolicy


class DynamicTable:
    # workload_table:
    # 3D matrix: [layer, gpus, experts_per_gpu_per_layer] -> value: workload (heat) at the corresponding position
    # Size: number of layers * number of GPUs * number of experts per GPU per layer
    # The element at (i, j, k) represents the workload (heat) of the k-th expert on the j-th GPU in the i-th layer
    # For experts that are not available or collected, the value is set to -1
    workload_table = None

    # placement_table:
    # 3D matrix: [layer, gpus, experts_per_gpu_per_layer] -> value: physical expert ID at the corresponding position
    # Size: number of layers * number of GPUs * number of experts per GPU per layer
    # The element at (i, j, k) represents the physical expert ID of the k-th expert on the j-th GPU in the i-th layer
    # For experts that are not available or collected, the value is set to -1
    placement_table = None


class SwiftBalanceEplb(EplbPolicy):
    def __init__(self, config: DynamicConfig):
        super().__init__(config)
        self.num_layers: int = 0
        self.num_original_experts: int = 0
        self.num_ranks: int = 0
        self.num_experts_per_rank: int = 0
        self.num_nodes: int = 0
        self.is_node_redundant: bool = False
        self.num_max_com: int = 1
        self.imbalance_threshold: float = 1.01
        self.increment = 0.01
        self.swap_threshold: float = 0
        self.max_swap_times: int = 100

    @staticmethod
    def calculate_max_heat_per_layer(workload_table: np.ndarray) -> list[float]:
        max_heat_per_layer: list[float] = []
        for layer_idx in range(workload_table.shape[0]):
            npu_heats_now = np.sum(workload_table[layer_idx], axis=1)
            max_heat_per_layer.append(np.max(npu_heats_now))
        return max_heat_per_layer

    @staticmethod
    def get_original_workload(
        current_expert_table: np.ndarray, expert_workload: np.ndarray, num_original_expert: int
    ) -> np.ndarray:
        """
        Accumulate workload for each routed expert
        """
        layer_num, npu_num, experts_per_npu = expert_workload.shape
        workload_new = np.zeros((layer_num, num_original_expert))
        for layer_idx in range(layer_num):
            workload_dict: dict[int, int] = defaultdict(int)
            placement_layer = current_expert_table[layer_idx].copy()
            workload_layer = expert_workload[layer_idx].copy()
            for npu_idx in range(npu_num):
                for expert_idx in range(experts_per_npu):
                    workload_dict[int(placement_layer[npu_idx][expert_idx])] += workload_layer[npu_idx][expert_idx]
            for expert_idx in range(num_original_expert):
                workload_new[layer_idx][expert_idx] = workload_dict[expert_idx]
        return workload_new

    @staticmethod
    def constraint_expert_local_exchange(old_deployment: np.ndarray, new_deployment: np.ndarray):
        """
        Align the new deployment with the old deployment
        """
        for layer_id in range(len(new_deployment)):
            for card_id in range(len(new_deployment[layer_id])):
                current_list = [int(x) for x in old_deployment[layer_id][card_id]]
                new_list = [int(x) for x in new_deployment[layer_id][card_id]]
                num = len(new_list)

                new_index = [-1] * num
                new_result = [-1] * num
                remaining_elements = []

                for i in range(num):
                    flag = True
                    for j in range(num):
                        if new_list[i] == current_list[j] and new_index[j] == -1:
                            new_index[j] = 0
                            new_result[j] = current_list[j]
                            flag = False
                            break
                    if flag:
                        remaining_elements.append(new_list[i])

                index = 0
                for k in range(num):
                    if new_result[k] == -1:
                        new_result[k] = remaining_elements[index]
                        index += 1

                new_deployment[layer_id][card_id] = new_result

    def calculate_imbalance(self, cur_deployment: np.ndarray, cur_experts_load: np.ndarray) -> list[float]:
        """
        Calculate the imbalance degree of each layer.
        """
        per_layer_imbalance = []
        num_per_expert = np.zeros_like(cur_experts_load)
        for layer_id, layer in enumerate(cur_deployment):
            for rank in layer:
                for expert_id in rank:
                    num_per_expert[layer_id][expert_id] += 1

        for layer_id, layer in enumerate(cur_deployment):
            cur_layer_max_load = 0
            total_load = 0
            for rank in layer:
                rank_load = 0
                for expert_id in rank:
                    update_workload = cur_experts_load[layer_id][expert_id] / num_per_expert[layer_id][expert_id]

                    rank_load += update_workload
                    total_load += update_workload
                if cur_layer_max_load < rank_load:
                    cur_layer_max_load = rank_load

            avg_load = total_load / self.num_ranks
            if abs(avg_load) < 1e-9:
                cur_layer_imbalance = 1.0
            else:
                cur_layer_imbalance = cur_layer_max_load / avg_load
            per_layer_imbalance.append(cur_layer_imbalance)

        return per_layer_imbalance

    def statistics_expert_distribution(
        self, single_layer_deployment: np.ndarray
    ) -> tuple[list[list[int]], np.ndarray, set[int], int]:
        """
        Statistics on the distribution of redundant experts and logical
        experts under the current deployment

        Parameters:
            single_layer_deployment: [num_ranks, num_experts_per_rank]
                the expert deployment status on each rank
        Returns:
            redundant_expert_pos: the positions of redundant experts
                on each rank
            expert_from_rank: [num_logical_experts] the rank where
                each logical expert resides
            num_redundant_experts: the number of redundant experts
        """

        num_ranks = len(single_layer_deployment)
        redundant_expert_pos: list[list[int]] = [[] for _ in range(num_ranks)]
        num_redundant_experts = 0
        expert_from_rank = np.zeros(self.num_original_experts, dtype=np.int64)
        existing_experts = set()

        for index in range(self.num_experts_per_rank):
            for rank_id in range(num_ranks):
                expert_id = int(single_layer_deployment[rank_id][index])
                if expert_id not in existing_experts:
                    existing_experts.add(expert_id)
                    expert_from_rank[expert_id] = rank_id
                else:
                    redundant_expert_pos[rank_id].append(index)
                    num_redundant_experts += 1

        return (redundant_expert_pos, expert_from_rank, existing_experts, num_redundant_experts)

    def compute_redundant_assignments(
        self,
        initial_weights: list,
        num_redundant_experts: int,
        num_ranks: int,
    ) -> tuple[list[tuple[int, float]], np.ndarray]:
        """
        Reconfigure redundant experts based on current expert workload and
        count the new expert workload after redundancy reconfiguration

        Parameters:
            initial_weights: [num_logical_experts] expert load statistics
            num_redundant_experts: Number of redundant experts
            num_ranks: Number of cards in the current node
        Returns:
            redundant_expert_list:[(expert_id, expert_load)]
                the redundantly generated experts
            update_weight: [num_logical_experts] expert workload status after
                reconfiguring redundant experts
        """

        current_weights = initial_weights.copy()
        redundant_assignments = np.zeros(self.num_original_experts, dtype=np.int64)

        for i in range(num_redundant_experts):
            sorted_indices = np.argsort([w for _, w in current_weights], kind="stable")[::-1]
            for index in sorted_indices:
                target_expert = current_weights[index]
                expert_id, original_weight = target_expert

                current_redundancy = redundant_assignments[expert_id] + 1
                if current_redundancy < num_ranks:
                    new_avg_weight = original_weight * (current_redundancy + 1) / (current_redundancy + 2)

                    redundant_assignments[expert_id] += 1
                    current_weights[index] = (expert_id, new_avg_weight)
                    break

        update_weight = np.zeros(self.num_original_experts, dtype=np.float32)
        for expert_id, expert_weight in current_weights:
            update_weight[expert_id] = expert_weight

        redundant_expert_list = []
        if num_redundant_experts > 0:
            for expert_id in range(self.num_original_experts):
                for _ in range(redundant_assignments[expert_id]):
                    redundant_expert_list.append((expert_id, float(update_weight[expert_id])))

            redundant_expert_list.sort(key=lambda x: x[1], reverse=True)

        return redundant_expert_list, update_weight

    def fill_in_undeployed_ranks(
        self,
        initial_weights: list,
        rank_assignments: np.ndarray,
        undeployed_ranks: list[int],
        redundant_expert_pos: list[list[int]],
        num_com_between_rank: np.ndarray,
        rev_expert_per_rank: defaultdict[int, set[int]],
        expert_from_rank: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        reselect and assign redundant experts to the ranks
        with remaining redundant slots
        """

        update_workload, num_per_existing_expert = self.recomputing_initial_weight(initial_weights, rank_assignments)

        for rank_idx in undeployed_ranks:
            for pos in redundant_expert_pos[rank_idx]:
                sorted_expert_idx = np.argsort(update_workload, kind="stable")[::-1]

                for expert_id in sorted_expert_idx:
                    send_rank = expert_from_rank[expert_id]
                    if expert_id in rank_assignments[rank_idx]:
                        continue

                    if np.isclose(update_workload[expert_id], -1):
                        raise ValueError(f"Expert ID {expert_id} is not in the node")

                    rank_assignments[rank_idx][pos] = expert_id
                    num_com_between_rank[send_rank][rank_idx] += 1
                    rev_expert_per_rank[rank_idx].add(expert_id)

                    num_cur_expert = num_per_existing_expert[expert_id]
                    update_workload[expert_id] *= num_cur_expert / (num_cur_expert + 1)
                    num_per_existing_expert[expert_id] += 1
                    break

        rank_loads = np.zeros(len(rank_assignments), dtype=np.float32)
        for rank_id, rank in enumerate(rank_assignments):
            for index, expert_id in enumerate(rank):
                rank_loads[rank_id] += update_workload[expert_id]

        return update_workload, rank_loads

    def non_redundant_expert_information(
        self,
        origin_deployment: np.ndarray,
        updated_weights: np.ndarray,
        redundant_expert_pos: list[list[int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Statistics on the status of logical experts on each rank

        Parameters:
            origin_deployment: [num_ranks, num_experts_per_rank]
                the expert deployment status on each rank
            updated_weights: [num_logical_experts] expert workload status after
                reconfiguring redundant experts
            redundant_expert_pos: the positions of redundant experts
                on each rank
        Returns:
            rank_assignments: [num_ranks, num_experts_per_rank]
                the deployment of logical experts on each rank
            rank_loads: [num_ranks] The workload of
                logical experts on each rank
        """

        num_cur_deployment_ranks = origin_deployment.shape[0]
        rank_assignments = np.full((num_cur_deployment_ranks, self.num_experts_per_rank), fill_value=-1, dtype=np.int64)
        rank_loads = np.zeros(num_cur_deployment_ranks, dtype=np.float32)

        for rank_id, rank in enumerate(origin_deployment):
            for index, expert_id in enumerate(rank):
                if index in redundant_expert_pos[rank_id]:
                    continue
                rank_assignments[rank_id][index] = expert_id
                rank_loads[rank_id] += updated_weights[expert_id]

        return rank_assignments, rank_loads

    def recomputing_initial_weight(
        self, initial_weights: list, rank_assignments: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the load of the logic expert again based
        on the current deployment
        """

        num_per_existing_expert = np.zeros(self.num_original_experts, dtype=np.int64)

        for rank in rank_assignments:
            for expert_id in rank:
                if expert_id != -1:
                    num_per_existing_expert[expert_id] += 1

        update_workload = np.full(self.num_original_experts, fill_value=-1, dtype=np.float32)

        for expert_id, weight in initial_weights:
            num_cur_expert = num_per_existing_expert[expert_id]
            assert num_cur_expert != 0
            update_workload[expert_id] = weight / num_cur_expert

        return update_workload, num_per_existing_expert

    def distribute_redundant_experts(
        self,
        rank_assignments: np.ndarray,
        rank_loads: np.ndarray,
        redundant_expert_list: list[tuple[int, float]],
        expert_from_rank: np.ndarray,
        redundant_expert_pos: list[list[int]],
    ) -> tuple[np.ndarray, defaultdict[int, set[int]], list[int]]:
        """
        Assign redundant experts to ranks

        Parameters:
            rank_assignments: [num_ranks, num_experts_per_rank]
                the deployment of logical experts on each rank
            rank_loads: [num_ranks] The workload of
                logical experts on each rank
            redundant_expert_list:[(expert_id, expert_load)]
                the redundantly generated experts
            expert_from_rank: [num_logical_experts] the rank where
                each logical expert resides
            redundant_expert_pos: the positions of redundant experts
                on each rank
        Returns:
            num_com_between_rank:[num_ranks, num_ranks] the communication
                status between ranks
            rev_expert_per_rank:the experts assigned to each rank
                after reconfiguring redundancy
            undeployed_ranks: record the ranks that have not been
                assigned redundant experts in redundancy positions
        """

        num_ranks = len(rank_assignments)
        rev_expert_per_rank = defaultdict(set)
        num_com_between_rank = np.zeros((num_ranks, num_ranks), dtype=np.int64)

        for expert_id, weight in redundant_expert_list:
            candidate = -1
            send_rank = expert_from_rank[expert_id]
            for rank_id in range(num_ranks):
                if len(redundant_expert_pos[rank_id]) == 0:
                    continue
                if expert_id in rank_assignments[rank_id]:
                    continue
                if num_com_between_rank[send_rank][rank_id] >= self.num_max_com:
                    continue
                if candidate == -1 or rank_loads[rank_id] < rank_loads[candidate]:
                    candidate = rank_id

            if candidate != -1:
                pos = redundant_expert_pos[candidate].pop()
                rank_assignments[candidate][pos] = expert_id
                rank_loads[candidate] += weight

                num_com_between_rank[send_rank][candidate] += 1
                rev_expert_per_rank[candidate].add(expert_id)

        undeployed_ranks = []
        for rank_id in range(num_ranks):
            if len(redundant_expert_pos[rank_id]) > 0:
                undeployed_ranks.append(rank_id)

        return num_com_between_rank, rev_expert_per_rank, undeployed_ranks

    def redundancy_again(
        self, cur_layer_workload: np.ndarray, cur_layer_deployment: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, defaultdict[int, set[int]]]:
        """
        Calculate the status of a single node after redundant expert
        configuration.

        Parameters:
            cur_layer_workload: [num_logical_experts] expert load statistics
            cur_layer_deployment: [num_ranks, num_experts_per_rank]
                the expert deployment status on each rank
        Returns:
            rank_assignments: [num_ranks, num_experts_per_rank]
                the expert deployment status on each rank after
            reconfiguring redundant experts
            rank_loads: [num_ranks] the workload status of each rank after
                reconfiguring redundant experts
            updated_weights: [num_logical_experts] expert workload status after
                reconfiguring redundant experts
            num_com_between_rank:[num_ranks, num_ranks] the communication
                status between ranks
            rev_expert_per_rank:the experts assigned to each rank
                after reconfiguring redundancy
        """
        num_ranks = cur_layer_deployment.shape[0]

        (redundant_expert_pos, expert_from_rank, existing_experts, num_redundant_experts) = (
            self.statistics_expert_distribution(cur_layer_deployment)
        )

        initial_weights = []
        for expert_id in existing_experts:
            initial_weights.append((expert_id, cur_layer_workload[expert_id]))

        redundant_expert_list, updated_weights = self.compute_redundant_assignments(
            initial_weights, num_redundant_experts, num_ranks
        )

        rank_assignments, rank_loads = self.non_redundant_expert_information(
            cur_layer_deployment, updated_weights, redundant_expert_pos
        )

        num_com_between_rank, rev_expert_per_rank, undeployed_ranks = self.distribute_redundant_experts(
            rank_assignments, rank_loads, redundant_expert_list, expert_from_rank, redundant_expert_pos
        )

        if len(undeployed_ranks) > 0:
            updated_weights, rank_loads = self.fill_in_undeployed_ranks(
                initial_weights,
                rank_assignments,
                undeployed_ranks,
                redundant_expert_pos,
                num_com_between_rank,
                rev_expert_per_rank,
                expert_from_rank,
            )

        return (rank_assignments, rank_loads, updated_weights, num_com_between_rank, rev_expert_per_rank)

    def redundant_expert_deployment(
        self, cur_layer_workload: np.ndarray, cur_layer_deployment: np.ndarray
    ) -> tuple[
        list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[defaultdict[int, set[int]]]
    ]:
        """
        Calculate the status of each node after reconfiguring redundant experts;
        treat non-intra-node redundancy as a single node, store the result of
        each node in a list, and return it.
        """

        all_node_assignments = []
        all_node_loads = []
        updated_weights = []
        num_com_between_rank = []
        rev_expert_per_rank = []

        if self.is_node_redundant:
            num_ranks_per_node = self.config.num_die_per_host

            for node_id in range(self.num_nodes):
                cur_node_deployment = cur_layer_deployment[
                    node_id * num_ranks_per_node : (node_id + 1) * num_ranks_per_node
                ]

                (
                    cur_node_rank_assignments,
                    cur_node_rank_loads,
                    cur_node_updated_weights,
                    cur_node_num_com_between_rank,
                    cur_node_rev_expert_per_rank,
                ) = self.redundancy_again(cur_layer_workload, cur_node_deployment)

                all_node_assignments.append(cur_node_rank_assignments)
                all_node_loads.append(cur_node_rank_loads)
                updated_weights.append(cur_node_updated_weights)
                num_com_between_rank.append(cur_node_num_com_between_rank)
                rev_expert_per_rank.append(cur_node_rev_expert_per_rank)

        else:
            (
                cur_rank_assignments,
                cur_rank_loads,
                cur_updated_weights,
                cur_num_com_between_rank,
                cur_rev_expert_per_rank,
            ) = self.redundancy_again(cur_layer_workload, cur_layer_deployment)

            all_node_assignments.append(cur_rank_assignments)
            all_node_loads.append(cur_rank_loads)
            updated_weights.append(cur_updated_weights)
            num_com_between_rank.append(cur_num_com_between_rank)
            rev_expert_per_rank.append(cur_rev_expert_per_rank)

        return (all_node_assignments, all_node_loads, updated_weights, num_com_between_rank, rev_expert_per_rank)

    def swap_experts_between_ranks(
        self,
        max_rank_deployment_set: set[int],
        swap_rank_deployment_set: set[int],
        max_rank_rev_expert: set[int],
        swap_rank_rev_expert: set[int],
        workload: np.ndarray,
        max_rank_load: float,
        swap_rank_load: float,
    ) -> tuple[int, int, float]:
        """
        Find the optimal experts for workload reduction
        after exchange between two ranks
        """

        max_rank_expert = -1
        swap_rank_expert = -1
        max_weight = max_rank_load

        for cur_expert_id in max_rank_deployment_set:
            if cur_expert_id in swap_rank_deployment_set or cur_expert_id in max_rank_rev_expert:
                continue
            cur_weight = float(workload[cur_expert_id])

            for next_expert_id in swap_rank_deployment_set:
                if next_expert_id in max_rank_deployment_set or next_expert_id in swap_rank_rev_expert:
                    continue
                next_weight = float(workload[next_expert_id])

                cur_load_after_swap = max_rank_load - cur_weight + next_weight
                next_load_after_swap = swap_rank_load - next_weight + cur_weight
                max_load_after_swap = max(cur_load_after_swap, next_load_after_swap)
                if max_load_after_swap < max_weight:
                    max_weight = max_load_after_swap
                    max_rank_expert = cur_expert_id
                    swap_rank_expert = next_expert_id

        return max_rank_expert, swap_rank_expert, max_weight

    def expert_exchange_between_ranks(
        self,
        rank_assignments: np.ndarray,
        rank_loads: np.ndarray,
        num_com_between_rank: np.ndarray,
        rev_expert_per_rank: defaultdict[int, set[int]],
        updated_weights: np.ndarray,
    ) -> tuple[list[list[int]], float]:
        """
        Perform inter-rank expert exchange within a single node

        Parameters:
            rank_assignments: [num_ranks, num_experts_per_rank] the expert
                deployment status on each rank after reconfiguring
                redundant experts
            rank_loads: [num_ranks] the workload status of each rank after
                reconfiguring redundant experts
            num_com_between_rank:[num_ranks, num_ranks] the communication
                status between ranks
            rev_expert_per_rank:the experts assigned to each rank
                after reconfiguring redundancy
            updated_weights: [num_logical_experts] expert workload status after
                reconfiguring redundant experts
        Returns:
            ranks_deployment_after_swap: [num_ranks, num_experts_per_rank]
                the deployment status of experts on each rank after the exchange
            max_rank_load: the workload of the hottest rank
        """

        rank_deploy_sets = []
        for rank_id in range(len(rank_assignments)):
            rank_deploy_sets.append(set(rank_assignments[rank_id]))

        max_swap_times = self.max_swap_times
        max_rank_load = 0.0
        exchange = True
        while max_swap_times > 0:
            max_swap_times -= 1
            sorted_rank_idx = np.argsort(rank_loads, kind="stable")
            max_load_rank_id = int(sorted_rank_idx[-1])
            max_rank_load = float(rank_loads[max_load_rank_id])

            if not exchange:
                break
            exchange = False
            for swap_rank_id in sorted_rank_idx[:-1]:
                if (
                    num_com_between_rank[swap_rank_id][max_load_rank_id] < self.num_max_com
                    and num_com_between_rank[max_load_rank_id][swap_rank_id] < self.num_max_com
                ):
                    swap_rank_load = rank_loads[swap_rank_id]

                    max_rank_expert, swap_rank_expert, max_weight = self.swap_experts_between_ranks(
                        rank_deploy_sets[max_load_rank_id],
                        rank_deploy_sets[swap_rank_id],
                        rev_expert_per_rank[max_load_rank_id],
                        rev_expert_per_rank[swap_rank_id],
                        updated_weights,
                        max_rank_load,
                        swap_rank_load,
                    )

                    if max_rank_load - max_weight < self.swap_threshold or max_rank_expert == -1:
                        continue

                    rank_deploy_sets[max_load_rank_id].remove(max_rank_expert)
                    rank_deploy_sets[swap_rank_id].remove(swap_rank_expert)
                    rank_deploy_sets[max_load_rank_id].add(swap_rank_expert)
                    rank_deploy_sets[swap_rank_id].add(max_rank_expert)

                    rank_loads[max_load_rank_id] += updated_weights[swap_rank_expert] - updated_weights[max_rank_expert]
                    rank_loads[swap_rank_id] += updated_weights[max_rank_expert] - updated_weights[swap_rank_expert]

                    rev_expert_per_rank[max_load_rank_id].add(swap_rank_expert)
                    rev_expert_per_rank[swap_rank_id].add(max_rank_expert)

                    num_com_between_rank[swap_rank_id][max_load_rank_id] += 1
                    num_com_between_rank[max_load_rank_id][swap_rank_id] += 1

                    exchange = True
                    break

        ranks_deployment_after_swap = [list(s) for s in rank_deploy_sets]

        return ranks_deployment_after_swap, max_rank_load

    def exchange_experts(
        self,
        all_node_assignments: list[np.ndarray],
        all_node_loads: list[np.ndarray],
        num_com_between_rank: list[np.ndarray],
        rev_expert_per_rank: list[defaultdict[int, set[int]]],
        updated_weights: list[np.ndarray],
    ) -> tuple[np.ndarray, float]:
        """
        For each node after redundancy, perform inter-rank expert
        exchange within the node to reduce the workload of the hottest rank
        """

        max_workload = 0.0
        after_swap_ranks_deployment = []
        for idx in range(len(all_node_assignments)):
            cur_node_deployment, cur_node_max_workload = self.expert_exchange_between_ranks(
                all_node_assignments[idx],
                all_node_loads[idx],
                num_com_between_rank[idx],
                rev_expert_per_rank[idx],
                updated_weights[idx],
            )
            after_swap_ranks_deployment += cur_node_deployment
            if cur_node_max_workload > max_workload:
                max_workload = cur_node_max_workload

        new_deployment = np.array(after_swap_ranks_deployment)

        return new_deployment, max_workload

    def rebalance_experts(
        self, current_expert_table: torch.Tensor, expert_workload: torch.Tensor, is_node_redundant: bool = False
    ) -> tuple[int, np.ndarray, list[list[list[int]]]]:
        """
        Rebalance experts based on workload and deployment strategy.

        Parameters:
            current_expert_table: [num_layer, num_ranks, num_experts_per_rank],
            tensor of current expert assignment.
            expert_workload: [num_layer, num_ranks, num_experts_per_rank],
            tensor of workload for each expert.
            is_node_redundant: Whether to enable intra-node redundancy.

        Returns:
            change: Scalar flag for whether deployment changed.
            per_layer_priority: List of priority for each layer.
            new_deployment: [num_layer, num_ranks, num_experts_per_rank],
            list of adjusted expert deployment result.
        """

        info = DynamicTable()
        info.workload_table = expert_workload.numpy()
        info.placement_table = current_expert_table.numpy()
        assert info.workload_table is not None and info.placement_table is not None
        self.is_node_redundant = is_node_redundant
        self.num_layers, self.num_ranks, self.num_experts_per_rank = info.placement_table.shape
        self.num_nodes = self.num_ranks // self.config.num_die_per_host
        expert_ids, counts = np.unique(info.placement_table[0], return_counts=True)
        self.num_original_experts = len(expert_ids)
        layer_workloads = self.get_original_workload(
            info.placement_table, info.workload_table, self.num_original_experts
        )

        per_layer_total_load = layer_workloads[0].sum()

        ave_workload = per_layer_total_load / self.num_ranks
        self.swap_threshold = ave_workload * self.increment

        layer_initial_imbalance = self.calculate_imbalance(info.placement_table, layer_workloads)

        new_deployment = info.placement_table.copy()

        max_heat_per_layer_before = self.calculate_max_heat_per_layer(info.workload_table)
        npu_heat_all_origin = sum(max_heat_per_layer_before)
        max_heat_per_layer_after = []

        for layer in range(self.num_layers):
            cur_layer_deployment = info.placement_table[layer]
            cur_layer_workload = layer_workloads[layer]

            if layer_initial_imbalance[layer] < self.imbalance_threshold:
                max_heat_per_layer_after.append(max_heat_per_layer_before[layer])
                continue

            (all_node_assignments, all_node_loads, updated_weights, num_com_between_rank, rev_experts_per_rank) = (
                self.redundant_expert_deployment(cur_layer_workload, cur_layer_deployment)
            )

            (new_layer_deployment, new_max_workload) = self.exchange_experts(
                all_node_assignments, all_node_loads, num_com_between_rank, rev_experts_per_rank, updated_weights
            )

            after_swap_imbalance = new_max_workload / ave_workload

            if after_swap_imbalance < layer_initial_imbalance[layer]:
                new_deployment[layer] = new_layer_deployment

            max_heat_per_layer_after.append(new_max_workload)

        self.constraint_expert_local_exchange(info.placement_table, new_deployment)

        layer_changed_ratio = []
        for layer_idx in range(self.num_layers):
            if max_heat_per_layer_before[layer_idx] > 0:
                layer_changed_ratio.append(max_heat_per_layer_after[layer_idx] / max_heat_per_layer_before[layer_idx])
            else:
                layer_changed_ratio.append(1.0)

        per_layer_priority = np.argsort(layer_changed_ratio)
        npu_heat_all_after = sum(max_heat_per_layer_after)

        change = 0
        if npu_heat_all_after < 0.95 * npu_heat_all_origin:
            change = 1

        return change, per_layer_priority, new_deployment.tolist()
