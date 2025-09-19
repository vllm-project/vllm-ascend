# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# Todo: Once https://github.com/vllm-project/vllm/pull/24069 is merged in vllm. Remove this policy.

import torch
import numba
import numpy as np
import logging

from numba import njit
from collections import defaultdict, deque
from .policy_abstract import DynamicConfig, EplbPolicy

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


@njit
def compute_piece_counts(X, P, stage_weights):
    n_stage, N = X.shape
    S = P - N
    pieces = np.ones(N, dtype=np.int32)
    unit = X / pieces  # unit[i, j] = X[i, j] / pieces[j]

    for _ in range(S):
        deltas = np.zeros(N, dtype=np.float32)
        for i in range(n_stage):
            # Find top1 and top2
            idx1 = -1
            idx2 = -1
            val1 = -1.0
            val2 = -1.0
            for j in range(N):
                v = unit[i, j]
                if v > val1:
                    val2 = val1
                    idx2 = idx1
                    val1 = v
                    idx1 = j
                elif v > val2:
                    val2 = v
                    idx2 = j

            origin = unit[i, idx1]
            secv = unit[i, idx2]
            alt = X[i, idx1] / (pieces[idx1] + 1)
            delta = origin - (alt if alt > secv else secv)
            deltas[idx1] += delta * stage_weights[i] if np.any(delta) != 0 else stage_weights[i]

        max_idx = np.argmax(deltas)
        pieces[max_idx] += 1
        for i in range(n_stage):
            unit[i, max_idx] = X[i, max_idx] / pieces[max_idx]

    # Compute max load
    max_load = 0.0
    for j in range(N):
        total = 0.0
        for i in range(n_stage):
            total += unit[i, j]
        if total > max_load:
            max_load = total

    return pieces


@njit
def jsq_placement(X, pieces, M, stage_weights):
    n_stage, N = X.shape
    total_piece = pieces.sum()
    num_per_group = total_piece // M

    # 1. Compute unit_hotness
    unit_hotness = np.empty((n_stage, N), dtype=np.float32)
    for i in range(N):
        for s in range(n_stage):
            unit_hotness[s, i] = X[s, i] / pieces[i]

    # 2. Sort by total hotness
    scores = np.zeros(N, dtype=np.float32)
    for i in range(N):
        for s in range(n_stage):
            scores[i] += unit_hotness[s, i]
    idx = np.argsort(-scores)

    # 3. Initialization
    loads = np.zeros((n_stage, M), dtype=np.float32)
    dev_phy_exp_n = np.zeros(M, dtype=np.int32)
    deployment = -np.ones((M, num_per_group), dtype=np.int32)
    dep_ptr = np.zeros(M, dtype=np.int32)

    # 4. Main loop
    for t in range(N):
        i = idx[t]
        used_device = list()
        for _ in range(pieces[i]):
            # 4.1 Construct w vector
            w = np.empty(n_stage, dtype=np.float32)
            for s in range(n_stage):
                w[s] = unit_hotness[s, i]

            # 4.2 Compute stage-level maximum load
            stage_max = np.empty(n_stage, dtype=np.float32)
            for s in range(n_stage):
                max_val = loads[s, 0]
                for k in range(1, M):
                    if loads[s, k] > max_val:
                        max_val = loads[s, k]
                stage_max[s] = max_val

            # 4.3 Compute denominator
            denom = np.empty(n_stage, dtype=np.float32)
            for s in range(n_stage):
                sum_tmp = 0.0
                for j in range(M):
                    sum_tmp += loads[s, j] + w[s]
                denom[s] = sum_tmp / M + 1e-2

            # 4.4 Find best device j
            best_j = -1
            best_val = 1e30
            for j in range(M):
                if dev_phy_exp_n[j] >= num_per_group:
                    continue
                if j in used_device:
                    continue
                score = 0.0
                for s in range(n_stage):
                    tmp_sj = loads[s, j] + w[s]
                    numer_sj = tmp_sj if tmp_sj > stage_max[s] else stage_max[s]
                    score += stage_weights[s] * (numer_sj / denom[s])
                if score < best_val:
                    best_val = score
                    best_j = j
            if best_j == -1:
                continue

            used_device.append(best_j)

            # 4.5 Update status
            for s in range(n_stage):
                loads[s, best_j] += w[s]
            ptr = dep_ptr[best_j]
            deployment[best_j, ptr] = i
            dep_ptr[best_j] += 1
            dev_phy_exp_n[best_j] += 1

    # Handle remaining -1 values: fill with random elements from range(N) not in current column
    for rank in range(M):
        for col in range(num_per_group):
            if deployment[rank, col] == -1:
                # Get elements already in current column
                current_rank_elements = set(deployment[rank, :])
                # Filter elements from range(N) not in current column
                available = [x for x in range(N) if x not in current_rank_elements]
                # Randomly select an available element to fill
                rand_idx = np.random.randint(0, len(available))
                deployment[rank, col] = available[rand_idx]

    return deployment


@njit
def slice_values(X, pieces):
    total_len = 0
    for i in range(X.shape[0]):
        total_len += pieces[i]
    result = np.empty(total_len, dtype=np.float32)
    idx = 0
    for i in range(X.shape[0]):
        val = X[i] / pieces[i]
        for _ in range(pieces[i]):
            result[idx] = val
            idx += 1
    return result


@njit
def group_based_adaptive_bloating_kernel(
    X, P, M, simulated_pieces, simulated_deployment, stage_weights
):
    n_stage, N = X.shape
    num_group = P // M

    X_all = np.zeros(N, dtype=np.float32)
    for i in range(n_stage):
        for j in range(N):
            X_all[j] += X[i, j]

    sort_idx = np.argsort(np.negative(X_all))
    X_sorted = X[:, sort_idx]

    unit_load = np.empty(N, dtype=np.float32)
    for j in range(N):
        unit_load[j] = X_all[j] / simulated_pieces[j]

    flat_deployment = simulated_deployment.reshape(-1)
    simulated_load = np.zeros(M, dtype=np.float32)
    for i in range(flat_deployment.shape[0]):
        simulated_load[i // (flat_deployment.shape[0] // M)] += unit_load[flat_deployment[i]]

    slice_vals = slice_values(X_all, simulated_pieces)
    sorted_slices = np.sort(slice_vals)[::-1]
    simulated_slopes = (sorted_slices[:-M + 1] - sorted_slices[M - 1:]) / M

    cumulative_slices_used = np.zeros(N, dtype=np.int32)
    acc = 0
    for i in range(N):
        acc += simulated_pieces[sort_idx[i]]
        cumulative_slices_used[i] = acc

    group_boundary_indices = np.empty(num_group, dtype=np.int32)
    for i in range(1, num_group + 1):
        for j in range(N):
            if cumulative_slices_used[j] >= i * M:
                group_boundary_indices[i - 1] = j
                break

    slices_used_per_group = np.empty(num_group, dtype=np.int32)
    slices_used_per_group[0] = group_boundary_indices[0]
    for i in range(1, num_group):
        slices_used_per_group[i] = group_boundary_indices[i] - group_boundary_indices[i - 1]
    slices_used_per_group = M - slices_used_per_group

    loads = np.zeros(M, dtype=np.float32)
    pieces = np.zeros(N, dtype=np.int32)
    num_remain_slice = P - N
    current_idx = 0

    for g in range(num_group):
        window = X_sorted[:, current_idx: current_idx + 2 * M]
        slices_g = slices_used_per_group[g]
        low = max(0, current_idx + M - N)
        high = min(num_remain_slice, M - 1)

        while high - low > 1:
            mid = (high + low) // 2
            keep = M - mid
            current_group = window[:, :keep]
            current_pieces = compute_piece_counts(current_group, M, stage_weights)
            current_slice = slice_values(current_group.sum(0), current_pieces)
            current_slice_sorted = np.sort(current_slice)
            current_loads = loads + current_slice_sorted
            current_slope = (np.max(current_loads) - np.min(current_loads)) / M
            next_slope = np.max(simulated_slopes[current_idx + keep:])

            if abs(current_slope) > abs(next_slope):
                low = mid
            else:
                high = mid

        S = high
        keep = M - S
        current_group = window[:, :keep]
        current_pieces = compute_piece_counts(current_group, M, stage_weights)

        for i in range(keep):
            pieces[sort_idx[current_idx + i]] = current_pieces[i]

        current_slice = slice_values(current_group.sum(0), current_pieces)
        current_slice_sorted = np.sort(current_slice)
        loads += current_slice_sorted
        loads = np.sort(loads)[::-1]

        current_idx += keep
        num_remain_slice -= S

    return pieces


@njit
def compute_objective(deployment, X, pieces):
    M, P = deployment.shape
    loads = np.zeros(M)

    for i in range(M):
        for j in range(P):
            expert = deployment[i, j]
            if pieces[expert] == 0:
                continue
            loads[i] += X[expert] / pieces[expert]

    mean_load = np.mean(loads)
    max_load = np.max(loads)
    obj = max_load / mean_load
    return obj, loads


@njit
def auto_fix_new_placement(old_placement, new_placement):
    """
    Adjust the new_placement matrix to ensure elements (including duplicates) that exist in both
    old_placement and new_placement remain in their original positions from old_placement.
    New elements (unique to new_placement) will fill the remaining empty positions.

    Args:
        old_placement: Old deployment matrix with shape (num_ranks, num_experts)
        new_placement: New deployment matrix to be fixed, must have the same shape as old_placement

    Returns:
        fixed_new: adjusted version of the new_placement matrix
    """
    num_ranks, num_experts = old_placement.shape
    fixed_new = np.empty_like(new_placement)

    max_expert_old = old_placement.max() if num_experts > 0 else 0
    max_expert_new = new_placement.max() if num_experts > 0 else 0
    max_expert = max(max_expert_old, max_expert_new)

    for rank_id in range(num_ranks):
        old_row = old_placement[rank_id]
        new_row = new_placement[rank_id]

        index_array = np.full((max_expert + 1, num_experts), -1, dtype=np.int32)
        count_array = np.zeros(max_expert + 1, dtype=np.int32)

        for idx in range(num_experts):
            val = old_row[idx]
            if val >= 0 and val <= max_expert:
                pos = count_array[val]
                index_array[val, pos] = idx
                count_array[val] += 1

        old_counter = np.zeros(max_expert + 1, dtype=np.int32)
        for idx in range(num_experts):
            val = old_row[idx]
            if val >= 0 and val <= max_expert:
                old_counter[val] += 1

        retain_elements = np.empty(num_experts, dtype=new_placement.dtype)
        new_elements = np.empty(num_experts, dtype=new_placement.dtype)
        retain_ptr = 0
        new_ptr = 0

        for val in new_row:
            if val >= 0 and val <= max_expert and old_counter[val] > 0:
                retain_elements[retain_ptr] = val
                retain_ptr += 1
                old_counter[val] -= 1
            else:
                new_elements[new_ptr] = val
                new_ptr += 1

        current_fixed = np.full(num_experts, -1, dtype=new_placement.dtype)

        for i in range(retain_ptr):
            val = retain_elements[i]
            if val >= 0 and val <= max_expert:
                pos = count_array[val] - 1
                if pos >= 0:
                    idx = index_array[val, pos]
                    current_fixed[idx] = val
                    count_array[val] -= 1

        empty_indices = np.empty(num_experts, dtype=np.int32)
        empty_ptr = 0
        for idx in range(num_experts):
            if current_fixed[idx] == -1:
                empty_indices[empty_ptr] = idx
                empty_ptr += 1

        for i in range(new_ptr):
            if i < empty_ptr:
                current_fixed[empty_indices[i]] = new_elements[i]

        fixed_new[rank_id] = current_fixed

    return fixed_new


class FlashLB(EplbPolicy):
    par_history = defaultdict(float)
    hotness_window = {}

    def __init__(self, config: DynamicConfig):
        super().__init__(config)
        self.max_stage_window = (
            config.max_stage_window if hasattr(config, "max_stage_window") else 1
        )
        self.buffer_expert_layer_num = (
            config.buffer_expert_layer_num
            if hasattr(config, "buffer_expert_layer_num")
            else 58
        )
        self.threshold_ratio = (
            config.threshold_ratio if hasattr(config, "threshold_ratio") else 0
        )

    def compute_expert_hotness(self, num_of_expert: int, deployment: np.ndarray, rank_load: np.ndarray):
        hotness = np.zeros(num_of_expert, dtype=rank_load.dtype)
        deployment_flat = deployment.ravel()
        rank_load_flat = rank_load.ravel()
        np.add.at(hotness, deployment_flat, rank_load_flat)
        return hotness

    def compute_rank_load(self, deployment: np.ndarray, hotness: np.ndarray):
        n_stage, N = hotness.shape
        if np.any(deployment < 0):
            print(f"{deployment=}")
            exit()
        unit_hotness = hotness / np.bincount(deployment.reshape(-1))
        stage_par = np.zeros(n_stage)
        for i in range(n_stage):
            stage_load = unit_hotness[i][deployment].sum(-1)
            stage_par[i] = stage_load.max() / stage_load.mean()
        return stage_par.mean()

    def group_based_adaptive_bloating(self, X, P, M, stage_weights=None, recorsive=False):
        n_stage, N = X.shape
        if stage_weights is None:
            stage_weights = np.ones(n_stage, dtype=np.float32)

        if recorsive:
            (
                simulated_deployment,
                simulated_pieces,
            ) = self.group_based_adaptive_bloating(X, P, M, stage_weights, recorsive=False)
        else:
            simulated_pieces = compute_piece_counts(X, P, stage_weights)
            simulated_deployment = jsq_placement(X, simulated_pieces, M, stage_weights)

        pieces = group_based_adaptive_bloating_kernel(
            X.astype(np.float32),
            P,
            M,
            simulated_pieces.astype(np.int32),
            simulated_deployment.astype(np.int32),
            stage_weights.astype(np.float32),
        )

        deployment = jsq_placement(X, pieces, M, stage_weights)

        X_all = X.sum(0)
        unit_load = X_all / pieces
        load = unit_load[deployment].sum(-1)

        sim_unit_load = X_all / simulated_pieces
        sim_load = sim_unit_load[simulated_deployment].sum(-1)

        if load.max() > sim_load.max():
            return simulated_deployment, simulated_pieces
        return deployment, pieces

    def need_update(self, current_par, layer_id=0):
        threshold = self.par_history.get(layer_id, 0.0)
        return current_par >= self.threshold_ratio * threshold

    def compute_stage_weight(self, hotness):
        n_stage = hotness.shape[0]
        stage_weights = np.zeros(n_stage)
        for i in range(n_stage):
            stage_weights[i] = hotness[i].sum()

        stage_weights = stage_weights / stage_weights.max()
        return stage_weights

    def rebalance_layer(self, deployment, hotness, layer_id=0):
        num_rank, expert_per_rank = deployment.shape
        num_expert = np.unique(deployment.reshape(-1)).shape[0]
        num_of_redundant_expert = num_rank * expert_per_rank - num_expert

        current_par = self.compute_rank_load(deployment, hotness)

        if not self.need_update(current_par, layer_id):
            return deployment, current_par, current_par

        stage_weights = self.compute_stage_weight(hotness)
        new_deployment, _ = self.group_based_adaptive_bloating(
            hotness,
            num_expert + num_of_redundant_expert,
            num_rank,
            stage_weights,
            recorsive=False,
        )
        if np.any(new_deployment < 0):
            print(f"{new_deployment=}")
        new_par = self.compute_rank_load(new_deployment, hotness)

        return new_deployment, new_par, current_par

    def register_hotness(self, deployment, rank_load, num_layer, num_expert):
        for layer in range(num_layer):
            if layer not in self.hotness_window:
                self.hotness_window[layer] = deque(maxlen=self.max_stage_window)
            hotness = self.compute_expert_hotness(num_expert, deployment[layer], rank_load[layer])
            self.hotness_window[layer].append(hotness)

    def compress_by_avg_pooling_fast_nd(self, arr, m):
        n, d = arr.shape
        idx = (np.arange(n) * m // n)
        result = np.zeros((m, d))
        counts = np.zeros((m, 1))
        np.add.at(result, idx, arr)
        np.add.at(counts, idx, 1)
        return result / counts

    def rebalance_experts(self, current_expert_table, expert_workload):
        current_deployment = np.array(current_expert_table)
        expert_workload = np.array(expert_workload)
        expert_workload += 1
        num_layer = expert_workload.shape[0]
        num_expert = np.unique(current_expert_table[0].reshape(-1)).shape[0]
        self.register_hotness(current_deployment, expert_workload, num_layer, num_expert)

        new_deployment = current_deployment.copy()

        layers_need_update = np.arange(num_layer)

        new_par = np.zeros(layers_need_update.shape[0])
        current_par = np.zeros(layers_need_update.shape[0])
        for i, layer in enumerate(layers_need_update):
            hotness = np.array(self.hotness_window[layer])
            if hotness.shape[0] > self.max_stage_window:
                hotness = self.compress_by_avg_pooling_fast_nd(hotness, self.max_stage_window)

            (
                new_deployment[layer],
                new_par[i],
                current_par[i],
            ) = self.rebalance_layer(
                current_deployment[layer],
                hotness,
                layer_id=layer
            )

        priority = new_par / current_par
        priority_idx = np.argsort(priority)
        priority_idx = priority_idx[priority[priority_idx] < 1][: self.buffer_expert_layer_num]

        if np.all(expert_workload == 1):
            for _, layer in enumerate(layers_need_update):
                self.hotness_window[layer].pop()
            return False, np.array([], dtype=int), current_deployment
        change = len(priority_idx) > 0
        if change:
            for idx in priority_idx:
                self.par_history[layers_need_update[idx]] = new_par[idx]

        layers_need_update = priority_idx
        deployment = current_deployment
        for layer in layers_need_update:
            deployment[layer] = auto_fix_new_placement(current_deployment[layer], new_deployment[layer])

        return change, layers_need_update, deployment





def generate_layered_experts(num_layers=58, layer_shape=(32, 9), expert_min=0, expert_max=255):
    """
    Generate expert deployment matrix meeting the following conditions:
    - Total of num_layers layers
    - Each layer has shape layer_shape (32,9)
    - Each expert from expert_min to expert_max (0 to 255) appears at least once in each layer

    Args:
        num_layers: Number of layers, default 58
        layer_shape: Shape of a single layer, default (32,9)
        expert_min: Minimum expert ID, default 0
        expert_max: Maximum expert ID, default 255
    Returns:
        torch.Tensor: Tensor with shape (num_layers, layer_shape[0], layer_shape[1])
    """
    # 1. Basic parameter calculation
    expert_num = expert_max - expert_min + 1  # Total number of experts: 256 (0~255)
    layer_total = layer_shape[0] * layer_shape[1]  # Total elements in a single layer: 32*9=288
    extra_slots = layer_total - expert_num  # Number of random positions to fill per layer: 288-256=32

    # 2. Verify feasibility (total elements must be ≥ number of experts to cover all experts)
    assert layer_total >= expert_num, (
        f"Number of elements in a single layer {layer_total} < number of experts {expert_num}, "
        "cannot cover all experts"
    )

    # 3. Generate layers one by one
    layers = []
    for _ in range(num_layers):
        # 3.1 Generate "complete expert sequence" (ensure each expert from 0 to 255 is included)
        full_experts = torch.arange(expert_min, expert_max + 1, dtype=torch.int64)  # shape (256,)

        # 3.2 Generate "supplementary random experts" (fill remaining 32 positions, randomly selected from 0~255)
        extra_experts = torch.randint(
            expert_min, expert_max + 1, size=(extra_slots,), dtype=torch.int64
        )  # shape (32,)

        # 3.3 Concatenate and shuffle (ensure random distribution of experts in each layer)
        layer_flat = torch.cat([full_experts, extra_experts], dim=0)  # shape (288,)
        # Shuffle order (use randperm to generate random indices to avoid repeated shuffling issues)
        shuffle_idx = torch.randperm(layer_flat.shape[0])
        layer_shuffled = layer_flat[shuffle_idx]

        # 3.4 Reshape to layer_shape (32,9)
        layer = layer_shuffled.reshape(layer_shape)
        layers.append(layer)

    # 4. Stack all layers to get the final tensor
    return torch.stack(layers, dim=0)  # shape (58,32,9)

def warm_up():
    exam_config = DynamicConfig()
    exam_config.ep_worldsize = 32
    exam_config.num_die_per_host = 16
    algo = FlashLB(exam_config)
    # Generate target tensor
    expert_tensor = generate_layered_experts(num_layers=58, layer_shape=(32, 9))

    algo.rebalance_experts(expert_tensor, torch.randint(1, 1000, (58, 32, 9)))
    FlashLB.par_history = defaultdict(float)
    FlashLB.hotness_window = {}

warm_up()