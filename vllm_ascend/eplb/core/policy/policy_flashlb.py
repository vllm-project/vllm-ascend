# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# Todo: Once https://github.com/vllm-project/vllm/pull/24069 is merged in vllm. Remove this policy.

import logging
from collections import deque
from typing import Dict

import numpy as np
import torch
from numba import njit  # type: ignore

from .policy_abstract import DynamicConfig, EplbPolicy

from scipy import stats
from scipy.stats import norm
from scipy.optimize import linear_sum_assignment
import math

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

from vllm.logger import logger


@njit(fastmath=True, cache=True)
def min_max_replica_org(mu, var, num_available_replicas, current_replicas, z_score):
    N = mu.shape[0]
    unit_value = (mu + z_score * np.sqrt(var)) / current_replicas
    replicas_history = np.ones((num_available_replicas + 1, N), dtype=np.int32)
    replicas_history[0, :] = current_replicas[:]

    for r in range(num_available_replicas):
        max_idx = -1
        max_value = -1.0
        for idx in range(N):
            value = unit_value[idx]
            if value > max_value:
                max_value = value
                max_idx = idx

        current_replicas[max_idx] += 1
        unit_value[max_idx] = (mu[max_idx] + z_score * np.sqrt(var[max_idx])) / current_replicas[max_idx]
        replicas_history[r + 1, :] = current_replicas[:]

    return current_replicas, replicas_history


@njit
def max_delta_replica(mu, var, num_available_replicas, current_replicas, z_score):
    N = mu.shape[0]
    unit_value = (mu + z_score * np.sqrt(var)) / current_replicas
    replicas_history = np.ones((num_available_replicas + 1, N), dtype=np.int32)
    replicas_history[0, :] = current_replicas[:]

    for r in range(num_available_replicas):
        max_idx = -1
        max_value = -1.0
        for idx in range(N):
            value = unit_value[idx] / (current_replicas[idx] + 1)
            if value > max_value:
                max_value = value
                max_idx = idx

        current_replicas[max_idx] += 1
        unit_value[max_idx] = (mu[max_idx] + z_score * np.sqrt(var[max_idx])) / current_replicas[max_idx]
        replicas_history[r + 1, :] = current_replicas[:]

    return current_replicas, replicas_history


@njit
def percentage_replica(mu, var, num_available_replicas, current_replicas, z_score):
    N = mu.shape[0]
    total_load = mu + z_score * np.sqrt(var)
    sum_total_load = np.sum(total_load)

    replicas_history = np.ones((num_available_replicas + 1, N), dtype=np.int32)
    replicas_history[0, :] = current_replicas[:] 
    for r in range(1, num_available_replicas + 1):
        add_slots = np.zeros(N, dtype=np.int32)

        if sum_total_load == 0.0:
            base_add = r // N
            extra = r % N
            add_slots[:] = base_add
            add_slots[:extra] += 1
        else:
            quotas = (total_load / sum_total_load) * r
            base_add = np.floor(quotas).astype(np.int32)
            add_slots[:] = base_add
            remaining = r - np.sum(base_add)

            if remaining > 0:
                fractions = quotas - base_add
                indices = np.argsort(-fractions)
                add_slots[indices[:remaining]] += 1

        replicas_history[r] = current_replicas + add_slots

    return replicas_history[-1], replicas_history


@njit(fastmath=True, cache=True)
def compute_updated_device_variance(new_expert_id, device_slots, current_device_var, expert_var, expert_cov,
                                    expert_replicas):
    new_device_var = current_device_var + expert_var[new_expert_id] / expert_replicas[new_expert_id] ** 2

    for slot in device_slots:
        if slot == -1:
            break
        new_device_var += 2 * expert_cov[new_expert_id, slot] / expert_replicas[new_expert_id] / expert_replicas[slot]

    return new_device_var


@njit(fastmath=True, cache=True)
def lpt_deployment(mu, var, cov, deployment, deployed_replicas, total_replicas, z_score):
    num_devices, num_slots_per_device = deployment.shape

    unit_value = mu / total_replicas
    sorted_indices = np.argsort(-unit_value)

    new_deployment = -np.ones_like(deployment)
    device_mu = np.zeros(num_devices, dtype=np.float32)
    device_var = np.zeros(num_devices, dtype=np.float32)
    dev_ptr = np.zeros(num_devices, dtype=np.int32)

    for dev in range(num_devices):
        for slot in deployment[dev]:
            if slot != -1:
                device_mu[dev] += mu[slot] / total_replicas[slot]
                device_var[dev] += compute_updated_device_variance(slot, new_deployment[dev], device_var[dev], var, cov,
                                                                   total_replicas)
                new_deployment[dev, dev_ptr[dev]] = slot
                dev_ptr[dev] += 1

    for idx in sorted_indices:
        for _ in range(total_replicas[idx] - deployed_replicas[idx]):
            best_dev = -1
            best_risk = 1e30
            best_mu = -1.0
            best_var = -1.0
            for dev in range(num_devices):
                if dev_ptr[dev] >= num_slots_per_device:
                    continue
                if idx in new_deployment[dev]:
                    continue
                temp_mu = device_mu[dev] + mu[idx] / total_replicas[idx]
                temp_var = compute_updated_device_variance(idx, new_deployment[dev], device_var[dev], var, cov,
                                                           total_replicas)

                risk = temp_mu + z_score * np.sqrt(temp_var)
                if risk < best_risk:
                    best_risk = risk
                    best_dev = dev
                    best_mu = temp_mu
                    best_var = temp_var

            device_mu[best_dev] = best_mu
            device_var[best_dev] = best_var
            new_deployment[best_dev, dev_ptr[best_dev]] = idx
            dev_ptr[best_dev] += 1

    return new_deployment


@njit(fastmath=True, cache=True)
def compute_score(val_data, simulated_replicas, simulated_deployment):
    """
    同时执行:
      - unit_value = val_data / simulated_replicas
      - loads = unit_value[:, simulated_deployment].sum(-1)
    返回:
      loads (T, D)
    """
    T, N = val_data.shape
    D, K = simulated_deployment.shape
    scores = np.empty((T,), dtype=np.float32)
    for t in range(T):
        max_load = 0
        tot_load = 0
        for d in range(D):
            s = 0.0
            for k in range(K):
                idx = simulated_deployment[d, k]
                s += val_data[t, idx] / simulated_replicas[idx]
            tot_load += s
            max_load = max(max_load, s)
        scores[t] = (max_load * D + 1e-2) / (tot_load + 1e-2)

    return np.mean(scores)


def get_score(f, val_data, deployed_replicas,
              current_idx, current_replicas,
              remaind_idx, remaind_replicas):
    simulated_replicas = deployed_replicas.copy()
    simulated_replicas[current_idx] = current_replicas
    simulated_replicas[remaind_idx] = remaind_replicas
    simulated_deployment = f(simulated_replicas)

    score = compute_score(val_data, simulated_replicas, simulated_deployment)
    return score, simulated_deployment


def neighbor_search(low, high, initial, max_range, get_score, *args):
    max_range = min(max(initial - low, high - initial), max_range)
    best_x = initial
    best_score, best_sim = get_score(initial, *args)
    for r in range(1, max_range + 1):
        left = initial - r
        if left >= low:
            score, sim = get_score(left, *args)
            if score < best_score:
                best_x, best_score, best_sim = left, score, sim

        right = initial + r
        if right <= high:
            score, sim = get_score(right, *args)
            if score < best_score:
                best_x, best_score, best_sim = right, score, sim

    return best_x, best_score, best_sim


def compute_expert_hotness(num_of_expert: int,
                           deployment: np.ndarray, rank_load: np.ndarray):
    hotness = np.zeros(num_of_expert, dtype=rank_load.dtype)
    deployment_flat = deployment.ravel()
    rank_load_flat = rank_load.ravel()
    np.add.at(hotness, deployment_flat, rank_load_flat)
    return hotness


class FlashLB(EplbPolicy):
    def __init__(self, config: DynamicConfig):
        super().__init__(config)
        self.max_observation_window = (config.max_stage_window if hasattr(
            config, "max_stage_window") else 1000)
        # self.max_observation_window = (config.max_stage_window if hasattr(
        #     config, "max_stage_window") else 1)
        self.update_threshold_ratio = (config.threshold_ratio if hasattr(
            config, "threshold_ratio") else 0.95)
        self.update_threshold_value = (config.threshold_value if hasattr(
            config, "threshold_value") else 0.9)
        self.update_layers_upper_bound = (config.layers_upper_bound if hasattr(
            config, "layers_upper_bound") else -1)
        self.z_score = (config.z_score if hasattr(
            config, "z_score") else stats.norm.ppf(0.75))
        self.depth = (config.depth if hasattr(
            config, "depth") else 4)
        # self.depth = (config.depth if hasattr(
        #     config, "depth") else 1)
        self.width = (config.width if hasattr(
            config, "width") else 8)
        # self.width = (config.width if hasattr(
            # config, "width") else 1)
        self.average_to_peak_history = {}
        self.hotness_window = {}
        self.current_deployment = {}
        self.current_deployed_replicas = {}

    def min_max_replica(self, mu, var, num_available_replicas, current_replicas, z_score):
        return min_max_replica_org(mu, var, num_available_replicas, current_replicas, z_score)

    @staticmethod
    def compute_statistics(X):
        T, N = X.shape
        mean_ = np.mean(X, axis=0)
        if T > 1:
            X_centered = X - mean_
            variance_ = np.sum(X_centered ** 2, axis=0) / (T - 1)
            cov_matrix = (X_centered.T @ X_centered) / (T - 1)
        else:
            variance_ = np.zeros((N,))
            cov_matrix = np.zeros((N, N))
        return mean_, variance_, cov_matrix

    @staticmethod
    def sliding_update_stats(mean, cov, x_old, x_new, T):
        assert x_new.shape == x_old.shape
        mean = mean.astype(np.float64, copy=False)
        cov = cov.astype(np.float64, copy=False)
        x_old = x_old.astype(np.float64, copy=False)
        x_new = x_new.astype(np.float64, copy=False)

        sum_old = np.sum(x_old, axis=0)
        sum_new = np.sum(x_new, axis=0)
        deltaS = sum_new - sum_old
        new_mean = mean + deltaS / T

        x_old_centered = x_old - mean
        x_new_centered = x_new - mean

        SA_mu = np.dot(x_old_centered.T, x_old_centered)
        SB_mu = np.dot(x_new_centered.T, x_new_centered)

        Sigma = cov * (T - 1)
        Sigma_new = Sigma + SB_mu - SA_mu - np.outer(deltaS, deltaS) / T
        new_cov = Sigma_new / (T - 1)

        new_var = np.diag(new_cov)
        return new_mean, new_var, new_cov

    @staticmethod
    def incremental_update_stats(mean, cov, x_new, T):
        t, N = x_new.shape
        sum_new = np.sum(x_new, axis=0)
        new_T = T + t

        new_mean = (T * mean + sum_new) / new_T

        if T > 1:
            x_new_centered = x_new - new_mean
            cov_new = cov * (T - 1)
            cov_new += np.dot(x_new_centered.T, x_new_centered)
            cov_new += T * np.outer(mean - new_mean, mean - new_mean)
            new_cov = cov_new / (new_T - 1)
        else:
            x_old = mean.reshape(1, -1)
            x_old_centered = x_old - new_mean
            x_new_centered = x_new - new_mean
            sum_squares = (
                    np.dot(x_old_centered.T, x_old_centered)
                    + np.dot(x_new_centered.T, x_new_centered)
            )
            new_cov = sum_squares / (new_T - 1)

        new_var = np.diag(new_cov)
        return new_mean, new_var, new_cov, new_T

    def register_hotness(self, deployment, rank_load, num_layers, num_experts):
        num_stage = rank_load.shape[0]
        hotness = np.zeros((num_stage, num_layers, num_experts), dtype=rank_load.dtype)
        for stage in range(num_stage):
            for layer in range(num_layers):
                hotness[stage, layer] = compute_expert_hotness(num_experts, deployment[layer], rank_load[stage, layer])
        # hotness_stage = np.sum(hotness_raw, axis=(1, 2))
        # threshold = np.max(hotness_stage) / 2
        # mask = hotness_stage >= threshold
        # hotness = hotness_raw[mask]
        mask_1 = hotness == 0
        hotness[mask_1] += 1
        window_length = self.max_observation_window
        T_total = hotness.shape[0]

        for layer in range(num_layers):
            new_X = hotness[-window_length:, layer, :]
            t = new_X.shape[0]

            if layer not in self.hotness_window or t == window_length:
                mu, var, cov = self.compute_statistics(new_X)
                buffer = np.zeros((window_length, num_experts), dtype=new_X.dtype)
                buffer[:t] = new_X
                self.hotness_window[layer] = {
                    "buffer": buffer,
                    "start": 0,
                    "length": t,
                    "mean": mu.astype(np.float32, copy=False),
                    "var": var.astype(np.float32, copy=False),
                    "cov": cov.astype(np.float32, copy=False),
                }
                continue

            info = self.hotness_window[layer]
            buf = info["buffer"]
            start = info["start"]
            length = info["length"]
            mu = info["mean"]
            var = info["var"]
            cov = info["cov"]

            if length + t <= window_length:
                mu, var, cov, length = self.incremental_update_stats(
                    mu.astype(np.float64, copy=False), cov.astype(np.float64, copy=False), new_X, length)
                end = (start + length - t) % window_length
                buf[end:end + t] = new_X
            else:
                old_idx = np.arange(start, start + t) % window_length
                x_old = buf[old_idx]
                mu, var, cov = self.sliding_update_stats(
                    mu.astype(np.float64, copy=False), cov.astype(np.float64, copy=False), x_old, new_X,
                    window_length)
                buf[old_idx] = new_X
                start = (start + t) % window_length
                length = window_length

            self.hotness_window[layer] = {
                "buffer": buf,
                "start": start,
                "length": length,
                "mean": mu.astype(np.float32, copy=False),
                "var": var.astype(np.float32, copy=False),
                "cov": cov.astype(np.float32, copy=False),
            }

    @staticmethod
    @njit
    def compute_match(src_counts, dst_counts, N, M):
        """
        计算 src_counts 和 dst_counts 之间的匹配矩阵，逐元素计算。
        """
        matches = np.zeros((N, N), dtype=np.int32)
        for i in range(N):
            for j in range(N):
                match = 0
                for k in range(N * M):
                    match += min(src_counts[i, k], dst_counts[j, k])
                matches[i, j] = match
        return matches

    @staticmethod
    def minimize_redeploy_with_inner_permutation(src: np.ndarray, dst: np.ndarray):
        if src.shape != dst.shape:
            raise ValueError("src and dst must have same shape (N, M)")
        N, M = src.shape
        valid_src = src  
        valid_dst = dst

        max_val = N * M  
        src_counts = np.array([
            np.bincount(row[row != -1], minlength=max_val)
            for row in valid_src
        ], dtype=np.int32)
        dst_counts = np.array([
            np.bincount(row[row != -1], minlength=max_val)
            for row in valid_dst
        ], dtype=np.int32)

        matches = FlashLB.compute_match(src_counts, dst_counts, N, M)
        cost = M - matches

        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = list(zip(row_ind.tolist(), col_ind.tolist()))
        total_moves = int(cost[row_ind, col_ind].sum())

        dst_reordered = np.empty_like(dst)
        for src_idx, dst_idx in mapping:
            s_row = src[src_idx]
            d_row = dst[dst_idx]
            val_to_positions = {}
            for pos, v in enumerate(d_row):
                val_to_positions.setdefault(v, []).append(pos)

            reordered = np.empty(M, dtype=dst.dtype)
            assigned = [False] * M
            used_dst_positions = set()

            for pos_src, v in enumerate(s_row):
                positions = val_to_positions.get(v)
                if positions:

                    dst_pos = positions.pop()
                    reordered[pos_src] = v
                    assigned[pos_src] = True
                    used_dst_positions.add(dst_pos)

            remaining = [d_row[p] for p in range(M) if p not in used_dst_positions]

            ri = 0
            for pos in range(M):
                if not assigned[pos]:
                    reordered[pos] = remaining[ri]
                    ri += 1
            dst_reordered[src_idx] = reordered
        return dst_reordered

    def need_update(self, layer_id=0):
        past_average_to_peak_ratio = self.average_to_peak_history.get(layer_id, 0.0)
        if past_average_to_peak_ratio == 0.0:
            return True

        hotness = self.hotness_window[layer_id]["buffer"]
        average_to_peak_ratio = 1 / compute_score(
            hotness, self.current_deployed_replicas[layer_id], self.current_deployment[layer_id])

        return (average_to_peak_ratio < past_average_to_peak_ratio * self.update_threshold_ratio or
                average_to_peak_ratio < self.update_threshold_value)

    def flash_tree(self, X_row, mu, var, cov, num_total_replicas, num_devices, z_score=0.674, deep=4, width=8):
        num_experts = mu.shape[0]
        num_avalaible_replicas = num_total_replicas - num_experts

        if deep <= 1:
            default_replicas = np.ones(num_experts, dtype=np.int32)
            default_replicas = self.min_max_replica(mu, var, num_avalaible_replicas, default_replicas, z_score)[0]
            default_deployment = -np.ones((num_devices, num_total_replicas // num_devices), dtype=np.int32)
            default_deployment = lpt_deployment(
                mu, var, cov, default_deployment, np.zeros(num_experts, dtype=np.int32), default_replicas, z_score
            )
            default_par = compute_score(X_row, default_replicas, default_deployment)
            return default_deployment, default_replicas, default_par

        interval_size = math.ceil(num_experts / deep)
        weight = (mu + z_score * np.sqrt(var))
        idx = np.argsort(-weight)

        deployed_replicas = np.zeros(num_experts, dtype=np.int32)
        deployment = -np.ones((num_devices, num_total_replicas // num_devices), dtype=np.int32)

        def _lpt_deployment(replicas):
            nonlocal mu, var, cov, deployment, deployed_replicas, z_score
            return lpt_deployment(mu, var, cov, deployment,
                                  np.zeros_like(replicas), replicas, z_score)

        for node in range(deep - 1):
            low, high = 0, num_avalaible_replicas
            simulation_idx = idx[node * interval_size:]
            current_idx = idx[node * interval_size: (node + 1) * interval_size]
            remaind_idx = idx[(node + 1) * interval_size:]

            simulation_replicas = self.min_max_replica(
                mu[simulation_idx], var[simulation_idx],
                high, np.ones(simulation_idx.shape[0], dtype=np.int32),
                z_score
            )[0]
            current_replicas_f = self.min_max_replica(
                mu[current_idx], var[current_idx],
                high, np.ones(current_idx.shape[0], dtype=np.int32),
                z_score
            )[1]
            remaind_replicas_f = self.min_max_replica(
                mu[remaind_idx], var[remaind_idx],
                high, np.ones(remaind_idx.shape[0], dtype=np.int32),
                z_score
            )[1]

            initial_replicas = (simulation_replicas[:interval_size] - 1).sum()

            best_replica, best_score, best_deployment = neighbor_search(
                low, high, initial_replicas, width,
                lambda mid: get_score(
                    _lpt_deployment, X_row, deployed_replicas,
                    current_idx, current_replicas_f[mid],
                    remaind_idx, remaind_replicas_f[num_avalaible_replicas - mid],
                )
            )

            deployed_replicas[current_idx] = current_replicas_f[best_replica]
            num_avalaible_replicas -= best_replica

            if not num_avalaible_replicas or node == deep - 2:
                deployed_replicas[remaind_idx] = remaind_replicas_f[num_avalaible_replicas]
                break

        final_deployment = -np.ones((num_devices, num_total_replicas // num_devices), dtype=np.int32)
        final_deployment = lpt_deployment(
            mu, var, cov, final_deployment, np.zeros_like(deployed_replicas), deployed_replicas, z_score
        )
        final_par = compute_score(X_row, deployed_replicas, final_deployment)

        return final_deployment, deployed_replicas, final_par

    def rebalance_experts(self, current_expert_table, expert_workload):
        current_deployment = np.array(current_expert_table)
        expert_workload = np.array(expert_workload)

        if expert_workload.ndim == 3:
            expert_workload = expert_workload[np.newaxis, ...]
        num_layers = expert_workload.shape[1]
        num_expert = np.unique(current_deployment[0].reshape(-1)).shape[0]
        num_devices = current_deployment.shape[1]
        num_replicas = len(current_deployment[0].reshape(-1))
        self.register_hotness(current_deployment, expert_workload, num_layers, num_expert)
        
        for layer in range(num_layers):
            self.current_deployment[layer] = current_deployment[layer]
            self.current_deployed_replicas[layer] = np.bincount(current_deployment[layer].reshape(-1), minlength=num_expert)

        new_par = np.zeros((num_layers,), dtype=np.float32)
        new_deployment = np.zeros((num_layers, num_devices, num_replicas // num_devices), dtype=np.int32)
        new_deployed_replicas = np.zeros((num_layers, num_expert), dtype=np.int32)
        new_average_to_peak_ratio = np.zeros((num_layers,), dtype=np.float32)
        delta_average_to_peak_ratio = np.zeros((num_layers,), dtype=np.float32)

        for layer in range(num_layers):
            if not self.need_update(layer):
                new_deployment[layer] = self.current_deployment[layer]
                new_deployed_replicas[layer] = self.current_deployed_replicas[layer]
                new_average_to_peak_ratio[layer] = self.average_to_peak_history.get(layer, 0.0)
                new_par[layer] = 1 / new_average_to_peak_ratio[layer]
                delta_average_to_peak_ratio[layer] = 0
                continue

            layer_info = self.hotness_window[layer]
            buf = layer_info["buffer"]
            start = layer_info["start"]
            length = layer_info["length"]
            mu = layer_info["mean"]
            var = layer_info["var"]
            cov = layer_info["cov"]

            idx = np.arange(start, start + length) % self.max_observation_window
            val_data = buf[idx]

            deployment, deployed_replicas, new_score = self.flash_tree(val_data, mu, var, cov, num_replicas,
                                                                        num_devices,
                                                                        z_score=self.z_score, deep=self.depth,
                                                                        width=self.width)
   
            new_deployed_replicas[layer] = deployed_replicas
            new_average_to_peak_ratio[layer] = 1 / new_score
            new_par[layer] = new_score
            current_deployment = self.current_deployment[layer]

            new_deployment[layer] = FlashLB.minimize_redeploy_with_inner_permutation(current_deployment,
                                                                                    deployment)
            current_average_to_peak_ratio = 1 / compute_score(
                val_data, self.current_deployed_replicas.get(layer), current_deployment)

            delta_average_to_peak_ratio[layer] = \
                new_average_to_peak_ratio[layer] - current_average_to_peak_ratio
        priority_idx = np.argsort(-delta_average_to_peak_ratio)
        priority_idx = priority_idx[delta_average_to_peak_ratio[priority_idx] > 0]
        if self.update_layers_upper_bound > 0:
            priority_idx = priority_idx[:self.update_layers_upper_bound]

        for layer in priority_idx:
            self.current_deployment[layer] = new_deployment[layer]
            self.current_deployed_replicas[layer] = new_deployed_replicas[layer]
            self.average_to_peak_history[layer] = new_average_to_peak_ratio[layer]

        change = len(priority_idx) > 0
        return change, priority_idx, new_deployment


def generate_layered_experts(num_layers=58,
                             layer_shape=(32, 9),
                             expert_min=0,
                             expert_max=255):
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
    expert_num = expert_max - expert_min + 1
    layer_total = layer_shape[0] * layer_shape[1]
    extra_slots = layer_total - expert_num

    # 2. Verify feasibility (total elements must be ≥ number of experts to cover all experts)
    assert layer_total >= expert_num, (
        f"Number of elements in a single layer {layer_total} < number of experts {expert_num}, "
        "cannot cover all experts")

    # 3. Generate layers one by one
    layers = []
    for _ in range(num_layers):
        # 3.1 Generate "complete expert sequence" (ensure each expert from 0 to 255 is included)
        full_experts = torch.arange(expert_min,
                                    expert_max + 1,
                                    dtype=torch.int64)  # shape (256,)

        # 3.2 Generate "supplementary random experts" (fill remaining 32 positions, randomly selected from 0~255)
        extra_experts = torch.randint(expert_min,
                                      expert_max + 1,
                                      size=(extra_slots, ),
                                      dtype=torch.int64)  # shape (32,)

        # 3.3 Concatenate and shuffle (ensure random distribution of experts in each layer)
        layer_flat = torch.cat([full_experts, extra_experts],
                               dim=0)  # shape (288,)
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
    expert_tensor = generate_layered_experts(num_layers=58,
                                             layer_shape=(32, 9))

    algo.rebalance_experts(expert_tensor, torch.randint(1, 1000, (100, 58, 32, 9)))