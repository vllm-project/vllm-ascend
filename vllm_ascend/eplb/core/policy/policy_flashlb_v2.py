import logging
from collections import deque
from typing import Dict

import numpy as np
import torch
from numba import njit, prange
from collections import defaultdict

from .policy_abstract import DynamicConfig, EplbPolicy

from scipy import stats
from scipy.stats import norm
from scipy.optimize import linear_sum_assignment
import math
import time

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


@njit(fastmath=True, cache=True)
def min_max_replica(mu, var, num_available_replicas, current_replicas, z_score):
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


def make_replica(mu, var, num_available_replicas, current_replicas, z_score, method="percentage"):
    if method == "percentage":
        return percentage_replica(mu, var, num_available_replicas, current_replicas, z_score)
    elif method == "max_delta":
        return max_delta_replica(mu, var, num_available_replicas, current_replicas, z_score)
    else:
        return min_max_replica(mu, var, num_available_replicas, current_replicas, z_score)


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

            if best_dev == -1:
                continue

            device_mu[best_dev] = best_mu
            device_var[best_dev] = best_var
            new_deployment[best_dev, dev_ptr[best_dev]] = idx
            dev_ptr[best_dev] += 1

    return new_deployment


@njit(fastmath=True)
def affinity_deployment(mu, var, cov, deployment, affinity_groups, z_score):
    """
    使用numba优化的affinity部署算法

    参数:
    mu: (num_experts,) 专家期望值
    var: (num_experts,) 专家方差
    cov: (num_experts, num_experts) 专家协方差矩阵
    deployment: (num_devices, num_slots) 原始部署矩阵
    affinity_groups: (num_groups, group_size) 亲和组配置
    z_score: float Z分数

    返回:
    affinity_deployment: (num_groups, group_size) 分组后的设备索引
    """
    num_devices, num_slots = deployment.shape
    num_groups, group_size = affinity_groups[0], affinity_groups[1]
    num_experts = len(mu)

    # 预计算每个设备的专家列表（去重）
    device_experts_list = np.empty((num_devices, num_slots), dtype=np.int32)
    device_experts_count = np.zeros(num_devices, dtype=np.int32)

    for dev in range(num_devices):
        count = 0
        seen = np.zeros(num_experts, dtype=np.bool_)
        for slot in range(num_slots):
            expert = deployment[dev, slot]
            if expert != -1 and not seen[expert]:
                device_experts_list[dev, count] = expert
                count += 1
                seen[expert] = True
        device_experts_count[dev] = count

    # 初始化结果
    affinity_deployment = np.full((num_groups, group_size), -1, dtype=np.int32)

    # 使用位图记录设备是否已部署（替代deployed_devs列表）
    deployed_mask = np.zeros(num_devices, dtype=np.bool_)

    # 为每个组维护专家位图和专家列表
    group_expert_mask = np.zeros((num_groups, num_experts), dtype=np.bool_)
    group_experts_list = np.full((num_groups, group_size * num_slots), -1, dtype=np.int32)
    group_experts_count = np.zeros(num_groups, dtype=np.int32)

    # 预计算每个设备内部的专家协方差和
    device_self_cov = np.zeros(num_devices, dtype=np.float64)
    for dev in range(num_devices):
        count = device_experts_count[dev]
        if count > 0:
            experts = device_experts_list[dev, :count]
            for i in range(count):
                for j in range(i, count):
                    device_self_cov[dev] += cov[experts[i], experts[j]]
            device_self_cov[dev] *= 2  # 对称矩阵，但之前算法中对角线只算一次
            # 减去对角线重复计算的部分
            for i in range(count):
                device_self_cov[dev] -= cov[experts[i], experts[i]]

    # 预计算设备的总mu
    device_total_mu = np.zeros(num_devices, dtype=np.float64)
    for dev in range(num_devices):
        count = device_experts_count[dev]
        if count > 0:
            experts = device_experts_list[dev, :count]
            device_total_mu[dev] = np.sum(mu[experts])

    # 为每个组维护状态
    group_mu = np.zeros(num_groups, dtype=np.float64)
    group_var = np.zeros(num_groups, dtype=np.float64)

    # 开始分组部署
    for group in range(num_groups):
        for pos in range(group_size):
            best_dev = -1
            best_score = -np.inf
            best_delta_mu = 0.0
            best_delta_var = 0.0

            # 遍历所有未部署的设备
            for dev in range(num_devices):
                if deployed_mask[dev]:
                    continue

                # 计算该设备对当前组的增量贡献
                delta_mu = 0.0
                delta_var = 0.0

                # 获取该设备的专家
                count = device_experts_count[dev]
                if count == 0:
                    continue

                experts = device_experts_list[dev, :count]

                # 计算新专家和现有专家之间的协方差
                cross_cov_sum = 0.0
                for i in range(count):
                    expert_i = experts[i]

                    # 检查是否已存在
                    if group_expert_mask[group, expert_i]:
                        continue

                    delta_mu += mu[expert_i]
                    delta_var += var[expert_i]

                    # 与组内已有专家计算协方差
                    group_count = group_experts_count[group]
                    if group_count > 0:
                        group_experts = group_experts_list[group, :group_count]
                        for j in range(group_count):
                            expert_j = group_experts[j]
                            cross_cov_sum += cov[expert_i, expert_j]

                # 添加设备内部专家间的协方差
                delta_var += device_self_cov[dev]
                # 添加交叉协方差（乘以2，因为cov矩阵是对称的）
                delta_var += 2.0 * cross_cov_sum

                # 计算得分增量
                if delta_var < 0:
                    delta_var = 0.0

                current_score = group_mu[group] - z_score * np.sqrt(max(group_var[group], 0.0))
                new_mu = group_mu[group] + delta_mu
                new_var = group_var[group] + delta_var
                new_score = new_mu - z_score * np.sqrt(max(new_var, 0.0))
                score_delta = new_score - current_score

                if score_delta > best_score:
                    best_score = score_delta
                    best_dev = dev
                    best_delta_mu = delta_mu
                    best_delta_var = delta_var

            # 部署最佳设备
            if best_dev != -1:
                affinity_deployment[group, pos] = best_dev
                deployed_mask[best_dev] = True

                # 更新组状态
                group_mu[group] += best_delta_mu
                group_var[group] += best_delta_var

                # 更新组专家列表和位图
                count = device_experts_count[best_dev]
                if count > 0:
                    experts = device_experts_list[best_dev, :count]
                    for i in range(count):
                        expert = experts[i]
                        if not group_expert_mask[group, expert]:
                            group_expert_mask[group, expert] = True
                            group_experts_list[group, group_experts_count[group]] = expert
                            group_experts_count[group] += 1

    return affinity_deployment


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


class FlashTree:
    def __init__(self, X, num_replicas, num_devices, z_score=0.674, depth=4, width=8, affinity_group=None):
        super().__init__()
        self.num_replicas = num_replicas
        self.num_devices = num_devices
        self.z_score = z_score
        self.depth = depth
        self.width = width

        self.X = X
        self.mu, self.var, self.cov = FlashTree.compute_statistics(X)
        self.affinity_group = affinity_group

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

    def neighbor_search(self, low, high, initial, max_range, get_score, *args):
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

    def optimize_balanceness(self):
        X_row = self.X
        mu, var, cov = self.mu, self.var, self.cov
        num_total_replicas = self.num_replicas
        num_devices = self.num_devices
        z_score = self.z_score
        depth, width = self.depth, self.width

        num_experts = mu.shape[0]
        num_avalaible_replicas = num_total_replicas - num_experts

        if depth <= 1:
            default_replicas = np.ones(num_experts, dtype=np.int32)
            default_replicas = make_replica(mu, var, num_avalaible_replicas, default_replicas, z_score)[0]
            default_deployment = -np.ones((num_devices, num_total_replicas // num_devices), dtype=np.int32)
            default_deployment = lpt_deployment(
                mu, var, cov, default_deployment, np.zeros(num_experts, dtype=np.int32), default_replicas, z_score
            )
            default_par = compute_score(X_row, default_replicas, default_deployment)
            return default_deployment, default_replicas, default_par

        interval_size = math.ceil(num_experts / depth)
        weight = (mu + z_score * np.sqrt(var))
        idx = np.argsort(-weight)

        deployed_replicas = np.zeros(num_experts, dtype=np.int32)
        deployment = -np.ones((num_devices, num_total_replicas // num_devices), dtype=np.int32)

        def _lpt_deployment(replicas):
            nonlocal mu, var, cov, deployment, deployed_replicas, z_score
            return lpt_deployment(mu, var, cov, deployment,
                                  np.zeros_like(replicas), replicas, z_score)

        def get_score(f, val_data, deployed_replicas,
                      current_idx, current_replicas,
                      remaind_idx, remaind_replicas):
            simulated_replicas = deployed_replicas.copy()
            simulated_replicas[current_idx] = current_replicas
            simulated_replicas[remaind_idx] = remaind_replicas
            simulated_deployment = f(simulated_replicas)

            score = compute_score(val_data, simulated_replicas, simulated_deployment)
            return score, simulated_deployment

        for node in range(depth - 1):
            low, high = 0, num_avalaible_replicas
            simulation_idx = idx[node * interval_size:]
            current_idx = idx[node * interval_size: (node + 1) * interval_size]
            remaind_idx = idx[(node + 1) * interval_size:]

            simulation_replicas = make_replica(
                mu[simulation_idx], var[simulation_idx],
                high, np.ones(simulation_idx.shape[0], dtype=np.int32),
                z_score
            )[0]
            current_replicas_f = make_replica(
                mu[current_idx], var[current_idx],
                high, np.ones(current_idx.shape[0], dtype=np.int32),
                z_score
            )[1]
            remaind_replicas_f = make_replica(
                mu[remaind_idx], var[remaind_idx],
                high, np.ones(remaind_idx.shape[0], dtype=np.int32),
                z_score
            )[1]

            initial_replicas = (simulation_replicas[:interval_size] - 1).sum()

            best_replica, best_score, best_deployment = self.neighbor_search(
                low, high, initial_replicas, width,
                lambda mid: get_score(
                    _lpt_deployment, X_row, deployed_replicas,
                    current_idx, current_replicas_f[mid],
                    remaind_idx, remaind_replicas_f[num_avalaible_replicas - mid],
                )
            )

            deployed_replicas[current_idx] = current_replicas_f[best_replica]
            num_avalaible_replicas -= best_replica

            if not num_avalaible_replicas or node == depth - 2:
                deployed_replicas[remaind_idx] = remaind_replicas_f[num_avalaible_replicas]
                break

        final_deployment = -np.ones((num_devices, num_total_replicas // num_devices), dtype=np.int32)
        final_deployment = lpt_deployment(
            mu, var, cov, final_deployment, np.zeros_like(deployed_replicas), deployed_replicas, z_score
        )
        final_par = compute_score(X_row, deployed_replicas, final_deployment)

        return final_deployment, deployed_replicas, final_par

    def optimize_balanceness_and_affinity(self):
        deployment, replicas, par = self.optimize_balanceness()

        affinity = affinity_deployment(self.mu, self.var, self.cov, deployment, self.affinity_group, self.z_score)
        if np.all(affinity >= 0):
            deployment = deployment[affinity.reshape(-1)]

        return deployment, replicas, par


class FlashLB(EplbPolicy):  #
    def __init__(self, config):
        super().__init__(config)
        self.max_observation_window = (config.max_stage_window if hasattr(
            config, "max_stage_window") else 1000)
        self.update_threshold_ratio = (config.threshold_ratio if hasattr(
            config, "threshold_ratio") else 1)
        self.update_threshold_value = (config.threshold_value if hasattr(
            config, "threshold_value") else 0.9)
        self.update_layers_upper_bound = (config.layers_upper_bound if hasattr(
            config, "layers_upper_bound") else -1)
        self.z_score = (config.z_score if hasattr(
            config, "z_score") else stats.norm.ppf(0.75))
        self.depth = (config.depth if hasattr(
            config, "depth") else 4)
        self.width = (config.width if hasattr(
            config, "width") else 8)
        self.sample_size = (config.sample_size if hasattr(
            config, "sample_size") else 1000)
        self.affinity_group = (config.affinity_group if hasattr(
            config, "affinity_group") else None
                               )
        self.average_to_peak_history = {}
        self.hotness_window = {}
        self.current_deployment = {}
        self.current_deployed_replicas = {}
        self.eplb_count = 0
        # self.cur_iter = 0
        # self.pro_thr = 1500

    def register_hotness(self, deployment, rank_load, num_layers, num_experts):
        num_stage = rank_load.shape[0]
        hotness = np.zeros((num_stage, num_layers, num_experts), dtype=rank_load.dtype)

        # 计算hotness的部分（看起来有点问题，但保持原样）
        for stage in range(num_stage):
            for layer in range(num_layers):
                deployment_flat = deployment[layer].ravel()
                rank_load_flat = rank_load[stage, layer].ravel()
                np.add.at(hotness[stage, layer], deployment_flat, rank_load_flat)

        # hotness += 1
        window_length = self.max_observation_window

        for layer in range(num_layers):
            new_X = hotness[-window_length:, layer, :]
            t = new_X.shape[0]

            if layer not in self.hotness_window:
                self.hotness_window[layer] = {
                    "buffer": np.zeros((window_length, num_experts), dtype=new_X.dtype),
                    "start": 0,
                    "length": 0,
                }

            info = self.hotness_window[layer]
            buf = info["buffer"]
            start = info["start"]
            length = info["length"]

            # 向量化写入 - 使用切片操作
            if start + t <= window_length:
                # 情况1：可以直接一次性写入，不跨越边界
                buf[start:start + t] = new_X
            else:
                # 情况2：需要分两段写入（跨越边界）
                first_part = window_length - start
                buf[start:] = new_X[:first_part]
                buf[:t - first_part] = new_X[first_part:]

            # 更新元数据
            start = (start + t) % window_length
            length = min(window_length, length + t)

            self.hotness_window[layer]["buffer"] = buf
            self.hotness_window[layer]["start"] = start
            self.hotness_window[layer]["length"] = length

    def need_update(self, layer_id=0):
        current_deployment = self.current_deployment.get(layer_id, None)
        if current_deployment is None:
            return True

        hotness = self.hotness_window[layer_id]["buffer"]
        average_to_peak_ratio = 1 / compute_score(
            hotness, self.current_deployed_replicas.get(layer_id), current_deployment)
        past_average_to_peak_ratio = self.average_to_peak_history.get(layer_id, 0.0)

        return (average_to_peak_ratio < past_average_to_peak_ratio * self.update_threshold_ratio or
                average_to_peak_ratio < self.update_threshold_value)

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
    def minimize_redeploy_with_inner_permutation(src: np.ndarray, dst: np.ndarray, group_affinity=None):
        if src.shape != dst.shape:
            raise ValueError("src and dst must have same shape (N, M)")

        N, M = src.shape
        if group_affinity is None:
            group_affinity = [N, 1]

        dst_reordered = np.empty_like(dst).reshape((*group_affinity, M))

        valid_src = src.reshape((group_affinity[0], -1))
        valid_dst = dst.reshape((group_affinity[0], -1))

        max_val = N * M
        src_counts = np.array([
            np.bincount(row[row != -1], minlength=max_val)
            for row in valid_src
        ], dtype=np.int32)
        dst_counts = np.array([
            np.bincount(row[row != -1], minlength=max_val)
            for row in valid_dst
        ], dtype=np.int32)

        matches = FlashLB.compute_match(src_counts, dst_counts, group_affinity[0], group_affinity[1] * M)
        cost = group_affinity[1] * M - matches

        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = list(zip(row_ind.tolist(), col_ind.tolist()))

        for src_idx, dst_idx in mapping:
            valid_src_1 = valid_src[src_idx].reshape((-1, M))
            valid_dst_1 = valid_dst[dst_idx].reshape((-1, M))
            max_val_1 = group_affinity[1] * M
            src_counts_1 = np.array([
                np.bincount(row[row != -1], minlength=max_val_1)
                for row in valid_src_1
            ], dtype=np.int32)
            dst_counts_1 = np.array([
                np.bincount(row[row != -1], minlength=max_val_1)
                for row in valid_dst_1
            ], dtype=np.int32)

            matches_1 = FlashLB.compute_match(src_counts_1, dst_counts_1, group_affinity[1], M)
            cost_1 = M - matches_1

            row_ind_1, col_ind_1 = linear_sum_assignment(cost_1)
            mapping_1 = list(zip(row_ind_1.tolist(), col_ind_1.tolist()))

            for src_idx_1, dst_idx_1 in mapping_1:

                s_row = valid_src[src_idx].reshape((-1, M))[src_idx_1]
                d_row = valid_dst[dst_idx].reshape((-1, M))[dst_idx_1]

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
                dst_reordered[src_idx, src_idx_1] = reordered
        return dst_reordered.reshape((N, M))

    def rebalance_experts(self, current_expert_table, expert_workload):
        current_deployment = np.array(current_expert_table)
        expert_workload = np.array(expert_workload)
        self.eplb_count += 1
        if self.eplb_count <= 1:
            return False, [], current_expert_table
        # iter_len = expert_workload.shape[0]
        # self.cur_iter += iter_len
        # if self.cur_iter <= self.pro_thr:
        #     return False, [], current_expert_table
        
        if expert_workload.ndim == 3:
            expert_workload = expert_workload[np.newaxis, ...]

        mask = (expert_workload.sum(axis=(1,2,3)) > 0)
        expert_workload = expert_workload[mask]
        num_layers = expert_workload.shape[1]
        num_expert = np.unique(current_expert_table[0].reshape(-1)).shape[0]
        num_devices = current_deployment.shape[1]
        num_replicas = len(current_deployment[0].reshape(-1))

        self.register_hotness(current_deployment, expert_workload, num_layers, num_expert)

        for layer in range(num_layers):
            self.current_deployment[layer] = current_deployment[layer]
            self.current_deployed_replicas[layer] = np.bincount(current_deployment[layer].reshape(-1),
                                                                minlength=num_expert)

        new_deployment = np.zeros((num_layers, num_devices, num_replicas // num_devices), dtype=np.int32)
        new_deployed_replicas = np.zeros((num_layers, num_expert), dtype=np.int32)
        new_average_to_peak_ratio = np.zeros((num_layers,), dtype=np.float32)
        delta_average_to_peak_ratio = np.zeros((num_layers,), dtype=np.float32)

        for layer in range(num_layers):
            if not self.need_update(layer):
                new_deployment[layer] = self.current_deployment[layer]
                new_deployed_replicas[layer] = self.current_deployed_replicas[layer]
                new_average_to_peak_ratio[layer] = self.average_to_peak_history.get(layer, 0.0)
                delta_average_to_peak_ratio[layer] = 0
                continue

            layer_info = self.hotness_window[layer]
            buf = layer_info["buffer"]
            start = layer_info["start"]
            length = layer_info["length"]

            idx = np.arange(start, start + length) % self.max_observation_window
            data = buf[idx]

            shape = data.shape
            window = max(length // self.sample_size, 1)
            data = data[-window * self.sample_size:].reshape((-1, window, *shape[1:])).sum(1)
            flash_tree = FlashTree(data, num_replicas, num_devices, self.z_score, self.depth, self.width,
                                   self.affinity_group)
            if self.affinity_group is None:
                best_deployment, best_replicas, best_score = flash_tree.optimize_balanceness()
            else:
                best_deployment, best_replicas, best_score = flash_tree.optimize_balanceness_and_affinity()

            new_deployed_replicas[layer] = best_replicas
            new_average_to_peak_ratio[layer] = 1 / best_score

            current_deployment = self.current_deployment.get(layer, None)

            new_deployment[layer] = FlashLB.minimize_redeploy_with_inner_permutation(
                current_deployment, best_deployment)
            current_average_to_peak_ratio = 1 / compute_score(
                buf, self.current_deployed_replicas.get(layer), current_deployment)
            delta_average_to_peak_ratio[layer] = new_average_to_peak_ratio[layer] - current_average_to_peak_ratio

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
    expert_num = expert_max - expert_min + 1  # Total number of experts: 256 (0~255)
    layer_total = layer_shape[0] * layer_shape[
        1]  # Total elements in a single layer: 32*9=288
    extra_slots = layer_total - expert_num  # Number of random positions to fill per layer: 288-256=32

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
                                      size=(extra_slots,),
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
    """
    Run a full warmup to trigger JIT compilation and cache all major kernels.
    Covers:
      - compute_statistics
      - min_max_replica
      - lpt_deployment
      - minimize_redeploy_with_inner_permutation
      - compute_score
      - compute_logical_to_physical_map
    """
    # ---- Basic configuration ----
    num_stages = 128  # number of hotness samples (time dimension)
    num_layers = 1  # single layer warmup
    num_expert = 256  # number of experts per layer
    num_replicas = 320  # total expert replicas
    num_gpus = 64  # total GPUs / devices
    z_score = stats.norm.ppf(0.75)  # 75th percentile for load balancing

    # ---- Generate synthetic hotness data ----
    hotness = np.random.randint(0, 10_000, (num_stages, num_layers, num_expert), dtype=np.int32)
    val_data = hotness[:, 0].astype(np.float32, copy=False)

    # ---- Compute mean / variance / covariance ----
    mean_, var_, cov_ = FlashTree.compute_statistics(val_data)

    # ---- Initialize replica counts (min-max warmup) ----
    total_replicas = np.ones(num_expert, dtype=np.int32)
    total_replicas, *_ = min_max_replica(
        mean_.astype(np.float32, copy=False),
        var_.astype(np.float32, copy=False),
        num_replicas - num_expert,
        total_replicas,
        z_score,
    )
    total_replicas, *_ = max_delta_replica(
        mean_.astype(np.float32, copy=False),
        var_.astype(np.float32, copy=False),
        num_replicas - num_expert,
        total_replicas,
        z_score,
    )
    total_replicas, *_ = percentage_replica(
        mean_.astype(np.float32, copy=False),
        var_.astype(np.float32, copy=False),
        num_replicas - num_expert,
        total_replicas,
        z_score,
    )

    # ---- Initialize deployment matrix (LPT warmup) ----
    experts_per_gpu = num_replicas // num_gpus
    deployment = np.full((num_gpus, experts_per_gpu), -1, dtype=np.int32)

    deployment = lpt_deployment(
        mean_.astype(np.float32, copy=False),
        var_.astype(np.float32, copy=False),
        cov_.astype(np.float32, copy=False),
        deployment,
        np.zeros_like(total_replicas),
        total_replicas,
        z_score,
    )

    # ---- Trigger redeployment matching kernel ----
    FlashLB.minimize_redeploy_with_inner_permutation(deployment, deployment)

    # ---- Trigger score computation kernel ----
    compute_score(val_data, total_replicas, deployment)


warm_up()

if __name__ == "__main__":
    exam_config = DynamicConfig()
    exam_config.max_stage_window = 1000
    exam_config.affinity_group = [16, 2]
    exam_config.ep_worldsize = 32
    exam_config.num_die_per_host = 16
    algo = FlashLB(exam_config)
    # Generate target tensor
    expert_tensor = generate_layered_experts(num_layers=58,
                                             layer_shape=(32, 9))

    x = [torch.randint(1, 1000, (1000, 58, 32, 9)), torch.randint(1, 1000, (1000, 58, 32, 9))]

    algo.rebalance_experts(expert_tensor, torch.randint(1, 1000, (1000, 58, 32, 9)))

    st = time.time()
    for i in range(100):
        algo.rebalance_experts(expert_tensor, x[i % 2])

    et = time.time()
    print(et - st)