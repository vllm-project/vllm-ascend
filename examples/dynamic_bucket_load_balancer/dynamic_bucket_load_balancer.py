import math
from collections import namedtuple
from typing import AnyStr

ServerInfo = namedtuple("ServerInfo", ["instance_type", "instance_idx"])


class Task:
    """请求任务(包含请求长度信息)"""

    def __init__(self, task_id, task_length, task_load):
        self.id = task_id
        self.length = task_length
        self.bucket_idx = -1
        self.load = task_load
        self.server_info: ServerInfo = ServerInfo("Unknown", -1)

    def __repr__(self):
        return (
            f"Task(id={self.id}, length={self.length}, load={self.load}, "
            f"instance_type={self.server_info.instance_type}, instance_idx={self.server_info.instance_idx})"
        )


class Bucket:
    """桶"""

    def __init__(self, bucket_ranges: tuple[int, int]):
        # 桶的上下边界
        self.min_length = bucket_ranges[0]
        self.max_length = bucket_ranges[1]
        # 统计信息
        self.task_count = 0
        self.total_load = 0.0


class DynamicBucketLoadBalancer:
    """
    优先基于请求长度静态分桶，并根据桶的负载和长度亲和性动态调整新请求分配以实现负载均衡
    """

    def __init__(
        self, buckets: list[tuple[int, int]], sensitivity=1.0, affinity_strength=0.1, log_func=print, all_neighbor=False
    ):
        """
        初始化负载均衡器
        :param buckets: 每个桶的长度范围
        :param sensitivity: 对负载差距的敏感度系数，值越大，对差距越敏感
        :param affinity_strength: 长度亲和因子的强度系数，值越大，请求更倾向于分配到标准桶
        :param log_func: 日志打印函数
        :param all_neighbor: 是否将所有桶作为邻居（负载均衡的范围），False时仅将左右桶作为邻居
        """
        self.num_buckets = len(buckets)
        self.sensitivity = sensitivity
        self.affinity_strength = affinity_strength
        self.log_func = log_func
        self.all_neighbor = all_neighbor

        # 初始化桶
        self.buckets = {idx: Bucket(bucket_ranges) for idx, bucket_ranges in enumerate(buckets)}

        bucket_boundaries = ", ".join(
            f"bucket {idx}: [{bucket.min_length}, {bucket.max_length})" for idx, bucket in self.buckets.items()
        )
        self._log_info(f"Initialized {self.num_buckets} buckets: {bucket_boundaries}")

        # 负载均衡的阈值概率，请求重定向概率超过该阈值，则会被重定向
        self.base_probability_threshold = 0.12
        self._log_info(f"Load Balance base_probability_threshold: {self.base_probability_threshold:.2f} ")

        # 保存请求Task
        self.tasks: dict[AnyStr, Task] = {}  # type: ignore

        # 统计信息
        self.redirected_tasks = 0
        self.total_tasks = 0

    def _log_info(self, msg, *args, **kwargs):
        if self.log_func:
            self.log_func(msg, *args, **kwargs)

    def _get_standard_bucket_index(self, task_length):
        """根据请求长度确定其所属的标准桶索引"""
        for bucket_idx, bucket in self.buckets.items():
            if bucket.min_length <= task_length < bucket.max_length:
                return bucket_idx
        # 如果长度不在各桶长度范围内，则返回最后一个桶
        return self.num_buckets - 1

    def _get_neighbor_indices(self, bucket_idx):
        """获取指定桶的左右邻居索引"""
        if self.all_neighbor:
            return list(range(self.num_buckets))

        neighbors = []
        if bucket_idx > 0:
            neighbors.append(bucket_idx - 1)
        if bucket_idx < self.num_buckets - 1:
            neighbors.append(bucket_idx + 1)
        return neighbors

    def _calculate_length_affinity(self, task_length, neighbor_bucket_idx):
        """
        计算任务长度与邻居桶的亲和因子 (0.0 到 1.0)
        1.0 表示紧靠邻居桶，0.0 表示离邻居桶很远。
        """
        neighbor_bucket = self.buckets[neighbor_bucket_idx]
        neighbor_bucket_min = neighbor_bucket.min_length
        neighbor_bucket_max = neighbor_bucket.max_length

        if neighbor_bucket_min < task_length < neighbor_bucket_max:
            raise RuntimeError("task_length 必须在邻居桶范围外")

        neighbor_bucket_center = (neighbor_bucket_min + neighbor_bucket_max) / 2.0
        neighbor_bucket_half_width = (neighbor_bucket_max - neighbor_bucket_min) / 2.0

        # 计算请求长度到邻居桶中心的距离
        distance_to_center = abs(task_length - neighbor_bucket_center)

        # 计算亲和因子：距离邻居桶边界越近，因子越接近1
        if neighbor_bucket_half_width > 0:
            # 归一化距离，计算请求长度与邻居桶半径的相对距离
            normalized_distance = (distance_to_center - neighbor_bucket_half_width) / neighbor_bucket_half_width
            # 使用指数衰减计算亲和因子
            # 例：normalized_distance=0.1、self.affinity_strength=1.0时，有0.9的neighbor_affinity
            neighbor_affinity = math.exp(-self.affinity_strength * normalized_distance)
        else:
            neighbor_affinity = 1.0  # 理论上不会发生，但作为保护

        # 确保在 [0, 1] 范围内
        return max(0.0, min(neighbor_affinity, 1.0))

    def _calculate_redirect_probability(self, task_length, standard_bucket_idx, neighbor_bucket_idx):
        """
        根据负载差距和长度亲和性计算重定向到邻居桶的概率
        """
        standard_load = self.buckets[standard_bucket_idx].total_load
        neighbor_load = self.buckets[neighbor_bucket_idx].total_load

        # --- 1. 基于负载差距计算基础概率 ---
        if standard_load <= 0:
            load_probability = 0.0  # 标准桶无负载，无需重定向
        else:
            # 计算邻居桶与标准桶的负载比率
            load_ratio = neighbor_load / max(standard_load, 1e-9)  # 防止除以零

            # 应用敏感度系数并限制在 [0, 1] 范围内
            # 邻居桶的负载比标准桶的负载越小，重定向概率越大
            # neighbor_load / standard_load = 5/6，self.sensitivity=1.0时，load_probability=1/6=0.16666
            load_probability = 1 - load_ratio**self.sensitivity
            load_probability = max(0.0, min(load_probability, 1))

        # --- 2. 计算长度亲和因子 ---
        affinity_factor = self._calculate_length_affinity(task_length, neighbor_bucket_idx)

        # --- 3. 结合两者计算重定向到邻居桶的最终概率 ---
        # 最终概率 = 基础负载概率 * 长度亲和因子
        # 这意味着：即使邻居桶与标准桶负载差距很大，如果请求长度不接近邻居桶，重定向到邻居桶的概率也会被抑制。
        # 反之，如果请求长度接近邻居桶，即使负载差距一般，邻居桶也可能获得较高的重定向概率。
        final_probability = load_probability * affinity_factor

        return final_probability

    def dispatch_single_task(self, task_id: AnyStr, task_length: int, task_load):
        return self.dispatch_task(Task(task_id, task_length, task_load))

    def dispatch_task(self, cur_task):
        """
        为新请求分配桶，考虑动态负载均衡和长度亲和性
        """
        self.total_tasks += 1
        standard_bucket_idx = self._get_standard_bucket_index(cur_task.length)

        # 获取邻居桶
        neighbor_indices = self._get_neighbor_indices(standard_bucket_idx)

        best_neighbor_idx = None
        best_redirect_prob = 0.0

        # 检查所有邻居，找出综合考虑负载和长度亲和性后，重定向概率最高的那个
        for neighbor_idx in neighbor_indices:
            # 只考虑负载更低的邻居
            if self.buckets[neighbor_idx].total_load < self.buckets[standard_bucket_idx].total_load:
                prob = self._calculate_redirect_probability(cur_task.length, standard_bucket_idx, neighbor_idx)
                if prob > best_redirect_prob:
                    best_redirect_prob = prob
                    best_neighbor_idx = neighbor_idx

        # 决定最终分配的桶
        final_bucket_idx = standard_bucket_idx
        if best_neighbor_idx is not None and best_redirect_prob > 0:
            # 根据计算出的最佳概率决定是否重定向
            if self.base_probability_threshold < best_redirect_prob:
                final_bucket_idx = best_neighbor_idx
                self.redirected_tasks += 1
                self._log_info(
                    f"{cur_task} redirected from bucket {standard_bucket_idx} to {final_bucket_idx}"
                    f"(prob={best_redirect_prob:.4f})"
                )

        # 将请求任务分配给最终选定的桶（更新统计信息）
        self.buckets[final_bucket_idx].task_count += 1
        self.buckets[final_bucket_idx].total_load += cur_task.load
        cur_task.bucket_idx = final_bucket_idx

        if cur_task.id in self.tasks:
            raise RuntimeError(f"Task {cur_task.id} is existed!")
        else:
            self.tasks[cur_task.id] = cur_task

        return final_bucket_idx, cur_task

    def release_task(self, task_id):
        """释放请求负载"""
        if task_id in self.tasks:
            found_task = self.tasks.pop(task_id)
            if 0 <= found_task.bucket_idx < self.num_buckets:
                self.buckets[found_task.bucket_idx].task_count -= 1
                self.buckets[found_task.bucket_idx].total_load -= found_task.load
                return True
            else:
                raise RuntimeError(f"Bucket {found_task.bucket_idx} not found")
        else:
            raise RuntimeError(f"Task {task_id} not found")

    def release_all_tasks(self):
        for bucket in self.buckets.values():
            bucket.task_count = 0
            bucket.total_load = 0
        self.tasks.clear()


class NoStandardBucketLoadBalancer(DynamicBucketLoadBalancer):
    """仅根据负载进行请求分发，无标准桶"""

    def __init__(self, num_buckets: int, max_length: int, log_func=print):
        bucket_range = math.ceil(max_length / num_buckets)
        start_length = 0
        buckets = []
        for _ in range(num_buckets):
            end_length = start_length + bucket_range
            if end_length > max_length:
                end_length = max_length
            buckets.append((start_length, end_length))
            start_length += bucket_range
        super().__init__(buckets=buckets, log_func=log_func, sensitivity=100, affinity_strength=0, all_neighbor=True)
