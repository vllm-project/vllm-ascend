import math
from collections import namedtuple
from typing import AnyStr

ServerInfo = namedtuple("ServerInfo", ["instance_type", "instance_idx"])


class Task:
    """Request task (carries request length info)."""

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
    """A bucket."""

    def __init__(self, bucket_ranges: tuple[int, int]):
        self.min_length = bucket_ranges[0]
        self.max_length = bucket_ranges[1]
        self.task_count = 0
        self.total_load = 0.0


class DynamicBucketLoadBalancer:
    """
    Statically buckets requests by length first, then dynamically adjusts the
    assignment of new requests based on bucket load and length affinity to
    achieve load balancing.
    """

    def __init__(
        self, buckets: list[tuple[int, int]], sensitivity=1.0, affinity_strength=0.1, log_func=print, all_neighbor=False
    ):
        """
        Initialize the load balancer.
        :param buckets: length range of each bucket
        :param sensitivity: sensitivity to the load gap (higher = more sensitive)
        :param affinity_strength: strength of length affinity (higher = a request
            stays more in its standard bucket)
        :param log_func: logging function
        :param all_neighbor: if False, only the left/right buckets are neighbors;
            if True, all buckets are neighbors
        """
        self.num_buckets = len(buckets)
        self.sensitivity = sensitivity
        self.affinity_strength = affinity_strength
        self.log_func = log_func
        self.all_neighbor = all_neighbor

        self.buckets = {idx: Bucket(bucket_ranges) for idx, bucket_ranges in enumerate(buckets)}

        bucket_boundaries = ", ".join(
            f"bucket {idx}: [{bucket.min_length}, {bucket.max_length})" for idx, bucket in self.buckets.items()
        )
        self._log_info(f"Initialized {self.num_buckets} buckets: {bucket_boundaries}")

        # Redirect only when the redirect probability exceeds this threshold
        self.base_probability_threshold = 0.12
        self._log_info(f"Load Balance base_probability_threshold: {self.base_probability_threshold:.2f} ")

        # Track request tasks
        self.tasks: dict[AnyStr, Task] = {}  # type: ignore

        # Statistics
        self.redirected_tasks = 0
        self.total_tasks = 0

    def _log_info(self, msg, *args, **kwargs):
        if self.log_func:
            self.log_func(msg, *args, **kwargs)

    def _get_standard_bucket_index(self, task_length):
        """Return the standard bucket index for the given request length."""
        for bucket_idx, bucket in self.buckets.items():
            if bucket.min_length <= task_length < bucket.max_length:
                return bucket_idx
        # Fall back to the last bucket if the length is outside every range
        return self.num_buckets - 1

    def _get_neighbor_indices(self, bucket_idx):
        """Return the left and right neighbor indices of the given bucket."""
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
        Compute the affinity factor between the task length and the neighbor
        bucket (0.0 to 1.0). 1.0 means right next to the neighbor bucket, 0.0
        means far away from it.
        """
        neighbor_bucket = self.buckets[neighbor_bucket_idx]
        neighbor_bucket_min = neighbor_bucket.min_length
        neighbor_bucket_max = neighbor_bucket.max_length

        if neighbor_bucket_min < task_length < neighbor_bucket_max:
            raise RuntimeError("task_length must be outside the neighbor bucket range")

        neighbor_bucket_center = (neighbor_bucket_min + neighbor_bucket_max) / 2.0
        neighbor_bucket_half_width = (neighbor_bucket_max - neighbor_bucket_min) / 2.0

        distance_to_center = abs(task_length - neighbor_bucket_center)

        # The closer to the neighbor bucket boundary, the closer the affinity to 1
        if neighbor_bucket_half_width > 0:
            # Relative distance from the neighbor bucket half-width
            normalized_distance = (distance_to_center - neighbor_bucket_half_width) / neighbor_bucket_half_width
            # Exponential decay, e.g. normalized_distance=0.1, affinity_strength=1.0 -> 0.9
            neighbor_affinity = math.exp(-self.affinity_strength * normalized_distance)
        else:
            neighbor_affinity = 1.0  # Safeguard; unreachable in practice

        # Clamp to [0, 1]
        return max(0.0, min(neighbor_affinity, 1.0))

    def _calculate_redirect_probability(self, task_length, standard_bucket_idx, neighbor_bucket_idx):
        """
        Compute the probability of redirecting to the neighbor bucket based on
        the load gap and length affinity.
        """
        standard_load = self.buckets[standard_bucket_idx].total_load
        neighbor_load = self.buckets[neighbor_bucket_idx].total_load

        # --- 1. Base probability from the load gap ---
        if standard_load <= 0:
            load_probability = 0.0  # No load in the standard bucket -> no redirect
        else:
            load_ratio = neighbor_load / max(standard_load, 1e-9)  # Guard against division by zero
            # The smaller the neighbor load relative to the standard load, the
            # higher the redirect probability
            # e.g. neighbor/standard = 5/6, sensitivity=1.0 -> 1/6 ≈ 0.1667
            load_probability = 1 - load_ratio**self.sensitivity
            load_probability = max(0.0, min(load_probability, 1))

        # --- 2. Length affinity factor ---
        affinity_factor = self._calculate_length_affinity(task_length, neighbor_bucket_idx)

        # --- 3. Final probability: load gap * length affinity ---
        # A large load gap is suppressed when the request length is far from the
        # neighbor bucket; a modest gap can still redirect when it is close.
        final_probability = load_probability * affinity_factor

        return final_probability

    def dispatch_single_task(self, task_id: AnyStr, task_length: int, task_load):
        return self.dispatch_task(Task(task_id, task_length, task_load))

    def dispatch_task(self, cur_task):
        """
        Assign a bucket to a new request, considering dynamic load balancing and
        length affinity.
        """
        self.total_tasks += 1
        standard_bucket_idx = self._get_standard_bucket_index(cur_task.length)

        neighbor_indices = self._get_neighbor_indices(standard_bucket_idx)

        best_neighbor_idx = None
        best_redirect_prob = 0.0

        # Pick the neighbor with the highest redirect probability
        for neighbor_idx in neighbor_indices:
            # Only consider neighbors with lower load
            if self.buckets[neighbor_idx].total_load < self.buckets[standard_bucket_idx].total_load:
                prob = self._calculate_redirect_probability(cur_task.length, standard_bucket_idx, neighbor_idx)
                if prob > best_redirect_prob:
                    best_redirect_prob = prob
                    best_neighbor_idx = neighbor_idx

        # Decide the final bucket
        final_bucket_idx = standard_bucket_idx
        if best_neighbor_idx is not None and best_redirect_prob > 0:
            if self.base_probability_threshold < best_redirect_prob:
                final_bucket_idx = best_neighbor_idx
                self.redirected_tasks += 1
                self._log_info(
                    f"{cur_task} redirected from bucket {standard_bucket_idx} to {final_bucket_idx}"
                    f"(prob={best_redirect_prob:.4f})"
                )

        # Bookkeeping on the chosen bucket
        self.buckets[final_bucket_idx].task_count += 1
        self.buckets[final_bucket_idx].total_load += cur_task.load
        cur_task.bucket_idx = final_bucket_idx

        if cur_task.id in self.tasks:
            raise RuntimeError(f"Task {cur_task.id} is existed!")
        else:
            self.tasks[cur_task.id] = cur_task

        return final_bucket_idx, cur_task

    def release_task(self, task_id):
        """Release the load of a request."""
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
    """Dispatch requests by load only, with no standard bucket."""

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
