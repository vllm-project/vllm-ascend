import bisect


class DynamicSpeculativeDecodingManager:
    """Build a batch-size schedule from offline profiling results."""

    def __init__(
        self,
        *,
        batch_stats: dict[int, dict[int, float]],
        acceptance_rate_per_pos: list[float],
        vllm_max_batch_size: int,
    ):
        if not batch_stats:
            raise ValueError("batch_stats is required for dynamic speculative decoding")
        if not acceptance_rate_per_pos:
            raise ValueError("acceptance_rate_per_pos is required")

        self.batch_stats = batch_stats
        self.vllm_max_batch_size = vllm_max_batch_size
        self.max_num_speculative_tokens = len(acceptance_rate_per_pos)
        self.acceptance_rate_per_pos = acceptance_rate_per_pos
        self.available_batch_sizes = sorted(batch_stats.keys())

        # Sanity check
        if not all(0.0 <= a <= 1.0 for a in self.acceptance_rate_per_pos):
            raise ValueError("acceptance rates must be between 0 and 1")
        if 1 not in self.batch_stats:
            raise ValueError(f"BS 1 not found in {self.batch_stats.keys()}")
        if vllm_max_batch_size not in self.batch_stats:
            raise ValueError(f"max BS {vllm_max_batch_size} not found in {self.batch_stats.keys()}")

        for bs in self.available_batch_sizes:
            if bs <= 0:
                raise ValueError("batch sizes must be positive")
            if 0 not in self.batch_stats[bs]:
                raise ValueError(f"BS {bs} must have draft 0 stats")
            if 1 not in self.batch_stats[bs]:
                raise ValueError(f"BS {bs} must have draft 1 stats")
            if sorted(self.batch_stats[bs]) != list(self.batch_stats[bs]):
                raise ValueError(f"BS {bs} draft keys must be sorted")

        self._optimal_k_table: dict[int, int] = {}
        self._sorted_bs_keys: list[int] = []
        self.update_optimal_num_speculative_tokens()

    def update_optimal_num_speculative_tokens(self) -> None:
        self._optimal_k_table = {}
        for bs in self.available_batch_sizes:
            stats = self.batch_stats[bs]
            best_k = 0
            best_goodput = 1.0 / stats.get(0, float("inf"))  # K=0: AL=1

            for k in sorted(stats.keys()):
                if k == 0:
                    continue
                itl = stats[k]
                if itl <= 0:
                    continue
                al = self._compute_accepted_length(k)
                goodput = al / itl  # https://arxiv.org/pdf/2406.14066  Goodput
                if goodput > best_goodput:
                    best_goodput = goodput
                    best_k = k

            self._optimal_k_table[bs] = best_k
        self._sorted_bs_keys = sorted(self._optimal_k_table)

    def get_optimal_num_speculative_tokens(self, batch_size: int) -> int:
        """Return optimal K for a given batch size via binary search."""
        idx = bisect.bisect_right(self._sorted_bs_keys, batch_size) - 1
        if idx < 0:
            return self._optimal_k_table[self._sorted_bs_keys[0]]
        return self._optimal_k_table[self._sorted_bs_keys[idx]]

    def get_num_speculative_tokens_per_batch_size(self) -> list[list[int]]:
        """Return merged inclusive batch-size ranges for the optimal K table."""
        sampled_batch_sizes = [bs for bs in self._sorted_bs_keys if bs <= self.vllm_max_batch_size]
        ranges: list[list[int]] = []

        for index, range_start in enumerate(sampled_batch_sizes):
            if index + 1 < len(sampled_batch_sizes):
                range_end = sampled_batch_sizes[index + 1] - 1
            else:
                range_end = self.vllm_max_batch_size

            optimal_k = self._optimal_k_table[range_start]
            if ranges and ranges[-1][2] == optimal_k:
                ranges[-1][1] = range_end
            else:
                ranges.append([range_start, range_end, optimal_k])

        return ranges

    def _compute_accepted_length(self, k: int) -> float:
        """Compute expected accepted length for given K using unconditional rates."""
        al = 1.0  # base token from target model
        for i in range(min(k, len(self.acceptance_rate_per_pos))):
            al += self.acceptance_rate_per_pos[i]
        return al
