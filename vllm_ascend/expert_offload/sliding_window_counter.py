"""Sliding Window Counter for MoE Expert Hotness Tracking."""

from collections import Counter, deque


class SlidingWindowCounter:
    """按 layer 维护滑动窗口计数器，统计当前请求的 expert 激活频率。

    Args:
        num_layers: MoE 层的数量
        window_size: 滑动窗口大小，控制统计范围
    """

    def __init__(self, num_layers: int, window_size: int = 200):
        self._window_size = window_size
        self._windows: dict[int, deque[int]] = {
            i: deque(maxlen=window_size) for i in range(num_layers)
        }
        self._counters: dict[int, Counter] = {
            i: Counter() for i in range(num_layers)
        }

    def record(self, layer_idx: int, expert_ids: list[int]) -> None:
        """记录一次 expert 激活事件。

        Args:
            layer_idx: MoE 层索引
            expert_ids: 该层激活的 expert ID 列表
        """
        window = self._windows[layer_idx]
        old_count = len(window)
        excess = old_count + len(expert_ids) - self._window_size

        if excess > 0:
            for eid in list(window)[:excess]:
                self._counters[layer_idx][eid] -= 1
                if self._counters[layer_idx][eid] <= 0:
                    del self._counters[layer_idx][eid]

        window.extend(expert_ids)
        self._counters[layer_idx].update(expert_ids)

    def get_topk_hot_experts(
        self,
        layer_idx: int,
        top_k: int,
        min_threshold: int = 2,
    ) -> list[int]:
        """获取热点 expert 编号，按频率排序。

        Args:
            layer_idx: MoE 层索引
            top_k: 返回前 k 个热点 expert
            min_threshold: 最小激活次数阈值，低于此值的 expert 不返回

        Returns:
            按频率降序排列的 expert ID 列表
        """
        counter = self._counters[layer_idx]
        if not counter:
            return []

        hot = [
            (eid, cnt) for eid, cnt in counter.items() if cnt >= min_threshold
        ]
        hot.sort(key=lambda x: x[1], reverse=True)
        return [eid for eid, _ in hot[:top_k]]

    def reset(self, layer_idx: int | None = None) -> None:
        """重置计数器。

        Args:
            layer_idx: 如果为 None，重置所有层；否则只重置指定层
        """
        if layer_idx is None:
            for w, c in zip(self._windows.values(), self._counters.values()):
                w.clear()
                c.clear()
        else:
            self._windows[layer_idx].clear()
            self._counters[layer_idx].clear()
