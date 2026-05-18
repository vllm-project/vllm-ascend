"""Expert Hotness Tracker for MoE Prefetch."""

from vllm_ascend.expert_offload.sliding_window_counter import SlidingWindowCounter


class ExpertHotnessTracker:
    """根据 sliding window counter 输出热点 expert 编号。

    维护每个 layer 上个 step 的 expert 编号，结合热点计数器
    计算 prefetch 并集。
    """

    def __init__(self, counter: SlidingWindowCounter, top_k_default: int = 8):
        """初始化热点追踪器。

        Args:
            counter: 滑动窗口计数器实例
            top_k_default: 默认返回的热点 expert 数量
        """
        self._counter = counter
        self._top_k_default = top_k_default
        self._prev_step_experts: dict[int, set[int]] = {}

    def record_step_experts(self, layer_idx: int, expert_ids: list[int]) -> None:
        """记录当前 step 激活的 expert，作为下一个 step 的历史参考。

        Args:
            layer_idx: MoE 层索引
            expert_ids: 该层当前激活的 expert ID 列表
        """
        self._prev_step_experts[layer_idx] = set(expert_ids)

    def get_prev_step_experts(self, layer_idx: int) -> set[int]:
        """获取 layer_idx 上个 step 的 expert 编号。

        Args:
            layer_idx: MoE 层索引

        Returns:
            上个 step 激活的 expert ID 集合
        """
        return self._prev_step_experts.get(layer_idx, set())

    def get_union_experts(
        self,
        layer_idx: int,
        source1_experts: list[int],
        hotness_top_k: int | None = None,
    ) -> set[int]:
        """获取 prefetch 并集。

        并集来源:
        - 来源 1: source1_experts (Layer N 激活的 expert)
        - 来源 2: Layer N 上个 step 的 expert 编号
        - 来源 3: Layer N 的热点 expert 编号 (基于滑动窗口统计)

        Args:
            layer_idx: Layer N 的索引
            source1_experts: Layer N 当前激活的 expert 编号列表
            hotness_top_k: 热点 expert 数量，默认使用 top_k_default

        Returns:
            需要预取的 expert ID 集合
        """
        if hotness_top_k is None:
            hotness_top_k = self._top_k_default

        union = set(source1_experts)

        union.update(self.get_prev_step_experts(layer_idx))

        hot_experts = self._counter.get_topk_hot_experts(
            layer_idx, top_k=hotness_top_k
        )
        union.update(hot_experts)

        return union

    def reset(self) -> None:
        """请求结束时重置，清理历史记录。"""
        self._prev_step_experts.clear()
