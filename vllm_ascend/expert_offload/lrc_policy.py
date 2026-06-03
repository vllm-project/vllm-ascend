"""Local-routing-consistency policy for expert offload decode paging."""

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class LRCLayerState:
    recent_queue: deque[tuple[int, ...]] = field(default_factory=deque)
    freq: list[int] = field(default_factory=list)
    ema: list[float] = field(default_factory=list)
    router_score: list[float] = field(default_factory=list)
    last_used: list[int] = field(default_factory=list)
    step: int = 0


class LRCExpertCachePolicy:
    """Online approximation of SCH using recent frequency and EMA hotness.

    The policy is intentionally CPU-only. ExpertOffloadManager owns the actual
    CPU-to-NPU copies; this class only decides which resident expert should be
    evicted when a miss needs a device slot.
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        cache_size: int,
        topk: int,
        recent_window: int = 32,
        ema_beta: float = 0.9,
        recent_weight: float = 1.0,
        ema_weight: float = 0.5,
        router_weight: float = 0.3,
        age_weight: float = 0.01,
    ) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if cache_size < 1:
            raise ValueError("cache_size must be >= 1")
        if topk < 1:
            raise ValueError("topk must be >= 1")
        if recent_window < 1:
            raise ValueError("recent_window must be >= 1")
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError("ema_beta must be in [0, 1)")

        self.num_experts = num_experts
        self.cache_size = cache_size
        self.topk = topk
        self.recent_window = recent_window
        self.ema_beta = ema_beta
        self.recent_weight = recent_weight
        self.ema_weight = ema_weight
        self.router_weight = router_weight
        self.age_weight = age_weight
        self.layer_states = [self._new_state() for _ in range(num_layers)]

    def _new_state(self) -> LRCLayerState:
        return LRCLayerState(
            recent_queue=deque(),
            freq=[0 for _ in range(self.num_experts)],
            ema=[0.0 for _ in range(self.num_experts)],
            router_score=[0.0 for _ in range(self.num_experts)],
            last_used=[-1 for _ in range(self.num_experts)],
        )

    def observe(
        self,
        layer_idx: int,
        topk_ids: Iterable[Iterable[int]],
        router_scores: Iterable[Iterable[float]] | None = None,
    ) -> set[int]:
        """Update per-layer routing statistics and return unique routed ids."""
        state = self.layer_states[layer_idx]
        score_rows = router_scores if router_scores is not None else []
        unique: set[int] = set()

        for row_index, row in enumerate(topk_ids):
            experts = tuple(self._valid_expert(eid) for eid in row)
            unique.update(experts)
            state.step += 1

            hit_set = set(experts)
            for eid in range(self.num_experts):
                hit = 1.0 if eid in hit_set else 0.0
                state.ema[eid] = self.ema_beta * state.ema[eid] + (1.0 - self.ema_beta) * hit

            state.recent_queue.append(experts)
            for eid in experts:
                state.freq[eid] += 1
                state.last_used[eid] = state.step

            if router_scores is not None:
                score_row = list(score_rows[row_index])
                for eid, score in zip(experts, score_row, strict=False):
                    state.router_score[eid] = float(score)

            while len(state.recent_queue) > self.recent_window:
                old_experts = state.recent_queue.popleft()
                for old_eid in old_experts:
                    state.freq[old_eid] -= 1

        return unique

    def choose_victim(
        self,
        layer_idx: int,
        slot_owner: dict[int, int],
        protected: set[int],
        loading: set[int] | None = None,
    ) -> int | None:
        """Return resident expert id with the lowest predicted hotness."""
        loading = loading or set()
        candidates = [
            eid for eid in slot_owner.values()
            if eid not in protected and eid not in loading
        ]
        if not candidates:
            candidates = [eid for eid in slot_owner.values() if eid not in protected]
        if not candidates:
            return None

        return min(candidates, key=lambda eid: self._victim_key(layer_idx, eid))

    def hotness(self, layer_idx: int, expert_id: int) -> float:
        state = self.layer_states[layer_idx]
        age = 0 if state.last_used[expert_id] < 0 else state.step - state.last_used[expert_id]
        return (
            self.recent_weight * state.freq[expert_id]
            + self.ema_weight * state.ema[expert_id]
            + self.router_weight * state.router_score[expert_id]
            - self.age_weight * age
        )

    def layer_step(self, layer_idx: int) -> int:
        return self.layer_states[layer_idx].step

    def _victim_key(self, layer_idx: int, expert_id: int) -> tuple[float, int, int]:
        state = self.layer_states[layer_idx]
        return (self.hotness(layer_idx, expert_id), state.last_used[expert_id], expert_id)

    def _valid_expert(self, expert_id: int) -> int:
        expert_id = int(expert_id)
        if expert_id < 0 or expert_id >= self.num_experts:
            raise ValueError(f"expert id out of range: {expert_id}")
        return expert_id
