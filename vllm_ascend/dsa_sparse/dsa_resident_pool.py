"""DSA HBM resident 请求行与逐层 token->slot 持久状态资源池。

本模块为每个活跃请求分配固定 resident pool row，并额外保留一个 graph PAD
row。逐层 ``cache_slots`` 是 LIDU 跨 decode step 原址维护的权威映射：前
``W-1`` 列保存 ``原序列 token position -> resident logical slot``，末列保存
0/负预算/正预算三态 metadata。请求结束、preempt、row 复用和 graph capture
dummy 清理都必须与 pool row 生命周期同步。

它只管理 resident 元数据张量和 Python 行所有权，不负责 DRAM 满块分配、
scheduler HBM admission、KSC payload IO 或本轮 SFA 输出构造。
"""

from __future__ import annotations

from collections.abc import Hashable

import torch

from vllm_ascend.dsa_sparse.dsa_types import (
    DSA_LIDU_CACHE_ROW_ALIGNMENT,
    DSA_LIDU_TOKEN_CAPACITY,
)


class DSAResidentTokenPool:
    """Per-worker request-row ownership and per-layer LIDU persistent state."""

    def __init__(
        self,
        max_reqs: int,
        num_layers: int,
        max_tokens: int,
        max_model_len: int,
        block_size: int,
        *,
        device: torch.device | str | None = None,
    ):
        if max_reqs <= 0:
            raise ValueError(f"max_reqs must be positive, got {max_reqs}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if max_model_len <= 0:
            raise ValueError(
                f"max_model_len must be positive, got {max_model_len}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.max_reqs = int(max_reqs)
        # 图 replay 允许实际 batch 向上 padding 到 captured batch。PAD 行仍
        # 会经过统一的 LIDU -> KSC -> SFA-Offload 图，因此需要一个合法但
        # 永远不会分配给真实请求的 resident row，避免哨兵行误指向真实请求。
        self.padding_pool_index = self.max_reqs
        self._storage_rows = self.max_reqs + 1
        self.num_layers = int(num_layers)
        self.max_tokens = int(max_tokens)
        self.max_model_len = int(max_model_len)
        self.block_size = int(block_size)
        if self.max_model_len > DSA_LIDU_TOKEN_CAPACITY:
            raise ValueError(
                "DSA max_model_len exceeds the current LIDU token-position "
                f"capacity: max_model_len={self.max_model_len}, "
                f"capacity={DSA_LIDU_TOKEN_CAPACITY}")
        # LIDU 把最后一列当作带符号预算 metadata。按列数而不是 token
        # capacity 对齐，避免相邻请求行共享 DCache line；只有达到 18-bit
        # token position 上限时退回精确宽度，防止 padding 本身越过上限。
        raw_row_width = self.max_model_len + 1
        aligned_row_width = (
            (raw_row_width + DSA_LIDU_CACHE_ROW_ALIGNMENT - 1)
            // DSA_LIDU_CACHE_ROW_ALIGNMENT
            * DSA_LIDU_CACHE_ROW_ALIGNMENT)
        if aligned_row_width - 1 <= DSA_LIDU_TOKEN_CAPACITY:
            self.cache_row_width = aligned_row_width
        else:
            self.cache_row_width = raw_row_width
        self.token_capacity = self.cache_row_width - 1
        self.cache_metadata_index = self.cache_row_width - 1
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self._free_indices = list(range(self.max_reqs))
        self._request_to_index: dict[Hashable, int] = {}
        self._request_target_budgets: dict[Hashable, int] = {}
        # LIDU 跨 step 原址维护的权威映射。前 W-1 列保存
        # original token position -> resident logical slot，最后一列保存
        # 0/负预算/正预算三态 metadata。
        self._cache_slots = torch.full(
            (self.num_layers, self._storage_rows, self.cache_row_width),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self._cache_slots[:, :, self.cache_metadata_index].zero_()

    def acquire(self, request_id: Hashable,
                target_budget_tokens: int | None = None) -> int:
        current = self._request_to_index.get(request_id)
        if current is not None:
            if target_budget_tokens is not None:
                self.prepare_request(
                    request_id,
                    target_budget_tokens=target_budget_tokens,
                )
            return current
        if not self._free_indices:
            raise RuntimeError(
                "No free DSA resident metadata slot is available")

        pool_idx = self._free_indices.pop(0)
        self._request_to_index[request_id] = pool_idx
        self._clear_index(pool_idx)
        if target_budget_tokens is not None:
            self.prepare_request(
                request_id,
                target_budget_tokens=target_budget_tokens,
            )
        return pool_idx

    def release(self, request_id: Hashable) -> None:
        pool_idx = self._request_to_index.pop(request_id, None)
        if pool_idx is None:
            return
        self._request_target_budgets.pop(request_id, None)
        self._clear_index(pool_idx)
        self._free_indices.insert(0, pool_idx)

    def get_index(self, request_id: Hashable) -> int | None:
        return self._request_to_index.get(request_id)

    def prepare_request(
        self,
        request_id: Hashable,
        *,
        target_budget_tokens: int,
    ) -> None:
        """为请求所有 layer 写入一次 first-fill 负预算。"""
        pool_idx = self._require_index(request_id)
        target_budget_tokens = int(target_budget_tokens)
        if target_budget_tokens <= 0 or target_budget_tokens > self.max_tokens:
            raise ValueError(
                "DSA target resident budget is outside pool capacity: "
                f"target={target_budget_tokens}, capacity={self.max_tokens}")
        existing = self._request_target_budgets.get(request_id)
        if existing is not None:
            if existing != target_budget_tokens:
                raise RuntimeError(
                    "DSA request target resident budget changed after row "
                    f"binding: request={request_id!r}, old={existing}, "
                    f"new={target_budget_tokens}")
            return
        self._cache_slots[:, pool_idx,
                          self.cache_metadata_index].fill_(
                              -target_budget_tokens)
        self._request_target_budgets[request_id] = target_budget_tokens

    def get_cache_slots(self, *, layer_id: int) -> torch.Tensor:
        """返回 ``[storage_rows, W]`` 的逐层 LIDU mutable 状态。"""
        return self._cache_slots[self._normalize_layer_id(layer_id)]

    def clear_cache_slots_prefix(self, row_count: int) -> None:
        row_count = min(max(int(row_count), 0), self.max_reqs)
        if row_count == 0:
            return
        self._cache_slots[:, :row_count].fill_(-1)
        self._cache_slots[:, :row_count,
                          self.cache_metadata_index].zero_()

    def seed_cache_slots_prefix(self, row_count: int,
                                target_budget_tokens: int) -> None:
        """为 graph capture dummy 行写入合法 first-fill 状态。"""
        row_count = min(max(int(row_count), 0), self.max_reqs)
        target_budget_tokens = int(target_budget_tokens)
        if row_count == 0:
            return
        if target_budget_tokens <= 0 or target_budget_tokens > self.max_tokens:
            raise ValueError(
                f"Invalid graph dummy resident budget {target_budget_tokens}")
        self.clear_cache_slots_prefix(row_count)
        self._cache_slots[:, :row_count,
                          self.cache_metadata_index].fill_(
                              -target_budget_tokens)

    def clear_request(self, request_id: Hashable) -> None:
        # Preempt keeps the pool row ownership but invalidates all token-slot
        # mappings. Drop the initialization ledger as well, so resume enters
        # first-fill again and prepare_request() restores the negative budget
        # metadata instead of mistaking a zeroed row for an initialized row.
        pool_idx = self._require_index(request_id)
        self._request_target_budgets.pop(request_id, None)
        self._clear_index(pool_idx)

    def _clear_index(self, pool_idx: int) -> None:
        self._cache_slots[:, pool_idx].fill_(-1)
        self._cache_slots[:, pool_idx, self.cache_metadata_index].zero_()

    def _require_index(self, request_id: Hashable) -> int:
        pool_idx = self._request_to_index.get(request_id)
        if pool_idx is None:
            raise KeyError(f"DSA request {request_id!r} has no resident slot")
        return pool_idx

    def _normalize_layer_id(self, layer_id: int) -> int:
        layer_id = int(layer_id)
        if layer_id < 0 or layer_id >= self.num_layers:
            raise IndexError(
                f"layer_id {layer_id} out of range [0, {self.num_layers})")
        return layer_id
