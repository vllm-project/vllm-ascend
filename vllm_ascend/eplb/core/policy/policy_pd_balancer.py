from abc import abstractmethod
from collections import defaultdict

import numpy as np
from vllm.logger import logger
import heapq

class RequestManager:
    def __init__(self):
        self.m2 = {}  # req_id -> idx (从 1 开始)
        self.free_idxs = []  # 最小堆，回收的索引
        self.next_new_idx = 1  # 初始索引设为 1，0 留给 PAD

    def update_and_copy(self, m1, target_tensor):
        """
        m1: {req_id: token_num}
        target_tensor: 预先分配好的固定大小张量 [MAX_TOTAL_TOKENS]
        """
        # --- 1. 先删除：同步活跃请求并回收索引 ---
        active_req_ids = set(m1.keys())
        expired_req_ids = [rid for rid in self.m2 if rid not in active_req_ids]
        
        for rid in expired_req_ids:
            idx = self.m2.pop(rid)
            heapq.heappush(self.free_idxs, idx)

        # --- 2. 再添加：分配新索引 (从 1 开始) ---
        sum_tokens = 0
        for rid in m1.keys():
            if rid not in self.m2:
                if self.free_idxs:
                    allocated_idx = heapq.heappop(self.free_idxs)
                else:
                    allocated_idx = self.next_new_idx
                    self.next_new_idx += 1
                self.m2[rid] = allocated_idx
            sum_tokens += m1[rid]

        # --- 3. 原地拷贝：填充到固定大小张量中 ---
        # 首先将 target_tensor 重置为 0 (PAD)
        target_tensor.fill(0)
        
        current_ptr = 0
        for rid, num in m1.items():
            target_idx = self.m2[rid]
            # 执行原地拷贝到传入的张量
            target_tensor[current_ptr : current_ptr + num] = target_idx
            current_ptr += num
            
        return target_tensor, self.m2