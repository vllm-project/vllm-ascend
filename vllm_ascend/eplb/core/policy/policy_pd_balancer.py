from abc import abstractmethod
from collections import defaultdict

import numpy as np
from vllm.logger import logger
import heapq

class RequestManager:
    def __init__(self):
        self.m2 = {}  
        self.free_idxs = [] 
        self.next_new_idx = 1  

    def update_and_copy(self, m1, target_tensor):
        """
        m1: {req_id: token_num}
        target_tensor: buffer [MAX_TOTAL_TOKENS]
        """
        active_req_ids = set(m1.keys())
        expired_req_ids = [rid for rid in self.m2 if rid not in active_req_ids]
        
        for rid in expired_req_ids:
            idx = self.m2.pop(rid)
            heapq.heappush(self.free_idxs, idx)

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

        target_tensor.fill(0)
        
        current_ptr = 0
        for rid, num in m1.items():
            target_idx = self.m2[rid]
            target_tensor[current_ptr : current_ptr + num] = target_idx
            current_ptr += num
            
        return target_tensor, self.m2, expired_req_ids