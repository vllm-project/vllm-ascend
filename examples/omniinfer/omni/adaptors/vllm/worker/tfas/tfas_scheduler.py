from vllm.v1.core.sched.scheduler import Scheduler
import time
from collections import deque
from vllm.v1.request import Request, RequestStatus
import math
from vllm.v1.structured_output import StructuredOutputManager
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.logger import logger

class TFASScheduler(Scheduler):
    DEFAULT_TFAS_CONFIG = {
    "adjust_param": 3.60,
    "token_budget": 9154,
    }

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config, 
            kv_cache_config, 
            structured_output_manager, 
            mm_registry, 
            include_finished_set, 
            log_stats)
        if (self.vllm_config.kv_transfer_config is not None and 
            self.vllm_config.kv_transfer_config.is_kv_consumer):
            raise ValueError("TFASScheduler does not support KV consumer mode.")
        additional_cfg = vllm_config.additional_config
        tfas_config = additional_cfg.get("tfas_scheduler_config", None)
        if tfas_config is None:
            logger.warning("Missing tfas_scheduler_config. Using default config.")
            tfas_config = self.DEFAULT_TFAS_CONFIG.copy()
        else:
            # 确保是字典
            if not isinstance(tfas_config, dict):
                raise TypeError("tfas_scheduler_config must be a dict.")

            # 补充缺失的字段
            for field, default_val in self.DEFAULT_TFAS_CONFIG.items():
                if field not in tfas_config:
                    logger.warning(f"Missing '{field}', using default: {default_val}")
                    tfas_config[field] = default_val

        # 从字典中读取配置
        self.tfas_adjust_param = tfas_config["adjust_param"]
        self.tfas_waiting_time_out = 60
        self.tfas_token_budget = tfas_config["token_budget"]


        logger.info(
            f"TFASScheduler enabled (adjust_param={self.tfas_adjust_param}"
            f", token_budget={self.tfas_token_budget})"
        )

            
    def schedule(self):
        now_time = time.time()
        self.waiting = deque(
            sorted(self.waiting, key=lambda req: self._length_sort_time_decay(
                now_time, req))
        )
        upper_bound = self._compute_upper_bound(self.waiting)
        upper_bound = self._accumulate_until_bound(
            self.waiting, upper_bound)
        self.max_num_scheduled_tokens = min(
            self.scheduler_config.max_num_batched_tokens, upper_bound)
        scheduler_output = super().schedule()
        return scheduler_output
            
    def _accumulate_until_bound(self, queue: deque[Request], bound):
        total_request_len = 0
        for request in queue:
            if (request.status == RequestStatus.WAITING_FOR_REMOTE_KVS and 
                    request.request_id not in self.finished_recving_kv_req_ids):
                continue
            if (request.status == RequestStatus.WAITING_FOR_FSM):
                structured_output_req = request.structured_output_request
                if (not structured_output_req or 
                    not structured_output_req.grammar):
                    continue
            total_request_len += request.num_tokens_with_spec
            if total_request_len > bound:
                return total_request_len+1
        return bound
    
    def _compute_upper_bound(self, waiting_queue: deque[Request]) -> int:
        """
        Compute the token budget upper bound based on the waiting queue.
        """
        tokens_in_waiting_queue = sum(
            req.num_prompt_tokens for req in waiting_queue)
        req_in_waiting_queue = tokens_in_waiting_queue / 1024

        bound1 = self.tfas_token_budget
        bound2 = int(
            math.sqrt(
                req_in_waiting_queue * self.tfas_adjust_param
            ) * 1024
        )
        alpha = 0.9
        bound2 = int(alpha*bound2)
        return max(bound1, bound2)

    def _length_sort_time_decay(self, now_time: float, request: Request) -> int:
        """
        Sort key function: apply time decay.
        Requests waiting longer than the timeout get the lowest priority (key=0).
        """
        if now_time - request.arrival_time > self.tfas_waiting_time_out:
            return 0
        else:
            return request.num_prompt_tokens

class TFASProfilerScheduler(Scheduler):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config, 
            kv_cache_config, 
            structured_output_manager, 
            mm_registry, 
            include_finished_set, 
            log_stats)
        if (self.vllm_config.kv_transfer_config is not None and 
            self.vllm_config.kv_transfer_config.is_kv_consumer):
            raise ValueError(
                "TFASProfilerScheduler does not support KV consumer mode.")
        self.trigger_num = 0
        self.grow_frequency =20
        logger.info("TFASProfilerScheduler enabled"
                    f" (grow frequency={self.grow_frequency})")

    def schedule(self):
        self.trigger_num += 1
        self.max_num_running_reqs = min(
            self.trigger_num // self.grow_frequency + 1, 
            self.scheduler_config.max_num_seqs)
        logger.info("[TFASProfilerScheduler] tfas_max_num_seqs"
                    f"set to {self.max_num_running_reqs}")
        scheduler_output = super().schedule()
        return scheduler_output
