from abc import abstractmethod

from vllm.v1.core.sched.interface import SchedulerInterface


class SchedulerInterfaceEx:
    @abstractmethod
    def release_kv_cache(self, session_id: str, before_token_ids: list[int], released_token_index: int) -> int:
        raise NotImplementedError


def register_scheduler_interface():
    SchedulerInterface.release_kv_cache = SchedulerInterfaceEx.release_kv_cache
