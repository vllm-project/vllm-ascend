from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.scheduler import Scheduler


class SchedulerEx(Scheduler):
    def release_kv_cache(self, session_id: str, block_hashes: list[BlockHash]) -> int:
        return self.kv_cache_manager.release_kv_cache(session_id, block_hashes)


def register_scheduler():
    Scheduler.release_kv_cache = SchedulerEx.release_kv_cache
