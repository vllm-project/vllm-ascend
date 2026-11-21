import vllm.distributed.utils

def apply_sched_yield_patch():
    """Forces time.sleep(0) instead of os.sched_yield() to reduce CPU contention."""
  
    vllm.distributed.utils.USE_SCHED_YIELD = False