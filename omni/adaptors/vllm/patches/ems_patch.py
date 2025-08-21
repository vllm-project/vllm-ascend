def patch_ems():
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.multiproc_executor import Executor
    from omni.adaptors.vllm.ems.ems_interface import _pre_cc_handle, step, load_kv_cache
    EngineCore._pre_cc_handle = _pre_cc_handle
    EngineCore.step = step
    Executor.load_kv_cache = load_kv_cache
    print("++++++++++++++++++++++patch_ems++++++++++++++++++++++++++++")


patch_ems()