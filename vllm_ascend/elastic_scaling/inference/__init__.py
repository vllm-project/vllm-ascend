'''
Monkey-patching elastic code
'''
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm_ascend.elastic_scaling.inference.loader import zero_copy_model
DefaultModelLoader.zero_copy_model = zero_copy_model

from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm_ascend.elastic_scaling.inference.gpu_model_runner import get_supported_tasks
GPUModelRunner.get_supported_tasks = get_supported_tasks

from vllm_ascend.worker.worker import NPUWorker
from vllm_ascend.elastic_scaling.inference.npu_worker import reload_model
NPUWorker.reload_model = reload_model

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.elastic_scaling.inference.npu_model_runner import (load_model, initialize_kv_cache, 
                            __init__) 
NPUModelRunner.__init__ = __init__
NPUModelRunner.load_model = load_model
NPUModelRunner.initialize_kv_cache = initialize_kv_cache

from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE, AscendUnquantizedFusedMoEMethod
from vllm_ascend.elastic_scaling.inference.fused_moe import __init__, apply 
AscendFusedMoE.__init__ = __init__
AscendUnquantizedFusedMoEMethod.apply = apply

from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm_ascend.elastic_scaling.inference.layer import __init__
FusedMoE.__init__ = __init__

from vllm.v1.engine.async_llm import AsyncLLM
from vllm_ascend.elastic_scaling.inference.scale.async_llm import reload_models, reload_kvcache
AsyncLLM.reload_models = reload_models
AsyncLLM.reload_kvcache = reload_kvcache

from vllm.v1.engine.core_client import AsyncMPClient
from vllm_ascend.elastic_scaling.inference.scale.core_client import (reload_models_async, reload_kvcache_async)
AsyncMPClient.reload_models_async = reload_models_async
AsyncMPClient.reload_kvcache_async = reload_kvcache_async

from vllm.v1.engine.core import EngineCore
from vllm_ascend.elastic_scaling.inference.scale.core import reload_model, reload_kvcache
EngineCore.reload_model = reload_model
EngineCore.reload_kvcache = reload_kvcache

from vllm.v1.executor.abstract import Executor
from vllm_ascend.elastic_scaling.inference.scale.abstract import reload_model, reload_kvcache
Executor.reload_model = reload_model
Executor.reload_kvcache = reload_kvcache
