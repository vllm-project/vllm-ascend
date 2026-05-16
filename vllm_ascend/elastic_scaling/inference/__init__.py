"""
Monkey-patching elastic scaling behavior into vLLM components.

This module centralizes all runtime monkey patches required for
elastic scaling on Ascend / NPU backends.
"""

from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.v1.executor.abstract import Executor
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.elastic_scaling.inference.fused_moe import (
    __init__ as ascend_fused_moe_init,
)
from vllm_ascend.elastic_scaling.inference.gpu_model_runner import (
    get_supported_tasks,
)
from vllm_ascend.elastic_scaling.inference.layer import (
    __init__ as fused_moe_layer_init,
)
from vllm_ascend.elastic_scaling.inference.loader import zero_copy_model
from vllm_ascend.elastic_scaling.inference.npu_model_runner import (
    __init__ as npu_model_runner_init,
)
from vllm_ascend.elastic_scaling.inference.npu_model_runner import (
    initialize_kv_cache,
    load_model,
)
from vllm_ascend.elastic_scaling.inference.npu_worker import reload_model
from vllm_ascend.elastic_scaling.inference.scale.abstract import (
    reload_kvcache as executor_reload_kvcache,
)
from vllm_ascend.elastic_scaling.inference.scale.abstract import (
    reload_model as executor_reload_model,
)
from vllm_ascend.elastic_scaling.inference.scale.async_llm import (
    reload_kvcache,
    reload_models,
)
from vllm_ascend.elastic_scaling.inference.scale.core import (
    reload_kvcache as engine_reload_kvcache,
)
from vllm_ascend.elastic_scaling.inference.scale.core import (
    reload_model as engine_reload_model,
)
from vllm_ascend.elastic_scaling.inference.scale.core_client import (
    reload_kvcache_async,
    reload_models_async,
)
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.worker import NPUWorker

# ======================
# Monkey patch entrypoint
# ======================


def apply_elastic_monkey_patches() -> None:
    """Apply all elastic-scaling monkey patches."""

    # Model loading / execution
    DefaultModelLoader.zero_copy_model = zero_copy_model
    GPUModelRunner.get_supported_tasks = get_supported_tasks
    NPUWorker.reload_model = reload_model

    # NPU model runner
    NPUModelRunner.__init__ = npu_model_runner_init
    NPUModelRunner.load_model = load_model
    NPUModelRunner.initialize_kv_cache = initialize_kv_cache

    # Fused MoE
    AscendFusedMoE.__init__ = ascend_fused_moe_init
    FusedMoE.__init__ = fused_moe_layer_init

    # Async engine
    AsyncLLM.reload_models = reload_models
    AsyncLLM.reload_kvcache = reload_kvcache

    # Async MP client
    AsyncMPClient.reload_models_async = reload_models_async
    AsyncMPClient.reload_kvcache_async = reload_kvcache_async

    # Engine core
    EngineCore.reload_model = engine_reload_model
    EngineCore.reload_kvcache = engine_reload_kvcache

    # Executor
    Executor.reload_model = executor_reload_model
    Executor.reload_kvcache = executor_reload_kvcache


# Apply patches eagerly on import
apply_elastic_monkey_patches()
