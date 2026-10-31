from functools import cache

import torch
import vllm
import vllm.v1.worker.utils as v1_worker_utils
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.v1.worker.gpu import model_runner as gpu_model_runner_v2

from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.patch.worker.patch_bind_kv_cache import (
    bind_kv_cache,
    bind_kv_cache_to_layers,
)
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.utils import prepare_ascend_kernel_block_sizes
from vllm_ascend.worker.v2.attn_utils import (
    _allocate_kv_cache,
    _reshape_kv_cache_v2,
    allocate_kv_cache_main,
    get_kv_cache_spec,
)


def _get_ascend_sfa_indexer_backend(_self):
    return AscendSFAIndexerBackend


# vLLM main (#63384) moved OLMoE to the Transformers backend, whose
# `vllm_attention_forward` bakes a layout-dependent `aten.view`. v0.32.0 still
# uses the native OLMoE implementation, where this mismatch cannot happen, so
# the release lane gets none of this patch.
if not vllm_version_is("0.32.0"):
    _VLLM_ATTENTION_IMPL = "vllm"

    @cache
    def _materialize_attention_layout(forward):
        """Wrap a Transformers attention forward so q/k/v arrive with a concrete layout.

        `vllm_attention_forward` rewrites q/k/v as `transpose(1, 2)` followed by
        `reshape(hidden, -1)`, and whether that reshape stays a view is decided
        at trace time from the *meta* strides of the RoPE output. ATen's meta
        for elementwise ops keeps the first operand's strides, while the
        AscendC kernels always return a contiguous tensor, so the traced layout
        can differ from the runtime layout and the baked `aten.view` then
        raises "view size is not compatible with input tensor's size and
        stride" (OLMoE LoRA with graph mode, vLLM #63384). An unconditional
        contiguous clone makes both layouts identical; `.contiguous()` cannot
        be used because it is a no-op for the already-"contiguous" fake tensor
        and disappears from the traced graph.
        """

        def forward_with_contiguous_layout(module, query, key, value, attention_mask, **kwargs):
            query, key, value = (
                x.transpose(1, 2).clone(memory_format=torch.contiguous_format).transpose(1, 2)
                for x in (query, key, value)
            )
            return forward(module, query, key, value, attention_mask, **kwargs)

        return forward_with_contiguous_layout

    _assign_attention_impl = AttentionInterface.register.__func__

    def _register_attention_impl(cls, name, forward):
        # vLLM imports its Transformers backend lazily while loading the model,
        # so intercept the registration instead of importing the backend (and
        # with it `transformers.models`) in every worker.
        if name == _VLLM_ATTENTION_IMPL:
            forward = _materialize_attention_layout(forward)
        _assign_attention_impl(cls, name, forward)

    AttentionInterface.register = classmethod(_register_attention_impl)

    # Cover the case where the Transformers backend was imported before this
    # patch: registering again goes through the hook and replaces vLLM's
    # forward in place.
    if _VLLM_ATTENTION_IMPL in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register(_VLLM_ATTENTION_IMPL, ALL_ATTENTION_FUNCTIONS[_VLLM_ATTENTION_IMPL])

DeepseekV32IndexerCache.get_attn_backend = _get_ascend_sfa_indexer_backend
vllm.v1.worker.gpu.attn_utils._allocate_kv_cache = _allocate_kv_cache
vllm.v1.worker.gpu.attn_utils._reshape_kv_cache = _reshape_kv_cache_v2
# vLLM #51718 made this the live allocation symbol used by init_kv_cache.
vllm.v1.worker.gpu.attn_utils.allocate_kv_cache = allocate_kv_cache_main
gpu_model_runner_v2.init_kv_cache.__globals__["allocate_kv_cache"] = allocate_kv_cache_main
gpu_model_runner_v2.init_kv_cache.__globals__["bind_kv_cache"] = bind_kv_cache

# vLLM #57169's packed-KV heuristic reads Ascend layerwise alias layouts
# (`layer_stride == 0`) as block-outermost packing and drops the kernel-block
# split, so correct the selection before `init_attn_backend` consumes it.
vllm.v1.worker.gpu.attn_utils.prepare_kernel_block_sizes = prepare_ascend_kernel_block_sizes

vllm.v1.worker.gpu.attn_utils.bind_kv_cache = bind_kv_cache
# vLLM main (#53781) routes init_kv_cache through bind_kv_cache_to_layers,
# which Ascend overrides with direct raw-allocation binding.
vllm.v1.worker.gpu.attn_utils.bind_kv_cache_to_layers = bind_kv_cache_to_layers

# ``vllm.v1.worker.utils`` is the binding source for both the V1 Model Runner
# and the 310P V2 model runner (``vllm_ascend/_310p/worker/v2/model_runner.py``,
# which imports ``bind_kv_cache`` by name from it). Installing the shared Ascend
# implementations here keeps both of them working without depending on the
# MRv1-only ``patch_v1`` package.
v1_worker_utils.bind_kv_cache = bind_kv_cache
v1_worker_utils.bind_kv_cache_to_layers = bind_kv_cache_to_layers

vllm.v1.worker.gpu.model_runner.get_kv_cache_spec = get_kv_cache_spec
