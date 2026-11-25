# SPDX-License-Identifier: Apache-2.0
"""Select the opt-in NOVA backend before Attention constructs its implementation."""

import inspect
from functools import wraps

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.ascend_config import NovaKVConfig
from vllm_ascend.utils import enable_sp

_ORIGINAL_INIT = Attention.__init__
_ORIGINAL_SPEC = Attention.get_kv_cache_spec
_INIT_SIGNATURE = inspect.signature(_ORIGINAL_INIT)


def _validate(config):
    if config.model_config.hf_text_config.model_type != "qwen3_5_text":
        raise ValueError("NOVA-KV currently supports Qwen3.5 dense text models")
    if config.cache_config.enable_prefix_caching:
        raise ValueError("NOVA-KV requires --no-enable-prefix-caching")
    if config.speculative_config is not None or config.kv_transfer_config is not None:
        raise ValueError("NOVA-KV does not support speculative decoding or KV transfer")
    parallel = config.parallel_config
    if parallel.decode_context_parallel_size != 1 or parallel.prefill_context_parallel_size != 1:
        raise ValueError("NOVA-KV does not support context parallelism")
    if enable_sp(config):
        raise ValueError("NOVA-KV does not support FlashComm sequence parallelism")
    if config.cache_config.cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("NOVA-KV requires --kv-cache-dtype auto or bfloat16")
    if (config.additional_config or {}).get("enable_kv_nz", False):
        raise ValueError("NOVA-KV requires the packed ND cache layout")


@wraps(_ORIGINAL_INIT)
def nova_init(self, *args, **kwargs):
    config = get_current_vllm_config()
    raw_options = (config.additional_config or {}).get("nova_kv")
    if raw_options is None:
        return _ORIGINAL_INIT(self, *args, **kwargs)
    options = NovaKVConfig(**raw_options)
    _validate(config)
    if options.artifact_path and not config.use_v2_model_runner:
        raise ValueError("NOVA-KV requires Model Runner V2")
    if options.artifact_path:
        if config.scheduler_config.max_num_seqs > 4:
            raise ValueError("NOVA-KV requires max_num_seqs <= 4")
        if config.model_config.max_model_len > 96 * 1024:
            raise ValueError("NOVA-KV currently supports at most 96K context")
    if options.capture_path and config.compilation_config.cudagraph_mode.name != "NONE":
        raise ValueError("NOVA-KV calibration capture requires --enforce-eager")
    if options.artifact_path and config.compilation_config.cudagraph_mode.name not in ("NONE", "FULL_DECODE_ONLY"):
        raise ValueError("NOVA-KV requires cudagraph_mode NONE or FULL_DECODE_ONLY")
    if options.artifact_path and (
        config.scheduler_config.async_scheduling or config.parallel_config.pipeline_parallel_size != 1
    ):
        raise ValueError("NOVA-KV recovery requires synchronous scheduling and pipeline parallel size 1")
    if options.capture_path:
        from vllm_ascend.quantization.methods.kv_cache.nova_kv.capture import NovaCapture

        _ORIGINAL_INIT(self, *args, **kwargs)
        capture = NovaCapture(self.layer_name, options)
        original_forward = self.impl.forward

        def capture_forward(layer, query, key, value, kv_cache, attn_metadata, **kw):
            if attn_metadata is not None and key is not None and value is not None:
                count = attn_metadata.num_actual_tokens
                capture.add(query[:count], key[:count], value[:count])
            return original_forward(layer, query, key, value, kv_cache, attn_metadata, **kw)

        self.impl.forward = capture_forward
        return
    arguments = _INIT_SIGNATURE.bind(self, *args, **kwargs)
    if arguments.arguments.get("logits_soft_cap") not in (None, 0):
        raise ValueError("NOVA-KV does not support attention logits soft capping")
    head_size_v = arguments.arguments.get("head_size_v")
    if head_size_v is not None and head_size_v != arguments.arguments["head_size"]:
        raise ValueError("NOVA-KV requires equal key and value head dimensions")
    from vllm_ascend.attention.nova_kv import NovaAttentionBackend

    backend = NovaAttentionBackend
    if arguments.arguments.get("attn_backend") not in (None, backend):
        raise ValueError("NOVA-KV conflicts with an explicitly selected attention backend")
    arguments.arguments["attn_backend"] = backend
    _ORIGINAL_INIT(*arguments.args, **arguments.kwargs)
    self.align_kv_cache_with_mamba = False


def nova_spec(self, config):
    spec = _ORIGINAL_SPEC(self, config)
    if not getattr(self, "align_kv_cache_with_mamba", True):
        from vllm_ascend.attention.nova_kv import NovaAttentionBackend

        if isinstance(self.attn_backend, type) and issubclass(self.attn_backend, NovaAttentionBackend):
            if type(spec) is not FullAttentionSpec:
                raise ValueError("NOVA-KV requires full decoder attention")
            spec = NovaAttentionBackend.customize_spec(spec)
    return spec


Attention.__init__ = nova_init
Attention.get_kv_cache_spec = nova_spec
