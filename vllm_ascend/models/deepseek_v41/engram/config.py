# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-specific validation; existing UVA and HBM policies are unchanged."""


def validate_elastic_config(ascend_config, vllm_config) -> None:
    if not ascend_config.enable_engram or ascend_config.engram_offload_backend != "elastic":
        return
    engram = getattr(vllm_config, "engram_config", None)
    if engram is None or not engram.cpu_offload:
        raise ValueError("engram_offload_backend='elastic' requires EngramConfig.cpu_offload=True")
    if engram.dp_shared_memory:
        raise ValueError("ElasticBuffer row sharding cannot be combined with EngramConfig.dp_shared_memory")
    model = vllm_config.model_config
    text = getattr(model.hf_config, "text_config", model.hf_config)
    if not getattr(text, "engram_layer_ids", None):
        raise ValueError("ElasticBuffer requires a checkpoint with Engram layers")
    parallel = vllm_config.parallel_config
    if any(
        getattr(parallel, key, 1) != 1
        for key in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size")
    ):
        raise ValueError("ElasticBuffer Engram currently requires PP=PCP=DCP=1")
    world = int(getattr(parallel, "world_size_across_dp", getattr(parallel, "world_size", 1)))
    local = int(getattr(parallel, "local_world_size", world)) * int(getattr(parallel, "data_parallel_size_local", 1))
    size = world if ascend_config.engram_tp_size is None else ascend_config.engram_tp_size
    if size < 2 or size > world or world % size:
        raise ValueError("ElasticBuffer engram_tp_size must be >= 2 and divide world_size_across_dp")
    if world > local or size < min(world, local):
        raise ValueError("ElasticBuffer requires a node-local group spanning all participating ranks on this host")
    ascend_config.engram_tp_size = size
    if getattr(parallel, "use_sequence_parallel_moe", False):
        raise ValueError("ElasticBuffer Engram does not yet support MoE sequence parallelism")
    if vllm_config.load_config.load_format == "dummy":
        raise ValueError("ElasticBuffer requires a native MXFP8 checkpoint; dummy weights are unsupported")
    _validate_elastic_speculation(vllm_config.speculative_config)


def _validate_elastic_speculation(speculative_config) -> None:
    """Validate the Elastic combination without changing proposer selection."""
    if speculative_config is None:
        return
    if getattr(speculative_config, "method", None) != "dspark":
        raise ValueError("ElasticBuffer Engram supports only method='dspark' or disabled speculative decoding")
    use_dspark = getattr(speculative_config, "use_dspark", None)
    if callable(use_dspark) and not use_dspark():
        raise ValueError("ElasticBuffer Engram method='dspark' must report use_dspark()=True")

    from vllm_ascend.models.deepseek_v41.cache_config import STATE_RING_ROWS

    maximum = getattr(speculative_config, "num_speculative_tokens", None)
    schedule = getattr(speculative_config, "num_speculative_tokens_per_batch_size", None) or ()
    lengths = schedule.values() if isinstance(schedule, dict) else (entry[2] for entry in schedule)
    if maximum is None or not 0 < maximum < STATE_RING_ROWS:
        raise ValueError(f"ElasticBuffer Engram DSpark requires 1..{STATE_RING_ROWS - 1} speculative tokens")
    if any(not 0 <= length < STATE_RING_ROWS for length in lengths):
        raise ValueError(f"ElasticBuffer Engram DSpark dynamic lengths must be in 0..{STATE_RING_ROWS - 1}")
