# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Pure placement checks for retained-worker EP resizing."""

from collections.abc import Sequence


def validate_target(target: int, maximum: int) -> None:
    if type(target) is not int or not 2 <= target <= maximum:
        raise ValueError(f"EP size must be an integer between 2 and {maximum}.")


def replica_donors(maps: Sequence[Sequence[Sequence[int]]], active: int) -> dict[int, int]:
    """Require full logical coverage and whole-shard peers for sleeping ranks.

    Rows map logical experts to local slots (-1 means absent). Whole-shard
    matching permits storage refill without changing live tensor addresses.
    """
    if not maps or not maps[0]:
        raise ValueError("FastEEP requires static expert placement maps.")
    maximum = len(maps[0])
    validate_target(active, maximum)
    for layer in maps:
        if len(layer) != maximum or not layer[0]:
            raise ValueError("All MoE layers must use the same rank count.")
        logical = len(layer[0])
        capacity = sum(slot >= 0 for slot in layer[0])
        for row in layer:
            if len(row) != logical or any(type(slot) is not int or slot < -1 for slot in row):
                raise ValueError("Invalid expert placement row.")
            if sorted(slot for slot in row if slot >= 0) != list(range(capacity)):
                raise ValueError("Each rank must have the same contiguous local expert slots.")
        if any(all(row[expert] < 0 for row in layer[:active]) for expert in range(logical)):
            raise ValueError("Target ranks do not cover every logical expert.")
    donors = {}
    for rank in range(active, maximum):
        donor = next((peer for peer in range(active) if all(layer[rank] == layer[peer] for layer in maps)), None)
        if donor is None:
            raise ValueError(f"Rank {rank} has no complete replica among the target ranks.")
        donors[rank] = donor
    return donors


def validate_config(config) -> None:
    parallel = config.parallel_config
    if any(
        getattr(parallel, name) != 1
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        )
    ):
        raise ValueError("FastEEP currently requires TP=PP=PCP=DCP=1.")
    if not parallel.enable_expert_parallel or parallel.data_parallel_size < 2:
        raise ValueError("FastEEP requires DP >= 2 and expert parallelism.")
    if parallel.data_parallel_size_local != parallel.data_parallel_size:
        raise ValueError("FastEEP currently requires all ranks on one node.")
    if getattr(parallel, "data_parallel_external_lb", False) or getattr(parallel, "data_parallel_hybrid_lb", False):
        raise ValueError("FastEEP requires internal data-parallel request routing.")
    if parallel.enable_elastic_ep:
        raise ValueError("Use retained FastEEP or native Elastic EP in one instance, not both.")
    if parallel.enable_eplb or config.use_v2_model_runner:
        raise ValueError("FastEEP currently requires Model Runner V1 and static expert placement.")
    if not config.model_config.enforce_eager or not config.model_config.enable_sleep_mode:
        raise ValueError("FastEEP requires eager execution and sleep mode.")
    if str(config.model_config.dtype) != "torch.bfloat16" or config.quant_config is not None:
        raise ValueError("FastEEP currently supports unquantized BF16 weights.")
    if config.cache_config.enable_prefix_caching:
        raise ValueError("Disable prefix caching before discarding a drained rank's KV pool.")
    if any(
        getattr(config, name, None) is not None
        for name in ("lora_config", "speculative_config", "kv_transfer_config", "ec_transfer_config", "engram_config")
    ):
        raise ValueError("FastEEP does not support adapters, speculation, or cache/Engram transfer.")


def is_fasteep(config) -> bool:
    return (
        config.parallel_config.worker_extension_cls == "vllm_ascend.distributed.fasteep.runtime.FastEEPWorkerExtension"
    )


def local_physical_mask(topk_ids, rank: int, local_experts: int):
    """Each physical expert ID contributes on exactly one active rank."""
    first = rank * local_experts
    return (topk_ids >= first) & (topk_ids < first + local_experts)
