# SPDX-License-Identifier: Apache-2.0
"""Resolve the path-only SSD interface before engine/worker construction."""

from pathlib import Path

from vllm.logger import logger

from vllm_ascend.model_loader.engram_ssd_artifact import prepare_artifact

DEFAULT_HOST_MEMORY_BYTES = 32_000_000_000


def prepare_ssd_config(vllm_config) -> None:
    """Consume path/host_memory_bytes and produce the existing runtime config.

    Resolved configs passed to workers contain no preparation request, so worker
    initialization never repeats artifact conversion. Independent launchers still
    check and reuse the persistent artifact under its filesystem lock.
    """
    additional = vllm_config.additional_config or {}
    raw = additional.get("engram_ssd")
    if not isinstance(raw, dict) or "path" not in raw:
        return
    options = dict(raw)
    path = options.pop("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("engram_ssd.path must be a nonempty SSD directory")
    if options.get("enabled", True) is not True:
        raise ValueError("engram_ssd.path enables SSD offload; remove path to disable it")
    if any(
        key in options
        for key in ("manifest", "artifact_version", "table_paths", "total_host_budget_bytes", "cache_payload_bytes")
    ):
        raise ValueError(
            "Use engram_ssd.path with host_memory_bytes, or the legacy artifact/per-owner fields, not both"
        )
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    if model.architecture != "DeepseekV41ForCausalLM":
        raise ValueError("Automatic Engram SSD preparation currently supports DeepseekV41ForCausalLM")
    if parallel.pipeline_parallel_size != 1 or parallel.prefill_context_parallel_size != 1:
        raise ValueError("Engram SSD requires PP=1 and PCP=1")
    if getattr(parallel, "enable_eplb", False) or getattr(parallel, "enable_elastic_ep", False):
        raise ValueError("Engram SSD does not support EPLB or elastic EP")
    if parallel.data_parallel_size > 1 and parallel.data_parallel_size_local != parallel.data_parallel_size:
        raise ValueError("Engram SSD DP sharing requires all DP replicas on the same node")
    load = vllm_config.load_config
    if load.load_format not in ("auto", "safetensors", "ascend_engram_ssd"):
        raise ValueError("engram_ssd.path conflicts with the explicitly selected model loader")
    # Lazy imports: config imports platform code during normal initialization.
    from vllm.config import EngramConfig

    from vllm_ascend.ascend_config import EngramSSDConfig

    engram = vllm_config.engram_config
    if engram is None:
        engram = EngramConfig(cpu_offload=True)
    if not engram.cpu_offload or engram.embedding_across_dp or engram.use_thp:
        raise ValueError("Engram SSD requires cpu_offload=True, embedding_across_dp=False and use_thp=False")
    if parallel.data_parallel_size > 1 and engram.dp_shared_memory is False:
        raise ValueError("Engram SSD with DP>1 requires dp_shared_memory")
    hf = model.hf_text_config
    layer_ids = list(hf.engram_layer_ids)
    rows = list(hf.engram_num_embeddings)
    if len(layer_ids) != 2 or len(rows) != 2 or len(set(layer_ids)) != 2 or hf.engram_head_dim != 256:
        raise ValueError("Unsupported Engram model table layout")
    hash_columns = (hf.engram_max_ngram_size - 1) * hf.engram_n_heads
    if hash_columns <= 0 or hash_columns % parallel.tensor_parallel_size:
        raise ValueError("Engram SSD hash columns must divide evenly across TP ranks")
    budget = options.pop("host_memory_bytes", DEFAULT_HOST_MEMORY_BYTES)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("host_memory_bytes must be a positive integer (whole-service DDR bytes)")
    owners = parallel.tensor_parallel_size
    per_owner = budget // owners
    options.setdefault("total_host_budget_bytes", per_owner)
    options.setdefault("cache_payload_bytes", options["total_host_budget_bytes"] * 5 // 8)
    options.setdefault("threads_per_disk", 8)
    options.setdefault("queue_depth", 1024)
    options.setdefault("io_setup_events", 1280)
    # Default to OS scheduling within the launcher's cpuset. Do not hard-code
    # machine-specific cores or bind all TP owners to the same small CPU range.
    options.setdefault("io_cpu_list", [])
    options["enabled"] = True
    pending = dict(options, manifest="pending", artifact_version="pending", table_paths=["pending"])
    estimated = EngramSSDConfig(**pending).resource_estimate()
    if not estimated["within_budget"]:
        raise ValueError(f"Engram SSD estimated peak {estimated['peak_bytes']} exceeds per-owner DDR budget")
    # Fail before generating a large file when even the cache's estimated peak
    # does not fit currently available host memory. Allocation remains decisive.
    meminfo = Path("/proc/meminfo").read_text()
    available = next(
        (int(line.split()[1]) * 1024 for line in meminfo.splitlines() if line.startswith("MemAvailable:")), None
    )
    if available is not None and estimated["peak_bytes"] * owners > available:
        raise ValueError("Insufficient available host DDR for Engram SSD cache; reduce host_memory_bytes")
    artifact = prepare_artifact(model.model, path, dict(zip(layer_ids, rows)))
    resolved = dict(options, **artifact)
    # Publish config only after successful preparation, leaving failures retryable.
    vllm_config.additional_config = dict(additional, engram_ssd=resolved)
    load.load_format = "ascend_engram_ssd"
    engram.dp_shared_memory = parallel.data_parallel_size > 1
    vllm_config.engram_config = engram
    logger.info(
        "Engram SSD ready: %s; %d owners, %d payload bytes and %d peak bytes per owner",
        artifact["table_paths"][0],
        owners,
        options["cache_payload_bytes"],
        estimated["peak_bytes"],
    )
