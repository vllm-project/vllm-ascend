# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def kv_group_block_sizes(kv_cache_config) -> list[int]:
    sizes: list[int] = []
    for group in getattr(kv_cache_config, "kv_cache_groups", ()) or ():
        block_size = getattr(getattr(group, "kv_cache_spec", None), "block_size", None)
        if isinstance(block_size, int):
            sizes.append(block_size)
    return sizes


def next_retention_interval(retention_interval: int, scheduler_block_size: int) -> int:
    """Smallest non-negative multiple of ``scheduler_block_size`` at or above the interval."""
    if retention_interval < 0 or scheduler_block_size <= 0:
        return 0 if retention_interval < 0 else retention_interval
    remainder = retention_interval % scheduler_block_size
    if remainder == 0:
        return retention_interval
    return retention_interval + scheduler_block_size - remainder


def retention_interval_mismatch_message(
    base_message: str,
    retention_interval: int,
    scheduler_block_size: int | None,
    group_block_sizes: list[int],
) -> str:
    """Explain a hybrid-Mamba retention mismatch and name the next valid interval.

    Prefix caching can enlarge ``scheduler_block_size`` to the alignment of the
    KV groups, so a value chosen from the CLI ``--block-size`` (for example
    4096 with ``--block-size 32``) is rejected once the effective size is 1536.
    """
    if not isinstance(scheduler_block_size, int) or scheduler_block_size <= 0:
        return base_message
    suggested = next_retention_interval(retention_interval, scheduler_block_size)
    if suggested == retention_interval:
        return base_message
    group_text = ""
    if group_block_sizes:
        rendered = ", ".join(str(size) for size in group_block_sizes)
        group_text = (
            f" It is the alignment of the KV-cache group block sizes [{rendered}],"
            " which hybrid Mamba prefix caching can enlarge beyond the CLI --block-size."
        )
    return (
        f"{base_message} The effective scheduler_block_size is {scheduler_block_size}.{group_text}"
        f" The next valid VLLM_PREFIX_CACHE_RETENTION_INTERVAL is {suggested}."
    )


def validate_retention_interval(
    retention_interval,
    scheduler_block_size,
    kv_cache_config,
    validate,
) -> None:
    try:
        validate(retention_interval, scheduler_block_size, kv_cache_config)
    except ValueError as exc:
        message = str(exc)
        if "multiple of scheduler_block_size" not in message or not isinstance(retention_interval, int):
            raise
        raise ValueError(
            retention_interval_mismatch_message(
                message,
                retention_interval,
                scheduler_block_size if isinstance(scheduler_block_size, int) else None,
                kv_group_block_sizes(kv_cache_config),
            )
        ) from exc
