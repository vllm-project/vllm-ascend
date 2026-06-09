from collections.abc import Iterable

from vllm.logger import logger

from vllm_ascend.cpu_binding import get_memcache_client_cpus


def format_cpu_affinity(cpu_affinity: str | Iterable[int] | None) -> str:
    if cpu_affinity is None:
        return ""
    if isinstance(cpu_affinity, str):
        return cpu_affinity
    return ",".join(str(cpu) for cpu in cpu_affinity)


def get_default_memcache_cpu_affinity(rank_id: int) -> str:
    try:
        cpus = get_memcache_client_cpus(rank_id)
    except Exception as err:
        logger.warning("Failed to get CPUs for MemCache client: %s", err)
        return ""
    if not cpus:
        logger.warning("No MemCache client CPUs are configured.")
    return format_cpu_affinity(cpus)


def set_memcache_client_cpu_affinity(
    store: object,
    rank_id: int,
    cpu_affinity: str | Iterable[int] | None = None,
) -> None:
    client_cpu_affinity = format_cpu_affinity(cpu_affinity)
    if not client_cpu_affinity:
        client_cpu_affinity = get_default_memcache_cpu_affinity(rank_id)
    if client_cpu_affinity and hasattr(store, "set_client_cpu_affinity"):
        try:
            store.set_client_cpu_affinity(client_cpu_affinity)
            logger.info("Set MemCache client CPU affinity to %s", client_cpu_affinity)
        except Exception as err:
            logger.warning("Failed to set MemCache client CPU affinity to %s: %s", client_cpu_affinity, err)
    elif client_cpu_affinity:
        logger.warning(
            "MemCache client CPU affinity is configured, but the installed "
            "memcache_hybrid package does not expose set_client_cpu_affinity."
        )
