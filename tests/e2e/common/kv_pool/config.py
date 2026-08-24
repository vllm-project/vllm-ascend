from dataclasses import dataclass
from typing import Any

SUPPORTED_KV_POOL_TYPES = {"memcache", "mooncake"}


@dataclass(frozen=True)
class KVPoolConfig:
    """Base configuration shared by managed KV pool backends."""

    config: dict[str, Any]

    @property
    def type(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class MooncakeKVPoolConfig(KVPoolConfig):
    """Mooncake master configuration."""

    master_port: int
    metrics_port: int

    @property
    def type(self) -> str:
        return "mooncake"


@dataclass(frozen=True)
class MemcacheKVPoolConfig(KVPoolConfig):
    """Memcache MetaService configuration."""

    meta_service_port: int
    config_store_port: int

    @property
    def type(self) -> str:
        return "memcache"


def parse_kv_pool_config(raw_kv_pool: Any) -> KVPoolConfig | None:
    if raw_kv_pool is None:
        return None
    if not isinstance(raw_kv_pool, dict):
        raise TypeError("kv_pool must be a mapping")

    pool_type = str(raw_kv_pool.get("type", "")).lower()
    if pool_type not in SUPPORTED_KV_POOL_TYPES:
        raise ValueError(f"Unsupported kv_pool.type: {pool_type!r}")

    pool_config = raw_kv_pool.get("config")
    if not isinstance(pool_config, dict):
        raise TypeError("kv_pool.config must be a mapping")
    kv_pool: KVPoolConfig
    if pool_type == "mooncake":
        kv_pool = MooncakeKVPoolConfig(
            config=dict(pool_config),
            master_port=int(raw_kv_pool["master_port"]),
            metrics_port=int(raw_kv_pool["metrics_port"]),
        )
    else:
        kv_pool = MemcacheKVPoolConfig(
            config=dict(pool_config),
            meta_service_port=int(raw_kv_pool["meta_service_port"]),
            config_store_port=int(raw_kv_pool["config_store_port"]),
        )
    validate_kv_pool_config(kv_pool)
    return kv_pool


def validate_kv_pool_config(kv_pool: KVPoolConfig) -> None:
    if isinstance(kv_pool, MooncakeKVPoolConfig):
        ports = (
            ("master_port", kv_pool.master_port),
            ("metrics_port", kv_pool.metrics_port),
        )
    elif isinstance(kv_pool, MemcacheKVPoolConfig):
        ports = (
            ("meta_service_port", kv_pool.meta_service_port),
            ("config_store_port", kv_pool.config_store_port),
        )
    else:
        raise TypeError(f"Unsupported KV pool config: {type(kv_pool).__name__}")

    for field_name, port in ports:
        if port < 1 or port > 65535:
            raise ValueError(f"kv_pool.{field_name} must be between 1 and 65535")
    if ports[0][1] == ports[1][1]:
        raise ValueError(f"kv_pool.{ports[0][0]} and kv_pool.{ports[1][0]} must be different")

    if isinstance(kv_pool, MemcacheKVPoolConfig):
        for section in ("meta", "local"):
            if not isinstance(kv_pool.config.get(section), dict):
                raise TypeError(f"kv_pool.config.{section} must be a mapping for memcache")
