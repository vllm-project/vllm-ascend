"""Serializable configuration and identities for KV cache registration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, is_dataclass
from types import SimpleNamespace
from typing import Any

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheSpec, UniformTypeKVCacheSpecs

_KV_ROLES = {"kv_producer", "kv_consumer", "kv_both"}
_SUPPORTED_SPEC_MODULES = {"vllm.v1.kv_cache_interface", "vllm_ascend.core.kv_cache_interface"}


# ==============================
# KV cache layout in registration
# ==============================

# Registration carries the cache layout needed by the server, never the live
# tensors owned by the vLLM Worker. The server rebuilds metadata-only vLLM
# objects here; their storage crosses the process boundary separately through
# NPU IPC.


@dataclass(frozen=True)
class KVCacheGroupData:
    layer_names: tuple[str, ...]
    kv_cache_spec: KVCacheSpec
    is_eagle_group: bool

    @classmethod
    def from_group(cls, group: KVCacheGroupSpec) -> KVCacheGroupData:
        _validate_kv_cache_spec(group.kv_cache_spec)
        return cls(
            layer_names=tuple(group.layer_names),
            kv_cache_spec=group.kv_cache_spec,
            is_eagle_group=bool(getattr(group, "is_eagle_group", False)),
        )

    def build(self) -> KVCacheGroupSpec:
        _validate_kv_cache_spec(self.kv_cache_spec)
        return KVCacheGroupSpec(
            layer_names=list(self.layer_names),
            kv_cache_spec=self.kv_cache_spec,
            is_eagle_group=self.is_eagle_group,
        )


@dataclass(frozen=True)
class KVCacheConfigData:
    num_blocks: int
    groups: tuple[KVCacheGroupData, ...]

    @classmethod
    def from_config(cls, config: KVCacheConfig) -> KVCacheConfigData:
        return cls(
            num_blocks=_require_non_negative_int(config.num_blocks, "kv_cache_config.num_blocks"),
            groups=tuple(KVCacheGroupData.from_group(group) for group in config.kv_cache_groups),
        )

    def build(self) -> KVCacheConfig:
        return KVCacheConfig(
            num_blocks=self.num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[group.build() for group in self.groups],
        )


# ==============================
# Configuration rebuilt for the server process
# ==============================

# The inherited KVPool Scheduler and Worker expect a VllmConfig-shaped object,
# but the real VllmConfig also contains process-local runtime state. These
# immutable specifications carry only the fields consumed in KVCacheServer and
# rebuild the configuration used by its Scheduler and Worker services.


@dataclass(frozen=True)
class KVPoolModelConfigSpec:
    model: str
    max_model_len: int
    num_layers: int
    num_kv_heads: int
    num_hidden_layers: int
    use_mla: bool
    use_sparse: bool
    model_type: str | None
    compress_ratios: tuple[int, ...] | None

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> KVPoolModelConfigSpec:
        model_config = _require_attr(vllm_config, "model_config", "vllm_config")
        parallel_config = _require_attr(vllm_config, "parallel_config", "vllm_config")
        hf_text_config = _require_attr(model_config, "hf_text_config", "model_config")
        hf_config = getattr(model_config, "hf_config", None) or hf_text_config
        num_layers = _call_required_positive_int(
            model_config,
            "get_num_layers",
            "model_config.get_num_layers()",
            parallel_config,
        )
        compress_ratios = _optional_int_tuple(
            getattr(hf_text_config, "compress_ratios", None),
            "model_config.hf_text_config.compress_ratios",
        )
        if compress_ratios is None:
            compress_ratios = _optional_int_tuple(
                getattr(hf_config, "compress_ratios", None),
                "model_config.hf_config.compress_ratios",
            )
        num_hidden_layers = _optional_positive_int(
            getattr(hf_text_config, "num_hidden_layers", None),
            "model_config.hf_text_config.num_hidden_layers",
        )
        return cls(
            model=_require_non_empty_str(
                _require_attr(model_config, "model", "model_config"),
                "model_config.model",
            ),
            max_model_len=_require_positive_int(
                _require_attr(model_config, "max_model_len", "model_config"),
                "model_config.max_model_len",
            ),
            num_layers=num_layers,
            num_kv_heads=_call_required_positive_int(
                model_config,
                "get_total_num_kv_heads",
                "model_config.get_total_num_kv_heads()",
            ),
            num_hidden_layers=num_hidden_layers if num_hidden_layers is not None else num_layers,
            use_mla=_require_bool(
                _require_attr(model_config, "use_mla", "model_config"),
                "model_config.use_mla",
            ),
            use_sparse=hasattr(hf_text_config, "index_topk"),
            model_type=_optional_str(
                getattr(hf_config, "model_type", None),
                "model_config.hf_config.model_type",
            ),
            compress_ratios=compress_ratios,
        )

    @property
    def hf_text_config(self) -> SimpleNamespace:
        values: dict[str, object] = {"num_hidden_layers": self.num_hidden_layers}
        if self.model_type is not None:
            values["model_type"] = self.model_type
        if self.compress_ratios is not None:
            values["compress_ratios"] = self.compress_ratios
        if self.use_sparse:
            values["index_topk"] = True
        return SimpleNamespace(**values)

    @property
    def hf_config(self) -> SimpleNamespace:
        return self.hf_text_config

    def get_num_layers(self, _parallel_config: object) -> int:
        return self.num_layers

    def get_total_num_kv_heads(self) -> int:
        return self.num_kv_heads


@dataclass(frozen=True)
class KVPoolParallelConfigSpec:
    rank: int
    world_size: int
    data_parallel_rank: int
    data_parallel_index: int
    data_parallel_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    prefill_context_parallel_size: int
    decode_context_parallel_size: int

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> KVPoolParallelConfigSpec:
        config = _require_attr(vllm_config, "parallel_config", "vllm_config")
        return cls(
            rank=_require_non_negative_int(
                _require_attr(config, "rank", "parallel_config"),
                "parallel_config.rank",
            ),
            world_size=_require_positive_int(
                _require_attr(config, "world_size", "parallel_config"),
                "parallel_config.world_size",
            ),
            data_parallel_rank=_require_non_negative_int(
                _require_attr(config, "data_parallel_rank", "parallel_config"),
                "parallel_config.data_parallel_rank",
            ),
            data_parallel_index=_require_non_negative_int(
                _require_attr(config, "data_parallel_index", "parallel_config"),
                "parallel_config.data_parallel_index",
            ),
            data_parallel_size=_require_positive_int(
                _require_attr(config, "data_parallel_size", "parallel_config"),
                "parallel_config.data_parallel_size",
            ),
            tensor_parallel_size=_require_positive_int(
                _require_attr(config, "tensor_parallel_size", "parallel_config"),
                "parallel_config.tensor_parallel_size",
            ),
            pipeline_parallel_size=_require_positive_int(
                _require_attr(config, "pipeline_parallel_size", "parallel_config"),
                "parallel_config.pipeline_parallel_size",
            ),
            prefill_context_parallel_size=_require_positive_int(
                _require_attr(config, "prefill_context_parallel_size", "parallel_config"),
                "parallel_config.prefill_context_parallel_size",
            ),
            decode_context_parallel_size=_require_positive_int(
                _require_attr(config, "decode_context_parallel_size", "parallel_config"),
                "parallel_config.decode_context_parallel_size",
            ),
        )


@dataclass(frozen=True)
class KVPoolTransferConfigSpec:
    engine_id: str
    kv_role: str
    kv_connector: str
    kv_connector_extra_config: dict[str, Any]

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> KVPoolTransferConfigSpec:
        config = _require_attr(vllm_config, "kv_transfer_config", "vllm_config")
        engine_id = _require_non_empty_str(
            _require_attr(config, "engine_id", "kv_transfer_config"),
            "kv_transfer_config.engine_id",
        )
        kv_role = _require_non_empty_str(
            _require_attr(config, "kv_role", "kv_transfer_config"),
            "kv_transfer_config.kv_role",
        )
        if kv_role not in _KV_ROLES:
            raise ValueError(f"kv_transfer_config.kv_role must be one of {sorted(_KV_ROLES)}, got {kv_role!r}")
        extra_config = _require_attr(config, "kv_connector_extra_config", "kv_transfer_config")
        if not isinstance(extra_config, Mapping):
            raise TypeError(
                f"kv_transfer_config.kv_connector_extra_config must be a mapping, got {type(extra_config).__name__}"
            )
        projected_extra_config = _project_extra_value(extra_config)
        assert isinstance(projected_extra_config, dict)
        return cls(
            engine_id=engine_id,
            kv_role=kv_role,
            kv_connector=_require_non_empty_str(
                _require_attr(config, "kv_connector", "kv_transfer_config"),
                "kv_transfer_config.kv_connector",
            ),
            kv_connector_extra_config=projected_extra_config,
        )

    def get_from_extra_config(self, name: str, default: Any = None) -> Any:
        return self.kv_connector_extra_config.get(name, default)


@dataclass(frozen=True)
class KVPoolCacheConfigSpec:
    block_size: int
    prefix_match_unit: int | None

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> KVPoolCacheConfigSpec:
        config = _require_attr(vllm_config, "cache_config", "vllm_config")
        return cls(
            block_size=_require_positive_int(
                _require_attr(config, "block_size", "cache_config"),
                "cache_config.block_size",
            ),
            prefix_match_unit=_optional_positive_int(
                _require_attr(config, "prefix_match_unit", "cache_config", allow_none=True),
                "cache_config.prefix_match_unit",
            ),
        )


@dataclass(frozen=True)
class KVPoolSchedulerConfigSpec:
    disable_hybrid_kv_cache_manager: bool

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> KVPoolSchedulerConfigSpec:
        config = _require_attr(vllm_config, "scheduler_config", "vllm_config")
        disable_hybrid = _require_attr(config, "disable_hybrid_kv_cache_manager", "scheduler_config", allow_none=True)
        return cls(
            disable_hybrid_kv_cache_manager=(
                _require_bool(disable_hybrid, "scheduler_config.disable_hybrid_kv_cache_manager")
                if disable_hybrid is not None
                else False
            )
        )


@dataclass(frozen=True)
class KVPoolSpeculativeConfigSpec:
    num_speculative_tokens: int
    eagle_enabled: bool

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> KVPoolSpeculativeConfigSpec | None:
        config = getattr(vllm_config, "speculative_config", None)
        if config is None:
            return None
        use_eagle = _require_attr(config, "use_eagle", "speculative_config")
        if not callable(use_eagle):
            raise TypeError("speculative_config.use_eagle must be callable")
        eagle_enabled = use_eagle()
        return cls(
            num_speculative_tokens=_require_positive_int(
                _require_attr(config, "num_speculative_tokens", "speculative_config"),
                "speculative_config.num_speculative_tokens",
            ),
            eagle_enabled=_require_bool(eagle_enabled, "speculative_config.use_eagle()"),
        )

    def use_eagle(self) -> bool:
        return self.eagle_enabled


@dataclass(frozen=True)
class KVPoolEventsConfigSpec:
    enable_kv_cache_events: bool


@dataclass(frozen=True)
class KVPoolConfigSpec:
    model_config: KVPoolModelConfigSpec
    parallel_config: KVPoolParallelConfigSpec
    kv_transfer_config: KVPoolTransferConfigSpec
    cache_config: KVPoolCacheConfigSpec
    scheduler_config: KVPoolSchedulerConfigSpec
    speculative_config: KVPoolSpeculativeConfigSpec | None
    kv_events_config: KVPoolEventsConfigSpec | None
    kv_cache_config: KVCacheConfigData | None

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None) -> KVPoolConfigSpec:
        kv_events_config = getattr(vllm_config, "kv_events_config", None)
        kv_events_enabled = (
            _require_bool(
                _require_attr(kv_events_config, "enable_kv_cache_events", "kv_events_config"),
                "kv_events_config.enable_kv_cache_events",
            )
            if kv_events_config is not None
            else False
        )
        return cls(
            model_config=KVPoolModelConfigSpec.from_vllm_config(vllm_config),
            parallel_config=KVPoolParallelConfigSpec.from_vllm_config(vllm_config),
            kv_transfer_config=KVPoolTransferConfigSpec.from_vllm_config(vllm_config),
            cache_config=KVPoolCacheConfigSpec.from_vllm_config(vllm_config),
            scheduler_config=KVPoolSchedulerConfigSpec.from_vllm_config(vllm_config),
            speculative_config=KVPoolSpeculativeConfigSpec.from_vllm_config(vllm_config),
            kv_events_config=(KVPoolEventsConfigSpec(enable_kv_cache_events=True) if kv_events_enabled else None),
            kv_cache_config=KVCacheConfigData.from_config(kv_cache_config) if kv_cache_config is not None else None,
        )

    def build_kv_cache_config(self) -> KVCacheConfig | None:
        return self.kv_cache_config.build() if self.kv_cache_config is not None else None


# ==============================
# Service identities and registrations
# ==============================

# Scheduler services are owned per engine and data-parallel rank, while Worker
# services additionally own one rank's imported cache. The session id separates
# a restarted client from an older owner of the same logical identity so stale
# lifecycle calls cannot target its replacement.

_LEGACY_SESSION_ID = "legacy"

WorkerLookupHandler = Callable[["SchedulerIdentity", int, Sequence[BlockHash], list[int] | None, bool, int], int]


@dataclass(frozen=True)
class SchedulerIdentity:
    engine_id: str
    data_parallel_rank: int = 0

    def __post_init__(self) -> None:
        _require_non_empty_str(self.engine_id, "engine_id")
        _require_non_negative_int(self.data_parallel_rank, "data_parallel_rank")

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> SchedulerIdentity:
        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set")
        return cls(
            engine_id=kv_transfer_config.engine_id,
            data_parallel_rank=vllm_config.parallel_config.data_parallel_rank,
        )

    @classmethod
    def from_config_spec(cls, config: KVPoolConfigSpec) -> SchedulerIdentity:
        return cls(
            engine_id=config.kv_transfer_config.engine_id,
            data_parallel_rank=config.parallel_config.data_parallel_rank,
        )


@dataclass(frozen=True)
class WorkerIdentity:
    engine_id: str
    rank: int
    data_parallel_rank: int = 0

    def __post_init__(self) -> None:
        _require_non_empty_str(self.engine_id, "engine_id")
        _require_non_negative_int(self.rank, "rank")
        _require_non_negative_int(self.data_parallel_rank, "data_parallel_rank")

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> WorkerIdentity:
        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set")
        return cls(
            engine_id=kv_transfer_config.engine_id,
            rank=vllm_config.parallel_config.rank,
            data_parallel_rank=vllm_config.parallel_config.data_parallel_rank,
        )

    @classmethod
    def from_config_spec(cls, config: KVPoolConfigSpec) -> WorkerIdentity:
        return cls(
            engine_id=config.kv_transfer_config.engine_id,
            rank=config.parallel_config.rank,
            data_parallel_rank=config.parallel_config.data_parallel_rank,
        )


@dataclass(frozen=True)
class SchedulerRegistration:
    identity: SchedulerIdentity
    config: KVPoolConfigSpec
    page_size_bytes: int
    session_id: str = _LEGACY_SESSION_ID

    def __post_init__(self) -> None:
        _require_non_empty_str(self.session_id, "session_id")

    @classmethod
    def create(
        cls,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig | None,
        page_size_bytes: int,
        session_id: str = _LEGACY_SESSION_ID,
    ) -> SchedulerRegistration:
        _require_non_negative_int(page_size_bytes, "page_size_bytes")
        return cls(
            identity=SchedulerIdentity.from_vllm_config(vllm_config),
            config=KVPoolConfigSpec.from_vllm_config(vllm_config, kv_cache_config),
            page_size_bytes=page_size_bytes,
            session_id=session_id,
        )


@dataclass(frozen=True)
class WorkerRegistration:
    identity: WorkerIdentity
    config: KVPoolConfigSpec
    session_id: str = _LEGACY_SESSION_ID

    def __post_init__(self) -> None:
        _require_non_empty_str(self.session_id, "session_id")

    @classmethod
    def create(
        cls,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig | None,
        session_id: str = _LEGACY_SESSION_ID,
    ) -> WorkerRegistration:
        return cls(
            identity=WorkerIdentity.from_vllm_config(vllm_config),
            config=KVPoolConfigSpec.from_vllm_config(vllm_config, kv_cache_config),
            session_id=session_id,
        )


# ==============================
# Registration boundary validation
# ==============================

# Registrations are built from live vLLM objects and later trusted as runtime
# configuration in the server process. These checks admit only supported cache
# specifications and plain configuration values, and reject incomplete or
# invalid data before it crosses the process boundary.


def _validate_kv_cache_spec(spec: KVCacheSpec) -> None:
    spec_type = type(spec)
    if spec_type.__module__ not in _SUPPORTED_SPEC_MODULES:
        raise TypeError(
            f"Unsupported KV cache spec type {spec_type.__module__}.{spec_type.__name__}; "
            "AscendStore MP registrations support vLLM and vLLM Ascend cache specs only"
        )
    if not is_dataclass(spec):
        raise TypeError(f"KV cache spec {spec_type.__name__} must be a dataclass")
    if isinstance(spec, UniformTypeKVCacheSpecs):
        for nested_spec in spec.kv_cache_specs.values():
            _validate_kv_cache_spec(nested_spec)


def _project_extra_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, list):
        return [_project_extra_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_project_extra_value(item) for item in value)
    if isinstance(value, Mapping):
        projected = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Configuration mapping keys must be strings, got {type(key).__name__}")
            projected[key] = _project_extra_value(item)
        return projected
    raise TypeError(f"Unsupported registration configuration value {type(value).__name__}")


_MISSING = object()


def _require_attr(obj: object, name: str, owner: str, *, allow_none: bool = False) -> Any:
    value = getattr(obj, name, _MISSING)
    if value is _MISSING or (value is None and not allow_none):
        raise ValueError(f"{owner}.{name} must be set")
    return value


def _call_required_positive_int(obj: object, name: str, field_name: str, *args: object) -> int:
    method = getattr(obj, name, None)
    if not callable(method):
        raise TypeError(f"{field_name.removesuffix('()')} must be callable")
    return _require_positive_int(method(*args), field_name)


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean, got {type(value).__name__}")
    return value


def _require_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return value


def _require_positive_int(value: object, name: str) -> int:
    value = _require_int(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _optional_positive_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, name)


def _require_non_negative_int(value: object, name: str) -> int:
    value = _require_int(value, name)
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value}")
    return value


def _require_non_empty_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _optional_str(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_str(value, name)


def _optional_int_tuple(value: object, name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers, got {type(value).__name__}")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise TypeError(f"{name} must contain integers only")
    return tuple(value)
