from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

CED_LAYERWISE_CONFIG_KEY = "CED_layerwise"

_CED_ARCHITECTURES = frozenset(
    {
        "DeepseekV41ForCausalLM",
        "DeepseekV41ForConditionalGeneration",
    }
)

_LAYER_INDEX_PATTERN = re.compile(r"\.layers\.(\d+)\.")


def _layer_index(layer_name: str) -> int | None:
    match = _LAYER_INDEX_PATTERN.search(f".{layer_name}")
    return int(match.group(1)) if match else None


def _is_indexer_member(layer_name: str) -> bool:
    return ".indexer.k_cache" in layer_name


def _is_draft_member(layer_name: str) -> bool:
    return ".mtp." in f".{layer_name}"


def is_ced_model(vllm_config: VllmConfig) -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    architectures = getattr(model_config, "architectures", None) or ()
    return any(architecture in _CED_ARCHITECTURES for architecture in architectures)


def resolve_ced_layerwise(vllm_config: VllmConfig) -> bool:
    """Resolve the ``CED_layerwise`` pool extra-config switch.

    Enables the CED (DeepSeek-V4.1) per-source-slice session save and the
    dependency-unit pipelined load on top of the non-layerwise bulk path.
    The switch is rejected together with ``use_layerwise`` and silently
    ignored on non-V4.1 models so existing deployments keep their behavior.
    """
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    if not extra_config.get(CED_LAYERWISE_CONFIG_KEY, False):
        return False
    if extra_config.get("use_layerwise", False):
        raise ValueError(
            "CED_layerwise cannot be combined with use_layerwise: the CED path "
            "is a bulk (non-layerwise) pool extension. Set use_layerwise=false."
        )
    if not is_ced_model(vllm_config):
        logger.warning(
            "CED_layerwise only supports DeepSeek-V4.1 architectures %s; ignoring the switch.",
            sorted(_CED_ARCHITECTURES),
        )
        return False
    return True


@dataclass(frozen=True)
class CEDGroupSlice:
    """One incrementally publishable byte range of a pool group object.

    ``member_start``/``member_end`` are indices into the cache group's
    layer-name list and must match the entry order used by
    ``ChunkedTokenDatabase.prepare_value`` so that the concatenation of all
    slices of a group equals the atomic-put object layout byte for byte.
    """

    group_id: int
    member_start: int
    member_end: int
    trigger_layer: int


@dataclass(frozen=True)
class CEDTriggerMap:
    """Derived save/load unit map for the CED pool path.

    Built from the cacheable cache groups plus the model topology, so
    ``swa_bounded_replay`` and MTP variations converge without hardcoded
    layer numbers. Groups without entries (compressor state, draft, unknown
    layouts) keep the existing step-end bulk save.
    """

    slices_by_group: dict[int, tuple[CEDGroupSlice, ...]] = field(default_factory=dict)
    slices_by_trigger: dict[int, tuple[CEDGroupSlice, ...]] = field(default_factory=dict)
    publish_layers: dict[int, int] = field(default_factory=dict)

    def trigger_layers(self) -> tuple[int, ...]:
        return tuple(sorted(self.slices_by_trigger))


def build_ced_trigger_map(
    group_layer_names: dict[int, Sequence[str]],
    kv_source_layer_ids: Sequence[int] = (),
) -> CEDTriggerMap:
    """Derive per-slice trigger points from cacheable group layer names.

    Full-context (MLA) groups get one slice per KV source slot, finalized by
    the source layer's own forward; the members of one slot are contiguous
    (long_kv, indexer[, folded]) in the group entry order. SWA groups get a
    single slice finalized by the group's last member layer. Draft groups
    and unrecognized layouts are left to the step-end bulk save.
    """
    slices_by_group: dict[int, list[CEDGroupSlice]] = {}
    slices_by_trigger: dict[int, list[CEDGroupSlice]] = {}
    publish_layers: dict[int, int] = {}

    for group_id, names in sorted(group_layer_names.items()):
        name_list = list(names)
        if not name_list or any(_is_draft_member(name) for name in name_list):
            continue
        member_layers = [_layer_index(name) for name in name_list]
        if any(layer is None for layer in member_layers):
            continue
        is_long_kv = [name.endswith(".long_kv_cache") for name in name_list]
        is_swa = [name.endswith(".swa_cache") for name in name_list]

        group_slices: list[CEDGroupSlice] = []
        if any(is_long_kv):
            by_source: dict[int, tuple[int, int]] = {}
            for member, layer in enumerate(member_layers):
                if not (is_long_kv[member] or _is_indexer_member(name_list[member])):
                    continue
                start, end = by_source.get(layer, (member, member + 1))
                by_source[layer] = (min(start, member), max(end, member + 1))
            for source_layer in sorted(by_source):
                start, end = by_source[source_layer]
                group_slices.append(CEDGroupSlice(group_id, start, end, source_layer))
            missing_sources = set(kv_source_layer_ids) - set(by_source)
            if missing_sources:
                logger.warning(
                    "CED trigger map: KV source layers %s have no members in group %d "
                    "(PP stage split or partial layout); those slots keep the bulk save.",
                    sorted(missing_sources),
                    group_id,
                )
        elif all(is_swa):
            group_slices.append(CEDGroupSlice(group_id, 0, len(name_list), max(member_layers)))

        if not group_slices:
            continue
        for group_slice in group_slices:
            slices_by_trigger.setdefault(group_slice.trigger_layer, []).append(group_slice)
        slices_by_group[group_id] = group_slices
        publish_layers[group_id] = max(group_slice.trigger_layer for group_slice in group_slices)

    return CEDTriggerMap(
        slices_by_group={group_id: tuple(slices) for group_id, slices in slices_by_group.items()},
        slices_by_trigger={trigger: tuple(slices) for trigger, slices in slices_by_trigger.items()},
        publish_layers=publish_layers,
    )


def _kv_source_layer_ids(hf_config) -> tuple[int, ...]:
    source_ids = getattr(hf_config, "kv_source_layer_ids", None)
    return tuple(int(source_id) for source_id in source_ids) if source_ids else ()


class CEDSaveWorker(KVPoolWorker):
    """Bulk pool worker extended with the CED incremental session save.

    Inherits the non-layerwise AscendStore behavior unchanged; the CED
    per-slice session save and dependency-unit load are layered on top and
    only activate for trigger points present in the derived map. Groups and
    steps without early triggers keep the step-end bulk save (draft groups,
    decode steps, replay steps).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwise: bool,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config, use_layerwise, kv_cache_config)
        self.ced_trigger_map = build_ced_trigger_map(
            self._cacheable_group_layer_names(),
            _kv_source_layer_ids(self.hf_config),
        )
        self.ced_trigger_layers = self.ced_trigger_map.trigger_layers()
        logger.info(
            "CED incremental pool path active: groups=%d trigger_layers=%s",
            len(self.ced_trigger_map.slices_by_group),
            list(self.ced_trigger_layers[:8]),
        )

    def _cacheable_group_layer_names(self) -> dict[int, list[str]]:
        if self.kv_cache_config is None:
            return {}
        groups = self.kv_cache_config.kv_cache_groups
        return {
            group_id: list(groups[group_id].layer_names)
            for group_id in range(len(groups))
            if group_id in self.cacheable_group_ids
        }

    def on_ced_save_trigger(self, trigger_layer: int) -> None:
        """Handle a per-slice save trigger raised at a V4.1 layer boundary.

        Wired to the dsa_v41 forward hooks; runs the session save
        (alloc -> slice copy -> publish) for the slices finalized at
        ``trigger_layer``. Slices without early triggers keep the step-end
        bulk save.
        """
        slices = self.ced_trigger_map.slices_by_trigger.get(trigger_layer)
        if not slices:
            return
        logger.debug("CED save trigger layer=%d slices=%d", trigger_layer, len(slices))
