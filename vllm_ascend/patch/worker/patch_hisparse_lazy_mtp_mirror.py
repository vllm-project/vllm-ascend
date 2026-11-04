# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Optional certified MTP3 lazy Host mirror, independent of group prefetch.

This module is inert until explicitly installed in an independent Ascend
D-only candidate with eager_host_mirror=False. P must keep eager mirrors.
It does not implement outbound-export tail materialization. It keeps the original sealed-page
materialization, completion fences, Resident owners and Hot invalidation.
It changes neither decode_batch nor write_target nor MTP submission order.

Remove the OOT wrappers when common HiSparse exposes a backend mirror-policy
hook. Unknown common source versions must be re-reviewed before use.
"""

import ast
import inspect
from functools import wraps
from hashlib import sha256
from importlib import import_module
from inspect import signature
from pathlib import Path
from typing import Any

_EXPECTED_RUNTIME_SHA256 = "89b940424066b62bf042bcdade3b7efacf03968d9f449888c12b351c636cb926"
_EXPECTED_WORKER_SHA256 = "1953b3e582502d16df38bba4a5966e0b98492feb67b65cdac7f2b0397ca4b229"
_EXPECTED_COORDINATOR_SHA256 = "530a488c60c3411e53eab42a8fe0624454de942204da304391a4f1c295b30666"
_EXPECTED_CONNECTOR_SHA256 = "78610e02ebdd622dc8272a98d75c2a75a41d8b19ce1e72ac582bdedd19d11f13"
_EXPECTED_METADATA_SHA256 = "bb84247bab8c9edfc915de32755d9ac32240e93681c44f24c84e8d88792483f4"
_PATCH_MARKER = "_ascend_certified_lazy_mtp_mirror_patch"
_BATCH_FLAG = "_ascend_certified_lazy_mtp_mirror"


def _certificate_source_digest(source: str) -> str:
    """Keep the reviewed certificate stable across Python 3.11/3.12 ASTs.

    Python 3.12 adds an empty type_params field to ordinary functions. Remove
    only that empty field from this temporary parse tree; nonempty generics
    and all executable syntax still change the fingerprint.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if "type_params" in node._fields and getattr(node, "type_params", None) == []:
            node._fields = tuple(name for name in node._fields if name != "type_params")
    return sha256(ast.dump(tree).encode()).hexdigest()


def _legacy_eligible(metadata: Any | None, max_rows: int) -> bool:
    """Accept only the existing Ascend target graph's CPU layout certificate."""
    if metadata is None:
        return False
    names = (
        "num_actual_tokens",
        "num_decode_tokens",
        "num_reqs",
        "max_query_len",
        "hisparse_uniform_query_group_size",
        "num_prefills",
        "num_decodes",
    )
    values = [getattr(metadata, name, None) for name in names]
    if not all(type(value) is int for value in values):
        return False
    rows, decode_rows, requests, query_len, certificate, prefills, decodes = values
    return (
        certificate == 4
        and prefills == 0
        and decodes == requests
        and rows in (16, 32, 64)
        and 0 < rows <= max_rows
        and decode_rows == rows
        and query_len == 4
        and rows == 4 * requests
    )


def _eligible(metadata: Any | None, max_rows: int) -> bool:
    if _legacy_eligible(metadata, max_rows):
        return True
    layout = getattr(metadata, "hisparse_lazy_mtp_layout", None)
    if not isinstance(layout, tuple) or len(layout) != 4 or not all(type(value) is int for value in layout):
        return False
    real_requests, requests, rows, input_rows = layout
    return (
        0 < real_requests <= requests
        and rows == 4 * real_requests
        and 0 < rows <= input_rows <= max_rows
        and input_rows % 4 == 0
        and metadata.num_actual_tokens == metadata.num_decode_tokens == rows
        and metadata.num_reqs == requests
        and real_requests <= metadata.num_decodes <= requests
        and metadata.num_prefills == 0
        and metadata.max_query_len == 4
        and metadata.hisparse_uniform_query_group_size == 0
    )


def _verify_contract(runtime_module: Any, worker_module: Any) -> None:
    from vllm_ascend.distributed.kv_transfer.hisparse.optimization_profile import verify_core_source

    for module, expected in (
        (runtime_module, _EXPECTED_RUNTIME_SHA256),
        (worker_module, _EXPECTED_WORKER_SHA256),
    ):
        source = getattr(module, "__file__", None)
        if source is None:
            raise RuntimeError("Re-review Ascend lazy MTP mirror for this common HiSparse source version.")
        verify_core_source(Path(source), expected)
    for source, expected in (
        (Path(runtime_module.__file__).with_name("coordinator.py"), _EXPECTED_COORDINATOR_SHA256),
        (Path(worker_module.__file__).with_name("connector.py"), _EXPECTED_CONNECTOR_SHA256),
    ):
        verify_core_source(source, expected)
    prepare = runtime_module.HiSparseCacheHandle._prepare_for_batch
    enqueue = worker_module.HiSparseConnectorWorker._enqueue_host_mirror
    if tuple(signature(prepare).parameters) != ("self", "attn_metadata"):
        raise RuntimeError("Unsupported HiSparse batch-preparation signature.")
    if Path(inspect.getsourcefile(enqueue)).resolve() != Path(worker_module.__file__).resolve():
        raise RuntimeError("Lazy MTP mirror requires the reviewed Core Host-mirror method.")
    if tuple(signature(enqueue).parameters) != ("self", "ready_event"):
        raise RuntimeError("Unsupported HiSparse Host-mirror signature.")
    if getattr(runtime_module.HiSparseRuntime._resolve_and_stage_group, "_ascend_certified_mtp_prefetch_patch", False):
        raise RuntimeError("Qualify lazy MTP mirror independently of the group-prefetch candidate.")


def _validate_consumer(config: Any) -> None:
    """Limit this candidate to the current PD consumer, before any capture."""
    transfer = config.kv_transfer_config
    extra = getattr(transfer, "kv_connector_extra_config", None)
    connectors = extra.get("connectors") if isinstance(extra, dict) else None
    # A direct HiSparseConnector has no network/export connector. Permit it
    # only to isolate consumer-side costs before the final paired P/D test.
    single_d = (
        transfer is not None
        and transfer.kv_connector == "HiSparseConnector"
        and transfer.kv_role == "kv_both"
        and connectors is None
    )
    if not single_d:
        if (
            transfer is None
            or transfer.kv_connector != "MultiConnector"
            or not isinstance(connectors, list)
            or len(connectors) != 2
            or not all(isinstance(item, dict) for item in connectors)
        ):
            raise RuntimeError("Lazy MTP mirror requires a local HiSparseConnector or reviewed P/D MultiConnector.")
        by_name = {item.get("kv_connector"): item for item in connectors}
        if (
            set(by_name) != {"HiSparseConnector", "HiSparsePDConnector"}
            or by_name["HiSparsePDConnector"].get("kv_role") != "kv_consumer"
        ):
            raise RuntimeError("Lazy MTP mirror has no producer/outbound-export tail flush qualification.")
    spec = config.speculative_config
    if (
        spec is None
        or spec.method != "mtp"
        or spec.num_speculative_tokens != 3
        or getattr(spec, "enable_adaptive_verification", False)
        or getattr(spec, "parallel_drafting", False)
        or config.model_config.enforce_eager
        or getattr(config.compilation_config.cudagraph_mode, "name", None) != "FULL_DECODE_ONLY"
        or config.attention_config.hisparse_config.eager_host_mirror is not False
    ):
        raise RuntimeError("Lazy MTP mirror candidate requires fixed MTP3/FULL and explicit eager_host_mirror=False.")


def install(config: Any) -> None:
    """Install before capture only in an explicitly selected D-only candidate."""
    _validate_consumer(config)
    metadata_module = import_module("vllm_ascend.attention.sfa_hisparse_optimized")
    source = getattr(metadata_module, "__file__", None)
    certificate = getattr(metadata_module, "_hisparse_uniform_mtp_group_size", None)
    actual = None if certificate is None else _certificate_source_digest(inspect.getsource(certificate))
    if source is None or actual != _EXPECTED_METADATA_SHA256:
        raise RuntimeError("Re-review the actual Ascend MTP layout certificate before lazy mirror installation.")
    common = import_module("vllm.v1.hisparse.runtime")
    worker_module = import_module("vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker")
    handle_cls = common.HiSparseCacheHandle
    worker_cls = worker_module.HiSparseConnectorWorker
    prepare_original = handle_cls._prepare_for_batch
    enqueue_original = worker_cls._enqueue_host_mirror
    markers = (getattr(prepare_original, _PATCH_MARKER, False), getattr(enqueue_original, _PATCH_MARKER, False))
    if all(markers):
        from .patch_hisparse_lazy_row_map import install as install_row_map

        install_row_map()
        return
    if any(markers):
        raise RuntimeError("Incomplete Ascend lazy MTP mirror installation.")
    _verify_contract(common, worker_module)

    @wraps(prepare_original)
    def prepare(self, attn_metadata):
        result = prepare_original(self, attn_metadata)
        enabled = (
            self.runtime.device.type == "npu"
            and self.runtime.eager_host_mirror is False
            and not self.decode_batch
            and _eligible(attn_metadata, self.runtime.max_swap_rows)
            # The new partial/padded scope is restricted to full Resident;
            # existing qualified C4/C8/C16 behavior is preserved for misses.
            and (attn_metadata.hisparse_uniform_query_group_size == 4 or self.all_context_pages_resident is True)
        )
        setattr(self, _BATCH_FLAG, enabled)
        if enabled:
            self.host_mirror_required = False
        return result

    @wraps(enqueue_original)
    def enqueue(self, ready_event=None):
        # The original still validates all active cache layers' mirror keys.
        # For the certified lazy batch it returns before DMA AND invalidation;
        # restore only the latter, using the original current-row mappings.
        result = enqueue_original(self, ready_event)
        active = [handle for handle in self.cache_handles if handle.num_actual_tokens != 0]
        lazy = [getattr(handle, _BATCH_FLAG, False) for handle in active]
        if not any(lazy):
            return result
        if not all(lazy) or any(
            handle.decode_batch or handle.host_mirror_required or handle.runtime.eager_host_mirror is not False
            for handle in active
        ):
            raise RuntimeError("HiSparse cache layers disagree on certified lazy MTP mirror policy.")
        cache = active[0]
        slots = cache.mirror_slot_mapping
        if slots is None:
            raise RuntimeError("HiSparse host mirror has no source slot mapping.")
        num_rows = min(cache.num_actual_tokens, slots.numel())
        if num_rows == 0:
            return result
        assert cache.req_id_per_token is not None
        # Core owns the grouped invalidator, including its per-binding tables.
        self._invalidate_written_slots(
            [handle.runtime for handle in active if handle.runtime.is_group_leader],
            slots[:num_rows],
            cache.req_id_per_token[:num_rows],
        )
        return result

    setattr(prepare, _PATCH_MARKER, True)
    setattr(enqueue, _PATCH_MARKER, True)
    handle_cls._prepare_for_batch = prepare
    worker_cls._enqueue_host_mirror = enqueue
    from .patch_hisparse_lazy_row_map import install as install_row_map

    install_row_map()
