#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import gc
import time
from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
from torch.nn import Module
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.distributed.parallel_state import get_ep_group, get_pp_group
from vllm.logger import logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.hardware_profile import get_current_hardware_profile
from vllm_ascend.model_loader.rfork.config import RForkConfig
from vllm_ascend.model_loader.rfork.identity import build_compatibility_fingerprint
from vllm_ascend.model_loader.rfork.session import RForkSession
from vllm_ascend.model_loader.rfork.types import (
    RForkFallbackCleanupResult,
    RForkIdentity,
    RForkSeedServiceStartResult,
)


class _RForkSeedUnavailable(RuntimeError):
    pass


@dataclass
class _RForkProcessGlobalModelState:
    """Snapshot process-global model registries before an RFork model attempt."""

    static_forward_context: tuple[dict[Any, Any], dict[Any, Any]] | None
    static_all_moe_layers: tuple[list[Any], list[Any]] | None
    rope_cache: dict[Any, Any] | None


def _is_rfork_summary_rank(session: RForkSession) -> bool:
    identity = getattr(session, "identity", None)
    return getattr(identity, "tp_rank", 0) == 0


def _rfork_model_kind(session: RForkSession) -> str:
    identity = getattr(session, "identity", None)
    return "draft" if getattr(identity, "is_draft_model", False) else "main"


def _start_rfork_seed_service(
    session: RForkSession,
    model: Module,
    processed_layout: bool,
    exclude_blocks: list[tuple[int, int]],
    *,
    load_source: str,
) -> bool:
    try:
        result = session.start_seed_service(model, processed_layout, exclude_blocks)
    except Exception:
        logger.exception(
            "RFork %s model loaded from %s, but seed service startup raised; inference can continue.",
            _rfork_model_kind(session),
            load_source,
        )
        return False
    if result is RForkSeedServiceStartResult.DEFERRED:
        return False
    started = bool(result)
    if not started:
        logger.warning(
            "RFork %s model loaded from %s is ready, but this worker is not currently advertised as a seed; "
            "inference can continue.",
            _rfork_model_kind(session),
            load_source,
        )
    return started


def _is_mtp_hf_config(hf_config: object | None) -> bool:
    if hf_config is None:
        return False

    model_type = getattr(hf_config, "model_type", None)
    if isinstance(model_type, str) and model_type.lower().endswith("_mtp"):
        return True

    architectures = getattr(hf_config, "architectures", None)
    if isinstance(architectures, str):
        architectures = [architectures]
    if not isinstance(architectures, (list, tuple)):
        return False

    return any(isinstance(architecture, str) and architecture.endswith("MTPModel") for architecture in architectures)


def _is_draft_model_config(model_config: object | None) -> bool:
    if model_config is None:
        return False
    if getattr(model_config, "runner_type", None) == "draft":
        return True

    return any(
        _is_mtp_hf_config(getattr(model_config, hf_config_attr, None))
        for hf_config_attr in ("hf_config", "hf_text_config")
    )


def _is_draft_model(vllm_config: VllmConfig, model_config: ModelConfig | None = None) -> bool:
    return (
        _is_draft_model_config(model_config)
        or _is_draft_model_config(getattr(vllm_config, "model_config", None))
        or _is_draft_model_config(getattr(vllm_config, "scheduler_config", None))
    )


def _get_rfork_session_attr(vllm_config: VllmConfig, model_config: ModelConfig) -> str:
    return "rfork_draft_session" if _is_draft_model(vllm_config, model_config) else "rfork_session"


def _get_ep_rank(vllm_config: VllmConfig) -> int | None:
    parallel_config = vllm_config.parallel_config
    if not parallel_config.enable_expert_parallel or getattr(parallel_config, "is_moe_model", None) is False:
        return None

    try:
        return get_ep_group().rank_in_group
    except AssertionError as e:
        raise RuntimeError("Expert parallelism is enabled, but the EP group is not initialized.") from e


def _get_pp_rank(vllm_config: VllmConfig) -> int | None:
    if getattr(vllm_config.parallel_config, "pipeline_parallel_size", 1) <= 1:
        return None

    try:
        return get_pp_group().rank_in_group
    except AssertionError as e:
        raise RuntimeError("Pipeline parallelism is enabled, but the PP group is not initialized.") from e


def _make_fallback_load_config(load_config: LoadConfig) -> LoadConfig:
    fallback_load_config = copy(load_config)
    fallback_load_config.load_format = "auto"
    fallback_load_config.model_loader_extra_config = {}
    return fallback_load_config


def _snapshot_process_global_model_state(vllm_config: VllmConfig) -> _RForkProcessGlobalModelState:
    """Snapshot registries that model construction can mutate in the process."""
    static_forward_context: tuple[dict[Any, Any], dict[Any, Any]] | None = None
    static_all_moe_layers: tuple[list[Any], list[Any]] | None = None
    compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is not None:
        forward_context = getattr(compilation_config, "static_forward_context", None)
        if isinstance(forward_context, dict):
            static_forward_context = (
                forward_context,
                dict(forward_context),
            )
        moe_layers = getattr(compilation_config, "static_all_moe_layers", None)
        if isinstance(moe_layers, list):
            static_all_moe_layers = (moe_layers, list(moe_layers))

    rope_cache: dict[Any, Any] | None = None
    try:
        from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT

        if isinstance(_ROPE_DICT, dict):
            rope_cache = dict(_ROPE_DICT)
    except Exception as e:  # pragma: no cover - best-effort across vLLM versions
        logger.debug("RFork fallback: skip snapshotting _ROPE_DICT: %s", e)

    return _RForkProcessGlobalModelState(static_forward_context, static_all_moe_layers, rope_cache)


def _reset_process_global_model_state(
    vllm_config: VllmConfig,
    model: Module | None = None,
    snapshot: _RForkProcessGlobalModelState | None = None,
) -> None:
    """Remove state owned by a discarded RFork model attempt.

    Model construction is serialized in a worker process, so restoring the exact
    pre-attempt snapshots preserves shared main/draft registrations as well as
    entries that were added by an earlier successful model load.
    """
    stale_module_ids = {id(module) for module in model.modules()} if model is not None else set()
    removed_names: set[Any] = set()
    compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is not None:
        static_forward_context = getattr(compilation_config, "static_forward_context", None)
        if snapshot is not None and snapshot.static_forward_context is not None:
            baseline_context, baseline_values = snapshot.static_forward_context
            baseline_context.clear()
            baseline_context.update(baseline_values)
            if static_forward_context is not baseline_context:
                compilation_config.static_forward_context = baseline_context
        elif isinstance(static_forward_context, dict):
            for name, module in list(static_forward_context.items()):
                if stale_module_ids and id(module) in stale_module_ids:
                    removed_names.add(name)
                    del static_forward_context[name]
            if not stale_module_ids:
                static_forward_context.clear()

        static_all_moe_layers = getattr(compilation_config, "static_all_moe_layers", None)
        if snapshot is not None and snapshot.static_all_moe_layers is not None:
            baseline_moe_layers, baseline_values = snapshot.static_all_moe_layers
            baseline_moe_layers[:] = baseline_values
            if static_all_moe_layers is not baseline_moe_layers:
                compilation_config.static_all_moe_layers = baseline_moe_layers
        elif isinstance(static_all_moe_layers, list):
            if stale_module_ids:
                static_all_moe_layers[:] = [
                    layer
                    for layer in static_all_moe_layers
                    if id(layer) not in stale_module_ids and (not isinstance(layer, str) or layer not in removed_names)
                ]
            else:
                static_all_moe_layers.clear()

    try:
        from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT

        if isinstance(_ROPE_DICT, dict):
            if snapshot is not None and snapshot.rope_cache is not None:
                _ROPE_DICT.clear()
                _ROPE_DICT.update(snapshot.rope_cache)
            else:
                _ROPE_DICT.clear()
    except Exception as e:  # pragma: no cover - best-effort across vLLM versions
        logger.debug("RFork fallback: skip resetting _ROPE_DICT: %s", e)


def _iter_ascend_moe_quant_methods(model: Module) -> Iterator[Any]:
    """Yield each quant method owned by an Ascend MoE runner once."""
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner

    seen_quant_methods: set[int] = set()
    for module in model.modules():
        if not isinstance(module, AscendMoERunner):
            continue

        # AscendMoERunner exposes the routed experts' method via private _quant_method.
        quant_method = getattr(module, "_quant_method", None)
        if quant_method is None or id(quant_method) in seen_quant_methods:
            continue

        seen_quant_methods.add(id(quant_method))
        yield cast(Any, quant_method)


@contextmanager
def _rfork_pre_transfer_weight_processing(model: Module):
    """Use the unwrapped MoE post-load step so RFork pre-transfer skips shared-expert validation."""
    restored: list[tuple[Any, object]] = []
    for quant_method in _iter_ascend_moe_quant_methods(model):
        process_weights = getattr(quant_method, "process_weights_after_loading", None)
        original_process_weights = getattr(process_weights, "__wrapped__", None)
        if original_process_weights is None:
            continue

        restored.append((quant_method, process_weights))
        quant_method.process_weights_after_loading = original_process_weights

    try:
        yield
    finally:
        for quant_method, process_weights in restored:
            quant_method.process_weights_after_loading = process_weights


def _is_dynamic_eplb_enabled(vllm_config: VllmConfig) -> bool:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if bool(getattr(parallel_config, "enable_eplb", False)):
        return True

    eplb_config = get_ascend_config().eplb_config
    return eplb_config.dynamic_eplb or bool(eplb_config.expert_map_record_path)


@contextmanager
def _rfork_skip_unquantized_moe_post_load_processing(model: Module):
    """Suppress unquantized MoE post-load processing; dense layers still run theirs."""

    from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod

    restored_methods: list[tuple[Any, object]] = []
    for quant_method in _iter_ascend_moe_quant_methods(model):
        if not isinstance(quant_method, AscendUnquantizedFusedMoEMethod):
            continue

        process_weights = getattr(quant_method, "process_weights_after_loading", None)
        if process_weights is None:
            continue

        restored_methods.append((quant_method, process_weights))
        quant_method.process_weights_after_loading = _noop_process_weights_after_loading  # type: ignore[method-assign]

    try:
        yield
    finally:
        for quant_method, process_weights in restored_methods:
            quant_method.process_weights_after_loading = process_weights


def _noop_process_weights_after_loading(*args: Any, **kwargs: Any) -> None:
    pass


@register_model_loader("rfork")
class RForkModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self.rfork_config = RForkConfig.from_extra_config(load_config.model_loader_extra_config)

        config = self.rfork_config
        logger.debug(
            "Initializing rfork with config: "
            "MODEL_URL=%s, MODEL_DEPLOY_STRATEGY_NAME=%s, "
            "SCHEDULER_URL=%s, SEED_TIMEOUT_SEC=%s, REQUEST_TIMEOUT_SEC=%s, "
            "SEED_BIND_HOST=%s, SEED_ADVERTISE_HOST=%s",
            config.model_url,
            config.model_deploy_strategy_name,
            config.planner_url,
            config.seed_timeout_sec,
            config.request_timeout_sec,
            config.seed_bind_host,
            config.seed_advertise_host,
        )

    def download_model(self, model_config: ModelConfig) -> None:
        raise NotImplementedError

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        raise NotImplementedError

    def _ensure_rfork_session(self, vllm_config: VllmConfig, model_config: ModelConfig) -> RForkSession:
        session_attr = _get_rfork_session_attr(vllm_config, model_config)
        session = getattr(self.load_config, session_attr, None)
        if session is None:
            is_draft_model = _is_draft_model(vllm_config, model_config)
            global_rank = torch.distributed.get_rank()
            pp_rank = _get_pp_rank(vllm_config)
            ep_rank = _get_ep_rank(vllm_config)
            tp_rank = get_tensor_model_parallel_rank()
            compatibility_fingerprint = build_compatibility_fingerprint(
                vllm_config,
                model_config,
                model_url=self.rfork_config.model_url,
                model_deploy_strategy_name=self.rfork_config.model_deploy_strategy_name,
            )
            identity = RForkIdentity(
                tp_rank=tp_rank,
                global_rank=global_rank,
                is_draft_model=is_draft_model,
                pp_rank=pp_rank,
                ep_rank=ep_rank,
                compatibility_fingerprint=compatibility_fingerprint,
            )
            session = RForkSession(self.rfork_config, identity)
            setattr(self.load_config, session_attr, session)
            if tp_rank == 0:
                logger.info(
                    "RFork %s session group initialized: model=%s, strategy=%s, planner=%s, "
                    "pp_rank=%s, ep_rank=%s, fingerprint=%s",
                    "draft" if is_draft_model else "main",
                    self.rfork_config.model_url,
                    self.rfork_config.model_deploy_strategy_name,
                    self.rfork_config.planner_url,
                    pp_rank,
                    ep_rank,
                    compatibility_fingerprint,
                )
            else:
                logger.debug(
                    "RFork session initialized: model_kind=%s, tp_rank=%s, global_rank=%s, "
                    "pp_rank=%s, ep_rank=%s, session_attr=%s, fingerprint=%s",
                    "draft" if is_draft_model else "main",
                    tp_rank,
                    global_rank,
                    pp_rank,
                    ep_rank,
                    session_attr,
                    compatibility_fingerprint,
                )
        return session

    def _get_target_registered_blocks(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
    ) -> list[tuple[int, int]]:
        if not _is_draft_model(vllm_config, model_config):
            return []
        target_session = getattr(self.load_config, "rfork_session", None)
        target_transfer_backend = getattr(target_session, "transfer_backend", None)
        return list(getattr(target_transfer_backend, "registered_weight_blocks", None) or [])

    def _requires_processed_layout_transfer(self, model_config: ModelConfig) -> bool:
        if getattr(model_config, "quantization", None) is not None:
            return True

        try:
            weight_nz_mode = getattr(get_ascend_config(), "weight_nz_mode", 0)
            if not isinstance(weight_nz_mode, bool) and int(weight_nz_mode) == 2:
                return True
        except (TypeError, ValueError, RuntimeError):
            # AscendConfig may be unavailable during early or CPU-only loader inspection.
            pass

        try:
            hardware_policy = getattr(get_current_hardware_profile(), "weight_layout_policy", None)
            return getattr(hardware_policy, "name", str(hardware_policy).split(".")[-1]) == "FORCE_NZ"
        except (AttributeError, RuntimeError):
            return False

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> Module | None:
        device_config = vllm_config.device_config
        load_config = self.load_config
        load_device = device_config.device if load_config.device is None else load_config.device
        target_device = torch.device(load_device)

        with set_default_torch_dtype(model_config.dtype):
            need_del = False
            model_init_started = False
            model: Module | None = None
            session: RForkSession | None = None
            model_state_snapshot: _RForkProcessGlobalModelState | None = None
            exclude_blocks: list[tuple[int, int]] = []
            processed_layout_transfer = self._requires_processed_layout_transfer(model_config)
            bypass_reason = None
            if _is_dynamic_eplb_enabled(vllm_config):
                bypass_reason = "dynamic EPLB"

            if bypass_reason is not None:
                logger.warning(
                    "RFork transfer is disabled when %s is enabled; using the default model loader.",
                    bypass_reason,
                )
                fallback_load_config = _make_fallback_load_config(self.load_config)

                from vllm.model_executor.model_loader import get_model

                try:
                    return get_model(
                        vllm_config=vllm_config,
                        model_config=model_config,
                        load_config=fallback_load_config,
                        prefix=prefix,
                    )
                except Exception:
                    logger.exception("RFork disabled for %s, but default loader failed.", bypass_reason)
                    raise

            try:
                # Keep session and TransferEngine initialization inside fallback-protected RFork loading.
                session = self._ensure_rfork_session(vllm_config, model_config)
                # Avoid re-registering target-model storage shared by draft workers.
                exclude_blocks = self._get_target_registered_blocks(vllm_config, model_config)

                model_state_snapshot = _snapshot_process_global_model_state(vllm_config)
                model_init_started = True
                model_init_start_time = time.perf_counter()
                with target_device:
                    model = initialize_model(
                        vllm_config=vllm_config,
                        model_config=model_config,
                        prefix=prefix,
                    )
                    need_del = True
                logger.info(
                    "RFork %s model initialization took %.2f seconds",
                    _rfork_model_kind(session),
                    time.perf_counter() - model_init_start_time,
                )

                if processed_layout_transfer:
                    layout_start_time = time.perf_counter()
                    log_layout = logger.info if _is_rfork_summary_rank(session) else logger.debug
                    log_layout(
                        "RFork %s model uses post-load tensor layout transfer.",
                        _rfork_model_kind(session),
                    )
                    with _rfork_pre_transfer_weight_processing(model):
                        process_weights_after_loading(model, model_config, target_device)
                    # Complete async NPU layout conversion before exposing buffers.
                    torch.npu.synchronize()
                    logger.info(
                        "RFork %s model layout processing took %.2f seconds",
                        _rfork_model_kind(session),
                        time.perf_counter() - layout_start_time,
                    )

                acquire_seed_start_time = time.perf_counter()
                try:
                    acquired_seed = session.acquire_seed()
                finally:
                    logger.info(
                        "RFork %s seed acquisition took %.2f seconds",
                        _rfork_model_kind(session),
                        time.perf_counter() - acquire_seed_start_time,
                    )
                if not acquired_seed:
                    raise _RForkSeedUnavailable("planner returned no compatible seed")

                weight_load_start_time = time.perf_counter()
                if not session.transfer_from_seed(model, processed_layout_transfer, exclude_blocks):
                    raise RuntimeError("transfer failed.")
                logger.info(
                    "RFork %s model registration and transfer took %.2f seconds",
                    _rfork_model_kind(session),
                    time.perf_counter() - weight_load_start_time,
                )

                if not processed_layout_transfer:
                    with _rfork_skip_unquantized_moe_post_load_processing(model):
                        process_weights_after_loading(model, model_config, target_device)

                # Advertise only after post-load and eval; the session owns failure cleanup.
                model = model.eval()
                _start_rfork_seed_service(
                    session,
                    model,
                    processed_layout_transfer,
                    exclude_blocks,
                    load_source="transfer",
                )
                return model
            except _RForkSeedUnavailable:
                assert session is not None
                log_seed_miss = logger.info if _is_rfork_summary_rank(session) else logger.debug
                log_seed_miss(
                    "RFork has no available %s seed; loading locally and registering this worker as a seed.",
                    _rfork_model_kind(session),
                )
            except Exception as e:
                logger.warning("RFork transfer failed: %s, clean up and fall back to default loader", e)

            cleanup_result: bool | RForkFallbackCleanupResult = False
            if session is not None:
                cleanup_result = session.prepare_for_fallback()

            if model_init_started:
                _reset_process_global_model_state(vllm_config, model, model_state_snapshot)

            if need_del and model is not None:
                del model
                gc.collect()
                torch.npu.empty_cache()
                for _ in range(3):
                    gc.collect()
                    torch.npu.empty_cache()

            fallback_load_config = _make_fallback_load_config(self.load_config)

            from vllm.model_executor.model_loader import get_model

            try:
                model = get_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    load_config=fallback_load_config,
                    prefix=prefix,
                )
            except Exception:
                logger.exception("RFork fallback default loader failed.")
                raise

            # Advertise fallback only via an existing session; startup failure cleans its MR but keeps the model.
            can_start_fallback_seed = bool(cleanup_result)
            if isinstance(cleanup_result, RForkFallbackCleanupResult):
                can_start_fallback_seed = cleanup_result.can_schedule_seed
            if session is not None and can_start_fallback_seed:
                _start_rfork_seed_service(
                    session,
                    model,
                    processed_layout_transfer,
                    exclude_blocks,
                    load_source="fallback",
                )
            return model
