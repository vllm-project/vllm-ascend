#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import json
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, ClassVar, cast
from weakref import ReferenceType, ref

import torch
from torch import nn
from vllm.config import LoadConfig, ModelConfig, VllmConfig, get_current_vllm_config_or_none
from vllm.logger import logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import initialize_model, process_weights_after_loading
from vllm.utils.torch_utils import set_default_torch_dtype

from .executor.elastic_load import cache_processed_layout_transfer_manifest, synchronize_npu
from .interaction.elastic import ElasticServer
from .load import elastic_load
from .utils import find_free_port, is_valid_path_prefix

DRAFT_PORT_OFFSET = 10000
MAX_FREE_PORT_RETRIES = 5
MAX_FALLBACK_MEMORY_RECLAIM_ATTEMPTS = 2
BYTES_PER_MIB = 1024**2


@dataclass
class _FallbackCleanupContext:
    context_keys: dict[int, set[Any]]
    static_all_moe_layers: dict[int, tuple[list[Any], list[Any]]]
    rope_cache: dict[Any, Any] | None
    eplb_layers: list[nn.Module]
    eplb_layer_count: int
    memory_baseline: tuple[int, int] | None


@dataclass
class _FallbackReclaimResult:
    attempts: int
    before: tuple[int, int] | None
    after: tuple[int, int] | None
    reached_baseline: bool


@contextmanager
def pre_transfer_weight_processing(model: nn.Module):
    """Unwrap MoE shared-expert validation during pre-transfer process_weights."""
    try:
        from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    except ImportError:
        AscendMoERunner = None  # type: ignore[misc, assignment]

    restored: list[tuple[Any, object]] = []
    seen_quant_methods: set[int] = set()

    def _unwrap_quant_method(quant_method: Any) -> None:
        if quant_method is None or id(quant_method) in seen_quant_methods:
            return

        process_weights = getattr(quant_method, "process_weights_after_loading", None)
        original_process_weights = getattr(process_weights, "__wrapped__", None)
        if original_process_weights is None:
            return

        quant_method = cast(Any, quant_method)
        seen_quant_methods.add(id(quant_method))
        restored.append((quant_method, process_weights))
        quant_method.process_weights_after_loading = original_process_weights

    for module in model.modules():
        if AscendMoERunner is not None and isinstance(module, AscendMoERunner):
            _unwrap_quant_method(module._quant_method)

    try:
        yield
    finally:
        for quant_method, process_weights in restored:
            quant_method.process_weights_after_loading = process_weights


@register_model_loader("netloader")
class ModelNetLoaderElastic(BaseModelLoader):
    """
    A model loader that uses elastic loading for loading weights.
    """

    # Shared across loader instances in one worker: draft uses a separate
    # draft_vllm_config, so an instance/config flag would not be visible there.
    _target_elastic_fallback: ClassVar[bool] = False

    source: list[dict] | None
    model_path: str | None
    listen_port: int | None
    int8_cache: str
    int8_cache_name: list[str] | None
    output_prefix: str | None

    def __init__(self, load_config: LoadConfig):
        """
        Initializes the ModelNetLoaderElastic with configuration.

        Parameters:
        - load_config: Configuration for loading the model.
        """
        super().__init__(load_config)

        config = None

        # Try to read config file at first
        extra = load_config.model_loader_extra_config

        if extra is not None and not isinstance(extra, dict):
            err_msg = "NetLoader requires --model-loader-extra-config to be a JSON object."
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        if extra and "CONFIG_FILE" in extra:
            try:
                logger.info("Reading configs in file %s ...", load_config.model_loader_extra_config["CONFIG_FILE"])
                with open(extra["CONFIG_FILE"]) as f:
                    config = json.load(f)
            except FileNotFoundError:
                logger.error("CONFIG_FILE not found")
            except json.JSONDecodeError:
                logger.error("CONFIG_FILE is not a valid JSON file")
            except Exception as e:
                logger.error("Unexpected error while reading CONFIG_FILE: %s", e)

        if config is None and extra:
            logger.info("Reading configs in model_loader_extra_config ...")
            config = extra
        config = config or {}

        for key, attr, checker, caster, default in [
            ("SOURCE", "source", lambda v: isinstance(v, list), lambda v: v, None),
            ("MODEL", "model_path", lambda v: isinstance(v, str), lambda v: v, None),
            (
                "LISTEN_PORT",
                "listen_port",
                lambda v: isinstance(v, int) or (isinstance(v, str) and v.isdigit()),
                lambda v: int(v),
                None,
            ),
            (
                "INT8_CACHE",
                "int8_cache",
                lambda v: isinstance(v, str) and v.lower() in ["hbm", "dram", "no"],
                lambda v: v.lower(),
                "no",
            ),
            ("INT8_CACHE_NAME", "int8_cache_name", lambda v: isinstance(v, list), lambda v: v, None),
            (
                "OUTPUT_PREFIX",
                "output_prefix",
                lambda v: isinstance(v, str) and is_valid_path_prefix(v),
                lambda v: v,
                None,
            ),
        ]:
            v = config.get(key, default)
            if not checker(v):
                v = default
            else:
                v = caster(v)
            setattr(self, attr, v)

        logger.info(
            "Initializing elastic Netloader with config: "
            "MODEL=%s, LISTEN_PORT=%s,"
            "SOURCE=%s, INT8_CACHE=%s, INT8_CACHE_NAME=%s,"
            "OUTPUT_PREFIX=%s)",
            self.model_path,
            self.listen_port,
            self.source,
            self.int8_cache,
            self.int8_cache_name,
            self.output_prefix,
        )

    @staticmethod
    def _is_draft_model(model_config: ModelConfig) -> bool:
        """Check whether the model_config corresponds to a draft model for speculative decoding."""
        return getattr(model_config, "runner_type", None) == "draft"

    @staticmethod
    def _sync_target_netloader_before_draft(vllm_config: VllmConfig) -> None:
        if getattr(vllm_config, "speculative_config", None) is None:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        local_world_size = getattr(vllm_config.parallel_config, "local_world_size", world_size)
        if not isinstance(local_world_size, int) or local_world_size <= 0 or world_size % local_world_size != 0:
            local_world_size = world_size

        # Same-node ranks only; skip cross-node wait for disk/HBM/NIC contention.
        group = None
        if local_world_size < world_size:
            my_node = rank // local_world_size
            # new_group is world-collective: every rank must create every node group.
            for node_idx in range(world_size // local_world_size):
                ranks = list(range(node_idx * local_world_size, (node_idx + 1) * local_world_size))
                pg = torch.distributed.new_group(ranks=ranks)
                if node_idx == my_node:
                    group = pg

        logger.info(
            "Waiting for local target netloader ranks before loading draft model (local_world_size=%s)",
            local_world_size,
        )
        barrier_start = time.perf_counter()
        torch.distributed.barrier(group=group)
        logger.info(
            "Target netloader barrier before draft model time: %s",
            time.perf_counter() - barrier_start,
        )

    @staticmethod
    def _get_static_forward_context(vllm_config: VllmConfig):
        compilation_config = getattr(vllm_config, "compilation_config", None)
        static_forward_context = getattr(compilation_config, "static_forward_context", None)
        if static_forward_context is None or not hasattr(static_forward_context, "clear"):
            return None
        return static_forward_context

    @staticmethod
    def _iter_compilation_configs(vllm_config: VllmConfig):
        """Yield unique compilation configs for the passed and current vLLM configs."""
        candidates = [("vllm_config", vllm_config)]
        current_vllm_config = get_current_vllm_config_or_none()
        if current_vllm_config is not None:
            candidates.append(("current_vllm_config", current_vllm_config))

        seen_config_ids: set[int] = set()
        for source, config in candidates:
            compilation_config = getattr(config, "compilation_config", None)
            if compilation_config is None:
                continue
            config_id = id(compilation_config)
            if config_id in seen_config_ids:
                continue
            seen_config_ids.add(config_id)
            yield source, compilation_config

    @staticmethod
    def _iter_static_forward_contexts(vllm_config: VllmConfig):
        """Yield unique (source_name, context_dict) pairs for this and current vllm config."""
        seen_context_ids: set[int] = set()
        for source, compilation_config in ModelNetLoaderElastic._iter_compilation_configs(vllm_config):
            static_forward_context = getattr(compilation_config, "static_forward_context", None)
            if static_forward_context is None or not hasattr(static_forward_context, "clear"):
                continue
            context_id = id(static_forward_context)
            if context_id in seen_context_ids:
                continue
            seen_context_ids.add(context_id)
            yield source, static_forward_context

    @staticmethod
    def _snapshot_static_forward_context_keys(vllm_config: VllmConfig) -> dict[int, set[Any]]:
        """Snapshot context keys before initialize_model so fallback can drop only new ones."""
        snapshots: dict[int, set[Any]] = {}
        for _, static_forward_context in ModelNetLoaderElastic._iter_static_forward_contexts(vllm_config):
            try:
                snapshots[id(static_forward_context)] = set(static_forward_context.keys())
            except TypeError:
                snapshots[id(static_forward_context)] = set()
        return snapshots

    @staticmethod
    def _remove_new_static_forward_context_keys(
        vllm_config: VllmConfig,
        snapshots: dict[int, set[Any]],
        failed_model: nn.Module | None = None,
    ) -> None:
        """Remove new keys and any registrations owned by the failed model."""
        stale_modules = set(failed_model.modules()) if failed_model is not None else set()
        for _, static_forward_context in ModelNetLoaderElastic._iter_static_forward_contexts(vllm_config):
            keep_keys = snapshots.get(id(static_forward_context), set())
            new_keys = [
                key
                for key, module in list(static_forward_context.items())
                if key not in keep_keys or module in stale_modules
            ]
            for key in new_keys:
                del static_forward_context[key]

    @staticmethod
    def _snapshot_static_all_moe_layers(
        vllm_config: VllmConfig,
    ) -> dict[int, tuple[list[Any], list[Any]]]:
        """Snapshot shared MoE registries so fallback preserves existing target layers."""
        snapshots: dict[int, tuple[list[Any], list[Any]]] = {}
        for _, compilation_config in ModelNetLoaderElastic._iter_compilation_configs(vllm_config):
            static_all_moe_layers = getattr(compilation_config, "static_all_moe_layers", None)
            if isinstance(static_all_moe_layers, list) and id(static_all_moe_layers) not in snapshots:
                snapshots[id(static_all_moe_layers)] = (static_all_moe_layers, list(static_all_moe_layers))
        return snapshots

    @staticmethod
    def _restore_static_all_moe_layers(snapshots: dict[int, tuple[list[Any], list[Any]]]) -> None:
        for static_all_moe_layers, baseline_layers in snapshots.values():
            static_all_moe_layers[:] = baseline_layers

    @staticmethod
    def _snapshot_rope_cache() -> dict[Any, Any] | None:
        try:
            from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT

            return dict(_ROPE_DICT) if isinstance(_ROPE_DICT, dict) else None
        except Exception as exc:
            logger.debug("Netloader fallback: skip snapshotting _ROPE_DICT: %s", exc)
            return None

    @staticmethod
    def _restore_rope_cache(snapshot: dict[Any, Any] | None) -> None:
        if snapshot is None:
            return
        try:
            from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT

            if isinstance(_ROPE_DICT, dict):
                _ROPE_DICT.clear()
                _ROPE_DICT.update(snapshot)
        except Exception as exc:
            logger.debug("Netloader fallback: skip restoring _ROPE_DICT: %s", exc)

    @staticmethod
    def _cleanup_compilation_hooks(model: nn.Module) -> None:
        """Remove bytecode hooks that otherwise pin a failed compiled model."""
        try:
            from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

            for module in model.modules():
                if isinstance(module, TorchCompileWithNoGuardsWrapper):
                    module.cleanup()
        except Exception as exc:
            logger.warning("Netloader fallback: failed to remove compilation bytecode hooks: %s", exc)

    @classmethod
    def _create_fallback_cleanup_context(
        cls,
        vllm_config: VllmConfig,
        device_type: str,
    ) -> _FallbackCleanupContext:
        from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor

        eplb_layers = VllmEplbAdaptor._registered_moe_layers
        return _FallbackCleanupContext(
            context_keys=cls._snapshot_static_forward_context_keys(vllm_config),
            static_all_moe_layers=cls._snapshot_static_all_moe_layers(vllm_config),
            rope_cache=cls._snapshot_rope_cache(),
            eplb_layers=eplb_layers,
            eplb_layer_count=len(eplb_layers),
            memory_baseline=cls._get_npu_memory_usage(device_type),
        )

    @classmethod
    def _release_failed_model_references(
        cls,
        model: nn.Module,
        vllm_config: VllmConfig,
        context: _FallbackCleanupContext,
    ) -> ReferenceType[nn.Module]:
        """Detach process-global references before the caller drops the model."""
        model_ref = ref(model)
        cls._cleanup_compilation_hooks(model)
        cls._remove_new_static_forward_context_keys(vllm_config, context.context_keys, model)
        cls._restore_static_all_moe_layers(context.static_all_moe_layers)
        cls._restore_rope_cache(context.rope_cache)
        del context.eplb_layers[context.eplb_layer_count :]
        return model_ref

    @staticmethod
    def _get_npu_memory_usage(device_type: str) -> tuple[int, int] | None:
        if device_type != "npu":
            return None
        try:
            return int(torch.npu.memory_allocated()), int(torch.npu.memory_reserved())
        except Exception as exc:
            logger.debug("Netloader fallback: failed to query NPU memory usage: %s", exc)
            return None

    @staticmethod
    def _reclaim_failed_model_memory(
        device_type: str,
        memory_baseline: tuple[int, int] | None,
    ) -> _FallbackReclaimResult:
        if device_type != "npu":
            gc.collect()
            if device_type == "cuda":
                torch.cuda.empty_cache()
            return _FallbackReclaimResult(1, None, None, False)

        baseline_allocated = memory_baseline[0] if memory_baseline is not None else None
        before = ModelNetLoaderElastic._get_npu_memory_usage(device_type)
        after = None
        reached_baseline = False
        for attempt in range(1, MAX_FALLBACK_MEMORY_RECLAIM_ATTEMPTS + 1):
            gc.collect()
            torch.npu.empty_cache()
            after = ModelNetLoaderElastic._get_npu_memory_usage(device_type)
            reached_baseline = baseline_allocated is not None and after is not None and after[0] <= baseline_allocated
            if reached_baseline:
                break
        return _FallbackReclaimResult(attempt, before, after, reached_baseline)

    def load_model(self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = "") -> nn.Module:
        """
        Loads the model using the specified configuration.

        Parameters:
        - vllm_config: Configuration for the VLLM.
        - model_config: Configuration for the model.
        - prefix: Module prefix for pipeline parallelism (e.g., "model.layers.0.").

        Returns:
        - The loaded model.
        """

        device_config = vllm_config.device_config
        parallel_config = vllm_config.parallel_config

        need_process_weights_after_loading = False

        if self.model_path is None:
            self.model_path = model_config.model
            logger.info("model_path is set to %s", self.model_path)

        device_id = torch.distributed.get_rank()
        is_draft = self._is_draft_model(model_config)
        load_int8_cache = "hbm" if is_draft and self.int8_cache != "no" else self.int8_cache

        if is_draft:
            logger.info("Loading draft model via netloader, model_path: %s", model_config.model)
        else:
            logger.info("Loading target model via netloader, model_path: %s", model_config.model)

        # After target elastic fallback, skip another draft P2P attempt on the same rank.
        skip_draft_elastic = is_draft and ModelNetLoaderElastic._target_elastic_fallback
        has_valid_source = (
            self.source is not None
            and isinstance(self.source, list)
            and device_id
            in [
                one_device["device_id"]
                for one_device in self.source
                if isinstance(one_device, dict) and "device_id" in one_device
            ]
        )

        if skip_draft_elastic:
            logger.warning(
                "Target netloader already fell back to DefaultModelLoader; "
                "skip draft elastic load and use DefaultModelLoader"
            )
            model, need_process_weights_after_loading = self.revert_to_default(
                model_config, vllm_config, device_config, prefix
            )
        elif not has_valid_source:
            logger.warning("Did not get valid source info, use DefaultModelLoader")
            model, need_process_weights_after_loading = self.revert_to_default(
                model_config, vllm_config, device_config, prefix
            )

        else:
            target_device = torch.device(device_config.device)

            _quant_config = getattr(vllm_config, "quant_config", None)
            _quant_config = deepcopy(_quant_config) if _quant_config is not None else None
            model_config_backup = deepcopy(model_config)
            fallback_cleanup_context = self._create_fallback_cleanup_context(
                vllm_config,
                device_config.device_type,
            )

            with set_default_torch_dtype(model_config.dtype):
                with target_device:
                    model = initialize_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)
                elastic_model = model
                fallback_reason = "elastic load failed"

                if load_int8_cache == "no":
                    try:
                        with pre_transfer_weight_processing(model):
                            process_weights_after_loading(model, model_config, torch.device(device_config.device))
                        synchronize_npu(device_config.device_type)
                        manifest_build_start = time.perf_counter()
                        manifest_count = cache_processed_layout_transfer_manifest(model)
                        logger.info(
                            "Netloader manifest build time: %s, rank: %s, manifest=%s",
                            time.perf_counter() - manifest_build_start,
                            device_id,
                            manifest_count,
                        )
                    except Exception as exc:
                        fallback_reason = f"pre-recv preparation failed: {exc}"
                        model = None
                # Narrowed by has_valid_source above.
                assert self.source is not None
                sources: list[dict] = self.source
                if is_draft:
                    sources = [
                        {
                            "device_id": s["device_id"],
                            "sources": [
                                f"{parts[0]}:{int(parts[1]) + DRAFT_PORT_OFFSET}"
                                for addr in s.get("sources", [])
                                if isinstance(addr, str)
                                and len(parts := addr.rsplit(":", 1)) == 2
                                and parts[1].isdigit()
                            ],
                        }
                        for s in sources
                        if isinstance(s, dict) and "device_id" in s
                    ]
                    if any(int(addr.rsplit(":", 1)[1]) > 65535 for s in sources for addr in s.get("sources", [])):
                        fallback_reason = "draft source port exceeds 65535"
                        model = None

                if model is not None:
                    model = elastic_load(
                        model=elastic_model,
                        device_id=device_id,
                        model_path=model_config.model,
                        sources=sources,
                        tp=parallel_config.tensor_parallel_size,
                        pp=parallel_config.pipeline_parallel_size,
                        group_name="netloader_draft" if is_draft else "netloader",
                        int8_cache=load_int8_cache,
                    )
                need_process_weights_after_loading = load_int8_cache != "no"

                if model is None:
                    logger.warning(
                        "Netloader load failed, fallback to DefaultModelLoader, rank=%s, reason=%s",
                        device_id,
                        fallback_reason,
                    )

                    if hasattr(vllm_config, "quant_config"):
                        vllm_config.quant_config = _quant_config
                    model_config = model_config_backup

                    elastic_model_ref = self._release_failed_model_references(
                        elastic_model,
                        vllm_config,
                        fallback_cleanup_context,
                    )
                    elastic_model = None
                    reclaim_result = self._reclaim_failed_model_memory(
                        device_config.device_type,
                        fallback_cleanup_context.memory_baseline,
                    )
                    baseline_allocated = (
                        fallback_cleanup_context.memory_baseline[0]
                        if fallback_cleanup_context.memory_baseline is not None
                        else None
                    )
                    logger.info(
                        "[netloader_client] stage=fallback_cleanup rank=%s attempts=%s model_released=%s "
                        "baseline_allocated_mib=%s before_allocated_mib=%s after_allocated_mib=%s "
                        "after_reserved_mib=%s reached_baseline=%s",
                        device_id,
                        reclaim_result.attempts,
                        elastic_model_ref() is None,
                        f"{baseline_allocated / BYTES_PER_MIB:.2f}" if baseline_allocated is not None else "unknown",
                        f"{reclaim_result.before[0] / BYTES_PER_MIB:.2f}"
                        if reclaim_result.before is not None
                        else "unknown",
                        f"{reclaim_result.after[0] / BYTES_PER_MIB:.2f}"
                        if reclaim_result.after is not None
                        else "unknown",
                        f"{reclaim_result.after[1] / BYTES_PER_MIB:.2f}"
                        if reclaim_result.after is not None
                        else "unknown",
                        reclaim_result.reached_baseline,
                    )

                    if not is_draft:
                        ModelNetLoaderElastic._target_elastic_fallback = True

                    model, need_process_weights_after_loading = self.revert_to_default(
                        model_config, vllm_config, device_config, prefix
                    )
                elif not is_draft:
                    ModelNetLoaderElastic._target_elastic_fallback = False

        if load_int8_cache == "no" and need_process_weights_after_loading:
            process_weights_after_loading(model, model_config, torch.device(device_config.device))
            synchronize_npu(device_config.device_type)
            manifest_count = cache_processed_layout_transfer_manifest(model)
            need_process_weights_after_loading = False
            logger.info(
                "Netloader seed process_weights done, rank: %s, manifest=%s",
                device_id,
                manifest_count,
            )

        elastic_server = None
        # start elastic server
        if model is not None and (
            (self.listen_port and self.listen_port in range(1024, 65535)) or (self.listen_port is None)
        ):
            from vllm.utils.network_utils import get_ip

            driver_ip = get_ip()

            if driver_ip == "0.0.0.0":
                logger.error("Driver IP is not set, skip to start Netloader server")
            else:
                if self.listen_port is None:
                    listen_port = find_free_port()
                    for _ in range(MAX_FREE_PORT_RETRIES):
                        if listen_port <= 65535 - DRAFT_PORT_OFFSET:
                            break
                        listen_port = find_free_port()
                    else:
                        logger.warning(
                            "Failed to find listen port <= %s after %s retries; skip Netloader server",
                            65535 - DRAFT_PORT_OFFSET,
                            MAX_FREE_PORT_RETRIES,
                        )
                        listen_port = -1
                else:
                    listen_port = self.listen_port + device_id
                if is_draft:
                    listen_port += DRAFT_PORT_OFFSET
                if listen_port < 1024 or listen_port > 65535:
                    logger.warning(
                        "Skip %s Netloader server due to invalid listen port: %s",
                        "draft" if is_draft else "target",
                        listen_port,
                    )
                else:
                    self.listen_port = listen_port

                    group_name = "netloader_draft" if is_draft else "netloader"

                    logger.info(
                        "Start elastic Netloader server, rank: %s, listen port: %s:%s, group: %s",
                        device_id,
                        driver_ip,
                        listen_port,
                        group_name,
                    )

                    if self.output_prefix is not None and not is_draft:
                        try:
                            with open(self.output_prefix + str(device_id) + ".txt", "w") as file:
                                file.write(f"{driver_ip}:{listen_port}")
                            logger.info(
                                "Successfully wrote server address to file: %s", self.output_prefix + str(device_id)
                            )
                        except FileNotFoundError:
                            logger.error("File path %s does not exist.", self.output_prefix + str(device_id))
                        except PermissionError:
                            logger.error("No permission to write to file %s.", self.output_prefix + str(device_id))
                        except OSError as e:
                            logger.error(
                                "I/O error occurred while writing to file %s: %s",
                                self.output_prefix + str(device_id),
                                e,
                            )
                        except Exception as e:
                            logger.error("Unknown error: %s", e)

                    try:
                        elastic_server = ElasticServer(
                            driver_ip,
                            listen_port,
                            model,
                            device_id,
                            model_config.model,
                            parallel_config.tensor_parallel_size,
                            parallel_config.pipeline_parallel_size,
                            load_int8_cache,
                            self.int8_cache_name,
                            group_name=group_name,
                        )
                        if load_int8_cache == "no":
                            elastic_server.register_transfer_manifest(model)
                        elastic_server.start()
                        logger.info("Elastic server started, rank: %s, group: %s", device_id, group_name)
                        if is_draft:
                            self._draft_elastic_server = elastic_server
                        else:
                            self._target_elastic_server = elastic_server
                    except Exception as e:
                        logger.error("Failed to start Netloader server for rank: %s, details: %s", device_id, e)
        else:
            logger.info("Skip to start Netloader server")

        if need_process_weights_after_loading:
            process_weights_after_loading(model, model_config, torch.device(device_config.device))
            synchronize_npu(device_config.device_type)
            logger.info("Netloader final process_weights done, rank: %s", device_id)

        if not is_draft:
            self._sync_target_netloader_before_draft(vllm_config)

        if model is None:
            logger.error("NetLoader elastic loads model fails")
            raise RuntimeError("NetLoader elastic loads model fails")

        return model.eval()

    def revert_to_default(self, model_config, vllm_config, device_config, prefix: str = "") -> tuple[nn.Module, bool]:
        """
        Reverts to the default model loading logic when elastic loading fails or is not applicable.

        This method resets the loader's extra config and load format to defaults,
        then delegates model loading to a DefaultModelLoader.
        If quantization is enabled, it will load the model and then run the
        processing of weights (i.e. applying quantization adjustments) before returning.

        Parameters:
        - model_config: Configuration describing model architecture, quantization, etc.
        - vllm_config: Configuration for vLLM (device, parallelism, dtype, etc).
        - device_config: Configuration for the target device (device type, device id, etc).
        - prefix: Module prefix for pipeline parallelism.

        Returns:
        - A tuple (model, need_process_weights_after_loading):
            * model: The loaded `nn.Module` under default loading logic.
            * need_process_weights_after_loading: A boolean flag indicating whether
              weights post-processing (e.g. quantization adjustments) still needs to be applied.
        """
        load_config = deepcopy(self.load_config)
        load_config.model_loader_extra_config = {}
        load_config.load_format = "auto"
        default_model_loader = DefaultModelLoader(load_config)

        if model_config.quantization is None:
            model = default_model_loader.load_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)
            need_process_weights_after_loading = False
        else:
            logger.warning("Quantization is set, netloader use DefaultModelLoader with process_weights_after_loading ")
            need_process_weights_after_loading = True
            target_device = torch.device(device_config.device)
            with set_default_torch_dtype(model_config.dtype):
                with target_device:
                    model = initialize_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)
                default_model_loader.load_weights(model, model_config)
            model = model.eval()

        return model, need_process_weights_after_loading

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        pass
