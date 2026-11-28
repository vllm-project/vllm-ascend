# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Validation for the Ascend EPLB configuration.

``additional_config.eplb_config`` is validated once per engine. The checks live
with the EPLB feature so ``vllm_ascend.platform`` only keeps the platform hook
that calls them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import logger

from vllm_ascend.ascend_config import EplbConfig, StairConfig, is_draft_config_context

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Deployment knobs that only make sense with --enable-eplb.
DEPLOYMENT_EPLB_FIELDS = frozenset({"mode", "expert_placement_path"})
_EPLB_CONFIG_FIELDS = frozenset(EplbConfig.__dataclass_fields__)
_STAIR_CONFIG_FIELDS = frozenset(StairConfig.__dataclass_fields__)
_KV_ROLE_TO_PLACEMENT_SCOPE = {"kv_producer": "prefill", "kv_consumer": "decode", "kv_both": "mixed"}
_PHASE_TO_PLACEMENT_SCOPE = {"all": "mixed", "prefill": "prefill", "decode": "decode"}


def resolve_placement_scope(kv_role: object, load_collection_phase: str) -> str:
    """Infer which P/D traffic a deployment describes.

    The scope is recorded in the deployment file so a placement tuned on one
    role is never installed on the other. A disaggregated engine knows its role
    from its KV transfer config; a collocated engine falls back to the phase it
    collects load on.
    """
    if kv_role is not None:
        scope = _KV_ROLE_TO_PLACEMENT_SCOPE.get(str(kv_role))
        if scope is not None:
            return scope
    try:
        return _PHASE_TO_PLACEMENT_SCOPE[load_collection_phase]
    except KeyError:
        raise ValueError(f"load_collection_phase {load_collection_phase!r} has no EPLB placement scope") from None


def validate_eplb_config(vllm_config: VllmConfig) -> None:
    """Reject EPLB combinations that the Ascend runtime cannot serve."""
    if vllm_config.parallel_config.enable_eplb:
        offload = vllm_config.offload_config
        if offload.uva.cpu_offload_gb > 0 or offload.prefetch.offload_group_size > 0:
            raise ValueError(
                "EPLB is incompatible with weight offload/prefetch on Ascend. "
                "Set cpu_offload_gb=0 and offload_group_size=0, or disable EPLB."
            )
    additional_config = vllm_config.additional_config or {}
    eplb_config = additional_config.get("eplb_config", {})
    if not isinstance(eplb_config, dict):
        raise TypeError("additional_config.eplb_config must be a dictionary.")

    mode = eplb_config.get("mode", "dynamic")
    if mode not in ("dynamic", "static"):
        raise ValueError("EPLB mode must be 'dynamic' or 'static'")
    expert_placement_path = eplb_config.get("expert_placement_path")
    if mode == "static":
        if not vllm_config.parallel_config.enable_eplb:
            raise ValueError("Static EPLB requires --enable-eplb")
        if not expert_placement_path:
            raise ValueError("Static EPLB requires eplb_config.expert_placement_path")
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("Static EPLB currently requires PP=1")
        if eplb_config.get("load_collection_phase", "all") != "all":
            raise ValueError("Static EPLB does not collect load; load_collection_phase must be 'all'")
    elif expert_placement_path is not None:
        raise ValueError("eplb_config.expert_placement_path requires mode='static'")

    unsupported_eplb_fields = sorted(set(eplb_config) - _EPLB_CONFIG_FIELDS)
    stair_config = eplb_config.get("stair_config")
    unsupported_stair_fields = (
        sorted(set(stair_config) - _STAIR_CONFIG_FIELDS) if isinstance(stair_config, dict) else []
    )
    if unsupported_eplb_fields or unsupported_stair_fields:
        ignored_fields = unsupported_eplb_fields + [f"stair_config.{field}" for field in unsupported_stair_fields]
        logger.warning(
            "EPLB currently supports only these parameters: "
            "additional_config.eplb_config fields %s; stair_config fields %s. "
            "The following parameters are unsupported and will be ignored: %s.",
            sorted(_EPLB_CONFIG_FIELDS),
            sorted(_STAIR_CONFIG_FIELDS),
            ignored_fields,
        )
        eplb_config = {key: value for key, value in eplb_config.items() if key in _EPLB_CONFIG_FIELDS}
        if isinstance(stair_config, dict):
            eplb_config["stair_config"] = {
                key: value for key, value in stair_config.items() if key in _STAIR_CONFIG_FIELDS
            }
        additional_config = dict(additional_config)
        additional_config["eplb_config"] = eplb_config
        vllm_config.additional_config = additional_config
    load_collection_phase = eplb_config.get("load_collection_phase", "all")
    # Disabled drafts inherit this target-only setting but never collect
    # EPLB load. Keep target validation strict and preserve other checks.
    if load_collection_phase != "all" and not vllm_config.parallel_config.enable_eplb and not is_draft_config_context():
        raise ValueError("additional_config.eplb_config.load_collection_phase requires --enable-eplb.")
    if vllm_config.parallel_config.enable_eplb:
        upstream_eplb_config = vllm_config.parallel_config.eplb_config
        if upstream_eplb_config.communicator not in (None, "torch_gloo", "hixl"):
            raise ValueError(
                "Async EPLB on Ascend requires the torch_gloo or hixl communicator "
                f"but got {upstream_eplb_config.communicator!r}. "
                "Set eplb_config.communicator to 'torch_gloo' or 'hixl'."
            )
        if not upstream_eplb_config.use_async:
            logger.warning(
                "Synchronous EPLB is not supported on Ascend; "
                "parameter=eplb_config.use_async, value=False, "
                "action: forcing asynchronous EPLB."
            )
            upstream_eplb_config.use_async = True
        if vllm_config.parallel_config.enable_elastic_ep:
            raise ValueError("Async EPLB is not supported with elastic EP on Ascend.")
    elif DEPLOYMENT_EPLB_FIELDS & eplb_config.keys():
        raise ValueError(
            "additional_config.eplb_config mode and expert_placement_path require --enable-eplb."
        )
