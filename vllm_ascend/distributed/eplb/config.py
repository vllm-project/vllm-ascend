# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Ascend EPLB configuration and communicator selection."""

from dataclasses import replace
from typing import Any

import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_eplb_group
from vllm.logger import logger


def check_and_update_eplb_config(parallel_config) -> None:
    config = parallel_config.eplb_config
    if config.policy is None:
        config.policy = "stair"
    if config.policy not in ("default", "stair"):
        raise ValueError(f"Unknown EPLB policy: {config.policy}")
    if config.communicator is None:
        binding, reason = _probe_local_hixl_binding()
        config.communicator = "hixl" if binding != "none" else "torch_gloo"
        _mark_auto_selected(config)
        logger.info(
            "Ascend EPLB provisionally selected %s (%s).", config.communicator, binding if reason is None else reason
        )
    if config.communicator not in ("torch_gloo", "hixl"):
        raise ValueError(f"Unsupported Ascend EPLB communicator: {config.communicator}")


def _probe_local_hixl_binding() -> tuple[str, str | None]:
    """Return the locally usable HIXL binding and why HIXL is unavailable.

    The binding is ``official`` (the ``hixl`` Python package shipped with
    newer CANN distributions), ``ctypes`` (the vllm_ascend fallback that
    drives ``libcann_hixl.so`` directly), or ``none``.
    """
    try:
        import hixl  # type: ignore[import-not-found, import-untyped]

        if hasattr(hixl, "Hixl"):
            return "official", None
        package_reason = "the hixl package has no Hixl binding"
    except ImportError as error:
        package_reason = str(error) or "the hixl package is missing"

    from vllm_ascend.distributed.eplb import hixl_compat

    try:
        hixl_compat.ensure_available()
        return "ctypes", None
    except Exception as error:
        return "none", f"{package_reason}; ctypes fallback: {error}"


_BINDING_CONSENSUS_RANK = {"none": 0, "ctypes": 1, "official": 2}
_STAIR_TRANSFER_LIMIT_KEYS = ("rank_transfer_limit", "cross_node_transfer_limit")


def _group_hixl_binding_consensus() -> str:
    """Reduce the per-rank HIXL bindings to one group-wide decision.

    Returns ``official`` when every rank resolved the official package,
    ``ctypes`` when every rank at least resolved the ctypes fallback,
    ``mixed`` when the ranks resolved different binding classes, and
    ``none`` when at least one rank has no usable HIXL binding. Mixed
    bindings and ``none`` both degrade the whole group to torch_gloo:
    HIXL engines only interoperate over one CANN line protocol, and the
    sender/receiver roles must agree on the transfer path.
    """
    local_binding, local_reason = _probe_local_hixl_binding()
    group = get_eplb_group().cpu_group
    local_rank = _BINDING_CONSENSUS_RANK[local_binding]
    lowest_flag = torch.tensor([local_rank], dtype=torch.int32)
    highest_flag = torch.tensor([local_rank], dtype=torch.int32)
    torch.distributed.all_reduce(lowest_flag, group=group, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(highest_flag, group=group, op=torch.distributed.ReduceOp.MAX)
    lowest, highest = int(lowest_flag.item()), int(highest_flag.item())
    if local_reason is not None and local_binding == "none":
        logger.info("HIXL EPLB unavailable on this rank (%s).", local_reason)
    if lowest == 0:
        return "none"
    if lowest != highest:
        return "mixed"
    return "official" if lowest == 2 else "ctypes"


def _clamp_stair_limits_for_gloo_fallback(ascend_eplb_config) -> Any:
    """Clamp only the STAIR transfer limits the user left unset.

    An automatic Gloo fallback must not override explicitly configured
    limits (for example ``cross_node_transfer_limit: 0`` disables
    cross-node transfers on purpose); unset limits default to unlimited
    and are clamped to one because CPU-staged transfers cannot sustain a
    migration storm.
    """
    vllm_config = get_current_vllm_config()
    additional_config = vllm_config.additional_config if isinstance(vllm_config.additional_config, dict) else {}
    eplb_config = additional_config.get("eplb_config", {})
    stair_config = eplb_config.get("stair_config", {}) if isinstance(eplb_config, dict) else {}
    stair_config = stair_config if isinstance(stair_config, dict) else {}
    unset_keys = [key for key in _STAIR_TRANSFER_LIMIT_KEYS if key not in stair_config]
    if unset_keys:
        stair_config = {**stair_config, **dict.fromkeys(unset_keys, 1)}
        eplb_config = {**eplb_config, "stair_config": stair_config}
        vllm_config.additional_config = {**additional_config, "eplb_config": eplb_config}
        logger.info(
            "Ascend EPLB fell back to torch_gloo; STAIR %s clamped to 1.",
            " and ".join(unset_keys),
        )
    preserved_keys = [key for key in _STAIR_TRANSFER_LIMIT_KEYS if key not in unset_keys]
    if preserved_keys:
        logger.warning(
            "Ascend EPLB fell back to torch_gloo; keeping the explicitly configured STAIR %s.",
            " and ".join(preserved_keys),
        )
    clamped_fields = {key: 1 for key in unset_keys}
    if clamped_fields:
        return replace(ascend_eplb_config.stair_config, **clamped_fields)
    return ascend_eplb_config.stair_config


# Provisional auto-selection marker set on an EPLBConfig instance by the
# ParallelConfig post-init hook (config stage, per-process probe). The
# worker-stage consensus in resolve_ascend_eplb_communicator() consumes it:
# auto-selected communicators are re-validated group-wide, explicit user
# settings are never touched.
_AUTO_SELECTED_ATTRIBUTE = "_vllm_ascend_eplb_auto_selected"


def _mark_auto_selected(eplb_config) -> None:
    eplb_config.__dict__.pop(_AUTO_SELECTED_ATTRIBUTE, None)
    eplb_config.__dict__[_AUTO_SELECTED_ATTRIBUTE] = True


def _clear_auto_selected(eplb_config) -> None:
    eplb_config.__dict__.pop(_AUTO_SELECTED_ATTRIBUTE, None)


def resolve_ascend_eplb_communicator(parallel_config, ascend_eplb_config) -> Any:
    """Confirm or correct the provisional communicator by group consensus.

    Called once per rank before the EPLB policy and state are built, when
    the EPLB group exists but neither the policy nor the communicator has
    been constructed. The config stage already set a provisional decision
    (marked auto-selected) from the local probe; this consensus re-validates
    it group-wide because HIXL engines only interoperate over one CANN line
    protocol and the transfer path must agree on every rank. An explicit
    ``communicator`` setting (no marker) skips the consensus entirely.
    Falls back to torch_gloo — clamping the STAIR transfer limits the user
    left unset — when any rank lacks a usable HIXL binding or when the
    binding classes differ between ranks.
    """
    eplb_config = parallel_config.eplb_config
    auto_selected = bool(getattr(eplb_config, _AUTO_SELECTED_ATTRIBUTE, False))
    if eplb_config.communicator is not None and not auto_selected:
        return ascend_eplb_config.stair_config
    _clear_auto_selected(eplb_config)
    binding = _group_hixl_binding_consensus()
    if binding in ("official", "ctypes"):
        if eplb_config.communicator != "hixl":
            logger.info(
                "Ascend EPLB consensus corrected communicator %s -> hixl: every EPLB rank resolved the %s binding.",
                eplb_config.communicator,
                binding,
            )
        eplb_config.communicator = "hixl"
        logger.info("Ascend EPLB selected hixl: every EPLB rank resolved the %s binding.", binding)
        return ascend_eplb_config.stair_config
    eplb_config.communicator = "torch_gloo"
    if binding == "mixed":
        logger.info("Ascend EPLB selected torch_gloo: HIXL binding classes differ between EPLB ranks.")
    else:
        logger.info("Ascend EPLB selected torch_gloo: HIXL is unavailable on at least one EPLB rank.")
    return _clamp_stair_limits_for_gloo_fallback(ascend_eplb_config)
