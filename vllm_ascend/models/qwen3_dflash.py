# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time DFlash alignment with a QuaRot target's hidden-state basis."""

from pathlib import Path

import torch
from vllm.config import VllmConfig
from vllm.logger import logger

from vllm_ascend.models.qwen3_dspark import align_draft_weights
from vllm_ascend.utils import get_rotation_path


def align_dflash_weights(model: torch.nn.Module, vllm_config: VllmConfig) -> None:
    """Align the owned projection and vocab copies once, before any capture.

    The existing DSpark loader applies Q to each input-feature block of fc
    and Q.T to missing target vocab shards. It leaves the target's shared
    layers untouched. This hook changes neither auxiliary-layer selection
    nor Window policy and does no work for a non-QuaRot target.
    Drafts without an auxiliary projection retain their original loading
    path; this hook does not adapt their distinct hidden-state contract.
    """
    rotation_path = get_rotation_path(vllm_config)
    if rotation_path is None:
        return
    if not model.model.use_aux_hidden_state:
        return
    binding = str(Path(rotation_path).resolve())
    previous = getattr(model, "_ascend_dflash_quarot_alignment", None)
    if previous is not None:
        if previous != binding:
            raise ValueError("A DFlash model cannot reuse alignment from a different QuaRot target")
        return
    projection = getattr(model.model, "fc", None)
    if projection is None:
        raise ValueError("DFlash with a QuaRot target requires an auxiliary-hidden-state projection")
    if projection.weight.ndim != 2 or projection.weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("QuaRot DFlash alignment requires an unquantized floating-point fc weight")
    # A failed load discards this draft; never catch and continue with a
    # partially aligned model, or alias/mutate the target's vocab parameters.
    align_draft_weights(model, projection, vllm_config)
    model._ascend_dflash_quarot_alignment = binding
    logger.info("[spec_decode/quarot] Aligned MRV2 DFlash fc and owned vocab layers for %s", binding)
