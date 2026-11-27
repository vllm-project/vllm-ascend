# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install the Ascend RecoverSSM commit backends and the deferred machinery.

The stock commit context is swapped for the Ascend Triton one, and GLM
RecoverSSM model states are upgraded to the deferred variant: accepted-token
commits are delayed to the next verify step, which replays the pending
records in-kernel (see ``recoverssm_deferred``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm_ascend.models.glm5next.recoverssm_config import is_glm_recoverssm_model, use_glm_recoverssm

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

# Set once the deferred machinery is installed.
_deferred_installed = False


def patch_kda_recoverssm(vllm_config: VllmConfig) -> None:
    """Swap the RecoverSSM commit context for the Ascend one and hook the
    model-state init that installs the deferred machinery. Idempotent."""
    if not use_glm_recoverssm(vllm_config):
        return

    global _deferred_installed
    if _deferred_installed:
        return

    import vllm.models.kimi_k3.nvidia.ops.recoverssm as recoverssm

    from vllm_ascend.ops.triton.kda.recoverssm import KDARecoverSSMCommitContext

    recoverssm.KDARecoverSSMCommitContext = KDARecoverSSMCommitContext

    _hook_model_state_init()
    _deferred_installed = True


def _hook_model_state_init() -> None:
    """Wrap ``model_runner.init_model_state`` to upgrade GLM RecoverSSM
    states to the deferred variant after construction.

    ``patch_model_state`` binds the Ascend init onto the runner by value
    during ``adapt_patch``, which runs before this hook installs, so
    rebinding the ``model_states`` module attribute would never reach the
    runner. Wrap the runner binding instead.
    """
    import vllm.v1.worker.gpu.model_runner as model_runner

    from vllm_ascend.models.glm5next.recoverssm_deferred import install_model_state

    base_init = model_runner.init_model_state

    def init_model_state(
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> Any:
        state = base_init(vllm_config, model, encoder_cache, device)
        # install_model_state itself skips states without a recoverssm slot.
        if is_glm_recoverssm_model(vllm_config):
            install_model_state(state)
        return state

    model_runner.init_model_state = init_model_state
