# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations
from vllm_ascend.models.glm5next.recoverssm_config import use_glm_recoverssm

# Set True once the deferred machinery is installed.
_deferred_installed = False

def patch_kda_recoverssm(vllm_config):
    if not use_glm_recoverssm(vllm_config):
        return

    global _deferred_installed
    if _deferred_installed:
        return

    # Immediate path: swap the stock commit context for the Ascend one.
    import vllm.models.kimi_k3.nvidia.ops.recoverssm as recoverssm
    from vllm_ascend.ops.triton.kda.recoverssm import KDARecoverSSMCommitContext

    recoverssm.KDARecoverSSMCommitContext = KDARecoverSSMCommitContext

    _hook_model_state_init()
    _deferred_installed = True

def _hook_model_state_init() -> None:

    import vllm_ascend.worker.v2.model_states as model_states_mod
    from vllm_ascend.models.glm5next.recoverssm_config import is_glm_recoverssm_model
    from vllm_ascend.models.glm5next.recoverssm_deferred import install_model_state

    base_init = model_states_mod.init_asecnd_model_state

    def init_asecnd_model_state(vllm_config, model, encoder_cache, device):
        state = base_init(vllm_config, model, encoder_cache, device)

        if not is_glm_recoverssm_model(vllm_config):
            return state
        if not use_glm_recoverssm(vllm_config):
            return state
        if getattr(state, "recoverssm", None) is None:
            return state

        install_model_state(state)

        language_model = model
        if hasattr(model, "get_language_model"):
            language_model = model.get_language_model()
        layers = getattr(language_model, "layers", None)
        start = getattr(language_model, "start_layer", 0)
        end = getattr(language_model, "end_layer", None)
        if layers is not None:
            decoder_layers = layers[start:end] if end is not None else layers
            for decoder in decoder_layers:
                attn = getattr(decoder, "self_attn", None)
                if attn is None:
                    continue
                if not getattr(attn, "use_recoverssm", False):
                    continue
                if getattr(attn, "recoverssm_backend", "triton") != "fp32_deferred":
                    continue
                attn.deferred_recoverssm = True

        return state

    model_states_mod.init_asecnd_model_state = init_asecnd_model_state