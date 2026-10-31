# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_ascend.models.glm5next.recoverssm_config import use_glm_recoverssm


def patch_kda_recoverssm(vllm_config):
    if not use_glm_recoverssm(vllm_config):
        return

    import vllm.models.kimi_k3.nvidia.ops.recoverssm as recoverssm

    from vllm_ascend.ops.triton.kda.recoverssm import KDARecoverSSMCommitContext

    recoverssm.KDARecoverSSMCommitContext = KDARecoverSSMCommitContext
