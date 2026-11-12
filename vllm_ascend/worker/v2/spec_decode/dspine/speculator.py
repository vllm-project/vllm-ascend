# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from vllm.v1.worker.gpu.spec_decode.dspine.speculator import DSpineSpeculator

from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator


class AscendDSpineSpeculator(DSpineSpeculator, AscendDFlash2Speculator):
    """Keep upstream DSpine candidates and reuse the Ascend DFlash2 runtime.

    DSpineSpeculator supplies transition_top_k, tuple-output handling and exact
    transition scores through _prepare_candidates. Cooperative inheritance
    retains Ascend attention metadata, KV preparation, ACL graph management and
    LM head TP guards. The shared CandidateSampler already uses Ascend's
    selector-walk kernel through patch_dflash_speculator.
    """
