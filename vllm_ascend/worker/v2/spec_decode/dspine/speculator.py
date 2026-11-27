# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from vllm.v1.worker.gpu.spec_decode.dspine.speculator import DSpineSpeculator

from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator


class AscendDSpineSpeculator(DSpineSpeculator, AscendDFlash2Speculator):
    """Use upstream DSpine candidate preparation with the Ascend DFlash2 runtime."""
