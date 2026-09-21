# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.lilicorr.speculator import (
    LiLiCorrSpeculator,
)

from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)


class AscendLiLiCorrSpeculator(LiLiCorrSpeculator, AscendDFlashSpeculator):
    """LiLiCorr drafter using the shared Ascend DFlash runtime."""

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # LiLiCorr shares DFlash2's candidate walk. Until that NPU Triton
        # kernel is ACL-graph capturable, keep the draft eager while allowing
        # the target model to retain its independently configured graph mode.
        if self.speculative_config.enforce_eager:
            cudagraph_mode = CUDAGraphMode.NONE
        else:
            raise NotImplementedError(
                "LiLiCorr does not currently support graph mode on Ascend; please enable enforce_eager."
            )
        super().init_cudagraph_manager(cudagraph_mode)
