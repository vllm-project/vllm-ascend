# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator

if TYPE_CHECKING:
    from vllm_ascend.models.kimi_k3.recoverssm import KimiDSparkRecoverSSMState


class KimiDSparkSpeculator(AscendDSparkSpeculator):
    _recovery: "KimiDSparkRecoverSSMState | None" = None

    @contextmanager
    def sampling_scope(self, model_state) -> Iterator[None]:
        from vllm_ascend.models.kimi_k3.recoverssm import KimiRecoverSSMModelState

        if not isinstance(model_state, KimiRecoverSSMModelState):
            yield
            return
        recovery = model_state.recoverssm
        with recovery.draft_scope():
            self._recovery = recovery
            try:
                yield
            finally:
                self._recovery = None

    def propose(self, *args, **kwargs) -> torch.Tensor:
        draft_tokens = super().propose(*args, **kwargs)
        if self._recovery is not None:
            # The runner can publish drafts and finalize KV transfer on return.
            self._recovery.wait_for_commit()
        return draft_tokens
