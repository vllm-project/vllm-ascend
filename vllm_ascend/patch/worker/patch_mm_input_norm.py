# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Backport vLLM #62767 when fused image normalization is available."""

from functools import wraps


def patch_mm_input_norm() -> None:
    # vLLM v0.30.0 predates this module. Keep the release worker importable.
    try:
        from vllm.model_executor.layers.fusion.mm_input_norm import FusedMMInputNorm
    except ModuleNotFoundError as exc:
        if exc.name not in (
            "vllm.model_executor.layers.fusion",
            "vllm.model_executor.layers.fusion.mm_input_norm",
        ):
            raise
        return

    original_init = FusedMMInputNorm.__init__
    if getattr(original_init, "_ascend_nonpersistent_norm", False):
        return

    @wraps(original_init)
    def init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        # These constants come from the processor, not the checkpoint. Keep
        # their values/device while excluding them from sharded/dummy loading.
        self.register_buffer("weight", self.weight, persistent=False)
        self.register_buffer("bias", self.bias, persistent=False)

    init._ascend_nonpersistent_norm = True  # type: ignore[attr-defined]
    FusedMMInputNorm.__init__ = init


patch_mm_input_norm()
