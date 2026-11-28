# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Keep vLLM v0.30 image-normalization constants out of model weights."""

from functools import wraps

from vllm.model_executor.models.vision import FusedInputNorm


def patch_mm_input_norm() -> None:
    original_init = FusedInputNorm.__init__
    if getattr(original_init, "_ascend_nonpersistent_norm", False):
        return

    @wraps(original_init)
    def init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        # These constants come from the processor, not the checkpoint. Keep
        # their values/device while excluding them from sharded/dummy loading
        # and preserving them when layerwise weight updates restore buffers.
        self.register_buffer("weight", self.weight, persistent=False)
        self.register_buffer("bias", self.bias, persistent=False)

    init._ascend_nonpersistent_norm = True  # type: ignore[attr-defined]
    FusedInputNorm.__init__ = init


patch_mm_input_norm()
