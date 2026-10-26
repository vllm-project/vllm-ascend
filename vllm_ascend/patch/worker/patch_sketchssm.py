# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route Mamba-2 SketchSSM to the Ascend NPU implementation.

``MambaMixer2`` imports ``Mamba2SketchSSM`` into its own module namespace and
calls ``Mamba2SketchSSM.from_config`` there, so rebinding that name makes every
``--sketchssm`` layer of this worker an NPU layer. The state shapes are the
key-major ones #61124 already computes.

The Ascend attention-metadata builder spreads
``ModelSpecificAttnMetadata.get_extra_common_attn_kwargs`` into the common
metadata, which is how ``AscendMambaHybridModelState`` hands over the
persistent request indices it records per step. vLLM's SketchSSM row helpers
read them as ``req_idx``, so forward the recorded indices here.

The layer these names describe ships with vllm#61124. On a tree that does not
have it this module is inert: nothing from the SketchSSM package is imported
and no upstream name is rebound, so the rest of vllm-ascend keeps working.
"""

from typing import Any

from vllm.logger import logger
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridAttnMetadata

_get_extra_common_attn_kwargs = MambaHybridAttnMetadata.get_extra_common_attn_kwargs


def _patched_get_extra_common_attn_kwargs(
    self: MambaHybridAttnMetadata, kv_cache_group_id: int, num_reqs: int
) -> dict[str, Any]:
    kwargs = _get_extra_common_attn_kwargs(self, kv_cache_group_id, num_reqs)
    req_idx = getattr(self, "_sketchssm_req_idx", None)
    if req_idx is not None:
        kwargs["req_idx"] = req_idx[:num_reqs]
    return kwargs


def _apply() -> None:
    """Rebind the core SketchSSM layer, or no-op when #61124 is absent."""
    try:
        from vllm.model_executor.layers.mamba import mamba_mixer2
        from vllm.model_executor.layers.mamba.sketchssm import calibration  # noqa: F401
    except ImportError:
        logger.info_once(
            "vllm.model_executor.layers.mamba.sketchssm is not available in this "
            "vLLM tree (vllm#61124); Mamba-2 SketchSSM stays on the core layer.",
            scope="process",
        )
        return

    from vllm_ascend.ops.sketchssm_mamba2_npu import Mamba2SketchSSMNPU

    mamba_mixer2.Mamba2SketchSSM = Mamba2SketchSSMNPU  # type: ignore[misc]
    MambaHybridAttnMetadata.get_extra_common_attn_kwargs = _patched_get_extra_common_attn_kwargs


_apply()
