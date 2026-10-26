# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SketchSSM of Mamba-2 layers (``MambaMixer2``) on Ascend NPU.

Same public surface as ``Mamba2SketchSSM``, with the Triton kernels of
``sketchssm_mamba2_npu*.py`` in place of the CUDA ones.
"""

from enum import Enum
from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.mamba.sketchssm.calibration import (
    SketchSSMCalibration,
    load_sketchssm_calibration,
)

from vllm_ascend.ops.triton.mamba.sketchssm_mamba2_npu import (
    SKETCH_DTYPES,
    SketchArgs,
    SketchTables,
    sketch_build,
    sketch_rotate_states,
    sketch_shapes,
)
from vllm_ascend.ops.triton.mamba.sketchssm_mamba2_npu_triton import (
    sketch_triton_decode,
)

if TYPE_CHECKING:
    from vllm.config import CacheConfig
    from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata


class Backend(str, Enum):
    CUDA = "cuda"
    TRITON = "triton"


def select_backend(
    *, device_type: str, cuda_kernels_available: bool = False
) -> Backend:
    """The decode backend of a Mamba-2 layer on ``device_type``.

    Only the Triton kernels are vendored here, so Ascend NPU always decodes
    with Triton.
    """
    if device_type == "cuda" and cuda_kernels_available:
        return Backend.CUDA
    return Backend.TRITON


class Mamba2SketchSSMNPU(torch.nn.Module):
    """SketchSSM frames, sketch and decode kernels of one Mamba-2 NPU layer."""

    def __init__(
        self,
        calibration: SketchSSMCalibration,
        layer: int,
        num_heads: int,
        head_dim: int,
        n_groups: int,
        state_size: int,
        max_num_reqs: int,
    ):
        super().__init__()
        # The kernels take state_size as a power-of-two block and use it and
        # head_dim as tl.dot dimensions, which must be at least 16.
        if state_size < 16 or state_size & (state_size - 1):
            raise ValueError(
                "SketchSSM requires a Mamba-2 state size that is a power of two "
                f"and at least 16, got {state_size}"
            )
        if head_dim < 16:
            raise ValueError(
                f"SketchSSM requires a Mamba-2 head dim of at least 16, got {head_dim}"
            )
        frames = calibration.frames[layer]
        ranks = calibration.ranks[layer]
        if frames.shape != (n_groups, state_size, state_size):
            raise ValueError(
                f"SketchSSM frames {tuple(frames.shape)} do not match "
                f"{n_groups} groups of state size {state_size}"
            )
        if ranks.shape != (num_heads,):
            raise ValueError(
                f"SketchSSM ranks cover {ranks.numel()} of {num_heads} heads"
            )
        self.register_buffer(
            "frames_t",
            frames.transpose(-1, -2)
            .contiguous()
            .to(torch.get_default_device(), torch.float32),
            persistent=False,
        )
        self.tables = SketchTables(ranks, state_size)
        # One sketch per concurrent request, indexed by its persistent index.
        shapes = sketch_shapes(ranks, head_dim, state_size)
        for name, shape, dtype in zip(("u", "w", "ag"), shapes, SKETCH_DTYPES):
            self.register_buffer(
                name, torch.zeros(max_num_reqs, *shape, dtype=dtype), persistent=False
            )
        # No CUDA SketchSSM kernels are vendored for this layer to select.
        backend = select_backend(device_type=torch.get_default_device().type)
        if backend is not Backend.TRITON:
            raise ValueError("SketchSSM NPU layers decode with the Triton kernels")
        self._decode = sketch_triton_decode

    @property
    def sketch(self) -> SketchArgs:
        return SketchArgs(self.u, self.w, self.ag, self.tables, self.frames_t)

    def rotate_states(
        self, states: torch.Tensor, inverse: bool = False
    ) -> torch.Tensor:
        """Prefill states ``(items, heads, head_dim, state_size)`` FP32 into the
        rotated frame of the cache (back to the model frame if ``inverse``)."""
        return sketch_rotate_states(states, self.frames_t, inverse)

    def decode(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        x_cache: torch.Tensor,
        dt_cache: torch.Tensor,
        B_cache: torch.Tensor,
        state_indices: torch.Tensor,
        attn_metadata: "Mamba2AttentionMetadata",
        out: torch.Tensor,
    ) -> None:
        """One decode step with unrotated B and C; arguments as for
        ``selective_state_update_replayssm_output_only``."""
        if state_indices.dim() == 2:
            state_indices = state_indices[:, 0]
        # The kernels index rows contiguously; a column slice is strided.
        state_indices = state_indices.contiguous()
        self._decode(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            x_cache,
            dt_cache,
            B_cache,
            attn_metadata.bc_pre_scratch,
            attn_metadata.write_pos_d,
            attn_metadata.is_flush_d,
            attn_metadata.sketch_flush_rows_d,
            state_indices,
            attn_metadata.sketch_indices_d,
            out,
            self.sketch,
        )

    def prefilled(
        self,
        state: torch.Tensor,
        state_indices: torch.Tensor,
        attn_metadata: "Mamba2AttentionMetadata",
    ) -> None:
        """Build the sketch of each prefill row that completes its prompt."""
        sketch_build(
            state,
            attn_metadata.sketch_build_p,
            state_indices.contiguous(),
            attn_metadata.sketch_indices_p,
            self.sketch,
        )

    def warmup(self, state: torch.Tensor) -> None:
        """Compile the sketch build, which first runs when a prompt completes.

        ``state`` is the layer's SSM state cache; no row is built.
        """
        rows = torch.zeros(1, dtype=torch.int32, device=state.device)
        sketch_build(
            state.transpose(-1, -2), rows.to(torch.int8), rows, rows, self.sketch
        )

    @classmethod
    def from_config(
        cls,
        cache_config: "CacheConfig",
        model_layer_idx: int,
        num_heads: int,
        head_dim: int,
        n_groups: int,
        state_size: int,
        max_num_reqs: int,
        tp_size: int = 1,
        tp_rank: int = 0,
    ) -> "Mamba2SketchSSMNPU":
        """The SketchSSM of the model layer ``model_layer_idx``."""
        assert cache_config.sketchssm is not None
        if tp_size != 1:
            raise ValueError("SketchSSM on NPU does not support tensor parallelism")
        calibration = load_sketchssm_calibration(
            cache_config.sketchssm,
            cache_config.sketchssm_mean_rank,
            cache_config.replayssm_buffer_len,
        )
        return cls(
            calibration,
            calibration.layer_index(model_layer_idx),
            num_heads,
            head_dim,
            n_groups,
            state_size,
            max_num_reqs,
        )
